#!/usr/bin/env python3
"""Run SWE-Interact mechanics with the release's exact GPT-5.5 user model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import re
import sys
import time
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.strict_rows import parse_keyed_pipe_rows
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import swe_interact_mechanics_serving as v1
from scripts.swe_interact_source_audit import (
    EXPECTED_COMMIT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TREE,
    sha256_bytes,
)


INTERFACE_VERSION = "swe-interact-gpt55-mechanics-serving-2"
USER_MODEL_ID = "openai/gpt-5.5"
JUDGE_MODEL_ID = "openai/gpt-5.4-mini"
SEED = 24424


def judge_messages(
    task: v1.MechanicsTask,
    initial_reply: str,
    branch_replies: dict[str, str],
) -> list[dict[str, str]]:
    catalog = "\n".join(
        f"{index}|{requirement}"
        for index, requirement in enumerate(task.requirements, start=1)
    )
    responses = "\n\n".join(
        [
            f"[INITIAL]\n{initial_reply}",
            *(
                f"[{key}]\n{branch_replies[key]}"
                for key in v1.JUDGE_KEYS
                if key != "INITIAL"
            ),
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an exact semantic annotation function. The requirement catalog "
                "is private. For each keyed maintainer response, return the decimal "
                "indexes whose concrete content the maintainer asserts, confirms, or "
                "corrects. For non-INITIAL rows, count only information newly disclosed "
                "beyond INITIAL. Do not count a detail merely because it appeared in the "
                "assistant's question. A maintainer yes/no that confirms a concrete "
                "assistant proposal does count. Generic approval or deferral is NONE. "
                "Return exactly one row per requested key and no other text as KEY|NONE "
                "or KEY|comma-separated decimal indexes in ascending numeric order. "
                "Never prefix an index with a letter."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TASK_ID\n{task.task_id}\n\n"
                f"REQUIREMENT_CATALOG\n{catalog}\n\n"
                f"MAINTAINER_RESPONSES\n{responses}\n\n"
                f"OUTPUT_KEYS\n{','.join(v1.JUDGE_KEYS)}"
            ),
        },
    ]


def parse_numeric_requirement_set(
    value: str,
    requirement_count: int,
) -> frozenset[str]:
    if value == "NONE":
        return frozenset()
    parts = value.split(",")
    if not parts or any(re.fullmatch(r"[1-9][0-9]*", part) is None for part in parts):
        raise ValueError("Numeric requirement annotation is noncanonical")
    indexes = [int(part) for part in parts]
    if indexes != sorted(indexes):
        raise ValueError("Numeric requirement annotation is not sorted")
    if len(indexes) != len(set(indexes)):
        raise ValueError("Numeric requirement annotation repeats an index")
    if any(index < 1 or index > requirement_count for index in indexes):
        raise ValueError("Numeric requirement annotation has an unknown index")
    return frozenset(f"R{index}" for index in indexes)


def parse_judgement(
    text: str,
    requirement_count: int,
) -> dict[str, frozenset[str]]:
    rows = parse_keyed_pipe_rows(
        text,
        expected_keys=v1.JUDGE_KEYS,
        value_fields=1,
    )
    return {
        key: parse_numeric_requirement_set(rows[key][0], requirement_count)
        for key in v1.JUDGE_KEYS
    }


def build_models(config: Config) -> tuple[v1.ChatModel, v1.ChatModel]:
    if len(config.model_pairs) != 1:
        raise ValueError("SWE-Interact V2 requires exactly one model pair")
    pair = config.model_pairs[0]
    if pair.questioner.backend != "openrouter" or pair.questioner.model != USER_MODEL_ID:
        raise ValueError("Native released-user model/config changed")
    if pair.answerer.backend != "openrouter" or pair.answerer.model != JUDGE_MODEL_ID:
        raise ValueError("Semantic judge model/config changed")
    user_spec = replace(
        pair.questioner,
        thinking=None,
        reasoning_effort="high",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    judge_spec = replace(
        pair.answerer,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    return (
        build_model_adapter(user_spec, config),
        build_model_adapter(judge_spec, config),
    )


class DeterministicFixtureModel(v1.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if not self.judge:
            return super().chat_complete_messages_batched(
                batch_messages,
                temperature,
                block_size,
                max_new_tokens,
            )
        del temperature, block_size, max_new_tokens
        self.requests += len(batch_messages)
        outputs = []
        for messages in batch_messages:
            prompt = messages[-1]["content"]
            if "rf_task-694b4b99829f00e24fd118a1" in prompt:
                root_a, root_b = "1", "5"
            elif "swebenchpro_instance_qutebrowser" in prompt:
                root_a, root_b = "2", "5"
            else:
                root_a, root_b = "2", "4"
            outputs.append(
                "\n".join(
                    (
                        "INITIAL|1",
                        "GENERIC|NONE",
                        f"ROOT_A_1|{root_a}",
                        f"ROOT_A_2|{root_a}",
                        f"ROOT_B_1|{root_b}",
                        f"ROOT_B_2|{root_b}",
                        f"REVIEW_A|{root_a}",
                        f"REVIEW_B|{root_b}",
                    )
                )
            )
        return outputs


def run_serving(
    tasks: tuple[v1.MechanicsTask, ...],
    *,
    user_model: v1.ChatModel,
    judge_model: v1.ChatModel,
    raw_path: Path,
) -> dict[str, Any]:
    started = time.monotonic()
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "source_commit": EXPECTED_COMMIT,
        "seed": SEED,
        "initial": {},
        "branches": {},
        "judgements": {},
    }
    initial_outputs = user_model.chat_complete_messages_batched(
        [v1.initial_messages(task) for task in tasks],
        temperature=0.0,
        block_size=len(tasks),
        max_new_tokens=4096,
    )
    if len(initial_outputs) != len(tasks):
        raise ValueError("Initial GPT-5.5 batch has wrong response count")
    for task, output in zip(tasks, initial_outputs, strict=True):
        raw["initial"][task.task_id] = {
            "messages": v1.initial_messages(task),
            "response": output,
        }
    v1.checkpoint_private(raw_path, raw)

    branch_order = [
        (task, branch)
        for task in tasks
        for branch in v1.JUDGE_KEYS
        if branch != "INITIAL"
    ]
    messages_batch = [
        v1.branch_messages(task, raw["initial"][task.task_id]["response"], branch)
        for task, branch in branch_order
    ]
    branch_outputs = user_model.chat_complete_messages_batched(
        messages_batch,
        temperature=0.0,
        block_size=len(messages_batch),
        max_new_tokens=4096,
    )
    if len(branch_outputs) != len(branch_order):
        raise ValueError("Branch GPT-5.5 batch has wrong response count")
    for (task, branch), messages, output in zip(
        branch_order,
        messages_batch,
        branch_outputs,
        strict=True,
    ):
        raw["branches"].setdefault(task.task_id, {})[branch] = {
            "messages": messages,
            "response": output,
        }
    v1.checkpoint_private(raw_path, raw)

    judge_batch = []
    for task in tasks:
        branch_replies = {
            branch: raw["branches"][task.task_id][branch]["response"]
            for branch in v1.JUDGE_KEYS
            if branch != "INITIAL"
        }
        judge_batch.append(
            judge_messages(
                task,
                raw["initial"][task.task_id]["response"],
                branch_replies,
            )
        )
    judge_outputs = judge_model.chat_complete_messages_batched(
        judge_batch,
        temperature=0.0,
        block_size=len(judge_batch),
        max_new_tokens=1024,
    )
    if len(judge_outputs) != len(tasks):
        raise ValueError("V2 judge batch has wrong response count")
    for task, messages, output in zip(
        tasks,
        judge_batch,
        judge_outputs,
        strict=True,
    ):
        raw["judgements"][task.task_id] = {
            "messages": messages,
            "response": output,
        }
    v1.checkpoint_private(raw_path, raw)

    all_outputs = [
        *(row["response"] for row in raw["initial"].values()),
        *(
            row["response"]
            for task_rows in raw["branches"].values()
            for row in task_rows.values()
        ),
        *(row["response"] for row in raw["judgements"].values()),
    ]
    labels = {
        task.task_id: parse_judgement(
            raw["judgements"][task.task_id]["response"],
            len(task.requirements),
        )
        for task in tasks
    }
    result = v1.build_result(
        tasks,
        labels,
        user_model.usage_snapshot(),
        judge_model.usage_snapshot(),
        all_replies_nonempty=all(
            isinstance(output, str) and bool(output.strip())
            for output in all_outputs
        ),
        private_raw_sha256=sha256_bytes(raw_path.read_bytes()),
        elapsed_seconds=time.monotonic() - started,
    )
    result["protocol"].update(
        {
            "interface_version": INTERFACE_VERSION,
            "user_model": USER_MODEL_ID,
            "user_reasoning_effort": "high",
            "judge_model": JUDGE_MODEL_ID,
            "judge_id_grammar": "bare_decimal",
            "seed": SEED,
        }
    )
    public = json.dumps(result, sort_keys=True)
    if any(
        requirement in public
        for task in tasks
        for requirement in task.requirements
        if requirement
    ):
        raise ValueError("V2 public result leaks private requirement text")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("results/nonmyopic/swe_interact_release_manifest.json"),
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("results/nonmyopic/swe_interact_gpt55_mechanics"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.run_id = args.run_id
    run_dir = args.output_root / args.run_id
    config.log_path = run_dir / "run.log"
    raw_path = args.output_root / "private" / args.run_id / "RAW.json"
    tasks = v1.load_tasks(args.source_repo.resolve(), args.manifest.resolve())
    if args.dry_run:
        user_model: v1.ChatModel = DeterministicFixtureModel(judge=False)
        judge_model: v1.ChatModel = DeterministicFixtureModel(judge=True)
    else:
        user_model, judge_model = build_models(config)
    try:
        result = run_serving(
            tasks,
            user_model=user_model,
            judge_model=judge_model,
            raw_path=raw_path,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "private_raw_sha256": (
                sha256_bytes(raw_path.read_bytes()) if raw_path.exists() else None
            ),
            "user_usage": user_model.usage_snapshot(),
            "judge_usage": judge_model.usage_snapshot(),
        }
        v1.write_json(run_dir / "FAILURE.json", failure)
        raise
    v1.write_json(run_dir / "SERVING.json", result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["usage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
