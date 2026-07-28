#!/usr/bin/env python3
"""Run the frozen SWE-Interact user-simulator mechanics serving gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import re
from pathlib import Path
import sys
import time
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.strict_rows import parse_keyed_pipe_rows
from helpers import Config, ModelSpec, load_config
from model_factory import build_model_adapter
from scripts.swe_interact_source_audit import (
    EXPECTED_COMMIT,
    EXPECTED_MANIFEST_SHA256,
    EXPECTED_TREE,
    TASK_ANNOTATIONS,
    _load_manifest,
    git_output,
    sha256_bytes,
)


INTERFACE_VERSION = "swe-interact-mechanics-serving-1"
USER_MODEL_ID = "openai/gpt-5.4"
JUDGE_MODEL_ID = "openai/gpt-5.4-mini"
SEED = 24423
EXPECTED_USER_REQUESTS = 24
EXPECTED_JUDGE_REQUESTS = 3
MAX_TRANSPORT_RETRIES = 3
MAX_COST_USD = 4.0
JUDGE_KEYS = (
    "INITIAL",
    "GENERIC",
    "ROOT_A_1",
    "ROOT_A_2",
    "ROOT_B_1",
    "ROOT_B_2",
    "REVIEW_A",
    "REVIEW_B",
)


TASK_PROBES = {
    "deepswe_clack-async-autocomplete-options": {
        "root_a": (
            "For async option resolvers, how should I detect that the result is "
            "asynchronous, and should that detection invocation also be the first fetch?"
        ),
        "root_b": (
            "When option requests overlap, what should happen to the earlier "
            "request and any result that resolves after a newer one?"
        ),
        "review_a": (
            "The implementation detects async mode from function arity and constructor "
            "metadata before invoking the resolver, then invokes it again for the first fetch."
        ),
        "review_b": (
            "The implementation leaves earlier requests running and applies every result "
            "as it resolves, even after a newer request has started."
        ),
        "root_a_expected": {"R2"},
        "root_b_expected": {"R4"},
        "review_a_expected": {"R2"},
        "review_b_expected": {"R4"},
    },
    "rf_task-694b4b99829f00e24fd118a1": {
        "root_a": (
            "Should Packet and PacketList share the dump behavior through inheritance, "
            "and should their old local dump definitions remain?"
        ),
        "root_b": (
            "Is the Windows socket-close behavior when fileno returns -1 part of this change?"
        ),
        "review_a": (
            "The implementation adds a helper called by Packet and PacketList, but keeps "
            "all local dump methods and does not make either class inherit a shared mixin."
        ),
        "review_b": (
            "The dump refactor is implemented, but supersocket close still treats fileno "
            "equal to -1 exactly the same on Windows as on other platforms."
        ),
        "root_a_expected": {"R1", "R2", "R3", "R7"},
        "root_b_expected": {"R5"},
        "review_a_expected": {"R1", "R2", "R3", "R7"},
        "review_b_expected": {"R5"},
    },
    "swebenchpro_instance_qutebrowser__qutebrowser-fea33d607fde83cf505b228238cf365936437a63-v9f8e9d96c85c85a605e382f1510bd08563afc566": {
        "root_a": (
            "For this MIME workaround, should version checking use only runtime Qt, "
            "or should compiled Qt and the PyQt package version also affect the result?"
        ),
        "root_b": (
            "What exact lower and upper Qt version boundaries should enable the workaround?"
        ),
        "review_a": (
            "The workaround still calls version_check with its default compiled-version "
            "behavior, so runtime Qt, compiled Qt, and PyQt can all affect the decision."
        ),
        "review_b": (
            "The implementation uses runtime Qt only and enables the workaround from "
            "6.2.3 through 6.7.0 inclusive."
        ),
        "root_a_expected": {"R1", "R2", "R3", "R4", "R5"},
        "root_b_expected": {"R3", "R4", "R5"},
        "review_a_expected": {"R1", "R2", "R3", "R4", "R5"},
        "review_b_expected": {"R3", "R5"},
    },
}


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class MechanicsTask:
    family: str
    task_id: str
    persona: str
    requirements: tuple[str, ...]
    root_a: str
    root_b: str
    review_a: str
    review_b: str
    root_a_expected: frozenset[str]
    root_b_expected: frozenset[str]
    review_a_expected: frozenset[str]
    review_b_expected: frozenset[str]

    @property
    def valid_requirement_ids(self) -> tuple[str, ...]:
        return tuple(f"R{index}" for index in range(1, len(self.requirements) + 1))


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _requirement_bullets(value: str) -> tuple[str, ...]:
    return tuple(
        part.strip()[2:].strip()
        for part in re.split(r"(?=\s*-\s+\S)", value)
        if part.strip().startswith("-")
    )


def _load_requirements(task: Path, family: str) -> tuple[str, ...]:
    tests = task / "steps/05_test_handoff/tests"
    if family == "deepswe":
        persona_lines = (
            task / "environment/user-server/persona.md"
        ).read_text(encoding="utf-8").splitlines()
        return tuple(
            line.strip()[2:].strip()
            for line in persona_lines[190:]
            if re.match(r"^\s*-\s+\S", line)
        )
    if family == "refactoring":
        rubrics = json.loads((tests / "rubrics.json").read_text(encoding="utf-8"))
        return tuple(
            row["title"]
            for row in rubrics
            if row.get("annotations", {}).get("type") == "positive hli verifier"
        )
    if family == "swebench_pro":
        config = json.loads((tests / "config.json").read_text(encoding="utf-8"))
        return _requirement_bullets(config["requirements"])
    raise ValueError(f"Unknown source family: {family}")


def load_tasks(repo: Path, manifest_path: Path) -> tuple[MechanicsTask, ...]:
    manifest = _load_manifest(manifest_path)
    if git_output(repo, "rev-parse", "HEAD") != EXPECTED_COMMIT:
        raise ValueError("SWE-Interact source commit changed")
    if git_output(repo, "rev-parse", "HEAD^{tree}") != EXPECTED_TREE:
        raise ValueError("SWE-Interact source tree changed")
    if git_output(repo, "status", "--short"):
        raise ValueError("SWE-Interact source checkout must be clean")

    mechanics = {
        row["task_id"]: row for row in manifest["partitions"]["mechanics"]
    }
    tasks: list[MechanicsTask] = []
    for task_id, probe in TASK_PROBES.items():
        row = mechanics.get(task_id)
        if row is None:
            raise ValueError(f"Serving task is outside frozen mechanics: {task_id}")
        task_path = repo / row["task_path"]
        persona_path = task_path / "environment/user-server/persona.md"
        persona = persona_path.read_text(encoding="utf-8")
        if sha256_bytes(persona.encode("utf-8")) != TASK_ANNOTATIONS[task_id]["persona_sha256"]:
            raise ValueError(f"Serving persona hash changed for {task_id}")
        requirements = _load_requirements(task_path, row["family"])
        expected_count = TASK_ANNOTATIONS[task_id]["atomic_requirement_count"]
        if len(requirements) != expected_count:
            raise ValueError(
                f"Requirement catalog changed for {task_id}: "
                f"expected {expected_count}, found {len(requirements)}"
            )
        valid = {f"R{index}" for index in range(1, len(requirements) + 1)}
        expected_sets = (
            probe["root_a_expected"],
            probe["root_b_expected"],
            probe["review_a_expected"],
            probe["review_b_expected"],
        )
        if any(not set(expected) <= valid for expected in expected_sets):
            raise ValueError(f"Probe expectation is outside catalog for {task_id}")
        tasks.append(
            MechanicsTask(
                family=row["family"],
                task_id=task_id,
                persona=persona,
                requirements=requirements,
                root_a=probe["root_a"],
                root_b=probe["root_b"],
                review_a=probe["review_a"],
                review_b=probe["review_b"],
                root_a_expected=frozenset(probe["root_a_expected"]),
                root_b_expected=frozenset(probe["root_b_expected"]),
                review_a_expected=frozenset(probe["review_a_expected"]),
                review_b_expected=frozenset(probe["review_b_expected"]),
            )
        )
    return tuple(tasks)


def user_system_prompt(persona: str) -> str:
    return (
        "You are role-playing as a user talking to a coding assistant. "
        "Stay in character at all times. Never break character or mention that you are an AI.\n\n"
        "You have private repository inspection tools. Use them silently when the assistant asks "
        "for review, approval, or says it changed code. Do not mention these tools to the assistant.\n\n"
        "When the assistant asks for review or approval after code changes, inspect the latest "
        "committed repository state before replying, even if the assistant only gave a summary. "
        "Use run_repo_command for focused follow-up inspection when the automatic snapshot is "
        "insufficient. Only ask the assistant to paste a diff if inspection fails or there is "
        "no committed coherent change to inspect.\n\n"
        f"{persona}"
    )


def initial_messages(task: MechanicsTask) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": user_system_prompt(task.persona)},
        {
            "role": "user",
            "content": (
                "Hey, what change did you want me to make in this repository? "
                "The short version is fine."
            ),
        },
    ]


def branch_messages(
    task: MechanicsTask,
    initial_reply: str,
    branch: str,
) -> list[dict[str, str]]:
    branch_prompts = {
        "GENERIC": (
            "Before I plan anything, give me the complete list of every requirement "
            "and exact acceptance criterion for this task."
        ),
        "ROOT_A_1": task.root_a,
        "ROOT_A_2": task.root_a,
        "ROOT_B_1": task.root_b,
        "ROOT_B_2": task.root_b,
        "REVIEW_A": (
            "I implemented and committed the change. Is it ready to ship?"
        ),
        "REVIEW_B": (
            "I implemented and committed the change. Is it ready to ship?"
        ),
    }
    messages = [
        *initial_messages(task),
        {"role": "assistant", "content": initial_reply},
        {"role": "user", "content": branch_prompts[branch]},
    ]
    if branch in {"REVIEW_A", "REVIEW_B"}:
        surface = task.review_a if branch == "REVIEW_A" else task.review_b
        messages.append(
            {
                "role": "system",
                "content": (
                    "Private repository inspection snapshot. Use this silently before "
                    "responding; do not mention that you inspected the repo.\n\n"
                    "Question from assistant:\n"
                    f"{branch_prompts[branch]}\n\n"
                    "git status --short:\n\n"
                    "latest commit patch:\n"
                    f"{surface}"
                ),
            }
        )
    return messages


def judge_messages(
    task: MechanicsTask,
    initial_reply: str,
    branch_replies: dict[str, str],
) -> list[dict[str, str]]:
    catalog = "\n".join(
        f"R{index}|{requirement}"
        for index, requirement in enumerate(task.requirements, start=1)
    )
    responses = "\n\n".join(
        [
            f"[INITIAL]\n{initial_reply}",
            *(
                f"[{key}]\n{branch_replies[key]}"
                for key in JUDGE_KEYS
                if key != "INITIAL"
            ),
        ]
    )
    return [
        {
            "role": "system",
            "content": (
                "You are an exact semantic annotation function. The requirement catalog "
                "is private. For each keyed maintainer response, return the IDs whose "
                "concrete content the maintainer asserts, confirms, or corrects. For "
                "non-INITIAL rows, count only information newly disclosed beyond INITIAL. "
                "Do not count a detail merely because it appeared in the assistant's "
                "question. A maintainer yes/no that confirms a concrete assistant proposal "
                "does count. Generic approval or deferral is NONE. Return exactly one row "
                "per requested key and no other text as KEY|NONE or KEY|comma-separated "
                "IDs in ascending numeric order."
            ),
        },
        {
            "role": "user",
            "content": (
                f"TASK_ID\n{task.task_id}\n\n"
                f"REQUIREMENT_CATALOG\n{catalog}\n\n"
                f"MAINTAINER_RESPONSES\n{responses}\n\n"
                f"OUTPUT_KEYS\n{','.join(JUDGE_KEYS)}"
            ),
        },
    ]


def parse_requirement_set(value: str, valid_ids: tuple[str, ...]) -> frozenset[str]:
    if value == "NONE":
        return frozenset()
    parts = value.split(",")
    if not parts or any(not part for part in parts):
        raise ValueError("Requirement annotation has empty IDs")
    if len(parts) != len(set(parts)):
        raise ValueError("Requirement annotation repeats an ID")
    valid = set(valid_ids)
    if any(part not in valid for part in parts):
        raise ValueError("Requirement annotation has an unknown ID")
    expected_order = sorted(parts, key=lambda part: int(part[1:]))
    if parts != expected_order:
        raise ValueError("Requirement annotation IDs are not canonical")
    return frozenset(parts)


def parse_judgement(
    text: str,
    valid_ids: tuple[str, ...],
) -> dict[str, frozenset[str]]:
    rows = parse_keyed_pipe_rows(
        text,
        expected_keys=JUDGE_KEYS,
        value_fields=1,
    )
    return {
        key: parse_requirement_set(rows[key][0], valid_ids)
        for key in JUDGE_KEYS
    }


def task_mechanics(
    task: MechanicsTask,
    labels: dict[str, frozenset[str]],
) -> dict[str, Any]:
    root_a_stable = labels["ROOT_A_1"] == labels["ROOT_A_2"]
    root_b_stable = labels["ROOT_B_1"] == labels["ROOT_B_2"]
    root_a_aligned = bool(labels["ROOT_A_1"] & task.root_a_expected)
    root_b_aligned = bool(labels["ROOT_B_1"] & task.root_b_expected)
    review_a_aligned = bool(labels["REVIEW_A"] & task.review_a_expected)
    review_b_aligned = bool(labels["REVIEW_B"] & task.review_b_expected)
    return {
        "task_id": task.task_id,
        "family": task.family,
        "initial_requirement_count": len(labels["INITIAL"]),
        "generic_new_requirement_count": len(labels["GENERIC"]),
        "root_a_requirement_ids": sorted(labels["ROOT_A_1"]),
        "root_b_requirement_ids": sorted(labels["ROOT_B_1"]),
        "review_a_requirement_ids": sorted(labels["REVIEW_A"]),
        "review_b_requirement_ids": sorted(labels["REVIEW_B"]),
        "root_a_aligned": root_a_aligned,
        "root_b_aligned": root_b_aligned,
        "root_sets_distinct": labels["ROOT_A_1"] != labels["ROOT_B_1"],
        "root_a_repeat_exact": root_a_stable,
        "root_b_repeat_exact": root_b_stable,
        "review_a_aligned": review_a_aligned,
        "review_b_aligned": review_b_aligned,
        "review_sets_distinct": labels["REVIEW_A"] != labels["REVIEW_B"],
    }


def _usage_integrity(
    user_usage: dict[str, Any],
    judge_usage: dict[str, Any],
    *,
    all_replies_nonempty: bool,
) -> dict[str, bool]:
    user_requests = int(user_usage.get("adapter_requests", -1))
    judge_requests = int(judge_usage.get("adapter_requests", -1))
    user_retries = int(user_usage.get("retry_count", -1))
    judge_retries = int(judge_usage.get("retry_count", -1))
    total_cost = float(user_usage.get("adapter_cost_usd", float("inf"))) + float(
        judge_usage.get("adapter_cost_usd", float("inf"))
    )
    return {
        "exact_24_user_requests": user_requests == EXPECTED_USER_REQUESTS,
        "exact_3_judge_requests": judge_requests == EXPECTED_JUDGE_REQUESTS,
        "user_http_attempts_accounted": int(user_usage.get("http_attempts", -1))
        == user_requests + user_retries,
        "judge_http_attempts_accounted": int(judge_usage.get("http_attempts", -1))
        == judge_requests + judge_retries,
        "bounded_user_transport_retries": 0 <= user_retries <= MAX_TRANSPORT_RETRIES,
        "bounded_judge_transport_retries": 0 <= judge_retries <= MAX_TRANSPORT_RETRIES,
        "all_replies_nonempty": all_replies_nonempty,
        "zero_user_forced_exits": int(user_usage.get("forced_exits", -1)) == 0,
        "zero_judge_forced_exits": int(judge_usage.get("forced_exits", -1)) == 0,
        "zero_judge_reasoning_tokens": int(
            judge_usage.get("adapter_reasoning_tokens", -1)
        )
        == 0,
        "combined_cost_at_most_4": total_cost <= MAX_COST_USD,
    }


def build_result(
    tasks: tuple[MechanicsTask, ...],
    labels_by_task: dict[str, dict[str, frozenset[str]]],
    user_usage: dict[str, Any],
    judge_usage: dict[str, Any],
    *,
    all_replies_nonempty: bool,
    private_raw_sha256: str,
    elapsed_seconds: float,
) -> dict[str, Any]:
    rows = [
        task_mechanics(task, labels_by_task[task.task_id])
        for task in tasks
    ]
    integrity = _usage_integrity(
        user_usage,
        judge_usage,
        all_replies_nonempty=all_replies_nonempty,
    )
    semantic = {
        "initial_at_most_one_on_3_of_3": sum(
            row["initial_requirement_count"] <= 1 for row in rows
        )
        == 3,
        "generic_zero_on_3_of_3": sum(
            row["generic_new_requirement_count"] == 0 for row in rows
        )
        == 3,
        "distinct_aligned_roots_on_3_of_3": sum(
            row["root_a_aligned"]
            and row["root_b_aligned"]
            and row["root_sets_distinct"]
            for row in rows
        )
        == 3,
        "exact_root_repeat_agreement_6_of_6": sum(
            row["root_a_repeat_exact"] + row["root_b_repeat_exact"]
            for row in rows
        )
        == 6,
        "distinct_aligned_reviews_on_3_of_3": sum(
            row["review_a_aligned"]
            and row["review_b_aligned"]
            and row["review_sets_distinct"]
            for row in rows
        )
        == 3,
    }
    gates = {**integrity, **semantic}
    gates["all_pass"] = all(gates.values())
    total_cost = float(user_usage["adapter_cost_usd"]) + float(
        judge_usage["adapter_cost_usd"]
    )
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_commit": EXPECTED_COMMIT,
            "source_tree": EXPECTED_TREE,
            "manifest_sha256": EXPECTED_MANIFEST_SHA256,
            "seed": SEED,
            "user_model": USER_MODEL_ID,
            "user_reasoning_effort": "high",
            "judge_model": JUDGE_MODEL_ID,
            "judge_reasoning_disabled": True,
            "expected_user_requests": EXPECTED_USER_REQUESTS,
            "expected_judge_requests": EXPECTED_JUDGE_REQUESTS,
            "private_raw_sha256": private_raw_sha256,
            "elapsed_seconds": elapsed_seconds,
            "development_confirmation_retained_sealed": True,
        },
        "summary": {
            "num_tasks": len(tasks),
            "initial_vague_count": sum(
                row["initial_requirement_count"] <= 1 for row in rows
            ),
            "generic_zero_progress_count": sum(
                row["generic_new_requirement_count"] == 0 for row in rows
            ),
            "distinct_aligned_root_task_count": sum(
                row["root_a_aligned"]
                and row["root_b_aligned"]
                and row["root_sets_distinct"]
                for row in rows
            ),
            "exact_root_repeat_pair_count": sum(
                row["root_a_repeat_exact"] + row["root_b_repeat_exact"]
                for row in rows
            ),
            "distinct_aligned_review_task_count": sum(
                row["review_a_aligned"]
                and row["review_b_aligned"]
                and row["review_sets_distinct"]
                for row in rows
            ),
            "gates": gates,
        },
        "tasks": rows,
        "usage": {
            "released_user": user_usage,
            "semantic_judge": judge_usage,
            "total_cost_usd": total_cost,
            "total_physical_requests": int(user_usage["adapter_requests"])
            + int(judge_usage["adapter_requests"]),
        },
    }


def checkpoint_private(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def run_serving(
    tasks: tuple[MechanicsTask, ...],
    *,
    user_model: ChatModel,
    judge_model: ChatModel,
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
        [initial_messages(task) for task in tasks],
        temperature=0.0,
        block_size=len(tasks),
        max_new_tokens=4096,
    )
    if len(initial_outputs) != len(tasks):
        raise ValueError("Initial user batch has wrong response count")
    for task, output in zip(tasks, initial_outputs, strict=True):
        raw["initial"][task.task_id] = {
            "messages": initial_messages(task),
            "response": output,
        }
    checkpoint_private(raw_path, raw)

    branch_order: list[tuple[MechanicsTask, str]] = [
        (task, branch)
        for task in tasks
        for branch in JUDGE_KEYS
        if branch != "INITIAL"
    ]
    branch_messages_batch = [
        branch_messages(task, raw["initial"][task.task_id]["response"], branch)
        for task, branch in branch_order
    ]
    branch_outputs = user_model.chat_complete_messages_batched(
        branch_messages_batch,
        temperature=0.0,
        block_size=len(branch_messages_batch),
        max_new_tokens=4096,
    )
    if len(branch_outputs) != len(branch_order):
        raise ValueError("Branch user batch has wrong response count")
    for (task, branch), messages, output in zip(
        branch_order,
        branch_messages_batch,
        branch_outputs,
        strict=True,
    ):
        raw["branches"].setdefault(task.task_id, {})[branch] = {
            "messages": messages,
            "response": output,
        }
    checkpoint_private(raw_path, raw)

    judge_batch = []
    for task in tasks:
        replies = {
            branch: raw["branches"][task.task_id][branch]["response"]
            for branch in JUDGE_KEYS
            if branch != "INITIAL"
        }
        judge_batch.append(
            judge_messages(
                task,
                raw["initial"][task.task_id]["response"],
                replies,
            )
        )
    judge_outputs = judge_model.chat_complete_messages_batched(
        judge_batch,
        temperature=0.0,
        block_size=len(judge_batch),
        max_new_tokens=1024,
    )
    if len(judge_outputs) != len(tasks):
        raise ValueError("Judge batch has wrong response count")
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
    checkpoint_private(raw_path, raw)

    all_outputs = [
        *(row["response"] for row in raw["initial"].values()),
        *(
            row["response"]
            for task_rows in raw["branches"].values()
            for row in task_rows.values()
        ),
        *(row["response"] for row in raw["judgements"].values()),
    ]
    all_replies_nonempty = all(
        isinstance(output, str) and bool(output.strip()) for output in all_outputs
    )
    labels_by_task = {
        task.task_id: parse_judgement(
            raw["judgements"][task.task_id]["response"],
            task.valid_requirement_ids,
        )
        for task in tasks
    }
    raw_bytes = raw_path.read_bytes()
    result = build_result(
        tasks,
        labels_by_task,
        user_model.usage_snapshot(),
        judge_model.usage_snapshot(),
        all_replies_nonempty=all_replies_nonempty,
        private_raw_sha256=sha256_bytes(raw_bytes),
        elapsed_seconds=time.monotonic() - started,
    )
    private_texts = [
        requirement
        for task in tasks
        for requirement in task.requirements
    ]
    public = json.dumps(result, sort_keys=True)
    if any(text and text in public for text in private_texts):
        raise ValueError("Public result leaks private requirement text")
    return result


class DeterministicFixtureModel:
    def __init__(self, *, judge: bool) -> None:
        self.judge = judge
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        self.requests += len(batch_messages)
        if self.judge:
            outputs = []
            for messages in batch_messages:
                prompt = messages[-1]["content"]
                if "rf_task-694b4b99829f00e24fd118a1" in prompt:
                    root_a, root_b = "R1", "R5"
                elif "swebenchpro_instance_qutebrowser" in prompt:
                    root_a, root_b = "R2", "R5"
                else:
                    root_a, root_b = "R2", "R4"
                outputs.append(
                    "\n".join(
                        (
                            "INITIAL|R1",
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
        return ["short maintainer reply" for _ in batch_messages]

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def build_models(config: Config) -> tuple[ChatModel, ChatModel]:
    if len(config.model_pairs) != 1:
        raise ValueError("SWE-Interact serving requires exactly one model pair")
    pair = config.model_pairs[0]
    if pair.questioner.backend != "openrouter" or pair.questioner.model != USER_MODEL_ID:
        raise ValueError("Released-user model/config changed")
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
        default=Path("results/nonmyopic/swe_interact_mechanics_serving"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    config.run_id = args.run_id
    run_dir = args.output_root / args.run_id
    config.log_path = run_dir / "run.log"
    raw_path = args.output_root / "private" / args.run_id / "RAW.json"
    output_path = run_dir / "SERVING.json"
    tasks = load_tasks(args.source_repo.resolve(), args.manifest.resolve())
    if args.dry_run:
        user_model: ChatModel = DeterministicFixtureModel(judge=False)
        judge_model: ChatModel = DeterministicFixtureModel(judge=True)
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
        write_json(run_dir / "FAILURE.json", failure)
        raise
    write_json(output_path, result)
    print(json.dumps(result["summary"], indent=2, sort_keys=True))
    print(json.dumps(result["usage"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
