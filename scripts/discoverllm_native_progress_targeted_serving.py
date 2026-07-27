#!/usr/bin/env python3
"""Run the targeted DiscoverLLM native-progress realistic serving gate."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import discoverllm_native_progress_serving as base
from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_ordinal_serving as ordinal
from scripts import discoverllm_priority_world_tier_serving as tier


INTERFACE_VERSION = "discoverllm-native-progress-targeted-serving-1"
MODEL_ID = "openai/gpt-5.4"
SERVING_TASK_ID = "technical_writing:artifact_352"
RESERVED_MECHANICS_TASK_IDS = (
    "technical_writing:artifact_83",
    "creative_writing:artifact_348",
    "svg_drawing:artifact_27",
)
WORLD_PRESENTATION_SEED = 24_418
EXPECTED_REQUESTS = 8
MAX_TRANSPORT_RETRIES = 3
PROJECTED_COST_USD = 0.22
MAX_COST_USD = 0.35


def _load_task(
    paths: dict[str, Path],
    manifest_path: Path,
) -> base.ProgressTask:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = tuple(frozen["splits"]["development"]["artifact_ids"])
    expected_prefix = (
        ordinal.SERVING_TASK_ID,
        tier.SERVING_TASK_ID,
        *tier.RESERVED_MECHANICS_TASK_IDS,
        base.SERVING_TASK_ID,
        *base.RESERVED_MECHANICS_TASK_IDS,
        SERVING_TASK_ID,
        *RESERVED_MECHANICS_TASK_IDS,
    )
    if development_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("targeted native-progress reservation changed")
    return base._load_task_by_id(paths, manifest_path, SERVING_TASK_ID)


def _world_presentation(
    task: base.ProgressTask,
    stage: str,
) -> dict[str, dict[str, Any]]:
    labels = list(base.WORLD_LABELS)
    digest = hashlib.sha256(
        f"{WORLD_PRESENTATION_SEED}:{task.key}:{stage}".encode("utf-8")
    ).digest()[:8]
    random.Random(int.from_bytes(digest, "big")).shuffle(labels)
    by_label = dict(zip(base.WORLD_LABELS, task.worlds))
    return {
        label: base._world_payload(by_label[label]) for label in labels
    }


def _action_messages(task: base.ProgressTask) -> list[dict[str, str]]:
    return base._messages(
        "ACTION_BANK",
        (
            "Generate four distinct shared next assistant messages for the "
            "conversation and four candidate current priorities. D1 is one "
            "broad open clarification question. D2 is one compact contrastive "
            "question that explicitly presents all four candidate priorities "
            "in natural language and asks the user to choose or distinguish "
            "them. R1 is one concise artifact synthesis addressing common "
            "ground across the four priorities. R2 is one message containing "
            "exactly four clearly separated artifact alternatives A-D; "
            "alternative i must be a concrete artifact designed to fully "
            "satisfy candidate priority Wi, including its subcriteria. The "
            "actions are shared and the true world is unknown. Do not mention "
            "world labels, hidden states, experiments, or scoring. D1,D2,R1 "
            "are at most 100 words each; R2 is at most 320 words. Return "
            "exactly four lines and no other text, in order D1,D2,R1,R2, as "
            "LABEL|message. Messages may not contain a pipe character."
        ),
        {
            "conversation": task.conversation,
            "candidate_current_priorities": _world_presentation(
                task,
                "ACTION_BANK",
            ),
            "output_labels": list(base.ACTION_LABELS),
        },
    )


def _transition_messages(
    stage: str,
    task: base.ProgressTask,
    actions: dict[str, str],
    *,
    branches: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    output_keys = (
        sorted(base._root_transition_keys())
        if branches is None
        else sorted(base._followup_transition_keys())
    )
    payload: dict[str, Any] = {
        "conversation": task.conversation,
        "candidate_worlds": _world_presentation(task, stage),
        "output_keys": output_keys,
    }
    if branches is None:
        payload["assistant_actions"] = actions
    else:
        payload["branches"] = branches
    return base._messages(
        stage,
        (
            "Apply DiscoverLLM's native active-root update independently to "
            "every requested cell. Output A only when the last assistant "
            "message is an artifact and fully satisfies every leaf of the "
            "candidate active priority, so the simulator advances. Output C "
            "when the root stays active but the response directly probes it "
            "or gives concrete artifact-specific shortcomings, making it "
            "clear to the user. Output V when the root stays active and the "
            "user remains vague. Output T only when the candidate state was "
            "already terminal. Return exactly the requested number of lines "
            "and no other text, each KEY|A, KEY|C, KEY|V, or KEY|T."
        ),
        payload,
    )


def _parse_state_deltas(
    text: str,
    expected_keys: set[str],
) -> dict[str, tuple[str, str]]:
    lines = text.strip().splitlines()
    if len(lines) != len(expected_keys):
        raise ValueError("state-delta response has the wrong line count")
    parsed = {}
    for line in lines:
        if line.count("|") != 1:
            raise ValueError("state-delta line has invalid separators")
        key, code = line.split("|")
        if key not in expected_keys or key in parsed:
            raise ValueError("state-delta key is invalid or duplicated")
        mapping = {
            "A": ("R", "S"),
            "C": ("R", "N"),
            "V": ("D", "N"),
            "T": ("D", "T"),
        }
        if code not in mapping:
            raise ValueError("state-delta code is invalid")
        parsed[key] = mapping[code]
    if set(parsed) != expected_keys:
        raise ValueError("state-delta response is missing keys")
    return parsed


def _gates(
    usage: dict[str, Any],
    root_transitions: dict[str, tuple[str, str]],
) -> dict[str, bool]:
    requests = int(usage.get("adapter_requests", -1))
    retries = int(usage.get("retry_count", -1))
    attempts = int(usage.get("http_attempts", -1))
    r2_advances = sum(
        root_transitions[f"R2_{world}"] == ("R", "S")
        for world in base.WORLD_LABELS
    )
    dialog_advances = sum(
        root_transitions[f"{action}_{world}"] == ("R", "S")
        for action in ("D1", "D2")
        for world in base.WORLD_LABELS
    )
    gates = {
        "exact_8_logical_requests": requests == EXPECTED_REQUESTS,
        "bounded_transport_retries": 0 <= retries <= MAX_TRANSPORT_RETRIES,
        "http_attempts_match_requests_plus_retries": (
            attempts == requests + retries
        ),
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "cost_at_most_0_35": float(
            usage.get("adapter_cost_usd", float("inf"))
        )
        <= MAX_COST_USD,
        "dialog_actions_never_advance": dialog_advances == 0,
        "targeted_R2_advances_at_least_two_worlds": r2_advances >= 2,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_serving(
    config: Config,
    *,
    paths: dict[str, Path],
    manifest_path: Path,
    raw_path: Path,
    model: base.ChatModel,
) -> dict[str, Any]:
    result = base.run_serving(
        config,
        paths=paths,
        manifest_path=manifest_path,
        raw_path=raw_path,
        model=model,
        task_loader=_load_task,
        action_messages_builder=_action_messages,
        transition_messages_builder=_transition_messages,
        transition_parser=_parse_state_deltas,
        gates_builder=_gates,
    )
    root_raw = json.loads(raw_path.read_text(encoding="utf-8"))
    root_transitions = _parse_state_deltas(
        root_raw["root_transitions"][0],
        base._root_transition_keys(),
    )
    result["protocol"].update(
        {
            "interface_version": INTERFACE_VERSION,
            "task_id": SERVING_TASK_ID,
            "reserved_mechanics_task_ids": list(
                RESERVED_MECHANICS_TASK_IDS
            ),
            "world_presentation_seed": WORLD_PRESENTATION_SEED,
            "state_delta_codes": {
                "A": "advance",
                "C": "stay_clear",
                "V": "stay_vague",
                "T": "terminal",
            },
            "maximum_transport_retries": MAX_TRANSPORT_RETRIES,
        }
    )
    result["root_transition_summary"] = {
        "D1_advance_cells": sum(
            root_transitions[f"D1_{world}"] == ("R", "S")
            for world in base.WORLD_LABELS
        ),
        "D2_advance_cells": sum(
            root_transitions[f"D2_{world}"] == ("R", "S")
            for world in base.WORLD_LABELS
        ),
        "R1_advance_cells": sum(
            root_transitions[f"R1_{world}"] == ("R", "S")
            for world in base.WORLD_LABELS
        ),
        "R2_advance_cells": sum(
            root_transitions[f"R2_{world}"] == ("R", "S")
            for world in base.WORLD_LABELS
        ),
    }
    return result


class DeterministicFixtureModel(base.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        stage = self._stage(batch_messages[0])
        if stage not in {
            "ROOT_PROGRESS_TRANSITIONS",
            "FOLLOWUP_PROGRESS_TRANSITIONS",
        }:
            return super().chat_complete_messages_batched(
                batch_messages,
                temperature,
                block_size,
                max_new_tokens,
            )
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            lines = []
            for key in payload["output_keys"]:
                action = key.split("_", 1)[0]
                code = "A" if action == "R2" else ("C" if action in {"D2", "R1"} else "V")
                lines.append(f"{key}|{code}")
            responses.append("\n".join(lines))
        self.requests += len(responses)
        return responses


def _build_model(config: Config) -> base.ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("targeted native-progress config selects wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--technical-writing", type=Path, required=True)
    parser.add_argument("--svg-drawing", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 1
    config.openrouter_max_retries = MAX_TRANSPORT_RETRIES
    config.openrouter_max_output_tokens = 4_000
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: base.ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        result = run_serving(
            config,
            paths={
                "creative_writing": args.creative_writing,
                "technical_writing": args.technical_writing,
                "svg_drawing": args.svg_drawing,
            },
            manifest_path=args.manifest,
            raw_path=raw_path,
            model=model,
        )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, base.ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        ordinal._checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    output_path = args.output_dir / "SERVING.json"
    ordinal._checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "parse_counts": result["parse_counts"],
                "root_transition_summary": result[
                    "root_transition_summary"
                ],
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
