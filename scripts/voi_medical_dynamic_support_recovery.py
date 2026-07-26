#!/usr/bin/env python3
"""Test whether semantic questions induce root-specific truth recovery."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.voi_medical_dynamic_support_serving import (
    HYPOTHESIS_COUNT,
    MODEL_ID,
    SOURCE_SHA256,
    hypothesis_messages,
    normalize_hypothesis,
    parse_hypotheses,
    target_is_covered,
)
from scripts.voi_medical_future_tree_mechanics import (
    OUTCOMES,
    MechanicsExecutionError,
    _checkpoint,
    _usage_snapshot,
    sha256_file,
)


INTERFACE_VERSION = "voi-medical-dynamic-support-recovery-1"
SERVING_SHA256 = "e7effe8977f40fa25e8b8d43f7862d664ea3ee0f991c89a93fc65567d484bb0d"
TASK_IDS = (50, 284)
TASK_POSITIONS = (3, 4)
ROOT_COUNT = 4
EXPECTED_REQUESTS = len(TASK_IDS) * ROOT_COUNT * (1 + len(OUTCOMES))
PROJECTED_COST_USD = 0.10
MAX_COST_USD = 0.25


def parse_answer_labels(text: str) -> list[str]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError(f"expected exactly {HYPOTHESIS_COUNT} answer lines")
    labels = []
    for index, line in enumerate(lines, start=1):
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 2 or fields[0] != f"H{index}":
            raise ValueError("answer line has invalid fields or index")
        if fields[1] not in OUTCOMES:
            raise ValueError("answer label must be Yes, No, or Maybe")
        labels.append(fields[1])
    return labels


def answer_map_messages(
    question: str,
    hypotheses: Sequence[Mapping[str, str]],
) -> list[dict[str, str]]:
    support = "\n".join(
        f"H{index}|{row['diagnosis']}|{row['rationale']}"
        for index, row in enumerate(hypotheses, start=1)
    )
    return [
        {
            "role": "system",
            "content": (
                "Freeze a semantic likelihood partition for one diagnostic "
                "question. For each clinical hypothesis, label how a typical "
                "patient would answer: Yes, No, or Maybe. Return exactly six "
                "lines and no other text: H1|<Yes/No/Maybe> through H6."
            ),
        },
        {
            "role": "user",
            "content": f"Question: {question}\n\nHypotheses:\n{support}",
        },
    ]


def branch_refresh_messages(
    *,
    self_report: str,
    initial_support: Sequence[Mapping[str, str]],
    question: str,
    outcome: str,
) -> list[dict[str, str]]:
    support = "\n".join(
        f"- {row['diagnosis']}: {row['rationale']}" for row in initial_support
    )
    return [
        {
            "role": "system",
            "content": (
                "Regenerate a differential diagnosis from scratch after a "
                "hypothetical patient answer. Produce exactly six distinct "
                "specific clinical hypotheses. You may retain, discard, or add "
                "hypotheses; do not merely copy the prior list. Use the complete "
                "hypothetical history. Return exactly six lines and no other "
                "text: H1|<diagnosis>|<one-sentence evidence rationale> through H6."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Patient self-report:\n{self_report}\n\n"
                f"Prior generated differential:\n{support}\n\n"
                f"Hypothetical question: {question}\n"
                f"Hypothetical patient answer: {outcome}"
            ),
        },
    ]


def immediate_eig(labels: Sequence[str]) -> float:
    initial_entropy = math.log(len(labels))
    expected_entropy = 0.0
    for outcome in OUTCOMES:
        count = sum(label == outcome for label in labels)
        if count:
            expected_entropy += (count / len(labels)) * math.log(count)
    return initial_entropy - expected_entropy


def outcome_masses(labels: Sequence[str]) -> dict[str, float]:
    return {
        outcome: sum(label == outcome for label in labels) / len(labels)
        for outcome in OUTCOMES
    }


def expected_coverage(
    target: str,
    labels: Sequence[str],
    branch_supports: Mapping[str, Sequence[Mapping[str, str]]],
) -> float:
    masses = outcome_masses(labels)
    return sum(
        masses[outcome] * target_is_covered(target, branch_supports[outcome])
        for outcome in OUTCOMES
    )


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("dynamic-support recovery config selects wrong model")
    return build_model_adapter(spec, config)


def run_recovery(
    config: Config,
    *,
    source_path: Path,
    serving_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("MedDG source hash mismatch")
    if sha256_file(serving_path) != SERVING_SHA256:
        raise ValueError("serving artifact hash mismatch")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    serving = json.loads(serving_path.read_text(encoding="utf-8"))
    initial_supports = [serving["hypotheses"][index] for index in TASK_POSITIONS]
    questions = [serving["questions"][index] for index in TASK_POSITIONS]
    visible = [{"self_repo": rows[index]["self_repo"]} for index in TASK_IDS]
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        map_keys = [
            (task_index, root_index)
            for task_index in range(len(TASK_IDS))
            for root_index in range(ROOT_COUNT)
        ]
        map_responses = model.chat_complete_messages_batched(
            [
                answer_map_messages(
                    questions[task_index][root_index],
                    initial_supports[task_index],
                )
                for task_index, root_index in map_keys
            ],
            temperature=0.0,
            block_size=len(map_keys),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["answer_maps"] = map_responses
        _checkpoint(raw_path, raw)
        answer_maps = {
            key: parse_answer_labels(response)
            for key, response in zip(map_keys, map_responses, strict=True)
        }

        branch_keys = [
            (task_index, root_index, outcome)
            for task_index in range(len(TASK_IDS))
            for root_index in range(ROOT_COUNT)
            for outcome in OUTCOMES
        ]
        branch_responses = model.chat_complete_messages_batched(
            [
                branch_refresh_messages(
                    self_report=visible[task_index]["self_repo"],
                    initial_support=initial_supports[task_index],
                    question=questions[task_index][root_index],
                    outcome=outcome,
                )
                for task_index, root_index, outcome in branch_keys
            ],
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["branch_supports"] = branch_responses
        _checkpoint(raw_path, raw)
        branches = {
            key: parse_hypotheses(response)
            for key, response in zip(branch_keys, branch_responses, strict=True)
        }
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    # Targets are read only after the complete target-blind tree is frozen.
    targets = [rows[index]["target"] for index in TASK_IDS]
    task_rows = []
    for task_index, task_id in enumerate(TASK_IDS):
        roots = []
        for root_index in range(ROOT_COUNT):
            labels = answer_maps[(task_index, root_index)]
            branch_supports = {
                outcome: branches[(task_index, root_index, outcome)]
                for outcome in OUTCOMES
            }
            roots.append(
                {
                    "root_index": root_index,
                    "question": questions[task_index][root_index],
                    "answer_labels": labels,
                    "outcome_masses": outcome_masses(labels),
                    "immediate_eig_nats": immediate_eig(labels),
                    "expected_target_coverage": expected_coverage(
                        targets[task_index],
                        labels,
                        branch_supports,
                    ),
                    "branch_target_coverage": {
                        outcome: target_is_covered(
                            targets[task_index], branch_supports[outcome]
                        )
                        for outcome in OUTCOMES
                    },
                    "branch_supports": branch_supports,
                }
            )
        myopic_index = max(
            range(ROOT_COUNT),
            key=lambda index: (roots[index]["immediate_eig_nats"], -index),
        )
        oracle_index = max(
            range(ROOT_COUNT),
            key=lambda index: (roots[index]["expected_target_coverage"], -index),
        )
        task_rows.append(
            {
                "task_id": task_id,
                "target": targets[task_index],
                "roots": roots,
                "myopic_root_index": myopic_index,
                "oracle_recovery_root_index": oracle_index,
                "myopic_expected_target_coverage": roots[myopic_index][
                    "expected_target_coverage"
                ],
                "oracle_expected_target_coverage": roots[oracle_index][
                    "expected_target_coverage"
                ],
                "oracle_minus_myopic_coverage": (
                    roots[oracle_index]["expected_target_coverage"]
                    - roots[myopic_index]["expected_target_coverage"]
                ),
                "root_coverage_range": (
                    max(root["expected_target_coverage"] for root in roots)
                    - min(root["expected_target_coverage"] for root in roots)
                ),
                "any_positive_branch_recovers_target": any(
                    root["outcome_masses"][outcome] > 0
                    and root["branch_target_coverage"][outcome]
                    for root in roots
                    for outcome in OUTCOMES
                ),
            }
        )

    gains = [row["oracle_minus_myopic_coverage"] for row in task_rows]
    ranges = [row["root_coverage_range"] for row in task_rows]
    generator = usage["generator"]
    gates = {
        "exact_frozen_inputs": True,
        "exact_32_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_32_http_attempts_zero_retries": (
            usage["http_attempts"] == EXPECTED_REQUESTS
            and usage["retry_count"] == 0
        ),
        "zero_reasoning_or_forced": (
            usage["reasoning_tokens"] == 0
            and usage["forced_exits"] == 0
            and int(generator.get("forced_final_requests", 0)) == 0
        ),
        "all_maps_and_24_branch_supports_complete": (
            len(answer_maps) == 8 and len(branches) == 24
        ),
        "every_root_has_at_least_two_positive_outcomes": all(
            sum(mass > 0 for mass in root["outcome_masses"].values()) >= 2
            for row in task_rows
            for root in row["roots"]
        ),
        "both_missing_targets_recovered_somewhere": all(
            row["any_positive_branch_recovers_target"] for row in task_rows
        ),
        "at_least_one_root_coverage_range_0_15": max(ranges) >= 0.15,
        "at_least_one_oracle_gain_0_15": max(gains) >= 0.15,
        "mean_oracle_gain_at_least_0_10": sum(gains) / len(gains) >= 0.10,
        "cost_at_most_0_25": usage["adapter_cost_usd"] <= MAX_COST_USD,
        "adapter_reports_nonreasoning_model": (
            generator.get("model") == MODEL_ID
            and generator.get("reasoning_enabled") is False
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "source_sha256": SOURCE_SHA256,
            "serving_sha256": SERVING_SHA256,
            "task_ids": list(TASK_IDS),
            "target_exposed_to_generation": False,
            "candidate_list_exposed": False,
            "expected_requests": EXPECTED_REQUESTS,
        },
        "tasks": task_rows,
        "metrics": {
            "oracle_minus_myopic_coverage_by_task": gains,
            "root_coverage_range_by_task": ranges,
            "mean_oracle_minus_myopic_coverage": sum(gains) / len(gains),
        },
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--serving", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_recovery(
            config,
            source_path=args.source,
            serving_path=args.serving,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, MechanicsExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        _checkpoint(args.output_dir / "RECOVERY_FAILURE.json", failure)
        raise
    output = args.output_dir / "RECOVERY.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
