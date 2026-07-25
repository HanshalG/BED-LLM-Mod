#!/usr/bin/env python3
"""Aggregate three frozen GPT-5.4 tau scorer test-retest replicates."""

from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tau_knowledge_cross_model_scorer import (
    CONFIRMATION_ARTIFACT_SHA256,
    NONSEMANTIC_ANALYSIS_SHA256,
)
from scripts.tau_knowledge_gpt54_scorer_retest import (
    INTERFACE_VERSION,
    MODEL_ID,
    REPLICATE_COUNT,
)
from scripts.tau_knowledge_receding_continuation import (
    FRESH_CONFIRMATION_IDS,
)


EXPECTED_REQUESTS_PER_REPLICATE = 140
TOTAL_COST_CAP_USD = 6.75


def mean_pairwise_agreement(selections: Sequence[Sequence[int]]) -> float:
    if len(selections) < 2:
        raise ValueError("at least two selection vectors are required")
    width = len(selections[0])
    if width == 0 or any(len(row) != width for row in selections):
        raise ValueError("selection vectors must have equal nonzero length")
    agreements = []
    for left, right in combinations(selections, 2):
        agreements.append(
            sum(a == b for a, b in zip(left, right, strict=True)) / width
        )
    return sum(agreements) / len(agreements)


def _validate_payload(payload: dict[str, Any], replicate_index: int) -> None:
    protocol = payload.get("protocol", {})
    if payload.get("status") not in {"passed", "gate_failed"}:
        raise ValueError("replicate did not complete its scientific payload")
    if protocol.get("stage") != "confirmation":
        raise ValueError("replicate is not a confirmation scorer run")
    if protocol.get("interface_version") != INTERFACE_VERSION:
        raise ValueError("replicate interface version does not match")
    if protocol.get("model") != MODEL_ID:
        raise ValueError("replicate model does not match")
    if protocol.get("replicate_index") != replicate_index:
        raise ValueError("replicate index does not match its position")
    if (
        protocol.get("source_artifact_sha256")
        != CONFIRMATION_ARTIFACT_SHA256
    ):
        raise ValueError("replicate source artifact hash does not match")
    if (
        protocol.get("nonsemantic_analysis_sha256")
        != NONSEMANTIC_ANALYSIS_SHA256
    ):
        raise ValueError("replicate nonsemantic hash does not match")
    if tuple(protocol.get("task_ids", ())) != tuple(FRESH_CONFIRMATION_IDS):
        raise ValueError("replicate task IDs do not match")


def analyze_replicates(payloads: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if len(payloads) != REPLICATE_COUNT:
        raise ValueError(f"exactly {REPLICATE_COUNT} replicates are required")
    for index, payload in enumerate(payloads, start=1):
        _validate_payload(payload, index)

    replicate_rows = []
    root_selections = []
    focused_selections = []
    per_task_rows: dict[str, list[dict[str, int]]] = {
        task_id: [] for task_id in FRESH_CONFIRMATION_IDS
    }
    for index, payload in enumerate(payloads, start=1):
        summary = payload["summary"]
        policies = summary["policy_diagnostics"]
        roots = summary["root_diagnostics"]
        if len(policies) != len(FRESH_CONFIRMATION_IDS) or len(roots) != 100:
            raise ValueError("replicate diagnostics are incomplete")
        endpoint_total = sum(
            row["nonmyopic_receding_value"] for row in policies
        )
        myopic_total = sum(row["myopic_receding_value"] for row in policies)
        myopic_advantage = endpoint_total - myopic_total
        replicate_rows.append(
            {
                "replicate_index": index,
                "original_all_gates_pass": bool(
                    summary["gates"]["all_pass"]
                ),
                "root_accuracy": summary[
                    "nonmyopic_root_pairwise_accuracy"
                ],
                "myopic_root_accuracy": summary[
                    "myopic_root_pairwise_accuracy"
                ],
                "root_accuracy_gain": summary[
                    "root_pairwise_accuracy_gain"
                ],
                "focused_accuracy": summary["focused_pairwise_accuracy"],
                "focused_optimal_rate": summary[
                    "focused_optimal_followup_rate"
                ],
                "focused_mean_regret": summary["focused_mean_regret"],
                "endpoint_total": endpoint_total,
                "myopic_endpoint_total": myopic_total,
                "myopic_advantage": myopic_advantage,
                "cost_usd": float(payload["usage"]["adapter_cost_usd"]),
                "physical_requests": int(
                    payload["usage"]["physical_requests"]
                ),
                "reasoning_tokens": int(
                    payload["usage"]["reasoning_tokens"]
                ),
            }
        )
        root_selections.append(
            [row["nonmyopic_root_index"] for row in policies]
        )
        focused_selections.append(
            [row["selected_followup_index"] for row in roots]
        )
        for row in policies:
            per_task_rows[row["task_id"]].append(
                {
                    "replicate_index": index,
                    "nonmyopic_value": row["nonmyopic_receding_value"],
                    "myopic_value": row["myopic_receding_value"],
                    "advantage": row["nonmyopic_advantage_over_myopic"],
                }
            )

    total_requests = sum(row["physical_requests"] for row in replicate_rows)
    total_reasoning = sum(row["reasoning_tokens"] for row in replicate_rows)
    total_cost = sum(row["cost_usd"] for row in replicate_rows)
    mean_root_accuracy = sum(
        row["root_accuracy"] for row in replicate_rows
    ) / REPLICATE_COUNT
    mean_root_gain = sum(
        row["root_accuracy_gain"] for row in replicate_rows
    ) / REPLICATE_COUNT
    mean_focused_accuracy = sum(
        row["focused_accuracy"] for row in replicate_rows
    ) / REPLICATE_COUNT
    endpoint_pass_count = sum(
        row["myopic_advantage"] >= 4 and row["endpoint_total"] >= 25
        for row in replicate_rows
    )
    gates = {
        "all_three_exact_140_zero_reasoning": all(
            row["physical_requests"] == EXPECTED_REQUESTS_PER_REPLICATE
            and row["reasoning_tokens"] == 0
            for row in replicate_rows
        ),
        "at_least_2_original_all_gate_passes": sum(
            row["original_all_gates_pass"] for row in replicate_rows
        )
        >= 2,
        "mean_root_accuracy_at_least_0_60": mean_root_accuracy >= 0.60,
        "mean_root_gain_at_least_0_05": mean_root_gain >= 0.05,
        "mean_focused_accuracy_at_least_0_60": (
            mean_focused_accuracy >= 0.60
        ),
        "at_least_2_endpoint_passes": endpoint_pass_count >= 2,
        "root_argmax_agreement_at_least_0_50": (
            mean_pairwise_agreement(root_selections) >= 0.50
        ),
        "focused_argmax_agreement_at_least_0_65": (
            mean_pairwise_agreement(focused_selections) >= 0.65
        ),
        "exact_420_physical_requests": total_requests == 420,
        "zero_reasoning_tokens": total_reasoning == 0,
        "total_cost_at_most_6_75": total_cost <= TOTAL_COST_CAP_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "replicate_count": REPLICATE_COUNT,
            "same_task_reproducibility_only": True,
            "source_artifact_sha256": CONFIRMATION_ARTIFACT_SHA256,
            "nonsemantic_analysis_sha256": NONSEMANTIC_ANALYSIS_SHA256,
        },
        "summary": {
            "mean_root_accuracy": mean_root_accuracy,
            "mean_root_accuracy_gain": mean_root_gain,
            "mean_focused_accuracy": mean_focused_accuracy,
            "root_argmax_agreement": mean_pairwise_agreement(root_selections),
            "focused_argmax_agreement": mean_pairwise_agreement(
                focused_selections
            ),
            "endpoint_pass_count": endpoint_pass_count,
            "total_physical_requests": total_requests,
            "total_reasoning_tokens": total_reasoning,
            "total_cost_usd": total_cost,
            "gates": gates,
        },
        "replicates": replicate_rows,
        "per_task": [
            {"task_id": task_id, "replicates": per_task_rows[task_id]}
            for task_id in FRESH_CONFIRMATION_IDS
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--replicate",
        type=Path,
        action="append",
        required=True,
        help="Repeat exactly three times in replicate-index order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.replicate
    ]
    result = analyze_replicates(payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
