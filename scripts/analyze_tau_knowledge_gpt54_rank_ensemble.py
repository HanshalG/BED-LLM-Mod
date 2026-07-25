#!/usr/bin/env python3
"""Analyze a fresh three-call rank ensemble against frozen development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_tau_knowledge_gpt54_scorer_retest import (
    mean_pairwise_agreement,
)
from scripts.tau_knowledge_cross_model_scorer import (
    load_records,
    summarize_cross_model,
)
from scripts.tau_knowledge_gpt54_scorer_retest import REPLICATE_COUNT
from scripts.tau_knowledge_receding_continuation import (
    FRESH_CONFIRMATION_IDS,
)


DEVELOPMENT_SHA256 = (
    "932ca6c046a0c8bd6cd9c48786600e549da3f3dd10e4eb80ddf540b52b8f4135",
    "9c81e75209c095f508f3838ac4da72bf6b5693aa2924f0d005c00e4009f72877",
    "4ef14acb5121629510d887c18414d9881f19cc7e15503365ac105013026fc4ec",
)
EXPECTED_REQUESTS_PER_REPLICATE = 140
TOTAL_COST_CAP_USD = 6.75


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def midranks(scores: Sequence[float]) -> list[float]:
    return [
        sum(other < score for other in scores)
        + 0.5 * (sum(other == score for other in scores) - 1)
        for score in scores
    ]


def _majority_index(indices: Sequence[int]) -> int:
    return max(
        sorted(set(indices)),
        key=lambda index: (indices.count(index), -index),
    )


def aggregate_score_rows(
    replicate_rows: Sequence[Sequence[dict[str, Any]]],
) -> list[dict[str, Any]]:
    if len(replicate_rows) != REPLICATE_COUNT:
        raise ValueError("exactly three score replicates are required")
    if not replicate_rows or any(
        len(rows) != len(replicate_rows[0]) for rows in replicate_rows
    ):
        raise ValueError("score replicate lengths do not align")
    aggregated = []
    for items in zip(*replicate_rows, strict=True):
        ranked = [midranks(item["scores"]) for item in items]
        width = len(ranked[0])
        if any(len(row) != width for row in ranked):
            raise ValueError("candidate score widths do not align")
        scores = [
            sum(row[index] for row in ranked) / REPLICATE_COUNT
            for index in range(width)
        ]
        best_followups = []
        if items[0].get("best_followup_indices"):
            for indices in zip(
                *(item["best_followup_indices"] for item in items),
                strict=True,
            ):
                best_followups.append(_majority_index(list(indices)))
        aggregated.append(
            {
                "scores": scores,
                "best_followup_indices": best_followups,
                "rationales": ["three-call midrank ensemble"] * width,
            }
        )
    return aggregated


def _validate_replicates(payloads: Sequence[dict[str, Any]]) -> None:
    if len(payloads) != REPLICATE_COUNT:
        raise ValueError("exactly three scorer payloads are required")
    for index, payload in enumerate(payloads, start=1):
        protocol = payload.get("protocol", {})
        usage = payload.get("usage", {})
        if payload.get("status") not in {"passed", "gate_failed"}:
            raise ValueError("scorer replicate did not complete")
        if protocol.get("replicate_index") != index:
            raise ValueError("scorer replicate index does not match")
        if tuple(protocol.get("task_ids", ())) != tuple(
            FRESH_CONFIRMATION_IDS
        ):
            raise ValueError("scorer replicate tasks do not match")
        if int(usage.get("physical_requests", -1)) != 140:
            raise ValueError("scorer replicate request count does not match")
        if int(usage.get("reasoning_tokens", -1)) != 0:
            raise ValueError("scorer replicate used reasoning")
        generator = usage.get("generator", {})
        if int(generator.get("forced_exits", 0)) != 0:
            raise ValueError("scorer replicate used forced finalization")


def build_ensemble(
    payloads: Sequence[dict[str, Any]],
    *,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    _validate_replicates(payloads)
    myopic = aggregate_score_rows(
        [payload["myopic_scores"] for payload in payloads]
    )
    nonmyopic = aggregate_score_rows(
        [payload["nonmyopic_scores"] for payload in payloads]
    )
    continuations = [
        aggregate_score_rows(
            [
                payload["continuation_scores"][case_index]
                for payload in payloads
            ]
        )
        for case_index in range(len(records))
    ]
    compatibility_usage = {
        "physical_requests": EXPECTED_REQUESTS_PER_REPLICATE,
        "reasoning_tokens": 0,
        "adapter_cost_usd": 0.0,
        "generator": {},
    }
    summary = summarize_cross_model(
        records,
        continuations,
        compatibility_usage,
        stage="confirmation",
        myopic_scores=myopic,
        nonmyopic_scores=nonmyopic,
    )
    return {
        "summary": summary,
        "myopic_scores": myopic,
        "nonmyopic_scores": nonmyopic,
        "continuation_scores": continuations,
    }


def _selected_roots(ensemble: dict[str, Any]) -> list[int]:
    return [
        row["nonmyopic_root_index"]
        for row in ensemble["summary"]["policy_diagnostics"]
    ]


def _selected_followups(ensemble: dict[str, Any]) -> list[int]:
    return [
        row["selected_followup_index"]
        for row in ensemble["summary"]["root_diagnostics"]
    ]


def analyze_blocks(
    development: Sequence[dict[str, Any]],
    confirmation: Sequence[dict[str, Any]],
    *,
    records: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    development_ensemble = build_ensemble(development, records=records)
    confirmation_ensemble = build_ensemble(confirmation, records=records)
    root_agreement = mean_pairwise_agreement(
        [
            _selected_roots(development_ensemble),
            _selected_roots(confirmation_ensemble),
        ]
    )
    focused_agreement = mean_pairwise_agreement(
        [
            _selected_followups(development_ensemble),
            _selected_followups(confirmation_ensemble),
        ]
    )
    total_requests = sum(
        int(payload["usage"]["physical_requests"])
        for payload in confirmation
    )
    total_reasoning = sum(
        int(payload["usage"]["reasoning_tokens"])
        for payload in confirmation
    )
    total_cost = sum(
        float(payload["usage"]["adapter_cost_usd"])
        for payload in confirmation
    )
    gates = {
        "all_three_exact_140_zero_reasoning_no_forced_exits": all(
            int(payload["usage"]["physical_requests"]) == 140
            and int(payload["usage"]["reasoning_tokens"]) == 0
            and int(
                payload["usage"]["generator"].get("forced_exits", 0)
            )
            == 0
            for payload in confirmation
        ),
        "confirmation_ensemble_passes_all_original_gates": bool(
            confirmation_ensemble["summary"]["gates"]["all_pass"]
        ),
        "root_agreement_with_development_at_least_0_60": (
            root_agreement >= 0.60
        ),
        "focused_agreement_with_development_at_least_0_75": (
            focused_agreement >= 0.75
        ),
        "exact_420_physical_requests": total_requests == 420,
        "zero_reasoning_tokens": total_reasoning == 0,
        "total_cost_at_most_6_75": total_cost <= TOTAL_COST_CAP_USD,
    }
    gates["all_pass"] = all(gates.values())

    def concise(ensemble: dict[str, Any]) -> dict[str, Any]:
        summary = ensemble["summary"]
        endpoint = summary["cross_model_endpoint_total"]
        return {
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
            "endpoint_total": endpoint,
            "myopic_endpoint_total": endpoint
            - summary["end_to_end_total_advantage_over_myopic"],
            "myopic_advantage": summary[
                "end_to_end_total_advantage_over_myopic"
            ],
            "myopic_wins": summary["end_to_end_myopic_win_count"],
            "myopic_losses": summary["end_to_end_myopic_loss_count"],
            "joint_advantage": summary[
                "end_to_end_total_advantage_over_joint"
            ],
            "random_advantage": summary[
                "end_to_end_total_advantage_over_random"
            ],
            "original_gates": summary["gates"],
        }

    return {
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "ensemble": "mean within-response midranks across three calls",
            "same_tasks_as_development": True,
            "confirmation_is_fresh_model_execution_only": True,
            "development_sha256": list(DEVELOPMENT_SHA256),
        },
        "summary": {
            "development": concise(development_ensemble),
            "confirmation": concise(confirmation_ensemble),
            "root_agreement_with_development": root_agreement,
            "focused_agreement_with_development": focused_agreement,
            "total_physical_requests": total_requests,
            "total_reasoning_tokens": total_reasoning,
            "total_cost_usd": total_cost,
            "gates": gates,
        },
        "confirmation_policy_diagnostics": confirmation_ensemble[
            "summary"
        ]["policy_diagnostics"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--development-replicate",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument(
        "--confirmation-replicate",
        action="append",
        type=Path,
        required=True,
    )
    parser.add_argument("--source-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if tuple(
        sha256_file(path) for path in args.development_replicate
    ) != DEVELOPMENT_SHA256:
        raise ValueError("development scorer artifact hashes do not match")
    development = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.development_replicate
    ]
    confirmation = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.confirmation_replicate
    ]
    records = load_records(args.source_artifact, stage="confirmation")
    result = analyze_blocks(
        development,
        confirmation,
        records=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
