#!/usr/bin/env python3
"""Analyze two-batch tau-Knowledge proposal pooling without API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tau_knowledge_execution_replication_analysis import (
    _selected_documents,
    retrieval_metrics,
)


ORIGINAL_SHA256 = (
    "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
)
REPLICATION_SHA256 = (
    "627dfe7641dca9fade90890dc90f818da1c6967c08b7878208224cd961fd7f20"
)
RANDOM_SEED = 24_398
ROOT_COUNT = 5
FOLLOWUP_COUNT = 4


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def _candidate_rows(
    payloads: Sequence[dict[str, Any]],
    task_id: str,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for execution_index, payload in enumerate(payloads):
        records = {
            record["task_id"]: (index, record)
            for index, record in enumerate(payload["records"])
        }
        case_index, record = records[task_id]
        required = record["required_documents"]
        for root_index in range(ROOT_COUNT):
            focused_scores = payload["continuation_scores"][case_index][
                root_index
            ]["scores"]
            focused_index = max(
                range(FOLLOWUP_COUNT),
                key=lambda index: (focused_scores[index], -index),
            )
            endpoint_values = [
                retrieval_metrics(
                    required,
                    _selected_documents(record, root_index, followup_index),
                )["count"]
                for followup_index in range(FOLLOWUP_COUNT)
            ]
            candidates.append(
                {
                    "execution_index": execution_index,
                    "root_index": root_index,
                    "myopic_score": payload["myopic_scores"][case_index][
                        "scores"
                    ][root_index],
                    "opaque_nonmyopic_score": payload["nonmyopic_scores"][
                        case_index
                    ]["scores"][root_index],
                    "focused_score": focused_scores[focused_index],
                    "focused_band": focused_scores[focused_index] // 30,
                    "focused_followup_index": focused_index,
                    "endpoint_value": endpoint_values[focused_index],
                    "oracle_tail_value": max(endpoint_values),
                }
            )
    return candidates


def _strategy_keys(
    candidates: list[dict[str, Any]],
) -> dict[str, Callable[[dict[str, Any]], tuple[float, ...]]]:
    myopic_ranks = _average_ranks(
        [candidate["myopic_score"] for candidate in candidates]
    )
    focused_ranks = _average_ranks(
        [candidate["focused_score"] for candidate in candidates]
    )
    for candidate, myopic_rank, focused_rank in zip(
        candidates, myopic_ranks, focused_ranks
    ):
        candidate["rank_sum"] = myopic_rank + focused_rank

    def tie(candidate: dict[str, Any]) -> tuple[int, int]:
        return (
            -candidate["execution_index"],
            -candidate["root_index"],
        )

    return {
        "opaque_full_tree": lambda row: (
            row["opaque_nonmyopic_score"],
            *tie(row),
        ),
        "myopic": lambda row: (row["myopic_score"], *tie(row)),
        "focused_only": lambda row: (row["focused_score"], *tie(row)),
        "raw_sum": lambda row: (
            row["myopic_score"] + row["focused_score"],
            *tie(row),
        ),
        "band_sum": lambda row: (
            row["myopic_score"] + 30 * row["focused_band"],
            *tie(row),
        ),
        "rank_sum": lambda row: (row["rank_sum"], *tie(row)),
        "new_document_band_then_myopic": lambda row: (
            row["focused_band"],
            row["myopic_score"],
            *tie(row),
        ),
    }


def analyze(
    original: dict[str, Any],
    replication: dict[str, Any],
) -> dict[str, Any]:
    payloads = [original, replication]
    task_ids = [record["task_id"] for record in original["records"]]
    if set(task_ids) != {
        record["task_id"] for record in replication["records"]
    }:
        raise ValueError("execution task IDs differ")

    scope_filters = {
        "original": lambda row: row["execution_index"] == 0,
        "replication": lambda row: row["execution_index"] == 1,
        "pooled": lambda row: True,
    }
    selections: dict[str, dict[str, list[float]]] = {
        scope: {} for scope in scope_filters
    }
    pooled_oracle_values: list[float] = []
    best_batch_winner_values: list[float] = []

    for task_id in task_ids:
        all_candidates = _candidate_rows(payloads, task_id)
        pooled_oracle_values.append(
            max(row["oracle_tail_value"] for row in all_candidates)
        )
        batch_winners = []
        for execution_index in range(2):
            batch = [
                row
                for row in all_candidates
                if row["execution_index"] == execution_index
            ]
            key = _strategy_keys(batch)["opaque_full_tree"]
            batch_winners.append(max(batch, key=key)["endpoint_value"])
        best_batch_winner_values.append(max(batch_winners))

        for scope, predicate in scope_filters.items():
            candidates = [
                row for row in all_candidates if predicate(row)
            ]
            for method, key in _strategy_keys(candidates).items():
                selections[scope].setdefault(method, []).append(
                    max(candidates, key=key)["endpoint_value"]
                )

    summaries: dict[str, dict[str, Any]] = {}
    for scope, methods in selections.items():
        baseline = methods["myopic"]
        summaries[scope] = {}
        for method, values in methods.items():
            differences = [
                value - control
                for value, control in zip(values, baseline)
            ]
            summaries[scope][method] = {
                "total_required_documents": sum(values),
                "gain_over_myopic": sum(differences),
                "wins_over_myopic": sum(value > 0 for value in differences),
                "losses_to_myopic": sum(value < 0 for value in differences),
                "ties_with_myopic": sum(value == 0 for value in differences),
                "task_values": dict(zip(task_ids, values)),
            }

    authorization: dict[str, dict[str, bool]] = {}
    for method in summaries["pooled"]:
        if method == "myopic":
            continue
        authorization[method] = {
            "pooled_gain_at_least_4": (
                summaries["pooled"][method]["gain_over_myopic"] >= 4
            ),
            "original_gain_nonnegative": (
                summaries["original"][method]["gain_over_myopic"] >= 0
            ),
            "replication_gain_nonnegative": (
                summaries["replication"][method]["gain_over_myopic"] >= 0
            ),
            "pooled_wins_at_least_4": (
                summaries["pooled"][method]["wins_over_myopic"] >= 4
            ),
            "pooled_losses_at_most_2": (
                summaries["pooled"][method]["losses_to_myopic"] <= 2
            ),
        }
        authorization[method]["all_pass"] = all(
            authorization[method].values()
        )

    return {
        "schema_version": 1,
        "status": (
            "passed"
            if any(row["all_pass"] for row in authorization.values())
            else "gate_failed"
        ),
        "protocol": {
            "analysis_type": "disclosed_endpoint_development",
            "api_calls": 0,
            "task_count": len(task_ids),
            "proposal_batches": 2,
            "roots_per_batch": ROOT_COUNT,
            "followups_per_root": FOLLOWUP_COUNT,
            "tie_break": "earlier execution, then earlier root",
            "random_seed_reserved_but_unused": RANDOM_SEED,
        },
        "summaries": summaries,
        "diagnostics": {
            "pooled_pair_oracle_total": sum(pooled_oracle_values),
            "best_execution_winner_oracle_total": sum(
                best_batch_winner_values
            ),
        },
        "authorization_gates": authorization,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.original) != ORIGINAL_SHA256:
        raise ValueError("original artifact hash does not match")
    if _sha256(args.replication) != REPLICATION_SHA256:
        raise ValueError("replication artifact hash does not match")
    result = analyze(
        json.loads(args.original.read_text(encoding="utf-8")),
        json.loads(args.replication.read_text(encoding="utf-8")),
    )
    result["protocol"]["original_sha256"] = ORIGINAL_SHA256
    result["protocol"]["replication_sha256"] = REPLICATION_SHA256
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
