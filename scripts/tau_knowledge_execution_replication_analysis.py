#!/usr/bin/env python3
"""Analyze a fresh tau-Knowledge V3.1 execution replication."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
from typing import Any, Iterable, Sequence


ORIGINAL_SHA256 = "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
ANALYSIS_SEED = 24_396
BOOTSTRAP_SAMPLES = 100_000
METRICS = ("count", "ndcg", "reciprocal_rank")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selected_documents(
    record: dict[str, Any],
    root_index: int,
    followup_index: int,
) -> list[str]:
    branch = record["first_branches"][root_index]
    return [
        result["id"]
        for result in (
            list(branch["first_results"])
            + list(branch["followups"][followup_index]["results"])
        )
    ]


def retrieval_metrics(
    required_documents: Iterable[str],
    ranked_documents: Sequence[str],
) -> dict[str, float]:
    required = set(required_documents)
    ranked_unique: list[str] = []
    for document_id in ranked_documents:
        if document_id not in ranked_unique:
            ranked_unique.append(document_id)
    relevance = [
        1.0 if document_id in required else 0.0
        for document_id in ranked_unique
    ]
    count = sum(relevance)
    recall = count / len(required) if required else 0.0
    precision = count / len(ranked_unique) if ranked_unique else 0.0
    f1 = (
        2.0 * recall * precision / (recall + precision)
        if recall + precision
        else 0.0
    )
    dcg = sum(
        relevant / math.log2(rank + 2)
        for rank, relevant in enumerate(relevance)
    )
    ideal_count = min(len(required), len(ranked_unique))
    ideal_dcg = sum(
        1.0 / math.log2(rank + 2)
        for rank in range(ideal_count)
    )
    reciprocal_rank = next(
        (
            1.0 / (rank + 1)
            for rank, relevant in enumerate(relevance)
            if relevant
        ),
        0.0,
    )
    return {
        "count": count,
        "recall": recall,
        "precision": precision,
        "f1": f1,
        "ndcg": dcg / ideal_dcg if ideal_dcg else 0.0,
        "reciprocal_rank": reciprocal_rank,
    }


def artifact_rows(payload: dict[str, Any]) -> dict[str, dict[str, dict[str, float]]]:
    records = {record["task_id"]: record for record in payload["records"]}
    rows: dict[str, dict[str, dict[str, float]]] = {}
    diagnostics = payload["summary"]["policy_diagnostics"]
    if set(records) != {row["task_id"] for row in diagnostics}:
        raise ValueError("record and diagnostic task IDs differ")
    for row in diagnostics:
        task_id = row["task_id"]
        record = records[task_id]
        focused = row["focused_followup_indices"]
        policies = {
            "nonmyopic": (
                row["nonmyopic_root_index"],
                focused[row["nonmyopic_root_index"]],
            ),
            "myopic": (
                row["myopic_root_index"],
                focused[row["myopic_root_index"]],
            ),
            "joint": (
                row["nonmyopic_root_index"],
                payload["nonmyopic_scores"][
                    list(records).index(task_id)
                ]["best_followup_indices"][row["nonmyopic_root_index"]],
            ),
            "random": (
                row["random_strategy_root_index"],
                row["random_strategy_followup_index"],
            ),
        }
        rows[task_id] = {
            name: retrieval_metrics(
                record["required_documents"],
                _selected_documents(record, *indices),
            )
            for name, indices in policies.items()
        }
    return rows


def exact_sign_flip_p(values: Sequence[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("sign-flip test requires values")
    observed = sum(values)
    signed_sums = [0.0]
    for value in values:
        signed_sums = [
            item + sign * value
            for item in signed_sums
            for sign in (-1.0, 1.0)
        ]
    return sum(
        value >= observed - 1e-12 for value in signed_sums
    ) / len(signed_sums)


def bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float]:
    rng = random.Random(seed)
    values = list(values)
    means = sorted(
        sum(rng.choice(values) for _ in values) / len(values)
        for _ in range(samples)
    )
    return [
        means[int(0.025 * samples)],
        means[int(0.975 * samples) - 1],
    ]


def _comparison(
    rows: dict[str, dict[str, dict[str, float]]],
    metric: str,
    baseline: str,
) -> dict[str, Any]:
    task_ids = sorted(rows)
    values = [
        rows[task_id]["nonmyopic"][metric]
        - rows[task_id][baseline][metric]
        for task_id in task_ids
    ]
    return {
        "metric": metric,
        "baseline": baseline,
        "task_count": len(values),
        "nonmyopic_mean": sum(
            rows[task_id]["nonmyopic"][metric] for task_id in task_ids
        )
        / len(values),
        "baseline_mean": sum(
            rows[task_id][baseline][metric] for task_id in task_ids
        )
        / len(values),
        "mean_difference": sum(values) / len(values),
        "wins": sum(value > 0 for value in values),
        "losses": sum(value < 0 for value in values),
        "ties": sum(value == 0 for value in values),
        "exact_one_sided_p": exact_sign_flip_p(values),
        "bootstrap_95_ci": bootstrap_interval(
            values,
            seed=ANALYSIS_SEED + METRICS.index(metric),
        ),
        "task_differences": dict(zip(task_ids, values)),
    }


def _clustered_rows(
    first: dict[str, dict[str, dict[str, float]]],
    second: dict[str, dict[str, dict[str, float]]],
) -> dict[str, dict[str, dict[str, float]]]:
    if set(first) != set(second):
        raise ValueError("replication task IDs do not match")
    result: dict[str, dict[str, dict[str, float]]] = {}
    for task_id in first:
        result[task_id] = {}
        for policy in first[task_id]:
            result[task_id][policy] = {
                metric: (
                    first[task_id][policy][metric]
                    + second[task_id][policy][metric]
                )
                / 2.0
                for metric in first[task_id][policy]
            }
    return result


def holm_two(pvalues: dict[str, float]) -> dict[str, Any]:
    if set(pvalues) != {"ndcg", "reciprocal_rank"}:
        raise ValueError("Holm family must contain NDCG and reciprocal rank")
    ordered = sorted(pvalues.items(), key=lambda item: item[1])
    first_name, first_p = ordered[0]
    second_name, second_p = ordered[1]
    rejected = {
        first_name: first_p <= 0.025,
        second_name: first_p <= 0.025 and second_p <= 0.05,
    }
    return {
        "raw_pvalues": pvalues,
        "holm_rejected_at_0_05": rejected,
        "any_rejected": any(rejected.values()),
    }


def analyze(original: dict[str, Any], replication: dict[str, Any]) -> dict[str, Any]:
    original_rows = artifact_rows(original)
    replication_rows = artifact_rows(replication)
    clustered = _clustered_rows(original_rows, replication_rows)
    replication_comparisons = {
        metric: _comparison(replication_rows, metric, "myopic")
        for metric in METRICS
    }
    clustered_comparisons = {
        metric: _comparison(clustered, metric, "myopic")
        for metric in METRICS
    }
    secondary = holm_two(
        {
            metric: replication_comparisons[metric]["exact_one_sided_p"]
            for metric in ("ndcg", "reciprocal_rank")
        }
    )
    usage = replication["usage"]
    gates = {
        "replication_original_v3_1_gates_pass": bool(
            replication["summary"]["gates"]["all_pass"]
        ),
        "exact_280_physical_requests": int(usage["physical_requests"]) == 280,
        "exact_280_http_attempts": int(
            usage["generator"].get(
                "http_attempts", usage["physical_requests"]
            )
        )
        == 280,
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "zero_retries": int(usage["generator"].get("retry_count", 0)) == 0,
        "zero_forced_exits": int(
            usage["generator"].get("forced_exits", 0)
        )
        == 0,
        "replication_coverage_gain_positive": (
            replication_comparisons["count"]["mean_difference"] > 0.0
        ),
        "clustered_coverage_gain_positive": (
            clustered_comparisons["count"]["mean_difference"] > 0.0
        ),
        "clustered_coverage_exact_p_at_most_0_05": (
            clustered_comparisons["count"]["exact_one_sided_p"] <= 0.05
        ),
        "adapter_cost_at_most_4": float(usage["adapter_cost_usd"]) <= 4.0,
    }
    gates["all_primary_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_primary_pass"] else "gate_failed",
        "protocol": {
            "analysis_seed": ANALYSIS_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "task_cluster_is_statistical_unit": True,
            "same_tasks_not_counted_as_independent": True,
            "primary_metric": "unique required-document coverage",
            "secondary_holm_family": ["ndcg", "reciprocal_rank"],
        },
        "replication_comparisons": replication_comparisons,
        "task_clustered_two_execution_comparisons": clustered_comparisons,
        "replication_secondary_holm": secondary,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--replication", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.original) != ORIGINAL_SHA256:
        raise ValueError("original confirmation hash does not match")
    original = json.loads(args.original.read_text(encoding="utf-8"))
    replication = json.loads(args.replication.read_text(encoding="utf-8"))
    result = analyze(original, replication)
    result["protocol"]["original_sha256"] = ORIGINAL_SHA256
    result["protocol"]["replication_sha256"] = _sha256(args.replication)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
