#!/usr/bin/env python3
"""Audit future-first tau policy robustness across six frozen scorer runs."""

from __future__ import annotations

import argparse
from collections import Counter
from fractions import Fraction
import hashlib
from itertools import combinations
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tau_knowledge_first_link_scorer import pairwise_ranking_points
from scripts.tau_knowledge_retrieval_opportunity import analyze_record


SOURCE_SHA256 = (
    "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
)
REPLICATE_SHA256 = (
    "932ca6c046a0c8bd6cd9c48786600e549da3f3dd10e4eb80ddf540b52b8f4135",
    "9c81e75209c095f508f3838ac4da72bf6b5693aa2924f0d005c00e4009f72877",
    "4ef14acb5121629510d887c18414d9881f19cc7e15503365ac105013026fc4ec",
    "d28a863fa4d6fce30aa2cecbba9c65c8c7c46613620db87d55bca1becec5c953",
    "0cccf93296ac336b2ea587f7206cb22059cfd789c27aab075aa6add9f3a106f6",
    "607adc749723bdd5490a111989d839432f1aa56a192a0b554af49913e4ba726a",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def argmax(values: Sequence[int]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def future_first_root(
    myopic_scores: Sequence[int],
    full_scores: Sequence[int],
) -> int:
    if len(myopic_scores) != len(full_scores) or not myopic_scores:
        raise ValueError("root score arrays must have equal nonzero length")
    uplift = [
        full - myopic
        for full, myopic in zip(full_scores, myopic_scores, strict=True)
    ]
    return max(
        range(len(uplift)),
        key=lambda index: (
            uplift[index],
            myopic_scores[index],
            -index,
        ),
    )


def mean_pairwise_agreement(selections: Sequence[Sequence[int]]) -> float:
    if len(selections) < 2:
        raise ValueError("at least two selection vectors are required")
    width = len(selections[0])
    if width == 0 or any(len(row) != width for row in selections):
        raise ValueError("selection vectors must have equal nonzero length")
    values = [
        sum(left == right for left, right in zip(a, b, strict=True)) / width
        for a, b in combinations(selections, 2)
    ]
    return sum(values) / len(values)


def exact_one_sided_sign_flip_p(values: Sequence[Fraction]) -> float:
    nonzero = [abs(value) for value in values if value]
    if not nonzero:
        return 1.0
    observed = sum(values, Fraction())
    distribution: Counter[Fraction] = Counter({Fraction(): 1})
    for value in nonzero:
        updated: Counter[Fraction] = Counter()
        for subtotal, count in distribution.items():
            updated[subtotal - value] += count
            updated[subtotal + value] += count
        distribution = updated
    extreme = sum(
        count for subtotal, count in distribution.items() if subtotal >= observed
    )
    return extreme / (2 ** len(nonzero))


def _score_array(payload: dict[str, Any], expected: int) -> list[int]:
    values = payload.get("scores")
    if (
        not isinstance(values, list)
        or len(values) != expected
        or any(isinstance(value, bool) or not isinstance(value, int) for value in values)
    ):
        raise ValueError("score array is incomplete or nonintegral")
    return values


def _replicate_rows(
    source: dict[str, Any],
    replicate: dict[str, Any],
    replicate_index: int,
) -> list[dict[str, Any]]:
    records = source["records"]
    task_ids = [str(record["task_id"]) for record in records]
    protocol = replicate.get("protocol", {})
    if protocol.get("source_artifact_sha256") != SOURCE_SHA256:
        raise ValueError("replicate source hash does not match")
    if list(protocol.get("task_ids", ())) != task_ids:
        raise ValueError("replicate task order does not match source")
    if replicate.get("status") not in {"passed", "gate_failed"}:
        raise ValueError("replicate does not contain a completed payload")

    myopic_payloads = replicate.get("myopic_scores")
    full_payloads = replicate.get("nonmyopic_scores")
    continuation_payloads = replicate.get("continuation_scores")
    if not (
        isinstance(myopic_payloads, list)
        and isinstance(full_payloads, list)
        and isinstance(continuation_payloads, list)
        and len(myopic_payloads)
        == len(full_payloads)
        == len(continuation_payloads)
        == len(records)
    ):
        raise ValueError("replicate score blocks are incomplete")

    root_diagnostics = replicate["summary"]["root_diagnostics"]
    diagnostics = {
        (str(row["task_id"]), int(row["root_index"])): row
        for row in root_diagnostics
    }
    if len(diagnostics) != len(records) * 5:
        raise ValueError("replicate root diagnostics are incomplete")
    policy_diagnostics = {
        str(row["task_id"]): row
        for row in replicate["summary"]["policy_diagnostics"]
    }
    if len(policy_diagnostics) != len(records):
        raise ValueError("replicate policy diagnostics are incomplete")

    rows = []
    for task_index, record in enumerate(records):
        task_id = task_ids[task_index]
        endpoint = analyze_record(record)
        immediate_values = list(endpoint["one_step_counts"])
        pair_values = [list(values) for values in endpoint["pair_counts"]]
        if len(immediate_values) != 5 or any(len(values) != 4 for values in pair_values):
            raise ValueError("source endpoint arrays have unexpected shape")
        total_values = [max(values) for values in pair_values]
        future_values = [
            total - immediate
            for total, immediate in zip(
                total_values, immediate_values, strict=True
            )
        ]

        myopic_scores = _score_array(myopic_payloads[task_index], 5)
        full_scores = _score_array(full_payloads[task_index], 5)
        focused_scores = [
            _score_array(payload, 4)
            for payload in continuation_payloads[task_index]
        ]
        if len(focused_scores) != 5:
            raise ValueError("focused score block has unexpected shape")

        for root_index in range(5):
            diagnostic = diagnostics[(task_id, root_index)]
            if list(diagnostic["pair_values"]) != pair_values[root_index]:
                raise ValueError("replicate pair values do not reproduce source")
            if int(diagnostic["oracle_value"]) != total_values[root_index]:
                raise ValueError("replicate root oracle does not reproduce source")

        uplift_scores = [
            full - myopic
            for full, myopic in zip(
                full_scores, myopic_scores, strict=True
            )
        ]
        points, comparable = pairwise_ranking_points(
            uplift_scores, future_values
        )
        roots = {
            "future_first": future_first_root(myopic_scores, full_scores),
            "raw_full": argmax(full_scores),
            "myopic": argmax(myopic_scores),
        }
        selections: dict[str, dict[str, int]] = {}
        for name, root_index in roots.items():
            followup_index = argmax(focused_scores[root_index])
            selections[name] = {
                "root_index": root_index,
                "followup_index": followup_index,
                "value": pair_values[root_index][followup_index],
            }

        frozen_random = policy_diagnostics[task_id]
        random_root = int(frozen_random["random_strategy_root_index"])
        random_followup = int(
            frozen_random["random_strategy_followup_index"]
        )
        random_value = pair_values[random_root][random_followup]
        if random_value != int(frozen_random["random_strategy_value"]):
            raise ValueError("replicate random endpoint does not reproduce source")
        selections["random"] = {
            "root_index": random_root,
            "followup_index": random_followup,
            "value": random_value,
        }
        rows.append(
            {
                "replicate_index": replicate_index,
                "task_id": task_id,
                "immediate_values": immediate_values,
                "total_values": total_values,
                "future_values": future_values,
                "uplift_scores": uplift_scores,
                "uplift_points": points,
                "uplift_comparable": comparable,
                "selections": selections,
            }
        )
    return rows


def analyze(
    source: dict[str, Any],
    replicates: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(replicates) != 6:
        raise ValueError("exactly six scorer replicates are required")
    replicate_rows = [
        _replicate_rows(source, payload, index)
        for index, payload in enumerate(replicates, start=1)
    ]
    policies = ("future_first", "raw_full", "myopic", "random")
    replicate_summaries = []
    root_vectors = []
    all_points = 0.0
    all_comparable = 0
    per_task_differences: dict[str, dict[str, list[int]]] = {
        str(record["task_id"]): {
            "myopic": [],
            "raw_full": [],
            "random": [],
        }
        for record in source["records"]
    }
    for index, rows in enumerate(replicate_rows, start=1):
        points = sum(row["uplift_points"] for row in rows)
        comparable = sum(row["uplift_comparable"] for row in rows)
        all_points += points
        all_comparable += comparable
        totals = {
            policy: sum(
                row["selections"][policy]["value"] for row in rows
            )
            for policy in policies
        }
        differences = {
            control: totals["future_first"] - totals[control]
            for control in ("myopic", "raw_full", "random")
        }
        counts = {}
        for control in ("myopic", "raw_full", "random"):
            task_values = [
                row["selections"]["future_first"]["value"]
                - row["selections"][control]["value"]
                for row in rows
            ]
            counts[control] = {
                "wins": sum(value > 0 for value in task_values),
                "ties": sum(value == 0 for value in task_values),
                "losses": sum(value < 0 for value in task_values),
            }
            for row, value in zip(rows, task_values, strict=True):
                per_task_differences[row["task_id"]][control].append(value)
        root_vectors.append(
            [
                row["selections"]["future_first"]["root_index"]
                for row in rows
            ]
        )
        replicate_summaries.append(
            {
                "replicate_index": index,
                "block": "scorer_retest" if index <= 3 else "rank_ensemble",
                "future_gain_pairwise_accuracy": (
                    points / comparable if comparable else 0.0
                ),
                "future_gain_points": points,
                "future_gain_comparable": comparable,
                "endpoint_totals": totals,
                "future_first_advantages": differences,
                "task_counts": counts,
            }
        )

    block_accuracies = {}
    for name, start in (("scorer_retest", 0), ("rank_ensemble", 3)):
        block = replicate_summaries[start : start + 3]
        block_accuracies[name] = sum(
            row["future_gain_pairwise_accuracy"] for row in block
        ) / 3
    mean_totals = {
        policy: sum(
            row["endpoint_totals"][policy] for row in replicate_summaries
        )
        / 6
        for policy in policies
    }
    mean_advantages = {
        control: mean_totals["future_first"] - mean_totals[control]
        for control in ("myopic", "raw_full", "random")
    }
    nonnegative_counts = {
        control: sum(
            row["future_first_advantages"][control] >= 0
            for row in replicate_summaries
        )
        for control in ("myopic", "raw_full", "random")
    }
    clustered = {}
    for control in ("myopic", "raw_full", "random"):
        task_means = [
            Fraction(sum(values), len(values))
            for values in (
                per_task_differences[task_id][control]
                for task_id in per_task_differences
            )
        ]
        clustered[control] = {
            "mean_difference": float(sum(task_means, Fraction()) / len(task_means)),
            "wins": sum(value > 0 for value in task_means),
            "ties": sum(value == 0 for value in task_means),
            "losses": sum(value < 0 for value in task_means),
            "one_sided_sign_flip_p": exact_one_sided_sign_flip_p(task_means),
        }

    pooled_accuracy = all_points / all_comparable if all_comparable else 0.0
    root_agreement = mean_pairwise_agreement(root_vectors)
    strong_gates = {
        "scorer_retest_block_accuracy_at_least_0_55": (
            block_accuracies["scorer_retest"] >= 0.55
        ),
        "rank_ensemble_block_accuracy_at_least_0_55": (
            block_accuracies["rank_ensemble"] >= 0.55
        ),
        "pooled_accuracy_at_least_0_60": pooled_accuracy >= 0.60,
        "future_first_root_agreement_at_least_0_50": (
            root_agreement >= 0.50
        ),
        "mean_gain_vs_myopic_at_least_2": (
            mean_advantages["myopic"] >= 2
        ),
        "mean_gain_vs_raw_full_at_least_1": (
            mean_advantages["raw_full"] >= 1
        ),
        "at_least_4_nonnegative_replicates_vs_myopic": (
            nonnegative_counts["myopic"] >= 4
        ),
        "at_least_4_nonnegative_replicates_vs_raw_full": (
            nonnegative_counts["raw_full"] >= 4
        ),
        "task_clustered_p_vs_myopic_at_most_0_10": (
            clustered["myopic"]["one_sided_sign_flip_p"] <= 0.10
        ),
    }
    directional_gates = {
        "both_block_accuracies_at_least_0_50": all(
            value >= 0.50 for value in block_accuracies.values()
        ),
        "pooled_accuracy_at_least_0_55": pooled_accuracy >= 0.55,
        "positive_mean_gain_vs_myopic": mean_advantages["myopic"] > 0,
        "positive_mean_gain_vs_raw_full": mean_advantages["raw_full"] > 0,
        "at_least_3_nonnegative_replicates_vs_myopic": (
            nonnegative_counts["myopic"] >= 3
        ),
        "at_least_3_nonnegative_replicates_vs_raw_full": (
            nonnegative_counts["raw_full"] >= 3
        ),
    }
    if all(strong_gates.values()):
        classification = "strong_robustness_pass"
    elif all(directional_gates.values()):
        classification = "directional"
    else:
        classification = "null_or_adverse"
    return {
        "schema_version": 1,
        "status": "completed",
        "classification": classification,
        "protocol": {
            "source_sha256": SOURCE_SHA256,
            "replicate_sha256": list(REPLICATE_SHA256),
            "model_calls": 0,
            "same_open_tasks_only": True,
            "future_first_tie_break": [
                "full_minus_myopic",
                "myopic",
                "root_order",
            ],
        },
        "summary": {
            "block_mean_future_gain_pairwise_accuracy": block_accuracies,
            "pooled_future_gain_pairwise_accuracy": pooled_accuracy,
            "pooled_future_gain_points": all_points,
            "pooled_future_gain_comparable": all_comparable,
            "future_first_root_agreement": root_agreement,
            "mean_endpoint_totals": mean_totals,
            "mean_future_first_advantages": mean_advantages,
            "nonnegative_replicate_counts": nonnegative_counts,
            "task_clustered": clustered,
            "strong_gates": strong_gates,
            "directional_gates": directional_gates,
        },
        "replicates": replicate_summaries,
        "limitations": [
            "post hoc same-task audit on already-open V3.1 trees and endpoints",
            "six scorer runs share tasks and generated retrieval trees",
            "a pass is mechanism robustness rather than task generalization",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--replicate", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.source) != SOURCE_SHA256:
        raise ValueError("source artifact hash mismatch")
    if len(args.replicate) != len(REPLICATE_SHA256):
        raise ValueError("exactly six replicate paths are required")
    for path, expected in zip(
        args.replicate, REPLICATE_SHA256, strict=True
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"replicate artifact hash mismatch: {path}")
    source = json.loads(args.source.read_text(encoding="utf-8"))
    replicates = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in args.replicate
    ]
    result = analyze(source, replicates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
