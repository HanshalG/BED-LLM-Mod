#!/usr/bin/env python3
"""Replay exact endpoints under tau future-subtree derangements."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.analyze_tau_knowledge_first_link_confirmation import (
    exact_sign_flip_pvalue,
)
from scripts.tau_knowledge_first_link_scorer import pairwise_ranking_points
from scripts.tau_knowledge_future_alignment_ablation import (
    derange_future_subtrees,
)
from scripts.tau_knowledge_retrieval_opportunity import analyze_record


SOURCE_CONFIRMATION_SHA256 = (
    "f8b120b0d2b98f9f10eb0ef9251262a6a41b20d4d90d724ee4926c141ec275ae"
)
ALIGNMENT_CONFIRMATION_SHA256 = (
    "97d820fb59874370f9ea5a04992398af85503389753a46b9af44fba079b1e736"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _argmax(values: Sequence[int]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _condition_metrics(
    records: Sequence[dict[str, Any]],
    scores: Sequence[dict[str, list[int]]],
) -> dict[str, Any]:
    if len(records) != len(scores):
        raise ValueError("record and score counts differ")
    total_points = 0.0
    total_comparable = 0
    selected_root_total = 0
    selected_pair_total = 0
    followup_optimal_count = 0
    followup_regret = 0
    task_rows = []
    for record, score in zip(records, scores, strict=True):
        endpoint = analyze_record(record)
        pair_values = endpoint["pair_counts"]
        root_values = [max(values) for values in pair_values]
        if len(score["scores"]) != len(root_values):
            raise ValueError("root score count differs from endpoint")
        if len(score["best_followup_indices"]) != len(root_values):
            raise ValueError("follow-up choice count differs from endpoint")
        points, comparable = pairwise_ranking_points(
            score["scores"],
            root_values,
        )
        selected_root = _argmax(score["scores"])
        selected_followup = score["best_followup_indices"][selected_root]
        if not 0 <= selected_followup < len(pair_values[selected_root]):
            raise ValueError("selected follow-up is outside endpoint support")
        selected_root_value = root_values[selected_root]
        selected_pair_value = pair_values[selected_root][selected_followup]
        task_followup_optimal = 0
        task_followup_regret = 0
        for values, followup_index in zip(
            pair_values,
            score["best_followup_indices"],
            strict=True,
        ):
            if not 0 <= followup_index < len(values):
                raise ValueError("follow-up choice is outside endpoint support")
            optimum = max(values)
            task_followup_optimal += values[followup_index] == optimum
            task_followup_regret += optimum - values[followup_index]
        oracle_pair = max(root_values)
        normalized_selected_pair = (
            selected_pair_value / oracle_pair if oracle_pair else 0.0
        )
        total_points += points
        total_comparable += comparable
        selected_root_total += selected_root_value
        selected_pair_total += selected_pair_value
        followup_optimal_count += task_followup_optimal
        followup_regret += task_followup_regret
        task_rows.append(
            {
                "task_id": record["task_id"],
                "root_points": points,
                "root_comparable": comparable,
                "root_accuracy": points / comparable if comparable else 0.0,
                "selected_root_index": selected_root,
                "selected_followup_index": selected_followup,
                "selected_root_value": selected_root_value,
                "selected_pair_value": selected_pair_value,
                "oracle_pair_value": oracle_pair,
                "normalized_selected_pair": normalized_selected_pair,
                "followup_optimal_count": task_followup_optimal,
                "followup_regret": task_followup_regret,
            }
        )
    return {
        "root_pairwise_accuracy": (
            total_points / total_comparable if total_comparable else 0.0
        ),
        "root_pairwise_points": total_points,
        "root_pairwise_comparable_count": total_comparable,
        "selected_root_oracle_tail_total": selected_root_total,
        "selected_pair_total": selected_pair_total,
        "best_followup_optimal_count": followup_optimal_count,
        "best_followup_total_regret": followup_regret,
        "task_rows": task_rows,
    }


def _permutations_match(
    generated: Sequence[dict[str, Any]],
    stored: Sequence[dict[str, Any]],
) -> bool:
    return [
        {
            "task_id": row["task_id"],
            "source_branch_for_target": row["source_branch_for_target"],
            "is_derangement": row["is_derangement"],
            "future_subtree_multiset_preserved": row[
                "future_subtree_multiset_preserved"
            ],
        }
        for row in generated
    ] == list(stored)


def analyze(
    source: dict[str, Any],
    alignment: dict[str, Any],
) -> dict[str, Any]:
    records = source["records"]
    transformed, permutations = derange_future_subtrees(records)
    if not _permutations_match(permutations, alignment["permutations"]):
        raise ValueError("stored future-subtree permutations do not reproduce")
    aligned_scores = alignment["aligned_scores"]
    shuffled_scores = alignment["shuffled_scores"]
    conditions = {
        "aligned_scores_on_aligned_endpoints": _condition_metrics(
            records,
            aligned_scores,
        ),
        "shuffled_scores_on_aligned_endpoints": _condition_metrics(
            records,
            shuffled_scores,
        ),
        "shuffled_scores_on_counterfactual_endpoints": _condition_metrics(
            transformed,
            shuffled_scores,
        ),
        "aligned_scores_on_counterfactual_endpoints": _condition_metrics(
            transformed,
            aligned_scores,
        ),
    }
    shuffled_aligned = conditions[
        "shuffled_scores_on_aligned_endpoints"
    ]
    shuffled_counterfactual = conditions[
        "shuffled_scores_on_counterfactual_endpoints"
    ]
    root_task_differences = [
        own["root_accuracy"] - mismatched["root_accuracy"]
        for own, mismatched in zip(
            shuffled_counterfactual["task_rows"],
            shuffled_aligned["task_rows"],
            strict=True,
        )
    ]
    normalized_pair_differences = [
        own["normalized_selected_pair"]
        - mismatched["normalized_selected_pair"]
        for own, mismatched in zip(
            shuffled_counterfactual["task_rows"],
            shuffled_aligned["task_rows"],
            strict=True,
        )
    ]
    accuracy_gain = (
        shuffled_counterfactual["root_pairwise_accuracy"]
        - shuffled_aligned["root_pairwise_accuracy"]
    )
    root_p = exact_sign_flip_pvalue(root_task_differences)
    pair_p = exact_sign_flip_pvalue(normalized_pair_differences)
    strong_checks = {
        "counterfactual_root_accuracy_at_least_point_60": (
            shuffled_counterfactual["root_pairwise_accuracy"] >= 0.60
        ),
        "counterfactual_accuracy_gain_at_least_point_05": (
            accuracy_gain >= 0.05
        ),
        "root_task_sign_flip_p_at_most_point_05": root_p <= 0.05,
        "counterfactual_followup_optimal_at_least_70": (
            shuffled_counterfactual["best_followup_optimal_count"] >= 70
        ),
    }
    if all(strong_checks.values()):
        classification = "strong_intervention_consistency"
    elif (
        shuffled_counterfactual["root_pairwise_accuracy"] >= 0.55
        and accuracy_gain > 0.0
    ):
        classification = "directional_intervention_consistency"
    else:
        classification = "null_or_adverse"
    return {
        "status": "posthoc_counterfactual_replay",
        "classification": classification,
        "intervention_reproduced": {
            "permutations_match": True,
            "all_derangements": all(
                row["is_derangement"] for row in permutations
            ),
            "all_future_subtree_multisets_preserved": all(
                row["future_subtree_multiset_preserved"]
                for row in permutations
            ),
        },
        "conditions": conditions,
        "contrasts": {
            "shuffled_own_minus_aligned_endpoint_root_accuracy": (
                accuracy_gain
            ),
            "shuffled_own_minus_aligned_endpoint_selected_pair_total": (
                shuffled_counterfactual["selected_pair_total"]
                - shuffled_aligned["selected_pair_total"]
            ),
            "shuffled_own_minus_aligned_endpoint_root_task_sign_flip_p": (
                root_p
            ),
            "shuffled_own_minus_aligned_endpoint_normalized_pair_sign_flip_p": (
                pair_p
            ),
            "root_task_differences": root_task_differences,
            "normalized_selected_pair_differences": (
                normalized_pair_differences
            ),
        },
        "strong_intervention_consistency_checks": strong_checks,
        "limitations": [
            "post hoc analysis on open confirmation tasks",
            "complete subtrees move beliefs, queries, and documents together",
            "does not isolate refreshed belief text",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-confirmation", type=Path, required=True)
    parser.add_argument("--alignment-confirmation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if _sha256(args.source_confirmation) != SOURCE_CONFIRMATION_SHA256:
        raise ValueError("V3.1 confirmation artifact hash mismatch")
    if (
        _sha256(args.alignment_confirmation)
        != ALIGNMENT_CONFIRMATION_SHA256
    ):
        raise ValueError("future-alignment artifact hash mismatch")
    source = json.loads(args.source_confirmation.read_text(encoding="utf-8"))
    alignment = json.loads(
        args.alignment_confirmation.read_text(encoding="utf-8")
    )
    result = {
        "source_confirmation_sha256": SOURCE_CONFIRMATION_SHA256,
        "alignment_confirmation_sha256": ALIGNMENT_CONFIRMATION_SHA256,
        **analyze(source, alignment),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
