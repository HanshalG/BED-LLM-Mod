#!/usr/bin/env python3
"""Compute paired statistics for tau-Knowledge receding confirmation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.tau_knowledge_first_link_scorer import pairwise_ranking_points
from scripts.tau_knowledge_retrieval_opportunity import analyze_record
from scripts.analyze_tau_knowledge_first_link_confirmation import (
    _quantile,
    exact_sign_flip_pvalue,
)


BOOTSTRAP_SEED = 24339
BOOTSTRAP_SAMPLES = 100_000


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _case_rows(payload: dict[str, Any]) -> list[dict[str, Any]]:
    summary = payload["summary"]
    root_rows = summary["root_diagnostics"]
    policy_rows = summary["policy_diagnostics"]
    records = payload["records"]
    myopic_scores = payload["myopic_scores"]
    nonmyopic_scores = payload["nonmyopic_scores"]
    rows = []
    for index, (record, myopic, nonmyopic, policy) in enumerate(
        zip(
            records,
            myopic_scores,
            nonmyopic_scores,
            policy_rows,
            strict=True,
        )
    ):
        endpoint = analyze_record(record)
        root_values = [max(values) for values in endpoint["pair_counts"]]
        myopic_points, comparable = pairwise_ranking_points(
            myopic["scores"], root_values
        )
        nonmyopic_points, nonmyopic_comparable = pairwise_ranking_points(
            nonmyopic["scores"], root_values
        )
        if comparable != nonmyopic_comparable:
            raise ValueError("root comparison counts do not align")
        myopic_root = max(
            range(len(root_values)),
            key=lambda root_index: myopic["scores"][root_index],
        )
        nonmyopic_root = max(
            range(len(root_values)),
            key=lambda root_index: nonmyopic["scores"][root_index],
        )
        focused = root_rows[index * 5 : (index + 1) * 5]
        rows.append(
            {
                "task_id": record["task_id"],
                "root_comparable": comparable,
                "myopic_root_points": myopic_points,
                "nonmyopic_root_points": nonmyopic_points,
                "focused_comparable": sum(
                    row["pairwise_comparable_count"] for row in focused
                ),
                "focused_points": sum(
                    row["pairwise_points"] for row in focused
                ),
                "focused_optimal_count": sum(
                    row["regret"] == 0 for row in focused
                ),
                "focused_regret": sum(row["regret"] for row in focused),
                "myopic_value": policy["myopic_receding_value"],
                "nonmyopic_value": policy["nonmyopic_receding_value"],
                "joint_value": policy["nonmyopic_joint_value"],
                "random_value": policy["random_strategy_value"],
                "myopic_root_oracle_tail": root_values[myopic_root],
                "nonmyopic_root_oracle_tail": root_values[nonmyopic_root],
                "selected_root_oracle_tail": policy[
                    "selected_root_oracle_tail"
                ],
                "oracle_first_differs_from_greedy": endpoint[
                    "oracle_first_differs_from_greedy"
                ],
                "nonmyopic_required_document_gap": endpoint[
                    "nonmyopic_required_document_gap"
                ],
                "best_one_step_value": endpoint[
                    "best_one_step_required_document_count"
                ],
                "greedy_continuation_value": endpoint[
                    "greedy_continuation_required_document_count"
                ],
                "oracle_pair_value": max(
                    max(values) for values in endpoint["pair_counts"]
                ),
            }
        )
    return rows


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = _case_rows(payload)
    root_differences = [
        row["nonmyopic_root_points"] - row["myopic_root_points"]
        for row in rows
    ]
    focused_above_chance = [
        row["focused_points"] - 0.5 * row["focused_comparable"]
        for row in rows
    ]
    root_endpoint_advantages = [
        row["nonmyopic_root_oracle_tail"]
        - row["myopic_root_oracle_tail"]
        for row in rows
    ]
    advantage_keys = {
        "myopic": "myopic_value",
        "joint": "joint_value",
        "random": "random_value",
    }
    advantages = {
        name: [
            row["nonmyopic_value"] - row[control_key] for row in rows
        ]
        for name, control_key in advantage_keys.items()
    }

    rng = random.Random(BOOTSTRAP_SEED)
    root_gains = []
    focused_accuracies = []
    focused_optimal_rates = []
    focused_mean_regrets = []
    mean_root_endpoint_advantages = []
    mean_advantages = {name: [] for name in advantage_keys}
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        root_comparable = sum(row["root_comparable"] for row in sample)
        root_gains.append(
            sum(
                row["nonmyopic_root_points"] - row["myopic_root_points"]
                for row in sample
            )
            / root_comparable
            if root_comparable
            else 0.0
        )
        focused_comparable = sum(
            row["focused_comparable"] for row in sample
        )
        focused_accuracies.append(
            sum(row["focused_points"] for row in sample)
            / focused_comparable
            if focused_comparable
            else 0.0
        )
        focused_optimal_rates.append(
            sum(row["focused_optimal_count"] for row in sample)
            / (5 * len(sample))
        )
        focused_mean_regrets.append(
            sum(row["focused_regret"] for row in sample)
            / (5 * len(sample))
        )
        mean_root_endpoint_advantages.append(
            sum(
                row["nonmyopic_root_oracle_tail"]
                - row["myopic_root_oracle_tail"]
                for row in sample
            )
            / len(sample)
        )
        for name, control_key in advantage_keys.items():
            mean_advantages[name].append(
                sum(
                    row["nonmyopic_value"] - row[control_key]
                    for row in sample
                )
                / len(sample)
            )

    return {
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "root_pairwise_accuracy_gain_95_ci": _interval(root_gains),
            "root_pairwise_accuracy_gain_probability_gt_zero": sum(
                value > 0 for value in root_gains
            )
            / BOOTSTRAP_SAMPLES,
            "focused_pairwise_accuracy_95_ci": _interval(
                focused_accuracies
            ),
            "focused_optimal_rate_95_ci": _interval(
                focused_optimal_rates
            ),
            "focused_mean_regret_95_ci": _interval(
                focused_mean_regrets
            ),
            "mean_root_endpoint_advantage_95_ci": _interval(
                mean_root_endpoint_advantages
            ),
            "mean_root_endpoint_advantage_probability_gt_zero": sum(
                value > 0 for value in mean_root_endpoint_advantages
            )
            / BOOTSTRAP_SAMPLES,
            "mean_endpoint_advantage_95_ci": {
                name: _interval(values)
                for name, values in mean_advantages.items()
            },
            "mean_endpoint_advantage_probability_gt_zero": {
                name: sum(value > 0 for value in values)
                / BOOTSTRAP_SAMPLES
                for name, values in mean_advantages.items()
            },
        },
        "exact_task_level_tests": {
            "root_ranking_label_swap_one_sided_p": exact_sign_flip_pvalue(
                root_differences
            ),
            "focused_ranking_above_chance_one_sided_p": (
                exact_sign_flip_pvalue(focused_above_chance)
            ),
            "root_endpoint_advantage_sign_flip_one_sided_p": (
                exact_sign_flip_pvalue(root_endpoint_advantages)
            ),
            "endpoint_advantage_sign_flip_one_sided_p": {
                name: exact_sign_flip_pvalue(values)
                for name, values in advantages.items()
            },
        },
        "endpoint_totals": {
            "myopic_receding": sum(row["myopic_value"] for row in rows),
            "nonmyopic_receding": sum(
                row["nonmyopic_value"] for row in rows
            ),
            "nonmyopic_joint": sum(row["joint_value"] for row in rows),
            "seeded_random": sum(row["random_value"] for row in rows),
            "myopic_root_oracle_tail": sum(
                row["myopic_root_oracle_tail"] for row in rows
            ),
            "selected_root_oracle_tail": sum(
                row["selected_root_oracle_tail"] for row in rows
            ),
            "oracle_pair": sum(row["oracle_pair_value"] for row in rows),
        },
        "structural_opportunity": {
            "oracle_first_differs_from_greedy_count": sum(
                row["oracle_first_differs_from_greedy"] for row in rows
            ),
            "positive_nonmyopic_gap_count": sum(
                row["nonmyopic_required_document_gap"] > 0 for row in rows
            ),
            "total_nonmyopic_gap": sum(
                row["nonmyopic_required_document_gap"] for row in rows
            ),
            "mean_nonmyopic_gap": sum(
                row["nonmyopic_required_document_gap"] for row in rows
            )
            / len(rows),
            "best_one_step_total": sum(
                row["best_one_step_value"] for row in rows
            ),
            "greedy_continuation_total": sum(
                row["greedy_continuation_value"] for row in rows
            ),
            "oracle_pair_total": sum(
                row["oracle_pair_value"] for row in rows
            ),
        },
        "case_diagnostics": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("confirmation", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    raw = args.confirmation.read_bytes()
    payload = json.loads(raw)
    result = {
        "confirmation_sha256": hashlib.sha256(raw).hexdigest(),
        **analyze(payload),
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
    print(encoded, end="")


if __name__ == "__main__":
    main()
