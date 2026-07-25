#!/usr/bin/env python3
"""Compute frozen zero-call statistics for tau-Knowledge first-link results."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import random
from typing import Any, Sequence


BOOTSTRAP_SEED = 24336
BOOTSTRAP_SAMPLES = 100_000


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * probability
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (
        ordered[upper] - ordered[lower]
    ) * (position - lower)


def exact_sign_flip_pvalue(values: Sequence[float]) -> float:
    nonzero = [abs(value) for value in values if value != 0]
    if not nonzero:
        return 1.0
    observed = sum(value for value in values)
    outcomes = [
        sum(sign * value for sign, value in zip(signs, nonzero, strict=True))
        for signs in itertools.product((-1, 1), repeat=len(nonzero))
    ]
    return sum(value >= observed for value in outcomes) / len(outcomes)


def analyze(payload: dict[str, Any]) -> dict[str, Any]:
    rows = payload["summary"]["case_diagnostics"]
    advantages = [row["root_policy_advantage"] for row in rows]
    point_differences = [
        row["nonmyopic_pairwise_points"] - row["myopic_pairwise_points"]
        for row in rows
    ]
    ranking_swap_p = exact_sign_flip_pvalue(point_differences)
    endpoint_p = exact_sign_flip_pvalue(advantages)

    rng = random.Random(BOOTSTRAP_SEED)
    mean_advantages = []
    ranking_gains = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [rows[rng.randrange(len(rows))] for _row in rows]
        mean_advantages.append(
            sum(row["root_policy_advantage"] for row in sample) / len(sample)
        )
        comparable = sum(
            row["pairwise_comparable_count"] for row in sample
        )
        ranking_gains.append(
            (
                sum(
                    row["nonmyopic_pairwise_points"]
                    - row["myopic_pairwise_points"]
                    for row in sample
                )
                / comparable
            )
            if comparable
            else 0.0
        )

    selected_root_total = sum(
        row["nonmyopic_selected_root_value"] for row in rows
    )
    selected_pair_total = sum(
        row["nonmyopic_selected_pair_value"] for row in rows
    )
    return {
        "bootstrap": {
            "seed": BOOTSTRAP_SEED,
            "samples": BOOTSTRAP_SAMPLES,
            "mean_endpoint_advantage_95_ci": [
                _quantile(mean_advantages, 0.025),
                _quantile(mean_advantages, 0.975),
            ],
            "mean_endpoint_advantage_probability_gt_zero": sum(
                value > 0 for value in mean_advantages
            )
            / BOOTSTRAP_SAMPLES,
            "pairwise_accuracy_gain_95_ci": [
                _quantile(ranking_gains, 0.025),
                _quantile(ranking_gains, 0.975),
            ],
            "pairwise_accuracy_gain_probability_gt_zero": sum(
                value > 0 for value in ranking_gains
            )
            / BOOTSTRAP_SAMPLES,
        },
        "exact_task_level_tests": {
            "endpoint_advantage_sign_flip_one_sided_p": endpoint_p,
            "ranking_label_swap_one_sided_p": ranking_swap_p,
        },
        "endpoint_totals": {
            "myopic_selected_root_oracle_tail": sum(
                row["myopic_selected_root_value"] for row in rows
            ),
            "nonmyopic_selected_root_oracle_tail": selected_root_total,
            "oracle_root": sum(row["oracle_root_value"] for row in rows),
            "oracle_strength_greedy_root": sum(
                row["oracle_strength_greedy_root_value"] for row in rows
            ),
            "nonmyopic_selected_pair": selected_pair_total,
        },
        "second_link_diagnostics": {
            "documents_lost_vs_selected_root_oracle_tail": (
                selected_root_total - selected_pair_total
            ),
            "tasks_with_selected_followup_loss": sum(
                row["nonmyopic_selected_pair_value"]
                < row["nonmyopic_selected_root_value"]
                for row in rows
            ),
        },
        "root_hit_counts": {
            "myopic_oracle_optimal": sum(
                row["myopic_root_is_oracle_optimal"] for row in rows
            ),
            "nonmyopic_oracle_optimal": sum(
                row["nonmyopic_root_is_oracle_optimal"] for row in rows
            ),
            "num_tasks": len(rows),
        },
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
