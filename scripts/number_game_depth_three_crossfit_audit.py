#!/usr/bin/env python3
"""Cross-fit Number Game depth-two and depth-three policy risk."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    evaluate_policy_root,
    predictive_bayes_risk_scores,
)
from scripts.number_game_ranking_fidelity_audit import (
    bootstrap_mean_interval,
    pairwise_concordance,
    spearman_correlation,
)
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
    retained_second_branches,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-depth-three-crossfit-audit-1"
VALIDATION_COUNTS = (1, 2, 4, 8, 19)
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 28250
RISK_TIE_TOLERANCE = 1e-12


def circular_validation_indices(
    tree_index: int,
    *,
    tree_count: int,
    validation_count: int,
) -> list[int]:
    if tree_count < 2:
        raise ValueError("at least two trees are required")
    if not 1 <= validation_count < tree_count:
        raise ValueError(
            "validation_count must be positive and below tree_count"
        )
    return [
        (tree_index + offset) % tree_count
        for offset in range(1, validation_count + 1)
    ]


def select_minimum_risk_root(
    roots: Sequence[int],
    risks: dict[int, float],
) -> int:
    minimum = min(risks[root] for root in roots)
    return next(
        root
        for root in roots
        if risks[root] <= minimum + RISK_TIE_TOLERANCE
    )


def _mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _target_mapping(
    hypotheses: Sequence[Any],
    *,
    prefix: str,
) -> dict[str, Any]:
    return {
        f"{prefix}_{index:03d}": hypothesis
        for index, hypothesis in enumerate(hypotheses)
    }


def _parse_tree(tree: dict[str, Any]) -> dict[str, Any]:
    initial = [_rule(item) for item in tree["initial"]]
    roots = [int(root) for root in tree["roots"]]
    first = _first_branches(tree)
    generated_second = _second_branches(
        tree,
        key="generated_second_branches",
    )
    retained_second, _, _ = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated_second,
    )
    targets = [_rule(item) for item in tree["targets"]]
    initial_extensions = {item.extension for item in initial}
    novel_targets = [
        item for item in targets if item.extension not in initial_extensions
    ]
    return {
        "tree_index": int(tree["tree_index"]),
        "tree_seed": int(tree["tree_seed"]),
        "target_seed": int(tree["target_seed"]),
        "initial": initial,
        "roots": roots,
        "first": first,
        "second": retained_second,
        "targets": targets,
        "novel_targets": novel_targets,
    }


def _endpoint_risks(
    *,
    parsed_tree: dict[str, Any],
    targets: Sequence[Any],
) -> tuple[dict[int, float], dict[int, float]]:
    roots = parsed_tree["roots"]
    target_mapping = _target_mapping(targets, prefix="target")
    depth_three = {
        root: depth.evaluate_policy_root_depth_three(
            policy=f"depth_three_root_{root}",
            root=root,
            targets=target_mapping,
            first_branches=parsed_tree["first"],
            second_branches=parsed_tree["second"],
        )["mean_posterior_predictive_brier"]
        for root in roots
    }
    depth_two = {
        root: evaluate_policy_root(
            policy=f"depth_two_root_{root}",
            root=root,
            targets=target_mapping,
            branches=parsed_tree["first"],
        )["mean_posterior_predictive_brier"]
        for root in roots
    }
    return depth_three, depth_two


def _comparison(
    *,
    rows: Sequence[dict[str, Any]],
    validation_count: int,
) -> dict[str, Any]:
    differences = [
        row["candidate_brier"] - row["baseline_brier"] for row in rows
    ]
    candidate_mean = _mean(row["candidate_brier"] for row in rows)
    baseline_mean = _mean(row["baseline_brier"] for row in rows)
    wins = sum(value < -1e-15 for value in differences)
    losses = sum(value > 1e-15 for value in differences)
    ties = len(differences) - wins - losses
    return {
        "validation_support_count": validation_count,
        "mean_candidate_brier": candidate_mean,
        "mean_baseline_brier": baseline_mean,
        "relative_brier_reduction": (
            baseline_mean - candidate_mean
        )
        / baseline_mean,
        "mean_candidate_minus_baseline_brier": _mean(differences),
        "tree_bootstrap_brier_difference_95pct": (
            bootstrap_mean_interval(
                differences,
                seed=BOOTSTRAP_SEED + validation_count,
                samples=BOOTSTRAP_SAMPLES,
            )
        ),
        "candidate_wins": wins,
        "ties": ties,
        "candidate_losses": losses,
        "root_changes": sum(
            row["candidate_root"] != row["baseline_root"] for row in rows
        ),
        "mean_candidate_oracle_regret": _mean(
            row["candidate_oracle_regret"] for row in rows
        ),
        "mean_baseline_oracle_regret": _mean(
            row["baseline_oracle_regret"] for row in rows
        ),
        "mean_candidate_risk_spearman": _mean(
            row["candidate_risk_spearman"] for row in rows
        ),
        "mean_baseline_risk_spearman": _mean(
            row["baseline_risk_spearman"] for row in rows
        ),
        "mean_candidate_pairwise_concordance": _mean(
            row["candidate_pairwise_concordance"] for row in rows
        ),
        "mean_baseline_pairwise_concordance": _mean(
            row["baseline_pairwise_concordance"] for row in rows
        ),
    }


def audit_crossfit(
    public_trees: Sequence[dict[str, Any]],
    *,
    validation_counts: Sequence[int] = VALIDATION_COUNTS,
) -> dict[str, Any]:
    parsed = [_parse_tree(tree) for tree in public_trees]
    tree_count = len(parsed)
    if max(validation_counts) >= tree_count:
        raise ValueError("validation count requires more source trees")

    risk_grid: list[list[dict[str, dict[int, float]]]] = []
    novel_endpoint_grid: list[dict[int, float]] = []
    for source in parsed:
        source_rows = []
        for target in parsed:
            depth_three_risk, depth_two_risk = _endpoint_risks(
                parsed_tree=source,
                targets=target["targets"],
            )
            source_rows.append(
                {
                    "depth_three": depth_three_risk,
                    "depth_two": depth_two_risk,
                }
            )
        risk_grid.append(source_rows)
        novel_endpoint_grid.append(
            _endpoint_risks(
                parsed_tree=source,
                targets=source["novel_targets"],
            )[0]
        )

    sensitivity: dict[str, Any] = {}
    selected_rows: dict[str, list[dict[str, Any]]] = {}
    for validation_count in validation_counts:
        rows = []
        for index, source in enumerate(parsed):
            roots = source["roots"]
            validation_indices = circular_validation_indices(
                index,
                tree_count=tree_count,
                validation_count=validation_count,
            )
            candidate_risks = {
                root: _mean(
                    risk_grid[index][target_index]["depth_three"][root]
                    for target_index in validation_indices
                )
                for root in roots
            }
            baseline_risks = {
                root: _mean(
                    risk_grid[index][target_index]["depth_two"][root]
                    for target_index in validation_indices
                )
                for root in roots
            }
            candidate_root = select_minimum_risk_root(
                roots,
                candidate_risks,
            )
            baseline_root = select_minimum_risk_root(
                roots,
                baseline_risks,
            )
            endpoint_risks = risk_grid[index][index]["depth_three"]
            endpoint_values = [endpoint_risks[root] for root in roots]
            oracle_brier = min(endpoint_values)
            candidate_risk_values = [
                candidate_risks[root] for root in roots
            ]
            baseline_risk_values = [
                baseline_risks[root] for root in roots
            ]
            rows.append(
                {
                    "tree_index": source["tree_index"],
                    "tree_seed": source["tree_seed"],
                    "target_seed": source["target_seed"],
                    "validation_tree_indices": validation_indices,
                    "candidate_root": candidate_root,
                    "baseline_root": baseline_root,
                    "candidate_brier": endpoint_risks[candidate_root],
                    "baseline_brier": endpoint_risks[baseline_root],
                    "candidate_novel_brier": novel_endpoint_grid[index][
                        candidate_root
                    ],
                    "baseline_novel_brier": novel_endpoint_grid[index][
                        baseline_root
                    ],
                    "candidate_oracle_regret": (
                        endpoint_risks[candidate_root] - oracle_brier
                    ),
                    "baseline_oracle_regret": (
                        endpoint_risks[baseline_root] - oracle_brier
                    ),
                    "candidate_risk_spearman": spearman_correlation(
                        candidate_risk_values,
                        endpoint_values,
                    ),
                    "baseline_risk_spearman": spearman_correlation(
                        baseline_risk_values,
                        endpoint_values,
                    ),
                    "candidate_pairwise_concordance": (
                        pairwise_concordance(
                            candidate_risk_values,
                            endpoint_values,
                        )
                    ),
                    "baseline_pairwise_concordance": (
                        pairwise_concordance(
                            baseline_risk_values,
                            endpoint_values,
                        )
                    ),
                }
            )
        comparison = _comparison(
            rows=rows,
            validation_count=validation_count,
        )
        novel_differences = [
            row["candidate_novel_brier"] - row["baseline_novel_brier"]
            for row in rows
        ]
        comparison["mean_novel_candidate_minus_baseline_brier"] = _mean(
            novel_differences
        )
        comparison["novel_brier_difference_95pct"] = (
            bootstrap_mean_interval(
                novel_differences,
                seed=BOOTSTRAP_SEED + 100 + validation_count,
                samples=BOOTSTRAP_SAMPLES,
            )
        )
        sensitivity[str(validation_count)] = comparison
        selected_rows[str(validation_count)] = rows

    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "posthoc_development",
        "protocol": {
            "tree_count": tree_count,
            "validation_counts": list(validation_counts),
            "validation_assignment": (
                "next k target supports cyclically, excluding own support"
            ),
            "endpoint": "each tree's own held-out target support",
            "candidate": "cross-fitted depth-three Brier risk",
            "baseline": "cross-fitted depth-two Brier risk",
            "selection": "minimum mean validation Brier, source root order tie-break",
            "risk_tie_tolerance": RISK_TIE_TOLERANCE,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "model_calls": 0,
        },
        "sensitivity": sensitivity,
        "trees": selected_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = json.loads(args.input.read_text())
    result = audit_crossfit(payload["trees"])
    result["input_sha256"] = hashlib.sha256(
        args.input.read_bytes()
    ).hexdigest()
    checkpoint(args.output, result)
    print(json.dumps(result["sensitivity"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
