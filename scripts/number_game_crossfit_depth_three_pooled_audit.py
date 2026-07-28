#!/usr/bin/env python3
"""Pool independent Number Game cross-fitted depth-three endpoint studies."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-crossfit-depth-three-pooled-audit-1"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 31600
FIXED_POLICY_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_crossfit_endpoint_precision"
    / "number-game-crossfit-endpoint-precision-20260728/RESULT.json"
)
FRESH_REPLICATION_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
    / "number-game-crossfit-depth-three-fresh-replication-20260728"
    / "RESULT.json"
)
FIXED_POLICY_RESULT_SHA256 = (
    "47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e"
)
FRESH_REPLICATION_RESULT_SHA256 = (
    "25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8"
)
STUDY_NAMES = ("fixed_policy_fresh_endpoints", "fresh_tree_replication")
BASELINES = (
    "crossfit_depth_two",
    "retained_risk_set_depth_three",
    "predictive_bayes_risk_depth_two",
    "myopic_eig",
    "fixed_support_depth_three",
    "uniform_random_candidate_root",
    "positive_test_strategy",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def stratified_bootstrap_interval(
    groups: Sequence[Sequence[float]],
    *,
    seed: int,
    samples: int = BOOTSTRAP_SAMPLES,
) -> list[float]:
    if not groups or any(not group for group in groups):
        raise ValueError("bootstrap groups must be nonempty")
    rng = random.Random(seed)
    total_count = sum(len(group) for group in groups)
    values = []
    for _ in range(samples):
        total = 0.0
        for group in groups:
            total += sum(rng.choice(group) for _ in group)
        values.append(total / total_count)
    values.sort()
    return [
        values[int(0.025 * samples)],
        values[min(samples - 1, int(0.975 * samples))],
    ]


def exact_one_sided_sign_pvalue(wins: int, losses: int) -> float:
    trials = wins + losses
    if trials == 0:
        return 1.0
    return sum(
        math.comb(trials, count) for count in range(wins, trials + 1)
    ) / (2**trials)


def _study_summary(
    result: dict[str, Any],
    *,
    study_name: str,
) -> dict[str, Any]:
    primary = result["aggregate"]["comparisons"]["crossfit_depth_two"]
    return {
        "study": study_name,
        "tree_count": len(result["trees"]),
        "candidate_mean_brier": primary["candidate_mean_brier"],
        "baseline_mean_brier": primary["baseline_mean_brier"],
        "relative_brier_reduction": primary["relative_brier_reduction"],
        "mean_brier_difference": primary[
            "mean_candidate_minus_baseline_brier"
        ],
        "brier_difference_95pct": primary[
            "tree_cluster_brier_difference_95pct_bootstrap"
        ],
        "wins": sum(
            tree["comparisons"]["crossfit_depth_two"][
                "candidate_minus_baseline_brier"
            ]
            < -1e-15
            for tree in result["trees"]
        ),
        "ties": sum(
            abs(
                tree["comparisons"]["crossfit_depth_two"][
                    "candidate_minus_baseline_brier"
                ]
            )
            <= 1e-15
            for tree in result["trees"]
        ),
        "losses": sum(
            tree["comparisons"]["crossfit_depth_two"][
                "candidate_minus_baseline_brier"
            ]
            > 1e-15
            for tree in result["trees"]
        ),
        "depth_three_spearman": result["aggregate"]["ranking"][
            "crossfit_depth_three_spearman_brier"
        ]["mean"],
        "depth_two_spearman": result["aggregate"]["ranking"][
            "crossfit_depth_two_spearman_brier"
        ]["mean"],
    }


def audit_pooled(
    results: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    if len(results) != len(STUDY_NAMES):
        raise ValueError("expected exactly two source studies")
    if any(len(result["trees"]) != 32 for result in results):
        raise ValueError("each source study must contain 32 trees")
    tree_seeds = [
        {
            int(tree["tree_seed"])
            for tree in result["trees"]
        }
        for result in results
    ]
    if tree_seeds[0] & tree_seeds[1]:
        raise ValueError("source studies share planning-tree seeds")

    comparisons = {}
    for baseline_index, baseline in enumerate(BASELINES):
        brier_groups = [
            [
                tree["comparisons"][baseline][
                    "candidate_minus_baseline_brier"
                ]
                for tree in result["trees"]
            ]
            for result in results
        ]
        hamming_groups = [
            [
                tree["comparisons"][baseline][
                    "candidate_minus_baseline_hamming"
                ]
                for tree in result["trees"]
            ]
            for result in results
        ]
        coverage_groups = [
            [
                tree["comparisons"][baseline]["coverage_difference"]
                for tree in result["trees"]
            ]
            for result in results
        ]
        candidate_brier = _mean(
            [
                tree["endpoint"]["crossfit_depth_three"][
                    "mean_posterior_predictive_brier"
                ]
                for result in results
                for tree in result["trees"]
            ]
        )
        baseline_brier = _mean(
            [
                tree["endpoint"][baseline][
                    "mean_posterior_predictive_brier"
                ]
                for result in results
                for tree in result["trees"]
            ]
        )
        differences = [value for group in brier_groups for value in group]
        wins = sum(value < -1e-15 for value in differences)
        losses = sum(value > 1e-15 for value in differences)
        comparisons[baseline] = {
            "candidate_mean_brier": candidate_brier,
            "baseline_mean_brier": baseline_brier,
            "relative_brier_reduction": (
                baseline_brier - candidate_brier
            )
            / baseline_brier,
            "mean_brier_difference": _mean(differences),
            "stratified_tree_bootstrap_brier_difference_95pct": (
                stratified_bootstrap_interval(
                    brier_groups,
                    seed=BOOTSTRAP_SEED + baseline_index * 3,
                )
            ),
            "wins": wins,
            "ties": len(differences) - wins - losses,
            "losses": losses,
            "one_sided_exact_sign_pvalue_excluding_ties": (
                exact_one_sided_sign_pvalue(wins, losses)
            ),
            "mean_hamming_difference": _mean(
                [value for group in hamming_groups for value in group]
            ),
            "stratified_tree_bootstrap_hamming_difference_95pct": (
                stratified_bootstrap_interval(
                    hamming_groups,
                    seed=BOOTSTRAP_SEED + baseline_index * 3 + 1,
                )
            ),
            "mean_coverage_difference": _mean(
                [value for group in coverage_groups for value in group]
            ),
            "stratified_tree_bootstrap_coverage_difference_95pct": (
                stratified_bootstrap_interval(
                    coverage_groups,
                    seed=BOOTSTRAP_SEED + baseline_index * 3 + 2,
                )
            ),
        }

    ranking = {}
    for offset, key in enumerate(
        (
            "crossfit_depth_three_spearman_brier",
            "crossfit_depth_two_spearman_brier",
            "crossfit_depth_three_pairwise_concordance",
            "crossfit_depth_two_pairwise_concordance",
        )
    ):
        groups = [
            [tree["ranking"][key] for tree in result["trees"]]
            for result in results
        ]
        ranking[key] = {
            "mean": _mean([value for group in groups for value in group]),
            "stratified_tree_bootstrap_95pct": (
                stratified_bootstrap_interval(
                    groups,
                    seed=BOOTSTRAP_SEED + 100 + offset,
                )
            ),
        }

    novel_keys = (
        "candidate_minus_baseline_brier",
        "candidate_minus_baseline_hamming",
        "coverage_difference",
    )
    novel = {}
    for offset, key in enumerate(novel_keys):
        groups = [
            [
                tree[
                    "novel_comparison_crossfit_depth_three_vs_depth_two"
                ][key]
                for tree in result["trees"]
            ]
            for result in results
        ]
        novel[key] = {
            "mean": _mean([value for group in groups for value in group]),
            "stratified_tree_bootstrap_95pct": (
                stratified_bootstrap_interval(
                    groups,
                    seed=BOOTSTRAP_SEED + 200 + offset,
                )
            ),
        }

    primary_groups = [
        [
            tree["comparisons"]["crossfit_depth_two"][
                "candidate_minus_baseline_brier"
            ]
            for tree in result["trees"]
        ]
        for result in results
    ]
    study_difference = [
        left - right
        for left, right in zip(
            primary_groups[0],
            primary_groups[1],
            strict=True,
        )
    ]
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "posthoc_pooled_positive",
        "protocol": {
            "model_calls": 0,
            "tree_count": 64,
            "study_count": 2,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap": (
                "resample 32 trees independently within each source study"
            ),
            "source_studies_remain_separately_preregistered": True,
        },
        "studies": [
            _study_summary(result, study_name=name)
            for result, name in zip(results, STUDY_NAMES, strict=True)
        ],
        "comparisons": comparisons,
        "ranking": ranking,
        "novel_targets": novel,
        "study_effect_difference": {
            "fixed_policy_minus_fresh_replication_mean_brier_difference": (
                _mean(primary_groups[0]) - _mean(primary_groups[1])
            ),
            "paired_index_descriptive_95pct": (
                stratified_bootstrap_interval(
                    [study_difference],
                    seed=BOOTSTRAP_SEED + 300,
                )
            ),
            "note": (
                "index pairing is descriptive only; studies use independent "
                "seeds"
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixed-policy-result",
        type=Path,
        default=FIXED_POLICY_RESULT,
    )
    parser.add_argument(
        "--fresh-replication-result",
        type=Path,
        default=FRESH_REPLICATION_RESULT,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if sha256_file(args.fixed_policy_result) != FIXED_POLICY_RESULT_SHA256:
        raise ValueError("fixed-policy source hash changed")
    if (
        sha256_file(args.fresh_replication_result)
        != FRESH_REPLICATION_RESULT_SHA256
    ):
        raise ValueError("fresh-replication source hash changed")
    result = audit_pooled(
        [
            json.loads(args.fixed_policy_result.read_text()),
            json.loads(args.fresh_replication_result.read_text()),
        ]
    )
    result["source_sha256"] = {
        STUDY_NAMES[0]: FIXED_POLICY_RESULT_SHA256,
        STUDY_NAMES[1]: FRESH_REPLICATION_RESULT_SHA256,
    }
    checkpoint(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
