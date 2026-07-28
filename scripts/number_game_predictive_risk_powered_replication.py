#!/usr/bin/env python3
"""Run a powered fresh-tree comparison of predictive-risk BED and PTS."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    BOOTSTRAP_SAMPLES,
    PLANNING_MODEL_ID,
    TARGET_MODEL_ID,
    TEMPERATURE,
    aggregate_tree_comparisons,
    run_tree,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-predictive-risk-powered-replication-1"
TREE_SEEDS = tuple(range(26400, 26432))
TARGET_SEEDS = tuple(range(26500, 26532))
EXPECTED_REQUESTS_PER_TREE = 18
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
MIN_INITIAL_VALID = 16
MIN_BRANCH_VALID = 8
MIN_TARGET_VALID = 16
MIN_NOVEL_TARGETS = 8
RUN_BUDGET_USD = 3.00


def run_powered_replication(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"trees": []}
    public_trees = []
    tree_results = []
    try:
        for tree_index, (tree_seed, target_seed) in enumerate(
            zip(TREE_SEEDS, TARGET_SEEDS, strict=True)
        ):
            tree, artifacts = run_tree(
                tree_index=tree_index,
                tree_seed=tree_seed,
                target_seed=target_seed,
                output_dir=output_dir,
                run_id=run_id,
            )
            tree_results.append(tree)
            raw["trees"].append(artifacts["raw"])
            public_trees.append(artifacts["public"])
            checkpoint(raw_path, raw)
        baselines = (
            "myopic_eig",
            "fixed_support_depth_two",
            "uniform_random_candidate_root",
            "positive_test_strategy",
        )
        aggregate = {
            baseline: aggregate_tree_comparisons(
                tree_results, baseline=baseline
            )
            for baseline in baselines
        }
        usage = {
            key: sum(
                tree["usage"].get(key, 0) for tree in tree_results
            )
            for key in (
                "adapter_requests",
                "http_attempts",
                "retry_count",
                "provider_error_retries",
                "adapter_reasoning_tokens",
                "forced_exits",
                "run_cost_usd",
            )
        }
        roots_differ_myopic = sum(
            tree["selection"]["predictive_bayes_risk_root"]
            != tree["selection"]["myopic_root"]
            for tree in tree_results
        )
        roots_differ_fixed = sum(
            tree["selection"]["predictive_bayes_risk_root"]
            != tree["selection"]["fixed_depth_two_root"]
            for tree in tree_results
        )
        novel_brier = [
            tree["novel_comparison_vs_myopic"][
                "candidate_minus_baseline_brier"
            ]
            for tree in tree_results
        ]
        novel_hamming = [
            tree["novel_comparison_vs_myopic"][
                "candidate_minus_baseline_hamming"
            ]
            for tree in tree_results
        ]
        pts = aggregate["positive_test_strategy"]
        myopic = aggregate["myopic_eig"]
        fixed = aggregate["fixed_support_depth_two"]
        random_control = aggregate["uniform_random_candidate_root"]
        gates = {
            "exact_576_accepted_requests": (
                usage["adapter_requests"] == EXPECTED_REQUESTS
            ),
            "transport_attempt_accounting_exact": (
                usage["http_attempts"]
                == usage["adapter_requests"] + usage["retry_count"]
            ),
            "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
            "zero_forced_exits": usage["forced_exits"] == 0,
            "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
            "all_trees_have_at_least_16_initial_rules": all(
                tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
                for tree in tree_results
            ),
            "all_branches_have_at_least_8_rules": all(
                tree["mechanics"]["minimum_branch_valid"]
                >= MIN_BRANCH_VALID
                for tree in tree_results
            ),
            "all_trees_have_at_least_16_targets_and_8_novel": all(
                tree["mechanics"]["target_valid"] >= MIN_TARGET_VALID
                and tree["mechanics"]["novel_targets"]
                >= MIN_NOVEL_TARGETS
                for tree in tree_results
            ),
            "predictive_root_differs_from_myopic_on_at_least_24_trees": (
                roots_differ_myopic >= 24
            ),
            "predictive_root_differs_from_fixed_on_at_least_24_trees": (
                roots_differ_fixed >= 24
            ),
            "brier_gain_vs_pts_at_least_2_percent": (
                pts["relative_brier_reduction"] >= 0.02
            ),
            "brier_cluster_ci_vs_pts_below_zero": (
                pts["tree_cluster_brier_difference_95pct_bootstrap"][1]
                < 0.0
            ),
            "brier_wins_vs_pts_on_at_least_20_trees": (
                pts["brier_tree_wins"] >= 20
            ),
            "hamming_gain_vs_pts_is_positive": (
                pts["relative_hamming_reduction"] > 0.0
            ),
            "hamming_cluster_ci_vs_pts_below_zero": (
                pts["tree_cluster_hamming_difference_95pct_bootstrap"][1]
                < 0.0
            ),
            "brier_gain_vs_myopic_at_least_10_percent": (
                myopic["relative_brier_reduction"] >= 0.10
            ),
            "brier_cluster_ci_vs_myopic_below_zero": (
                myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
                < 0.0
            ),
            "brier_wins_vs_myopic_on_at_least_24_trees": (
                myopic["brier_tree_wins"] >= 24
            ),
            "hamming_gain_vs_myopic_at_least_15_percent": (
                myopic["relative_hamming_reduction"] >= 0.15
            ),
            "no_mean_coverage_loss_vs_myopic": (
                myopic["mean_coverage_difference"] >= 0.0
            ),
            "brier_gain_vs_fixed_at_least_10_percent": (
                fixed["relative_brier_reduction"] >= 0.10
            ),
            "brier_cluster_ci_vs_fixed_below_zero": (
                fixed["tree_cluster_brier_difference_95pct_bootstrap"][1]
                < 0.0
            ),
            "brier_gain_vs_random_at_least_5_percent": (
                random_control["relative_brier_reduction"] >= 0.05
            ),
            "brier_cluster_ci_vs_random_below_zero": (
                random_control[
                    "tree_cluster_brier_difference_95pct_bootstrap"
                ][1]
                < 0.0
            ),
            "novel_targets_have_mean_brier_and_hamming_gains": (
                sum(novel_brier) < 0.0 and sum(novel_hamming) < 0.0
            ),
        }
        public = {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "planning_model": PLANNING_MODEL_ID,
                "target_model": TARGET_MODEL_ID,
                "reasoning": "disabled",
                "temperature": TEMPERATURE,
                "tree_seeds": list(TREE_SEEDS),
                "target_seeds": list(TARGET_SEEDS),
                "num_trees": len(TREE_SEEDS),
                "requests_per_tree": EXPECTED_REQUESTS_PER_TREE,
                "tree_bootstrap_samples": BOOTSTRAP_SAMPLES,
            },
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "trees": public_trees,
        }
        checkpoint(output_dir / "TREES.json", public)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "replication_failed",
            "protocol": public["protocol"],
            "gates": gates,
            "root_differences": {
                "versus_myopic": roots_differ_myopic,
                "versus_fixed_support_depth_two": roots_differ_fixed,
            },
            "aggregate": aggregate,
            "novel_target_mean_differences": {
                "candidate_minus_myopic_brier": sum(novel_brier)
                / len(novel_brier),
                "candidate_minus_myopic_hamming": sum(novel_hamming)
                / len(novel_hamming),
                "brier_tree_wins": sum(value < 0.0 for value in novel_brier),
                "hamming_tree_wins": sum(
                    value < 0.0 for value in novel_hamming
                ),
            },
            "trees": tree_results,
            "usage": usage,
            "trees_sha256": hashlib.sha256(
                (output_dir / "TREES.json").read_bytes()
            ).hexdigest(),
            "raw_responses_sha256": public["raw_responses_sha256"],
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(raw_path, raw)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "planning_model": PLANNING_MODEL_ID,
                "target_model": TARGET_MODEL_ID,
            },
            "error": f"{type(exc).__name__}: {exc}",
            "completed_trees": len(tree_results),
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
        }
        checkpoint(output_dir / "FAILURE.json", failure)
        return failure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_powered_replication(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
