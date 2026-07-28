#!/usr/bin/env python3
"""Confirm Number Game root-conditioned generation against a global pool."""

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
from scripts.number_game_pooled_support_control import evaluate_sources
from scripts.number_game_predictive_risk_powered_replication import (
    run_powered_replication,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-pooled-support-confirmation-1"
TREE_SEEDS = tuple(range(27200, 27232))
TARGET_SEEDS = tuple(range(27300, 27332))
RUN_BUDGET_USD = 3.00
EXPECTED_REQUESTS = 576


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    base_result = run_powered_replication(
        output_dir=output_dir,
        run_id=run_id,
        tree_seeds=TREE_SEEDS,
        target_seeds=TARGET_SEEDS,
        interface_version=INTERFACE_VERSION,
        run_budget_usd=RUN_BUDGET_USD,
    )
    trees_path = output_dir / "TREES.json"
    if not trees_path.exists():
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "protocol": {"interface_version": INTERFACE_VERSION},
            "base_status": base_result.get("status"),
            "base_error": base_result.get("error"),
        }
        checkpoint(output_dir / "CONFIRMATION_FAILURE.json", failure)
        return failure

    pooled_path = output_dir / "POOLED_RESULT.json"
    pooled_document = evaluate_sources(
        tree_paths=[trees_path],
        output_path=pooled_path,
    )
    pooled = next(iter(pooled_document["sources"].values()))
    aggregate = pooled["aggregate"]
    novel = pooled["novel_target_mean_differences"]
    usage = base_result["usage"]
    trees = base_result["trees"]
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
            tree["mechanics"]["initial_valid"] >= 16 for tree in trees
        ),
        "all_branches_have_at_least_8_rules": all(
            tree["mechanics"]["minimum_branch_valid"] >= 8
            for tree in trees
        ),
        "all_trees_have_at_least_16_targets_and_8_novel": all(
            tree["mechanics"]["target_valid"] >= 16
            and tree["mechanics"]["novel_targets"] >= 8
            for tree in trees
        ),
        "root_pooled_identity_holds_on_all_trees": (
            pooled["root_pooled_identity_count"] == len(TREE_SEEDS)
        ),
        "global_pooled_root_differs_on_at_least_20_trees": (
            pooled["roots_differ"] >= 20
        ),
        "brier_gain_at_least_8_percent": (
            aggregate["relative_brier_reduction"] >= 0.08
        ),
        "brier_cluster_ci_below_zero": (
            aggregate["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "brier_wins_on_at_least_18_trees": (
            aggregate["brier_tree_wins"] >= 18
        ),
        "hamming_gain_at_least_10_percent": (
            aggregate["relative_hamming_reduction"] >= 0.10
        ),
        "hamming_cluster_ci_below_zero": (
            aggregate["tree_cluster_hamming_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "hamming_wins_on_at_least_16_trees": (
            aggregate["hamming_tree_wins"] >= 16
        ),
        "no_mean_coverage_loss": (
            aggregate["mean_coverage_difference"] >= 0.0
        ),
        "novel_targets_have_brier_and_hamming_gains": (
            novel["candidate_minus_pooled_brier"] < 0.0
            and novel["candidate_minus_pooled_hamming"] < 0.0
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "confirmation_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "planning_model": base_result["protocol"]["planning_model"],
            "target_model": base_result["protocol"]["target_model"],
            "tree_seeds": list(TREE_SEEDS),
            "target_seeds": list(TARGET_SEEDS),
            "num_trees": len(TREE_SEEDS),
            "run_budget_usd": RUN_BUDGET_USD,
        },
        "gates": gates,
        "root_differences": pooled["roots_differ"],
        "root_pooled_identity_count": pooled[
            "root_pooled_identity_count"
        ],
        "aggregate": aggregate,
        "novel_target_mean_differences": novel,
        "usage": usage,
        "base_result_status": base_result["status"],
        "base_result_sha256": hashlib.sha256(
            (output_dir / "RESULT.json").read_bytes()
        ).hexdigest(),
        "pooled_result_sha256": hashlib.sha256(
            pooled_path.read_bytes()
        ).hexdigest(),
        "trees_sha256": hashlib.sha256(
            trees_path.read_bytes()
        ).hexdigest(),
        "raw_responses_sha256": base_result["raw_responses_sha256"],
    }
    checkpoint(output_dir / "CONFIRMATION_RESULT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    result = run_confirmation(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
