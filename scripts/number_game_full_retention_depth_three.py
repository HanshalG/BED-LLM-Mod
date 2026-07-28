#!/usr/bin/env python3
"""Run powered Number Game depth three with retention at every refresh."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Sequence
import urllib.request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_depth_three_development as depth
from scripts.number_game_retained_depth_three import (
    BRIER_TOLERANCE,
    BOOTSTRAP_SAMPLES,
    _usage,
    aggregate_scored_trees,
    score_public_tree,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-full-retention-depth-three-1"
TREE_SEEDS = tuple(range(28000, 28020))
TARGET_SEEDS = tuple(range(28100, 28120))
EXPECTED_REQUESTS_PER_TREE = 50
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
RUN_BUDGET_USD = 3.60
MIN_STARTING_BALANCE_USD = 3.25
MIN_INITIAL_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 8
MIN_TARGET_VALID = 16
MIN_NOVEL_TARGETS = 8


def openrouter_remaining_credit() -> float:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/credits",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urllib.request.urlopen(request, timeout=30.0) as response:
        payload = json.loads(response.read())
    data = payload.get("data") if isinstance(payload, dict) else None
    if (
        not isinstance(data, dict)
        or not isinstance(data.get("total_credits"), (int, float))
        or not isinstance(data.get("total_usage"), (int, float))
    ):
        raise RuntimeError("OpenRouter credits response has invalid fields")
    return float(data["total_credits"]) - float(data["total_usage"])


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projected run cost"
        )


def powered_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    live_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, bool]:
    comparisons = aggregate["comparisons"]
    depth_two = comparisons["predictive_bayes_risk_depth_two"]
    parent = comparisons["retained_parent_only_depth_three"]
    generated = comparisons["generated_only_depth_three"]
    myopic = comparisons["myopic_eig"]
    fixed = comparisons["fixed_support_depth_three"]
    random_control = comparisons["uniform_random_candidate_root"]
    roots = aggregate["root_differences"]
    novel = aggregate["novel_target_mean_differences"]
    return {
        "exact_1000_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
            for tree in scored_trees
        ),
        "all_fully_retained_first_supports_have_at_least_eight_rules": all(
            tree["mechanics"]["minimum_first_branch_valid"]
            >= MIN_FIRST_BRANCH_VALID
            for tree in scored_trees
        ),
        "all_fully_retained_second_supports_have_at_least_eight_rules": all(
            tree["mechanics"]["minimum_retained_second_branch_valid"]
            >= MIN_SECOND_BRANCH_VALID
            for tree in scored_trees
        ),
        "all_target_supports_valid": all(
            tree["mechanics"]["target_valid"] >= MIN_TARGET_VALID
            and tree["mechanics"]["novel_targets"] >= MIN_NOVEL_TARGETS
            for tree in scored_trees
        ),
        "live_and_public_support_minima_agree": all(
            live["mechanics"]["minimum_first_branch_valid"]
            == scored["mechanics"]["minimum_first_branch_valid"]
            and live["mechanics"]["minimum_second_branch_valid"]
            == scored["mechanics"][
                "minimum_retained_second_branch_valid"
            ]
            for live, scored in zip(
                live_trees,
                scored_trees,
                strict=True,
            )
        ),
        "retention_adds_parent_particles_at_both_refreshes": all(
            scored["mechanics"]["mean_first_branch_valid"]
            > scored["mechanics"]["mean_generated_first_branch_valid"]
            and scored["mechanics"]["mean_retained_second_branch_valid"]
            > scored["mechanics"]["mean_generated_second_branch_valid"]
            for scored in scored_trees
        ),
        "depth_three_root_differs_from_depth_two_on_at_least_six_trees": (
            roots["predictive_bayes_risk_depth_two_root"] >= 6
        ),
        "brier_gain_vs_depth_two_at_least_one_percent": (
            depth_two["relative_brier_reduction"] >= 0.01
        ),
        "brier_cluster_ci_vs_depth_two_below_zero": (
            depth_two["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "brier_wins_vs_depth_two_on_at_least_six_trees": (
            depth_two["brier_tree_wins"] >= 6
        ),
        "no_mean_hamming_regression_vs_depth_two": (
            depth_two["mean_candidate_minus_baseline_hamming"] <= 0.0
        ),
        "no_mean_coverage_regression_vs_depth_two": (
            depth_two["mean_coverage_difference"] >= 0.0
        ),
        "depth_three_root_differs_from_parent_only_on_at_least_five_trees": (
            roots["retained_parent_only_depth_three_root"] >= 5
        ),
        "brier_gain_vs_parent_only_at_least_one_percent": (
            parent["relative_brier_reduction"] >= 0.01
        ),
        "brier_wins_vs_parent_only_on_at_least_five_trees": (
            parent["brier_tree_wins"] >= 5
        ),
        "depth_three_root_differs_from_second_generated_only_on_at_least_three_trees": (
            roots["generated_only_depth_three_root"] >= 3
        ),
        "mean_brier_better_than_second_generated_only": (
            generated["mean_candidate_minus_baseline_brier"] < 0.0
        ),
        "brier_gain_vs_myopic_at_least_five_percent": (
            myopic["relative_brier_reduction"] >= 0.05
        ),
        "brier_gain_vs_fixed_at_least_five_percent": (
            fixed["relative_brier_reduction"] >= 0.05
        ),
        "brier_cluster_ci_vs_fixed_below_zero": (
            fixed["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "mean_brier_better_than_uniform_random": (
            random_control["mean_candidate_minus_baseline_brier"] < 0.0
        ),
        "mean_predictive_risk_spearman_brier_at_least_point_four": (
            aggregate["ranking"]["predictive_risk_spearman_brier"][
                "mean"
            ]
            >= 0.4
        ),
        "mean_pairwise_concordance_at_least_point_six_five": (
            aggregate["ranking"]["predictive_pairwise_concordance"][
                "mean"
            ]
            >= 0.65
        ),
        "novel_targets_have_no_mean_brier_hamming_or_coverage_regression": (
            novel["candidate_minus_baseline_brier"] <= 0.0
            and novel["candidate_minus_baseline_hamming"] <= 0.0
            and novel["coverage_difference"] >= 0.0
        ),
    }


def run_powered(
    *,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"trees": []}
    live_trees = []
    public_trees = []
    try:
        for tree_index, (tree_seed, target_seed) in enumerate(
            zip(TREE_SEEDS, TARGET_SEEDS, strict=True)
        ):
            tree, artifacts = depth.run_tree_depth_three(
                tree_index=tree_index,
                tree_seed=tree_seed,
                target_seed=target_seed,
                output_dir=output_dir,
                run_id=run_id,
                planning_model=depth.PLANNING_MODEL_ID,
                target_model=depth.TARGET_MODEL_ID,
                planning_concurrency=32,
                target_concurrency=1,
                projected_planning_cost=0.18,
                projected_target_cost=0.02,
                run_budget_usd=RUN_BUDGET_USD,
                shared_budget_run_id=run_id,
                first_support_mode=(
                    depth.FIRST_SUPPORT_RETAINED_REJUVENATION
                ),
                second_support_mode=(
                    depth.SECOND_SUPPORT_RETAINED_REJUVENATION
                ),
                brier_tolerance=BRIER_TOLERANCE,
            )
            live_trees.append(tree)
            raw["trees"].append(artifacts["raw"])
            public_trees.append(artifacts["public"])
            checkpoint(raw_path, raw)

        scored_trees = [score_public_tree(tree) for tree in public_trees]
        aggregate = aggregate_scored_trees(scored_trees)
        usage = _usage(live_trees)
        gates = powered_gates(
            scored_trees=scored_trees,
            live_trees=live_trees,
            usage=usage,
            aggregate=aggregate,
        )
        public = {
            "schema_version": SCHEMA_VERSION,
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "planning_model": depth.PLANNING_MODEL_ID,
                "target_model": depth.TARGET_MODEL_ID,
                "reasoning": "disabled",
                "temperature": depth.TEMPERATURE,
                "tree_seeds": list(TREE_SEEDS),
                "target_seeds": list(TARGET_SEEDS),
                "num_trees": len(TREE_SEEDS),
                "requests_per_tree": EXPECTED_REQUESTS_PER_TREE,
                "first_support_mode": (
                    depth.FIRST_SUPPORT_RETAINED_REJUVENATION
                ),
                "second_support_mode": (
                    depth.SECOND_SUPPORT_RETAINED_REJUVENATION
                ),
                "generated_only_control_scope": (
                    "second refresh only; first refresh remains retained"
                ),
                "brier_tolerance": BRIER_TOLERANCE,
                "run_budget_usd": RUN_BUDGET_USD,
                "minimum_starting_balance_usd": (
                    MIN_STARTING_BALANCE_USD
                ),
                "cumulative_budget_run_id": run_id,
                "tree_bootstrap_samples": BOOTSTRAP_SAMPLES,
            },
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "trees": public_trees,
        }
        trees_path = output_dir / "TREES.json"
        checkpoint(trees_path, public)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": public["protocol"],
            "gates": gates,
            "aggregate": aggregate,
            "usage": usage,
            "trees": scored_trees,
            "trees_sha256": hashlib.sha256(
                trees_path.read_bytes()
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
            "error": f"{type(exc).__name__}: {exc}",
            "completed_trees": len(live_trees),
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
    remaining_usd = openrouter_remaining_credit()
    require_starting_balance(remaining_usd)
    print(
        "OpenRouter preflight: "
        f"${remaining_usd:.6f} remaining; "
        f"${MIN_STARTING_BALANCE_USD:.2f} required"
    )
    result = run_powered(
        output_dir=args.output_dir.resolve(),
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
