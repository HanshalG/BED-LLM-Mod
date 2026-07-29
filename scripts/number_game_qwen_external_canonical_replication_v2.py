#!/usr/bin/env python3
"""Replicate fresh Qwen planning on the exact canonical Number Game bank."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
import sys
from typing import Any, Iterator, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_qwen_external_canonical_confirmation as engine
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-external-canonical-replication-v2-1"
TREE_SEEDS = tuple(range(55_000, 55_032))
TARGET_SEEDS = tuple(range(55_100, 55_132))
VALIDATION_SEED_START = 55_200
TREE_COUNT = 32
TARGET_COUNT = 33
VALIDATION_DRAWS_PER_TREE = 8
REQUESTS_PER_TREE = 49 + 1 + VALIDATION_DRAWS_PER_TREE
EXPECTED_REQUESTS = TREE_COUNT * REQUESTS_PER_TREE
RUN_BUDGET_USD = 3.50
MIN_STARTING_BALANCE_USD = 3.25
MAX_RETRIES = 8
MIN_INITIAL_VALID = 16
MIN_VALIDATION_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4
SMOKE_RESULT = engine.SMOKE_RESULT
SMOKE_RESULT_SHA256 = engine.SMOKE_RESULT_SHA256


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projection"
        )


def mechanics_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    targets: Sequence[Any],
) -> dict[str, bool]:
    return {
        "exactly_32_fresh_trees": len(scored_trees) == TREE_COUNT,
        "exactly_33_unique_canonical_targets": (
            len(targets) == TARGET_COUNT
            and len({target.extension for target in targets}) == TARGET_COUNT
        ),
        "accepted_request_count_exact": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "retries_within_cap": usage["retry_count"] <= MAX_RETRIES,
        "provider_error_retries_within_cap": (
            usage["provider_error_retries"] <= MAX_RETRIES
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_initial_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
            for tree in scored_trees
        ),
        "all_eight_validation_supports_valid": all(
            tree["mechanics"]["validation_support_count"]
            == VALIDATION_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_validation_support_valid"]
            >= MIN_VALIDATION_VALID
            for tree in scored_trees
        ),
        "all_retained_branches_non_degenerate": all(
            tree["mechanics"]["minimum_first_branch_valid"]
            >= MIN_FIRST_BRANCH_VALID
            and tree["mechanics"]["minimum_retained_second_branch_valid"]
            >= MIN_SECOND_BRANCH_VALID
            for tree in scored_trees
        ),
    }


def primary_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
    myopic = aggregate["comparisons"]["myopic_eig"]
    return {
        "depth_three_beats_myopic_by_eight_percent": (
            myopic["relative_brier_reduction"] >= 0.08
        ),
        "depth_three_vs_myopic_ci_below_zero": (
            myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twenty_trees_vs_myopic": (
            myopic["brier_tree_wins"] >= 20
        ),
    }


def diagnostic_gates(aggregate: dict[str, Any]) -> dict[str, bool]:
    comparisons = aggregate["comparisons"]
    depth_two = comparisons["crossfit_depth_two"]
    fixed = comparisons["fixed_support_depth_three"]
    pts = comparisons["positive_test_strategy"]
    random_root = comparisons["uniform_random_candidate_root"]
    ranking = aggregate["ranking"]
    return {
        "depth_three_beats_depth_two_by_one_percent": (
            depth_two["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_vs_depth_two_ci_below_zero": (
            depth_two["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twelve_vs_depth_two": (
            depth_two["brier_tree_wins"] >= 12
        ),
        "depth_three_beats_fixed_by_three_percent_with_ci": (
            fixed["relative_brier_reduction"] >= 0.03
            and fixed["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_pts_by_three_percent_with_ci": (
            pts["relative_brier_reduction"] >= 0.03
            and pts["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_random_by_five_percent_with_ci": (
            random_root["relative_brier_reduction"] >= 0.05
            and random_root[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
        ),
        "depth_three_rank_rho_at_least_point_three": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"] >= 0.3
        ),
        "depth_three_rank_rho_exceeds_depth_two": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            > ranking["crossfit_depth_two_spearman_brier"]["mean"]
        ),
    }


@contextmanager
def configured_engine() -> Iterator[None]:
    overrides = {
        "INTERFACE_VERSION": INTERFACE_VERSION,
        "TREE_SEEDS": TREE_SEEDS,
        "TARGET_SEEDS": TARGET_SEEDS,
        "VALIDATION_SEED_START": VALIDATION_SEED_START,
        "TREE_COUNT": TREE_COUNT,
        "TARGET_COUNT": TARGET_COUNT,
        "VALIDATION_DRAWS_PER_TREE": VALIDATION_DRAWS_PER_TREE,
        "REQUESTS_PER_TREE": REQUESTS_PER_TREE,
        "EXPECTED_REQUESTS": EXPECTED_REQUESTS,
        "RUN_BUDGET_USD": RUN_BUDGET_USD,
        "MIN_STARTING_BALANCE_USD": MIN_STARTING_BALANCE_USD,
        "MAX_RETRIES": MAX_RETRIES,
        "MIN_INITIAL_VALID": MIN_INITIAL_VALID,
        "MIN_VALIDATION_VALID": MIN_VALIDATION_VALID,
        "MIN_FIRST_BRANCH_VALID": MIN_FIRST_BRANCH_VALID,
        "MIN_SECOND_BRANCH_VALID": MIN_SECOND_BRANCH_VALID,
        "mechanics_gates": mechanics_gates,
        "primary_gates": primary_gates,
        "diagnostic_gates": diagnostic_gates,
    }
    originals = {name: getattr(engine, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(engine, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(engine, name, value)


def run_replication(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
) -> dict[str, Any]:
    with configured_engine():
        return engine.run_confirmation(
            output_dir=output_dir,
            run_id=run_id,
            smoke_result_path=smoke_result_path,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-result", type=Path, default=SMOKE_RESULT)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    result = run_replication(
        output_dir=args.output_dir,
        run_id=args.run_id,
        smoke_result_path=args.smoke_result,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "primary_gates": result["primary_gates"],
                "diagnostic_gates": result["diagnostic_gates"],
                "depth_two": result["aggregate"]["comparisons"][
                    "crossfit_depth_two"
                ],
                "myopic": result["aggregate"]["comparisons"]["myopic_eig"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
