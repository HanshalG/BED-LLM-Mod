#!/usr/bin/env python3
"""Replicate cross-fitted Number Game depth three with a DeepSeek planner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts import number_game_qwen_planner_depth_three as engine
from scripts.number_game_deepseek_planner_serving_smoke import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
    MODEL_ID as PLANNING_MODEL_ID,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-deepseek-planner-depth-three-1"
TARGET_MODEL_ID = "openai/gpt-5.4-mini"
FORMAL_TREE_SEEDS = tuple(range(43_000, 43_032))
FORMAL_TARGET_SEEDS = tuple(range(43_100, 43_132))
FORMAL_VALIDATION_SEED_START = 43_200
FORMAL_EXTRA_ENDPOINT_SEED_START = 43_500
FORMAL_EXPECTED_REQUESTS = (
    len(FORMAL_TREE_SEEDS) * engine.REQUESTS_PER_TREE
)
FORMAL_BUDGET_USD = 5.00
MIN_FORMAL_STARTING_BALANCE_USD = 5.00


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_FORMAL_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_FORMAL_STARTING_BALANCE_USD:.2f} formal projection"
        )


def validate_smoke_result(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text())
    protocol = result.get("protocol") or {}
    gates = result.get("gates") or {}
    if result.get("status") != "passed" or gates.get("all_pass") is not True:
        raise ValueError("DeepSeek planner exact-10 serving smoke did not pass")
    if protocol.get("interface_version") != SMOKE_INTERFACE_VERSION:
        raise ValueError("DeepSeek planner smoke interface changed")
    if protocol.get("model") != PLANNING_MODEL_ID:
        raise ValueError("DeepSeek planner smoke model changed")
    if protocol.get("expected_requests") != 10:
        raise ValueError("DeepSeek planner smoke was not exact-10")
    if protocol.get("efficacy_used_for_authorization") is not False:
        raise ValueError("DeepSeek planner smoke used efficacy for authorization")
    return result


def formal_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, bool]:
    primary = aggregate["comparisons"]["crossfit_depth_two"]
    myopic = aggregate["comparisons"]["myopic_eig"]
    fixed = aggregate["comparisons"]["fixed_support_depth_three"]
    pts = aggregate["comparisons"]["positive_test_strategy"]
    novel = aggregate["novel_target_mean_differences"]
    ranking = aggregate["ranking"]
    return {
        **engine.mechanics_gates(
            scored_trees=scored_trees,
            usage=usage,
            expected_requests=FORMAL_EXPECTED_REQUESTS,
            run_budget_usd=FORMAL_BUDGET_USD,
        ),
        "crossfit_depth_roots_differ_on_at_least_twelve_trees": (
            aggregate["root_differences"]["crossfit_depth_two"] >= 12
        ),
        "depth_three_brier_gain_at_least_one_percent": (
            primary["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_brier_ci_below_zero": (
            primary["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twelve_trees": (
            primary["brier_tree_wins"] >= 12
        ),
        "novel_target_brier_does_not_regress": (
            novel["candidate_minus_baseline_brier"] <= 0.0
        ),
        "depth_three_beats_myopic_with_ci": (
            myopic["mean_candidate_minus_baseline_brier"] < 0.0
            and myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_fixed_with_ci": (
            fixed["mean_candidate_minus_baseline_brier"] < 0.0
            and fixed["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_pts_with_ci": (
            pts["mean_candidate_minus_baseline_brier"] < 0.0
            and pts["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_rank_rho_at_least_point_seven": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"] >= 0.7
        ),
        "depth_three_rho_exceeds_depth_two_by_point_one_five": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            - ranking["crossfit_depth_two_spearman_brier"]["mean"]
            >= 0.15
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-result", type=Path, required=True)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    result = engine.run_study(
        stage="formal",
        tree_seeds=FORMAL_TREE_SEEDS,
        target_seeds=FORMAL_TARGET_SEEDS,
        validation_seed_start=FORMAL_VALIDATION_SEED_START,
        extra_endpoint_seed_start=FORMAL_EXTRA_ENDPOINT_SEED_START,
        output_dir=args.output_dir,
        run_id=args.run_id,
        run_budget_usd=FORMAL_BUDGET_USD,
        smoke_result_path=args.smoke_result,
        planning_model=PLANNING_MODEL_ID,
        target_model=TARGET_MODEL_ID,
        interface_version=INTERFACE_VERSION,
        smoke_validator=validate_smoke_result,
        formal_gate_fn=formal_gates,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "gates": result["gates"],
                "primary": result["aggregate"]["comparisons"][
                    "crossfit_depth_two"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
