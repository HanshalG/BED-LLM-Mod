#!/usr/bin/env python3
"""Run fresh Qwen Number Game trees against the canonical external bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts import number_game_qwen_planner_depth_three as qwen
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import (
    aggregate_scored_trees,
    score_public_tree as score_crossfit_public_tree,
)
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree
from scripts.number_game_external_canonical_replay import canonical_targets
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-external-canonical-confirmation-1"
TREE_SEEDS = tuple(range(49000, 49032))
TARGET_SEEDS = tuple(range(49100, 49132))
VALIDATION_SEED_START = 49200
TREE_COUNT = 32
TARGET_COUNT = 33
VALIDATION_DRAWS_PER_TREE = 8
REQUESTS_PER_TREE = 49 + 1 + VALIDATION_DRAWS_PER_TREE
EXPECTED_REQUESTS = TREE_COUNT * REQUESTS_PER_TREE
RUN_BUDGET_USD = 4.00
MIN_STARTING_BALANCE_USD = 3.60
MAX_RETRIES = 8
MIN_INITIAL_VALID = 16
MIN_VALIDATION_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4
SMOKE_RESULT = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_planner_depth_three_smoke"
    / "number-game-qwen-planner-depth-three-smoke-20260728"
    / "RESULT.json"
)
SMOKE_RESULT_SHA256 = (
    "2f311f9118a74f720edf219ca94e0ae4a8a01bd40b0ace01ade27e8d4c4795df"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projection"
        )


def validate_smoke_result(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SMOKE_RESULT_SHA256:
        raise ValueError("Qwen planner smoke hash changed")
    result = qwen.validate_smoke_result(path)
    usage = result["usage"]
    if usage["adapter_requests"] < 10:
        raise ValueError("Qwen planner smoke has fewer than ten calls")
    if usage["retry_count"] != 0:
        raise ValueError("Qwen planner smoke used a retry")
    if usage["provider_error_retries"] != 0:
        raise ValueError("Qwen planner smoke used a provider retry")
    if not all(result["gates"].values()):
        raise ValueError("Qwen planner smoke has a failed gate")
    return result


def validation_seeds_for_tree(tree_index: int) -> tuple[int, ...]:
    return qwen.validation_seeds_for_tree(
        tree_index,
        start=VALIDATION_SEED_START,
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
        "zero_provider_error_retries": (
            usage["provider_error_retries"] == 0
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
    depth_two = aggregate["comparisons"]["crossfit_depth_two"]
    myopic = aggregate["comparisons"]["myopic_eig"]
    return {
        "crossfit_depth_roots_differ_on_at_least_twelve_trees": (
            aggregate["root_differences"]["crossfit_depth_two"] >= 12
        ),
        "depth_three_beats_depth_two_by_one_percent": (
            depth_two["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_vs_depth_two_ci_below_zero": (
            depth_two["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_twelve_trees_vs_depth_two": (
            depth_two["brier_tree_wins"] >= 12
        ),
        "depth_three_beats_myopic_by_five_percent": (
            myopic["relative_brier_reduction"] >= 0.05
        ),
        "depth_three_vs_myopic_ci_below_zero": (
            myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_wins_at_least_sixteen_trees_vs_myopic": (
            myopic["brier_tree_wins"] >= 16
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
        "no_hamming_regression_vs_depth_two": (
            depth_two["mean_candidate_minus_baseline_hamming"] <= 0.0
        ),
        "no_coverage_regression_vs_depth_two": (
            depth_two["mean_coverage_difference"] >= 0.0
        ),
        "depth_three_directionally_beats_fixed_support": (
            fixed["mean_candidate_minus_baseline_brier"] <= 0.0
        ),
        "depth_three_directionally_beats_pts": (
            pts["mean_candidate_minus_baseline_brier"] <= 0.0
        ),
        "depth_three_beats_random_with_ci": (
            random_root[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
        ),
        "depth_three_rank_rho_exceeds_depth_two": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            > ranking["crossfit_depth_two_spearman_brier"]["mean"]
        ),
    }


def run_confirmation(
    *,
    output_dir: Path,
    run_id: str,
    smoke_result_path: Path = SMOKE_RESULT,
) -> dict[str, Any]:
    smoke = validate_smoke_result(smoke_result_path)
    targets = canonical_targets()
    public_targets = [target.public_dict() for target in targets]
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"trees": []}
    live_trees = []
    public_trees = []
    validation_snapshots = []
    try:
        for tree_index, (tree_seed, target_seed) in enumerate(
            zip(TREE_SEEDS, TARGET_SEEDS, strict=True)
        ):
            live, artifacts = depth.run_tree_depth_three(
                tree_index=tree_index,
                tree_seed=tree_seed,
                target_seed=target_seed,
                output_dir=output_dir,
                run_id=run_id,
                planning_model=qwen.PLANNING_MODEL_ID,
                target_model=qwen.TARGET_MODEL_ID,
                planning_concurrency=32,
                target_concurrency=1,
                projected_planning_cost=0.12,
                projected_target_cost=0.005,
                run_budget_usd=RUN_BUDGET_USD,
                shared_budget_run_id=run_id,
                first_support_mode=(
                    depth.FIRST_SUPPORT_RETAINED_REJUVENATION
                ),
                second_support_mode=(
                    depth.SECOND_SUPPORT_RETAINED_REJUVENATION
                ),
                brier_tolerance=0.0,
            )
            validation_seeds = validation_seeds_for_tree(tree_index)
            (
                validation_supports,
                validation_diagnostics,
                validation_responses,
                snapshots,
            ) = qwen._generate_supports(
                seeds=validation_seeds,
                output_dir=output_dir,
                run_id=run_id,
                run_budget_usd=RUN_BUDGET_USD,
                target_model=qwen.TARGET_MODEL_ID,
            )
            validation_snapshots.extend(snapshots)
            live_trees.append(live)
            artifacts["raw"]["validation_responses"] = [
                {"seed": seed, "response": response}
                for seed, response in zip(
                    validation_seeds,
                    validation_responses,
                    strict=True,
                )
            ]
            artifacts["public"]["validation_seeds"] = list(
                validation_seeds
            )
            artifacts["public"]["validation_supports"] = [
                [hypothesis.public_dict() for hypothesis in support]
                for support in validation_supports
            ]
            artifacts["public"]["validation_diagnostics"] = (
                validation_diagnostics
            )
            raw["trees"].append(artifacts["raw"])
            public_trees.append(artifacts["public"])
            checkpoint(raw_path, raw)

        source_metrics = [
            score_crossfit_public_tree(tree) for tree in public_trees
        ]
        scored_trees = [
            score_fixed_tree(tree, metrics, [public_targets])
            for tree, metrics in zip(
                public_trees,
                source_metrics,
                strict=True,
            )
        ]
        aggregate = aggregate_scored_trees(scored_trees)
        usage = qwen._usage(live_trees, validation_snapshots)
        mechanics = mechanics_gates(
            scored_trees=scored_trees,
            usage=usage,
            targets=targets,
        )
        primary = primary_gates(aggregate)
        diagnostics = diagnostic_gates(aggregate)
        protocol = {
            "interface_version": INTERFACE_VERSION,
            "planning_model": qwen.PLANNING_MODEL_ID,
            "target_and_validation_model": qwen.TARGET_MODEL_ID,
            "generated_target_used_for_efficacy": False,
            "reasoning": False,
            "temperature": depth.TEMPERATURE,
            "tree_seeds": list(TREE_SEEDS),
            "target_seeds": list(TARGET_SEEDS),
            "validation_seeds": [
                list(validation_seeds_for_tree(index))
                for index in range(TREE_COUNT)
            ],
            "tree_count": TREE_COUNT,
            "target_count": TARGET_COUNT,
            "validation_draws_per_tree": VALIDATION_DRAWS_PER_TREE,
            "target_weighting": "equal canonical concept weight",
            "tree_weighting": "equal tree weight",
            "expected_requests": EXPECTED_REQUESTS,
            "run_budget_usd": RUN_BUDGET_USD,
            "cumulative_budget_run_id": run_id,
            "smoke_result_sha256": SMOKE_RESULT_SHA256,
            "smoke_raw_responses_sha256": smoke[
                "raw_responses_sha256"
            ],
        }
        targets_document = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "source_doi": "10.1017/S0140525X01000061",
            "paper_domain": [1, 100],
            "evaluation_domain": [0, 100],
            "domain_adaptation": (
                "Natural predicate extension to n=0; no target is omitted "
                "or reweighted."
            ),
            "targets": public_targets,
        }
        checkpoint(output_dir / "TARGETS.json", targets_document)
        trees_document = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "raw_responses_sha256": sha256_file(raw_path),
            "trees": public_trees,
        }
        checkpoint(output_dir / "TREES.json", trees_document)
        result = {
            "schema_version": SCHEMA_VERSION,
            "status": (
                "passed"
                if all(mechanics.values()) and all(primary.values())
                else "gated_null"
            ),
            "protocol": protocol,
            "usage": usage,
            "aggregate": aggregate,
            "mechanics_gates": mechanics,
            "primary_gates": primary,
            "diagnostic_gates": diagnostics,
            "trees": scored_trees,
            "targets_sha256": sha256_file(output_dir / "TARGETS.json"),
            "trees_sha256": sha256_file(output_dir / "TREES.json"),
            "raw_responses_sha256": trees_document[
                "raw_responses_sha256"
            ],
        }
        checkpoint(output_dir / "RESULT.json", result)
        return result
    except Exception as exc:
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--smoke-result", type=Path, default=SMOKE_RESULT)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    result = run_confirmation(
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
