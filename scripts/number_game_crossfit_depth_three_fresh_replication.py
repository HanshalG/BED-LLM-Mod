#!/usr/bin/env python3
"""Run a wholly fresh cross-fitted Number Game depth-three replication."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import (
    aggregate_scored_trees,
    score_public_tree as score_crossfit_public_tree,
)
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-crossfit-depth-three-fresh-replication-1"
TREE_SEEDS = tuple(range(30600, 30632))
TARGET_SEEDS = tuple(range(31000, 31032))
VALIDATION_SEED_START = 30700
VALIDATION_DRAWS_PER_TREE = 8
EXTRA_ENDPOINT_SEED_START = 31100
EXTRA_ENDPOINT_DRAWS_PER_TREE = 15
ENDPOINT_DRAWS_PER_TREE = 1 + EXTRA_ENDPOINT_DRAWS_PER_TREE
EXPECTED_REQUESTS_PER_TREE = (
    49 + VALIDATION_DRAWS_PER_TREE + ENDPOINT_DRAWS_PER_TREE
)
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
RUN_BUDGET_USD = 7.60
MIN_STARTING_BALANCE_USD = 6.90
MIN_INITIAL_VALID = 16
MIN_VALIDATION_VALID = 16
MIN_ENDPOINT_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4
MIN_NOVEL_ENDPOINT_TOTAL = 128
RISK_TIE_TOLERANCE = 1e-12


def validation_seeds_for_tree(tree_index: int) -> tuple[int, ...]:
    start = VALIDATION_SEED_START + tree_index * VALIDATION_DRAWS_PER_TREE
    return tuple(range(start, start + VALIDATION_DRAWS_PER_TREE))


def extra_endpoint_seeds_for_tree(tree_index: int) -> tuple[int, ...]:
    start = (
        EXTRA_ENDPOINT_SEED_START
        + tree_index * EXTRA_ENDPOINT_DRAWS_PER_TREE
    )
    return tuple(range(start, start + EXTRA_ENDPOINT_DRAWS_PER_TREE))


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projected run cost"
        )


def _generate_supports(
    *,
    seeds: Sequence[int],
    output_dir: Path,
    run_id: str,
) -> tuple[
    list[list[Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    adapters = [
        depth._adapter(
            model=depth.TARGET_MODEL_ID,
            run_id=run_id,
            output_dir=output_dir,
            request_seed=int(seed),
            concurrency=1,
            projected_cost=0.005,
            run_budget_usd=RUN_BUDGET_USD,
        )
        for seed in seeds
    ]

    def request(adapter: Any) -> str:
        return adapter.chat_complete_messages_batched_structured(
            [depth.initial_messages()],
            temperature=depth.TEMPERATURE,
            block_size=1,
            response_format=depth.proposal_response_format(),
            max_new_tokens=depth.MAX_TOKENS,
        )[0]

    with ThreadPoolExecutor(max_workers=len(adapters)) as executor:
        responses = list(executor.map(request, adapters))
    supports = []
    diagnostics = []
    for response in responses:
        support, diagnostic = depth.parse_proposals(response)
        supports.append(support)
        diagnostics.append(diagnostic)
    return (
        supports,
        diagnostics,
        responses,
        [adapter.usage_snapshot() for adapter in adapters],
    )


def _usage(
    live_trees: Sequence[dict[str, Any]],
    support_snapshots: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    usage = {
        key: sum(tree["usage"].get(key, 0) for tree in live_trees)
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
    snapshot_fields = {
        "adapter_requests": "adapter_requests",
        "http_attempts": "http_attempts",
        "retry_count": "retry_count",
        "provider_error_retries": "provider_error_retries",
        "adapter_reasoning_tokens": "adapter_reasoning_tokens",
        "forced_exits": "forced_exits",
        "run_cost_usd": "adapter_cost_usd",
    }
    for output_key, snapshot_key in snapshot_fields.items():
        usage[output_key] += sum(
            snapshot.get(snapshot_key, 0)
            for snapshot in support_snapshots
        )
    return usage


def replication_gates(
    *,
    scored_trees: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    aggregate: dict[str, Any],
) -> dict[str, bool]:
    primary = aggregate["comparisons"]["crossfit_depth_two"]
    source_depth_three = aggregate["comparisons"][
        "retained_risk_set_depth_three"
    ]
    myopic = aggregate["comparisons"]["myopic_eig"]
    fixed = aggregate["comparisons"]["fixed_support_depth_three"]
    pts = aggregate["comparisons"]["positive_test_strategy"]
    novel = aggregate["novel_target_mean_differences"]
    ranking = aggregate["ranking"]
    return {
        "exact_2336_accepted_requests": (
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
        "all_eight_validation_supports_valid": all(
            tree["mechanics"]["validation_support_count"]
            == VALIDATION_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_validation_support_valid"]
            >= MIN_VALIDATION_VALID
            for tree in scored_trees
        ),
        "all_sixteen_endpoint_supports_valid": all(
            tree["mechanics"]["endpoint_draw_count"]
            == ENDPOINT_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_endpoint_support_valid"]
            >= MIN_ENDPOINT_VALID
            for tree in scored_trees
        ),
        "every_tree_has_at_least_128_novel_endpoint_hypotheses": all(
            tree["mechanics"]["total_novel_endpoint_hypotheses"]
            >= MIN_NOVEL_ENDPOINT_TOTAL
            for tree in scored_trees
        ),
        "all_retained_branches_non_degenerate": all(
            tree["mechanics"]["minimum_first_branch_valid"]
            >= MIN_FIRST_BRANCH_VALID
            and tree["mechanics"]["minimum_retained_second_branch_valid"]
            >= MIN_SECOND_BRANCH_VALID
            for tree in scored_trees
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
        "no_hamming_regression_vs_depth_two": (
            primary["mean_candidate_minus_baseline_hamming"] <= 0.0
        ),
        "no_coverage_regression_vs_depth_two": (
            primary["mean_coverage_difference"] >= 0.0
        ),
        "novel_targets_do_not_regress": (
            novel["candidate_minus_baseline_brier"] <= 0.0
            and novel["candidate_minus_baseline_hamming"] <= 0.0
            and novel["coverage_difference"] >= 0.0
        ),
        "crossfit_beats_source_depth_three_by_two_percent": (
            source_depth_three["relative_brier_reduction"] >= 0.02
        ),
        "crossfit_vs_source_depth_three_ci_below_zero": (
            source_depth_three[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
        ),
        "crossfit_depth_three_beats_myopic_by_five_percent": (
            myopic["relative_brier_reduction"] >= 0.05
        ),
        "crossfit_depth_three_beats_fixed_by_five_percent": (
            fixed["relative_brier_reduction"] >= 0.05
        ),
        "crossfit_depth_three_beats_pts_by_three_percent": (
            pts["relative_brier_reduction"] >= 0.03
        ),
        "crossfit_vs_pts_ci_below_zero": (
            pts["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_rank_rho_at_least_point_seven": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            >= 0.7
        ),
        "depth_three_rho_exceeds_depth_two_by_point_one_five": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            - ranking["crossfit_depth_two_spearman_brier"]["mean"]
            >= 0.15
        ),
    }


def run_replication(
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
    public_endpoint_trees = []
    support_snapshots = []
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
                planning_model=depth.PLANNING_MODEL_ID,
                target_model=depth.TARGET_MODEL_ID,
                planning_concurrency=32,
                target_concurrency=1,
                projected_planning_cost=0.18,
                projected_target_cost=0.01,
                run_budget_usd=RUN_BUDGET_USD,
                shared_budget_run_id=run_id,
                first_support_mode=depth.FIRST_SUPPORT_RETAINED_REJUVENATION,
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
                validation_snapshots,
            ) = _generate_supports(
                seeds=validation_seeds,
                output_dir=output_dir,
                run_id=run_id,
            )
            extra_endpoint_seeds = extra_endpoint_seeds_for_tree(tree_index)
            (
                extra_endpoint_supports,
                extra_endpoint_diagnostics,
                extra_endpoint_responses,
                endpoint_snapshots,
            ) = _generate_supports(
                seeds=extra_endpoint_seeds,
                output_dir=output_dir,
                run_id=run_id,
            )
            support_snapshots.extend(validation_snapshots)
            support_snapshots.extend(endpoint_snapshots)
            live_trees.append(live)

            artifacts["raw"]["validation_responses"] = [
                {"seed": seed, "response": response}
                for seed, response in zip(
                    validation_seeds,
                    validation_responses,
                    strict=True,
                )
            ]
            artifacts["raw"]["extra_endpoint_responses"] = [
                {"seed": seed, "response": response}
                for seed, response in zip(
                    extra_endpoint_seeds,
                    extra_endpoint_responses,
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
            public_endpoint_trees.append(
                {
                    "tree_index": tree_index,
                    "seeds": [
                        target_seed,
                        *extra_endpoint_seeds,
                    ],
                    "supports": [
                        artifacts["public"]["targets"],
                        *[
                            [
                                hypothesis.public_dict()
                                for hypothesis in support
                            ]
                            for support in extra_endpoint_supports
                        ],
                    ],
                    "diagnostics": [
                        live["target_diagnostics"],
                        *extra_endpoint_diagnostics,
                    ],
                }
            )
            raw["trees"].append(artifacts["raw"])
            public_trees.append(artifacts["public"])
            checkpoint(raw_path, raw)

        source_metrics = [
            score_crossfit_public_tree(tree) for tree in public_trees
        ]
        scored_trees = [
            score_fixed_tree(
                tree,
                metrics,
                endpoints["supports"],
            )
            for tree, metrics, endpoints in zip(
                public_trees,
                source_metrics,
                public_endpoint_trees,
                strict=True,
            )
        ]
        aggregate = aggregate_scored_trees(scored_trees)
        usage = _usage(live_trees, support_snapshots)
        gates = replication_gates(
            scored_trees=scored_trees,
            usage=usage,
            aggregate=aggregate,
        )
        protocol = {
            "interface_version": INTERFACE_VERSION,
            "tree_seeds": list(TREE_SEEDS),
            "target_seeds": list(TARGET_SEEDS),
            "validation_seeds": [
                list(validation_seeds_for_tree(index))
                for index in range(len(TREE_SEEDS))
            ],
            "extra_endpoint_seeds": [
                list(extra_endpoint_seeds_for_tree(index))
                for index in range(len(TREE_SEEDS))
            ],
            "validation_draws_per_tree": VALIDATION_DRAWS_PER_TREE,
            "endpoint_draws_per_tree": ENDPOINT_DRAWS_PER_TREE,
            "draw_weighting": "equal draw weight",
            "risk_tie_tolerance": RISK_TIE_TOLERANCE,
            "planning_model": depth.PLANNING_MODEL_ID,
            "target_model": depth.TARGET_MODEL_ID,
            "reasoning": False,
            "temperature": depth.TEMPERATURE,
            "first_support_mode": (
                depth.FIRST_SUPPORT_RETAINED_REJUVENATION
            ),
            "second_support_mode": (
                depth.SECOND_SUPPORT_RETAINED_REJUVENATION
            ),
            "expected_requests": EXPECTED_REQUESTS,
            "run_budget_usd": RUN_BUDGET_USD,
            "cumulative_budget_run_id": run_id,
        }
        trees_document = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "trees": public_trees,
        }
        endpoint_document = {
            "schema_version": SCHEMA_VERSION,
            "protocol": protocol,
            "raw_responses_sha256": trees_document[
                "raw_responses_sha256"
            ],
            "trees": public_endpoint_trees,
        }
        checkpoint(output_dir / "TREES.json", trees_document)
        checkpoint(output_dir / "ENDPOINTS.json", endpoint_document)
        public = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": protocol,
            "usage": usage,
            "aggregate": aggregate,
            "gates": gates,
            "trees": scored_trees,
            "trees_sha256": hashlib.sha256(
                (output_dir / "TREES.json").read_bytes()
            ).hexdigest(),
            "endpoints_sha256": hashlib.sha256(
                (output_dir / "ENDPOINTS.json").read_bytes()
            ).hexdigest(),
            "raw_responses_sha256": trees_document[
                "raw_responses_sha256"
            ],
        }
        checkpoint(output_dir / "RESULT.json", public)
        return public
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
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_replication(
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
