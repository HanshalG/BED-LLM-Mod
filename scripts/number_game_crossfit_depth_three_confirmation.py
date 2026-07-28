#!/usr/bin/env python3
"""Confirm cross-fitted depth-three Number Game policy selection."""

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
from scripts.number_game_depth_three_crossfit_audit import (
    RISK_TIE_TOLERANCE,
    select_minimum_risk_root,
)
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_generator_aware_bed import (
    evaluate_policy_root,
)
from scripts.number_game_predictive_risk_holdout import (
    aggregate_random_root_control,
    policy_comparison,
)
from scripts.number_game_predictive_risk_replication import (
    aggregate_tree_comparisons,
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
    score_public_tree as score_retained_public_tree,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-crossfit-depth-three-confirmation-1"
TREE_SEEDS = tuple(range(28300, 28332))
TARGET_SEEDS = tuple(range(28400, 28432))
VALIDATION_SEED_START = 28500
VALIDATION_SUPPORT_COUNT = 8
EXPECTED_REQUESTS_PER_TREE = 49 + VALIDATION_SUPPORT_COUNT + 1
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
RUN_BUDGET_USD = 6.50
MIN_STARTING_BALANCE_USD = 5.80
MIN_INITIAL_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4
MIN_TARGET_VALID = 16
MIN_NOVEL_TARGETS = 8
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 28800


def validation_seeds_for_tree(tree_index: int) -> tuple[int, ...]:
    start = VALIDATION_SEED_START + tree_index * VALIDATION_SUPPORT_COUNT
    return tuple(range(start, start + VALIDATION_SUPPORT_COUNT))


def require_starting_balance(remaining_usd: float) -> None:
    if remaining_usd + 1e-12 < MIN_STARTING_BALANCE_USD:
        raise RuntimeError(
            "OpenRouter balance "
            f"${remaining_usd:.6f} is below the frozen "
            f"${MIN_STARTING_BALANCE_USD:.2f} projected run cost"
        )


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _generate_validation_supports(
    *,
    tree_index: int,
    output_dir: Path,
    run_id: str,
) -> tuple[
    list[list[Any]],
    list[dict[str, Any]],
    list[str],
    list[dict[str, Any]],
]:
    seeds = validation_seeds_for_tree(tree_index)
    adapters = [
        depth._adapter(
            model=depth.TARGET_MODEL_ID,
            run_id=run_id,
            output_dir=output_dir,
            request_seed=seed,
            concurrency=1,
            projected_cost=0.01,
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

    with ThreadPoolExecutor(
        max_workers=VALIDATION_SUPPORT_COUNT
    ) as executor:
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


def _target_mapping(
    support: Sequence[Any],
    *,
    draw_index: int,
) -> dict[str, Any]:
    return {
        f"validation_{draw_index:02d}_{item_index:02d}_{hypothesis.name}": (
            hypothesis
        )
        for item_index, hypothesis in enumerate(support)
    }


def score_public_tree(tree: dict[str, Any]) -> dict[str, Any]:
    base = score_retained_public_tree(tree)
    initial = [_rule(item) for item in tree["initial"]]
    roots = [int(root) for root in tree["roots"]]
    first = _first_branches(tree)
    generated_second = _second_branches(
        tree,
        key="generated_second_branches",
    )
    second, _, _ = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated_second,
    )
    validation_supports = [
        [_rule(item) for item in support]
        for support in tree["validation_supports"]
    ]
    depth_three_draw_risks = []
    depth_two_draw_risks = []
    for draw_index, support in enumerate(validation_supports):
        validation_targets = _target_mapping(
            support,
            draw_index=draw_index,
        )
        depth_three_draw_risks.append(
            {
                root: depth.evaluate_policy_root_depth_three(
                    policy=f"crossfit_depth_three_root_{root}",
                    root=root,
                    targets=validation_targets,
                    first_branches=first,
                    second_branches=second,
                )["mean_posterior_predictive_brier"]
                for root in roots
            }
        )
        depth_two_draw_risks.append(
            {
                root: evaluate_policy_root(
                    policy=f"crossfit_depth_two_root_{root}",
                    root=root,
                    targets=validation_targets,
                    branches=first,
                )["mean_posterior_predictive_brier"]
                for root in roots
            }
        )
    depth_three_risk = {
        root: _mean(
            [draw_risk[root] for draw_risk in depth_three_draw_risks]
        )
        for root in roots
    }
    depth_two_risk = {
        root: _mean(
            [draw_risk[root] for draw_risk in depth_two_draw_risks]
        )
        for root in roots
    }
    candidate_root = select_minimum_risk_root(roots, depth_three_risk)
    baseline_root = select_minimum_risk_root(roots, depth_two_risk)

    targets = {
        f"target_{index:02d}_{item['name']}": _rule(item)
        for index, item in enumerate(tree["targets"])
    }
    per_root_endpoint = {
        root: depth.evaluate_policy_root_depth_three(
            policy=f"root_{root}",
            root=root,
            targets=targets,
            first_branches=first,
            second_branches=second,
        )
        for root in roots
    }
    policy_roots = {
        "crossfit_depth_three": candidate_root,
        "crossfit_depth_two": baseline_root,
        "retained_risk_set_depth_three": base["selection"][
            "retained_risk_set_depth_three_root"
        ],
        "predictive_bayes_risk_depth_two": base["selection"][
            "predictive_bayes_risk_depth_two_root"
        ],
        "myopic_eig": int(base["selection"]["myopic_root"]),
        "fixed_support_depth_three": base["selection"][
            "fixed_support_depth_three_root"
        ],
    }
    endpoint = {
        policy: per_root_endpoint[root] | {"policy": policy}
        for policy, root in policy_roots.items()
    }
    endpoint["uniform_random_candidate_root"] = (
        aggregate_random_root_control(per_root_endpoint)
    )
    pts_roots = [int(root) for root in base["selection"]["pts_roots"]]
    endpoint["positive_test_strategy"] = aggregate_random_root_control(
        {root: per_root_endpoint[root] for root in pts_roots}
    ) | {"policy": "uniform_over_two_seeded_pts_roots"}
    baselines = (
        "crossfit_depth_two",
        "retained_risk_set_depth_three",
        "predictive_bayes_risk_depth_two",
        "myopic_eig",
        "fixed_support_depth_three",
        "uniform_random_candidate_root",
        "positive_test_strategy",
    )
    comparisons = {
        baseline: policy_comparison(
            endpoint["crossfit_depth_three"],
            endpoint[baseline],
        )
        for baseline in baselines
    }

    initial_extensions = {hypothesis.extension for hypothesis in initial}
    novel_targets = {
        name: hypothesis
        for name, hypothesis in targets.items()
        if hypothesis.extension not in initial_extensions
    }
    novel_candidate = depth.evaluate_policy_root_depth_three(
        policy="crossfit_depth_three",
        root=candidate_root,
        targets=novel_targets,
        first_branches=first,
        second_branches=second,
    )
    novel_baseline = depth.evaluate_policy_root_depth_three(
        policy="crossfit_depth_two",
        root=baseline_root,
        targets=novel_targets,
        first_branches=first,
        second_branches=second,
    )
    endpoint_brier = [
        base["per_root"][str(root)]["endpoint_brier"] for root in roots
    ]
    depth_three_values = [depth_three_risk[root] for root in roots]
    depth_two_values = [depth_two_risk[root] for root in roots]
    return {
        **base,
        "mechanics": {
            **base["mechanics"],
            "validation_support_count": len(validation_supports),
            "minimum_validation_support_valid": min(
                len(support) for support in validation_supports
            ),
            "mean_validation_support_valid": _mean(
                [len(support) for support in validation_supports]
            ),
        },
        "selection": {
            **base["selection"],
            "crossfit_depth_three_root": candidate_root,
            "crossfit_depth_two_root": baseline_root,
            "crossfit_depth_three_brier": {
                str(root): depth_three_risk[root] for root in roots
            },
            "crossfit_depth_two_brier": {
                str(root): depth_two_risk[root] for root in roots
            },
        },
        "ranking": {
            **base["ranking"],
            "crossfit_depth_three_spearman_brier": spearman_correlation(
                depth_three_values,
                endpoint_brier,
            ),
            "crossfit_depth_two_spearman_brier": spearman_correlation(
                depth_two_values,
                endpoint_brier,
            ),
            "crossfit_depth_three_pairwise_concordance": (
                pairwise_concordance(depth_three_values, endpoint_brier)
            ),
            "crossfit_depth_two_pairwise_concordance": (
                pairwise_concordance(depth_two_values, endpoint_brier)
            ),
        },
        "endpoint": {
            policy: {
                key: value
                for key, value in result.items()
                if key != "targets"
            }
            for policy, result in endpoint.items()
        },
        "comparisons": comparisons,
        "novel_comparison_crossfit_depth_three_vs_depth_two": (
            policy_comparison(novel_candidate, novel_baseline)
        ),
    }


def aggregate_scored_trees(
    trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baselines = (
        "crossfit_depth_two",
        "retained_risk_set_depth_three",
        "predictive_bayes_risk_depth_two",
        "myopic_eig",
        "fixed_support_depth_three",
        "uniform_random_candidate_root",
        "positive_test_strategy",
    )
    comparisons = {
        baseline: aggregate_tree_comparisons(
            trees,
            baseline=baseline,
            candidate="crossfit_depth_three",
        )
        for baseline in baselines
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
        values = [tree["ranking"][key] for tree in trees]
        ranking[key] = {
            "mean": _mean(values),
            "tree_bootstrap_95pct": bootstrap_mean_interval(
                values,
                seed=BOOTSTRAP_SEED + offset,
                samples=BOOTSTRAP_SAMPLES,
            ),
        }
    novel_rows = [
        tree["novel_comparison_crossfit_depth_three_vs_depth_two"]
        for tree in trees
    ]
    return {
        "comparisons": comparisons,
        "root_differences": {
            baseline: sum(
                tree["selection"]["crossfit_depth_three_root"]
                != tree["selection"][f"{baseline}_root"]
                for tree in trees
            )
            for baseline in (
                "crossfit_depth_two",
                "retained_risk_set_depth_three",
                "predictive_bayes_risk_depth_two",
            )
        },
        "ranking": ranking,
        "novel_target_mean_differences": {
            "candidate_minus_baseline_brier": _mean(
                [
                    row["candidate_minus_baseline_brier"]
                    for row in novel_rows
                ]
            ),
            "candidate_minus_baseline_hamming": _mean(
                [
                    row["candidate_minus_baseline_hamming"]
                    for row in novel_rows
                ]
            ),
            "coverage_difference": _mean(
                [row["coverage_difference"] for row in novel_rows]
            ),
        },
    }


def _usage(
    live_trees: Sequence[dict[str, Any]],
    validation_snapshots: Sequence[dict[str, Any]],
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
            for snapshot in validation_snapshots
        )
    return usage


def confirmation_gates(
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
    novel = aggregate["novel_target_mean_differences"]
    ranking = aggregate["ranking"]
    return {
        "exact_1856_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_initial_and_target_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
            and tree["mechanics"]["target_valid"] >= MIN_TARGET_VALID
            and tree["mechanics"]["novel_targets"] >= MIN_NOVEL_TARGETS
            for tree in scored_trees
        ),
        "all_eight_validation_supports_valid": all(
            tree["mechanics"]["validation_support_count"]
            == VALIDATION_SUPPORT_COUNT
            and tree["mechanics"]["minimum_validation_support_valid"]
            >= MIN_TARGET_VALID
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
        "crossfit_depth_three_brier_gain_at_least_one_point_five_percent": (
            primary["relative_brier_reduction"] >= 0.015
        ),
        "crossfit_depth_three_brier_ci_below_zero": (
            primary["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "crossfit_depth_three_wins_at_least_ten_trees": (
            primary["brier_tree_wins"] >= 10
        ),
        "no_hamming_regression_vs_crossfit_depth_two": (
            primary["mean_candidate_minus_baseline_hamming"] <= 0.0
        ),
        "novel_target_brier_and_hamming_do_not_regress": (
            novel["candidate_minus_baseline_brier"] <= 0.0
            and novel["candidate_minus_baseline_hamming"] <= 0.0
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
        "crossfit_wins_vs_source_depth_three_at_least_ten_trees": (
            source_depth_three["brier_tree_wins"] >= 10
        ),
        "crossfit_depth_three_beats_myopic_by_five_percent": (
            myopic["relative_brier_reduction"] >= 0.05
        ),
        "crossfit_depth_three_rank_rho_at_least_point_seven": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            >= 0.7
        ),
        "crossfit_depth_three_rho_exceeds_depth_two_by_point_one_five": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            - ranking["crossfit_depth_two_spearman_brier"]["mean"]
            >= 0.15
        ),
    }


def run_confirmation(
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
            (
                validation_supports,
                validation_diagnostics,
                validation_responses,
                snapshots,
            ) = _generate_validation_supports(
                tree_index=tree_index,
                output_dir=output_dir,
                run_id=run_id,
            )
            validation_snapshots.extend(snapshots)
            live_trees.append(live)
            artifacts["raw"]["validation_responses"] = [
                {
                    "seed": seed,
                    "response": response,
                }
                for seed, response in zip(
                    validation_seeds_for_tree(tree_index),
                    validation_responses,
                    strict=True,
                )
            ]
            artifacts["public"]["validation_seeds"] = list(
                validation_seeds_for_tree(tree_index)
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

        scored_trees = [score_public_tree(tree) for tree in public_trees]
        aggregate = aggregate_scored_trees(scored_trees)
        usage = _usage(live_trees, validation_snapshots)
        gates = confirmation_gates(
            scored_trees=scored_trees,
            usage=usage,
            aggregate=aggregate,
        )
        public = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "tree_seeds": list(TREE_SEEDS),
                "target_seeds": list(TARGET_SEEDS),
                "validation_seeds": [
                    list(validation_seeds_for_tree(index))
                    for index in range(len(TREE_SEEDS))
                ],
                "validation_support_count": VALIDATION_SUPPORT_COUNT,
                "validation_draw_weighting": "equal draw weight",
                "expected_requests": EXPECTED_REQUESTS,
                "run_budget_usd": RUN_BUDGET_USD,
                "risk_tie_tolerance": RISK_TIE_TOLERANCE,
                "planning_model": depth.PLANNING_MODEL_ID,
                "target_and_validation_model": depth.TARGET_MODEL_ID,
                "reasoning": False,
                "temperature": depth.TEMPERATURE,
                "first_support_mode": (
                    depth.FIRST_SUPPORT_RETAINED_REJUVENATION
                ),
                "second_support_mode": (
                    depth.SECOND_SUPPORT_RETAINED_REJUVENATION
                ),
                "cumulative_budget_run_id": run_id,
            },
            "usage": usage,
            "aggregate": aggregate,
            "gates": gates,
            "trees": scored_trees,
        }
        public_trees_document = {
            "schema_version": SCHEMA_VERSION,
            "protocol": public["protocol"],
            "raw_responses_sha256": hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest(),
            "trees": public_trees,
        }
        checkpoint(output_dir / "TREES.json", public_trees_document)
        public["trees_sha256"] = hashlib.sha256(
            (output_dir / "TREES.json").read_bytes()
        ).hexdigest()
        public["raw_responses_sha256"] = public_trees_document[
            "raw_responses_sha256"
        ]
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
    result = run_confirmation(
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
