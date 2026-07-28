#!/usr/bin/env python3
"""Measure fixed Number Game policies on fresh high-precision endpoints."""

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
)
from scripts.number_game_full_retention_depth_three import (
    openrouter_remaining_credit,
)
from scripts.number_game_predictive_risk_holdout import (
    aggregate_random_root_control,
)
from scripts.number_game_ranking_fidelity_audit import (
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
INTERFACE_VERSION = "number-game-crossfit-endpoint-precision-1"
SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_crossfit_depth_three_confirmation"
    / "number-game-crossfit-depth-three-confirmation-20260728"
)
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_RESULT_SHA256 = (
    "1081da1e8381b88f7cd3fcd905b5ed539047863bcc087d686e509ed8f82d794a"
)
SOURCE_TREES_SHA256 = (
    "cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9"
)
TREE_COUNT = 32
ENDPOINT_DRAWS_PER_TREE = 16
ENDPOINT_SEED_START = 29000
EXPECTED_REQUESTS = TREE_COUNT * ENDPOINT_DRAWS_PER_TREE
RUN_BUDGET_USD = 1.50
MIN_STARTING_BALANCE_USD = 1.14
MIN_ENDPOINT_VALID = 16
MIN_NOVEL_PER_TREE = 128


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def endpoint_seeds_for_tree(tree_index: int) -> tuple[int, ...]:
    start = ENDPOINT_SEED_START + tree_index * ENDPOINT_DRAWS_PER_TREE
    return tuple(range(start, start + ENDPOINT_DRAWS_PER_TREE))


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


def _generate_endpoint_supports(
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
    seeds = endpoint_seeds_for_tree(tree_index)
    adapters = [
        depth._adapter(
            model=depth.TARGET_MODEL_ID,
            run_id=run_id,
            output_dir=output_dir,
            request_seed=seed,
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

    with ThreadPoolExecutor(max_workers=ENDPOINT_DRAWS_PER_TREE) as executor:
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


def _average_endpoint(
    rows: Sequence[dict[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    return {
        "policy": policy,
        "mean_posterior_predictive_brier": _mean(
            [row["mean_posterior_predictive_brier"] for row in rows]
        ),
        "mean_best_hamming_error": _mean(
            [row["mean_best_hamming_error"] for row in rows]
        ),
        "truth_extension_coverage_rate": _mean(
            [row["truth_extension_coverage_rate"] for row in rows]
        ),
    }


def _average_comparison(
    rows: Sequence[dict[str, Any]],
    *,
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_brier = baseline["mean_posterior_predictive_brier"]
    baseline_hamming = baseline["mean_best_hamming_error"]
    return {
        "candidate_minus_baseline_brier": _mean(
            [row["candidate_minus_baseline_brier"] for row in rows]
        ),
        "relative_brier_reduction": (
            baseline_brier - candidate["mean_posterior_predictive_brier"]
        )
        / baseline_brier,
        "candidate_minus_baseline_hamming": _mean(
            [row["candidate_minus_baseline_hamming"] for row in rows]
        ),
        "relative_hamming_reduction": (
            baseline_hamming - candidate["mean_best_hamming_error"]
        )
        / baseline_hamming,
        "coverage_difference": _mean(
            [row["coverage_difference"] for row in rows]
        ),
    }


def _summary_comparison(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, Any]:
    baseline_brier = baseline["mean_posterior_predictive_brier"]
    baseline_hamming = baseline["mean_best_hamming_error"]
    return {
        "candidate_minus_baseline_brier": (
            candidate["mean_posterior_predictive_brier"]
            - baseline_brier
        ),
        "relative_brier_reduction": (
            baseline_brier
            - candidate["mean_posterior_predictive_brier"]
        )
        / baseline_brier,
        "candidate_minus_baseline_hamming": (
            candidate["mean_best_hamming_error"] - baseline_hamming
        ),
        "relative_hamming_reduction": (
            baseline_hamming - candidate["mean_best_hamming_error"]
        )
        / baseline_hamming,
        "coverage_difference": (
            candidate["truth_extension_coverage_rate"]
            - baseline["truth_extension_coverage_rate"]
        ),
    }


def score_fixed_tree(
    source_tree: dict[str, Any],
    source_metrics: dict[str, Any],
    endpoint_supports: Sequence[Sequence[dict[str, Any]]],
) -> dict[str, Any]:
    initial = [_rule(item) for item in source_tree["initial"]]
    roots = [int(root) for root in source_tree["roots"]]
    first = _first_branches(source_tree)
    generated_second = _second_branches(
        source_tree,
        key="generated_second_branches",
    )
    second, _, _ = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated_second,
    )
    selection = source_metrics["selection"]
    policy_roots = {
        "crossfit_depth_three": int(
            selection["crossfit_depth_three_root"]
        ),
        "crossfit_depth_two": int(selection["crossfit_depth_two_root"]),
        "retained_risk_set_depth_three": int(
            selection["retained_risk_set_depth_three_root"]
        ),
        "predictive_bayes_risk_depth_two": int(
            selection["predictive_bayes_risk_depth_two_root"]
        ),
        "myopic_eig": int(selection["myopic_root"]),
        "fixed_support_depth_three": int(
            selection["fixed_support_depth_three_root"]
        ),
    }
    pts_roots = [int(root) for root in selection["pts_roots"]]
    initial_extensions = {hypothesis.extension for hypothesis in initial}
    draw_endpoints = []
    draw_novel_endpoints = []
    per_root_brier_rows = {root: [] for root in roots}
    valid_counts = []
    novel_counts = []
    for draw_index, public_support in enumerate(endpoint_supports):
        support = [_rule(item) for item in public_support]
        valid_counts.append(len(support))
        targets = {
            f"endpoint_{draw_index:02d}_{index:02d}_{hypothesis.name}": (
                hypothesis
            )
            for index, hypothesis in enumerate(support)
        }
        novel_targets = {
            name: hypothesis
            for name, hypothesis in targets.items()
            if hypothesis.extension not in initial_extensions
        }
        novel_counts.append(len(novel_targets))
        per_root = {
            root: depth.evaluate_policy_root_depth_three(
                policy=f"root_{root}",
                root=root,
                targets=targets,
                first_branches=first,
                second_branches=second,
            )
            for root in roots
        }
        per_root_novel = {
            root: depth.evaluate_policy_root_depth_three(
                policy=f"novel_root_{root}",
                root=root,
                targets=novel_targets,
                first_branches=first,
                second_branches=second,
            )
            for root in roots
        }
        for root in roots:
            per_root_brier_rows[root].append(
                per_root[root]["mean_posterior_predictive_brier"]
            )
        endpoints = {
            policy: per_root[root] | {"policy": policy}
            for policy, root in policy_roots.items()
        }
        novel_endpoints = {
            policy: per_root_novel[root] | {"policy": policy}
            for policy, root in policy_roots.items()
        }
        endpoints["uniform_random_candidate_root"] = (
            aggregate_random_root_control(per_root)
        )
        novel_endpoints["uniform_random_candidate_root"] = (
            aggregate_random_root_control(per_root_novel)
        )
        endpoints["positive_test_strategy"] = aggregate_random_root_control(
            {root: per_root[root] for root in pts_roots}
        ) | {"policy": "uniform_over_two_seeded_pts_roots"}
        novel_endpoints["positive_test_strategy"] = (
            aggregate_random_root_control(
                {root: per_root_novel[root] for root in pts_roots}
            )
            | {"policy": "uniform_over_two_seeded_pts_roots"}
        )
        draw_endpoints.append(endpoints)
        draw_novel_endpoints.append(novel_endpoints)

    policies = tuple(draw_endpoints[0])
    endpoint = {
        policy: _average_endpoint(
            [row[policy] for row in draw_endpoints],
            policy=policy,
        )
        for policy in policies
    }
    novel_endpoint = {
        policy: _average_endpoint(
            [row[policy] for row in draw_novel_endpoints],
            policy=policy,
        )
        for policy in policies
    }
    baselines = tuple(
        policy for policy in policies if policy != "crossfit_depth_three"
    )
    comparisons = {
        baseline: _average_comparison(
            [
                _summary_comparison(
                    row["crossfit_depth_three"],
                    row[baseline],
                )
                for row in draw_endpoints
            ],
            candidate=endpoint["crossfit_depth_three"],
            baseline=endpoint[baseline],
        )
        for baseline in baselines
    }
    novel_comparison = _average_comparison(
        [
            _summary_comparison(
                row["crossfit_depth_three"],
                row["crossfit_depth_two"],
            )
            for row in draw_novel_endpoints
        ],
        candidate=novel_endpoint["crossfit_depth_three"],
        baseline=novel_endpoint["crossfit_depth_two"],
    )
    endpoint_brier = [
        _mean(per_root_brier_rows[root]) for root in roots
    ]
    depth_three_risk = [
        float(selection["crossfit_depth_three_brier"][str(root)])
        for root in roots
    ]
    depth_two_risk = [
        float(selection["crossfit_depth_two_brier"][str(root)])
        for root in roots
    ]
    return {
        "tree_index": int(source_tree["tree_index"]),
        "tree_seed": int(source_tree["tree_seed"]),
        "source_target_seed": int(source_tree["target_seed"]),
        "mechanics": {
            **source_metrics["mechanics"],
            "endpoint_draw_count": len(endpoint_supports),
            "minimum_endpoint_support_valid": min(valid_counts),
            "mean_endpoint_support_valid": _mean(valid_counts),
            "total_novel_endpoint_hypotheses": sum(novel_counts),
            "mean_novel_endpoint_hypotheses": _mean(novel_counts),
        },
        "selection": selection,
        "ranking": {
            "crossfit_depth_three_spearman_brier": spearman_correlation(
                depth_three_risk,
                endpoint_brier,
            ),
            "crossfit_depth_two_spearman_brier": spearman_correlation(
                depth_two_risk,
                endpoint_brier,
            ),
            "crossfit_depth_three_pairwise_concordance": (
                pairwise_concordance(depth_three_risk, endpoint_brier)
            ),
            "crossfit_depth_two_pairwise_concordance": (
                pairwise_concordance(depth_two_risk, endpoint_brier)
            ),
        },
        "endpoint": endpoint,
        "comparisons": comparisons,
        "novel_comparison_crossfit_depth_three_vs_depth_two": (
            novel_comparison
        ),
        "per_root_endpoint_brier": {
            str(root): _mean(per_root_brier_rows[root]) for root in roots
        },
    }


def _usage(snapshots: Sequence[dict[str, Any]]) -> dict[str, Any]:
    fields = {
        "adapter_requests": "adapter_requests",
        "http_attempts": "http_attempts",
        "retry_count": "retry_count",
        "provider_error_retries": "provider_error_retries",
        "adapter_reasoning_tokens": "adapter_reasoning_tokens",
        "forced_exits": "forced_exits",
        "run_cost_usd": "adapter_cost_usd",
    }
    return {
        output_key: sum(
            snapshot.get(snapshot_key, 0) for snapshot in snapshots
        )
        for output_key, snapshot_key in fields.items()
    }


def precision_gates(
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
        "exact_512_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_sixteen_endpoint_supports_valid": all(
            tree["mechanics"]["endpoint_draw_count"]
            == ENDPOINT_DRAWS_PER_TREE
            and tree["mechanics"]["minimum_endpoint_support_valid"]
            >= MIN_ENDPOINT_VALID
            for tree in scored_trees
        ),
        "every_tree_has_at_least_128_novel_endpoint_hypotheses": all(
            tree["mechanics"]["total_novel_endpoint_hypotheses"]
            >= MIN_NOVEL_PER_TREE
            for tree in scored_trees
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
        "crossfit_depth_three_beats_myopic_by_five_percent": (
            myopic["relative_brier_reduction"] >= 0.05
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


def run_precision(
    *,
    source_result_path: Path,
    source_trees_path: Path,
    output_dir: Path,
    run_id: str,
) -> dict[str, Any]:
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source RESULT.json hash changed")
    if sha256_file(source_trees_path) != SOURCE_TREES_SHA256:
        raise ValueError("source TREES.json hash changed")
    source_result = json.loads(source_result_path.read_text())
    source_trees = json.loads(source_trees_path.read_text())
    if len(source_trees["trees"]) != TREE_COUNT:
        raise ValueError("source tree count changed")

    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    raw: dict[str, Any] = {"trees": []}
    public_endpoint_trees = []
    snapshots = []
    try:
        for tree_index in range(TREE_COUNT):
            (
                supports,
                diagnostics,
                responses,
                tree_snapshots,
            ) = _generate_endpoint_supports(
                tree_index=tree_index,
                output_dir=output_dir,
                run_id=run_id,
            )
            snapshots.extend(tree_snapshots)
            seeds = endpoint_seeds_for_tree(tree_index)
            raw["trees"].append(
                {
                    "tree_index": tree_index,
                    "responses": [
                        {"seed": seed, "response": response}
                        for seed, response in zip(
                            seeds,
                            responses,
                            strict=True,
                        )
                    ],
                }
            )
            public_endpoint_trees.append(
                {
                    "tree_index": tree_index,
                    "seeds": list(seeds),
                    "supports": [
                        [
                            hypothesis.public_dict()
                            for hypothesis in support
                        ]
                        for support in supports
                    ],
                    "diagnostics": diagnostics,
                }
            )
            checkpoint(raw_path, raw)

        scored_trees = [
            score_fixed_tree(
                source_tree,
                source_metrics,
                endpoints["supports"],
            )
            for source_tree, source_metrics, endpoints in zip(
                source_trees["trees"],
                source_result["trees"],
                public_endpoint_trees,
                strict=True,
            )
        ]
        aggregate = aggregate_scored_trees(scored_trees)
        usage = _usage(snapshots)
        gates = precision_gates(
            scored_trees=scored_trees,
            usage=usage,
            aggregate=aggregate,
        )
        endpoint_document = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "raw_responses_sha256": sha256_file(raw_path),
            "trees": public_endpoint_trees,
        }
        checkpoint(output_dir / "ENDPOINTS.json", endpoint_document)
        public = {
            "schema_version": SCHEMA_VERSION,
            "status": "passed" if all(gates.values()) else "gated_null",
            "protocol": {
                "interface_version": INTERFACE_VERSION,
                "source_result_sha256": SOURCE_RESULT_SHA256,
                "source_trees_sha256": SOURCE_TREES_SHA256,
                "tree_count": TREE_COUNT,
                "endpoint_draws_per_tree": ENDPOINT_DRAWS_PER_TREE,
                "endpoint_seeds": [
                    list(endpoint_seeds_for_tree(index))
                    for index in range(TREE_COUNT)
                ],
                "opened_source_endpoint_excluded": True,
                "draw_weighting": "equal draw weight",
                "target_model": depth.TARGET_MODEL_ID,
                "reasoning": False,
                "temperature": depth.TEMPERATURE,
                "expected_requests": EXPECTED_REQUESTS,
                "run_budget_usd": RUN_BUDGET_USD,
                "cumulative_budget_run_id": run_id,
            },
            "usage": usage,
            "aggregate": aggregate,
            "gates": gates,
            "trees": scored_trees,
            "endpoints_sha256": sha256_file(
                output_dir / "ENDPOINTS.json"
            ),
            "raw_responses_sha256": endpoint_document[
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
    parser.add_argument("--source-result", type=Path, default=SOURCE_RESULT)
    parser.add_argument("--source-trees", type=Path, default=SOURCE_TREES)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-balance-check", action="store_true")
    args = parser.parse_args()
    if not args.skip_balance_check:
        require_starting_balance(openrouter_remaining_credit())
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_precision(
        source_result_path=args.source_result,
        source_trees_path=args.source_trees,
        output_dir=args.output_dir,
        run_id=args.run_id,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
