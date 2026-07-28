#!/usr/bin/env python3
"""Evaluate and confirm retained-rejuvenation depth-three Number Game BED."""

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

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_depth_three_development as depth
from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    candidate_roots,
    choose_predictive_bayes_risk_root,
    compile_expression,
    predictive_bayes_risk_scores,
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


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-retained-depth-three-1"
TREE_SEEDS = tuple(range(27800, 27806))
TARGET_SEEDS = tuple(range(27900, 27906))
EXPECTED_REQUESTS_PER_TREE = 50
EXPECTED_REQUESTS = len(TREE_SEEDS) * EXPECTED_REQUESTS_PER_TREE
RUN_BUDGET_USD = 1.07
BRIER_TOLERANCE = 0.005
MIN_INITIAL_VALID = 16
MIN_FIRST_BRANCH_VALID = 8
MIN_SECOND_BRANCH_VALID = 4
MIN_TARGET_VALID = 16
MIN_NOVEL_TARGETS = 8
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 27950


def _rule(item: dict[str, Any]) -> RuleHypothesis:
    return RuleHypothesis(
        name=item["name"],
        expression=item["expression"],
        extension=compile_expression(item["expression"]),
    )


def _first_branches(
    tree: dict[str, Any],
) -> dict[tuple[int, bool], list[RuleHypothesis]]:
    branches = {}
    for key, items in tree["first_branches"].items():
        root, label = key.split(":")
        branches[(int(root), bool(int(label)))] = [
            _rule(item) for item in items
        ]
    return branches


def _second_branches(
    tree: dict[str, Any],
    *,
    key: str,
) -> dict[tuple[int, bool, int, bool], list[RuleHypothesis]]:
    branches = {}
    for branch_key, items in tree[key].items():
        root, first_label, query, second_label = branch_key.split(":")
        branches[
            (
                int(root),
                bool(int(first_label)),
                int(query),
                bool(int(second_label)),
            )
        ] = [_rule(item) for item in items]
    return branches


def retained_second_branches(
    *,
    first_branches: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    generated_second_branches: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
) -> tuple[
    dict[tuple[int, bool, int, bool], list[RuleHypothesis]],
    dict[tuple[int, bool, int, bool], list[RuleHypothesis]],
    dict[tuple[int, bool, int, bool], dict[str, int]],
]:
    retained = {}
    parent_only = {}
    diagnostics = {}
    for branch_key, generated in generated_second_branches.items():
        root, first_label, query, second_label = branch_key
        parent = first_branches[(root, first_label)]
        merged, row = depth.retain_parent_hypotheses(
            parent_support=parent,
            generated_support=generated,
            query=query,
            label=second_label,
        )
        retained[branch_key] = merged
        parent_only[branch_key] = [
            hypothesis
            for hypothesis in parent
            if hypothesis.extension[query] == second_label
        ]
        diagnostics[branch_key] = row
    return retained, parent_only, diagnostics


def _select_root(
    scores: dict[int, dict[str, Any]],
) -> int:
    return depth.choose_risk_set_root(
        scores,
        brier_tolerance=BRIER_TOLERANCE,
    )


def score_public_tree(tree: dict[str, Any]) -> dict[str, Any]:
    initial = [_rule(item) for item in tree["initial"]]
    roots = [int(root) for root in tree["roots"]]
    first = _first_branches(tree)
    generated_key = (
        "generated_second_branches"
        if "generated_second_branches" in tree
        else "second_branches"
    )
    generated = _second_branches(tree, key=generated_key)
    retained, parent_only, retention_rows = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated,
    )
    targets = {
        f"target_{index:02d}_{item['name']}": _rule(item)
        for index, item in enumerate(tree["targets"])
    }
    initial_extensions = {hypothesis.extension for hypothesis in initial}
    novel_targets = {
        name: hypothesis
        for name, hypothesis in targets.items()
        if hypothesis.extension not in initial_extensions
    }

    retained_scores = depth.depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first,
        second_branches=retained,
    )
    generated_scores = depth.depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first,
        second_branches=generated,
    )
    parent_scores = depth.depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first,
        second_branches=parent_only,
    )
    pure_brier_root = choose_predictive_bayes_risk_root(retained_scores)
    candidate_root = _select_root(retained_scores)
    generated_root = _select_root(generated_scores)
    parent_root = _select_root(parent_scores)
    depth_two_root = choose_predictive_bayes_risk_root(
        predictive_bayes_risk_scores(
            support=initial,
            roots=roots,
            branches=first,
        )
    )
    static_first, static_second = depth.static_depth_three_branches(
        support=initial,
        roots=roots,
    )
    static_root = _select_root(
        depth.depth_three_scores(
            support=initial,
            roots=roots,
            first_branches=static_first,
            second_branches=static_second,
        )
    )
    reconstructed_roots, metadata = candidate_roots(
        initial,
        seed=int(tree["tree_seed"]),
    )
    if reconstructed_roots != roots:
        raise ValueError("public roots do not match deterministic candidates")

    per_root = {
        root: depth.evaluate_policy_root_depth_three(
            policy=f"root_{root}",
            root=root,
            targets=targets,
            first_branches=first,
            second_branches=retained,
        )
        for root in roots
    }
    policy_roots = {
        "retained_risk_set_depth_three": candidate_root,
        "retained_pure_brier_depth_three": pure_brier_root,
        "generated_only_depth_three": generated_root,
        "retained_parent_only_depth_three": parent_root,
        "predictive_bayes_risk_depth_two": depth_two_root,
        "myopic_eig": int(metadata["myopic_root"]),
        "fixed_support_depth_three": static_root,
    }
    endpoints = {
        policy: per_root[root] | {"policy": policy}
        for policy, root in policy_roots.items()
    }
    endpoints["uniform_random_candidate_root"] = (
        aggregate_random_root_control(per_root)
    )
    pts_roots = [int(root) for root in metadata["pts_roots"]]
    endpoints["positive_test_strategy"] = aggregate_random_root_control(
        {root: per_root[root] for root in pts_roots}
    ) | {"policy": "uniform_over_two_seeded_pts_roots"}
    baselines = tuple(
        policy
        for policy in endpoints
        if policy != "retained_risk_set_depth_three"
    )
    comparisons = {
        baseline: policy_comparison(
            endpoints["retained_risk_set_depth_three"],
            endpoints[baseline],
        )
        for baseline in baselines
    }
    novel_candidate = depth.evaluate_policy_root_depth_three(
        policy="retained_risk_set_depth_three",
        root=candidate_root,
        targets=novel_targets,
        first_branches=first,
        second_branches=retained,
    )
    novel_depth_two = depth.evaluate_policy_root_depth_three(
        policy="predictive_bayes_risk_depth_two",
        root=depth_two_root,
        targets=novel_targets,
        first_branches=first,
        second_branches=retained,
    )
    source_brier = [
        retained_scores[root]["mean_posterior_predictive_brier"]
        for root in roots
    ]
    endpoint_brier = [
        per_root[root]["mean_posterior_predictive_brier"]
        for root in roots
    ]
    endpoint_hamming = [
        per_root[root]["mean_best_hamming_error"] for root in roots
    ]
    return {
        "tree_index": tree["tree_index"],
        "tree_seed": tree["tree_seed"],
        "target_seed": tree["target_seed"],
        "mechanics": {
            "initial_valid": len(initial),
            "minimum_first_branch_valid": min(
                len(branch) for branch in first.values()
            ),
            "minimum_generated_second_branch_valid": min(
                len(branch) for branch in generated.values()
            ),
            "minimum_parent_second_branch_valid": min(
                len(branch) for branch in parent_only.values()
            ),
            "minimum_retained_second_branch_valid": min(
                len(branch) for branch in retained.values()
            ),
            "mean_generated_second_branch_valid": sum(
                len(branch) for branch in generated.values()
            )
            / len(generated),
            "mean_retained_second_branch_valid": sum(
                len(branch) for branch in retained.values()
            )
            / len(retained),
            "mean_novel_parent_particles_retained": sum(
                row["retained_parent_novel_count"]
                for row in retention_rows.values()
            )
            / len(retention_rows),
            "target_valid": len(targets),
            "novel_targets": len(novel_targets),
        },
        "selection": {
            **metadata,
            "retained_risk_set_depth_three_root": candidate_root,
            "retained_pure_brier_depth_three_root": pure_brier_root,
            "generated_only_depth_three_root": generated_root,
            "retained_parent_only_depth_three_root": parent_root,
            "predictive_bayes_risk_depth_two_root": depth_two_root,
            "fixed_support_depth_three_root": static_root,
            "pts_roots": pts_roots,
        },
        "ranking": {
            "predictive_risk_spearman_brier": spearman_correlation(
                source_brier,
                endpoint_brier,
            ),
            "predictive_risk_spearman_hamming": spearman_correlation(
                source_brier,
                endpoint_hamming,
            ),
            "predictive_pairwise_concordance": pairwise_concordance(
                source_brier,
                endpoint_brier,
            ),
        },
        "endpoint": {
            policy: {
                key: value
                for key, value in endpoint.items()
                if key != "targets"
            }
            for policy, endpoint in endpoints.items()
        },
        "comparisons": comparisons,
        "novel_comparison_depth_three_vs_depth_two": policy_comparison(
            novel_candidate,
            novel_depth_two,
        ),
        "per_root": {
            str(root): {
                "source_predictive_brier": retained_scores[root][
                    "mean_posterior_predictive_brier"
                ],
                "source_hamming": retained_scores[root][
                    "mean_best_hamming_error"
                ],
                "source_coverage": retained_scores[root][
                    "truth_extension_coverage_rate"
                ],
                "endpoint_brier": per_root[root][
                    "mean_posterior_predictive_brier"
                ],
                "endpoint_hamming": per_root[root][
                    "mean_best_hamming_error"
                ],
                "endpoint_coverage": per_root[root][
                    "truth_extension_coverage_rate"
                ],
            }
            for root in roots
        },
    }


def aggregate_scored_trees(
    trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    baselines = (
        "retained_pure_brier_depth_three",
        "generated_only_depth_three",
        "retained_parent_only_depth_three",
        "predictive_bayes_risk_depth_two",
        "myopic_eig",
        "fixed_support_depth_three",
        "uniform_random_candidate_root",
        "positive_test_strategy",
    )
    ranking = {}
    for index, key in enumerate(
        (
            "predictive_risk_spearman_brier",
            "predictive_risk_spearman_hamming",
            "predictive_pairwise_concordance",
        )
    ):
        values = [tree["ranking"][key] for tree in trees]
        ranking[key] = {
            "mean": sum(values) / len(values),
            "tree_bootstrap_95pct": bootstrap_mean_interval(
                values,
                seed=BOOTSTRAP_SEED + index,
                samples=BOOTSTRAP_SAMPLES,
            ),
        }
    return {
        "comparisons": {
            baseline: aggregate_tree_comparisons(
                trees,
                baseline=baseline,
                candidate="retained_risk_set_depth_three",
            )
            for baseline in baselines
        },
        "ranking": ranking,
        "root_differences": {
            baseline: sum(
                tree["selection"]["retained_risk_set_depth_three_root"]
                != tree["selection"][baseline]
                for tree in trees
            )
            for baseline in (
                "retained_pure_brier_depth_three_root",
                "generated_only_depth_three_root",
                "retained_parent_only_depth_three_root",
                "predictive_bayes_risk_depth_two_root",
                "myopic_root",
                "fixed_support_depth_three_root",
            )
        },
        "novel_target_mean_differences": {
            key: sum(
                tree["novel_comparison_depth_three_vs_depth_two"][key]
                for tree in trees
            )
            / len(trees)
            for key in (
                "candidate_minus_baseline_brier",
                "candidate_minus_baseline_hamming",
                "coverage_difference",
            )
        },
    }


def evaluate_open_development(
    *,
    trees_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    document = json.loads(trees_path.read_text())
    trees = [score_public_tree(tree) for tree in document["trees"]]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "posthoc_development",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_sha256": hashlib.sha256(
                trees_path.read_bytes()
            ).hexdigest(),
            "second_support_mode": (
                depth.SECOND_SUPPORT_RETAINED_REJUVENATION
            ),
            "brier_tolerance": BRIER_TOLERANCE,
            "llm_calls": 0,
            "num_trees": len(trees),
        },
        "aggregate": aggregate_scored_trees(trees),
        "trees": trees,
    }
    checkpoint(output_path, result)
    return result


def _usage(tree_results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        key: sum(tree["usage"].get(key, 0) for tree in tree_results)
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


def confirmation_gates(
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
    roots = aggregate["root_differences"]
    return {
        "exact_300_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "transport_attempt_accounting_exact": (
            usage["http_attempts"]
            == usage["adapter_requests"] + usage["retry_count"]
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_run_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_initial_and_first_supports_valid": all(
            tree["mechanics"]["initial_valid"] >= MIN_INITIAL_VALID
            and tree["mechanics"]["minimum_first_branch_valid"]
            >= MIN_FIRST_BRANCH_VALID
            for tree in scored_trees
        ),
        "all_retained_second_supports_have_at_least_four_rules": all(
            tree["mechanics"]["minimum_retained_second_branch_valid"]
            >= MIN_SECOND_BRANCH_VALID
            for tree in scored_trees
        ),
        "all_target_supports_valid": all(
            tree["mechanics"]["target_valid"] >= MIN_TARGET_VALID
            and tree["mechanics"]["novel_targets"] >= MIN_NOVEL_TARGETS
            for tree in scored_trees
        ),
        "live_and_public_mechanics_agree": all(
            live["mechanics"]["minimum_second_branch_valid"]
            == scored["mechanics"][
                "minimum_retained_second_branch_valid"
            ]
            for live, scored in zip(
                live_trees,
                scored_trees,
                strict=True,
            )
        ),
        "retained_root_differs_from_depth_two_on_at_least_three_trees": (
            roots["predictive_bayes_risk_depth_two_root"] >= 3
        ),
        "brier_gain_vs_depth_two_at_least_one_percent": (
            depth_two["relative_brier_reduction"] >= 0.01
        ),
        "brier_wins_vs_depth_two_on_at_least_three_trees": (
            depth_two["brier_tree_wins"] >= 3
        ),
        "no_mean_hamming_regression_vs_depth_two": (
            depth_two["mean_candidate_minus_baseline_hamming"] <= 0.0
        ),
        "no_mean_coverage_regression_vs_depth_two": (
            depth_two["mean_coverage_difference"] >= 0.0
        ),
        "retained_root_differs_from_parent_only_on_at_least_two_trees": (
            roots["retained_parent_only_depth_three_root"] >= 2
        ),
        "brier_gain_vs_parent_only_at_least_one_percent": (
            parent["relative_brier_reduction"] >= 0.01
        ),
        "brier_wins_vs_parent_only_on_at_least_two_trees": (
            parent["brier_tree_wins"] >= 2
        ),
        "retained_root_differs_from_generated_only_on_at_least_two_trees": (
            roots["generated_only_depth_three_root"] >= 2
        ),
        "mean_brier_better_than_generated_only": (
            generated["mean_candidate_minus_baseline_brier"] < 0.0
        ),
        "mean_brier_better_than_myopic": (
            comparisons["myopic_eig"][
                "mean_candidate_minus_baseline_brier"
            ]
            < 0.0
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
        gates = confirmation_gates(
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
                "second_support_mode": (
                    depth.SECOND_SUPPORT_RETAINED_REJUVENATION
                ),
                "brier_tolerance": BRIER_TOLERANCE,
                "run_budget_usd": RUN_BUDGET_USD,
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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--trees", type=Path)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if (args.trees is None) == (args.run_id is None):
        parser.error("provide exactly one of --trees or --run-id")
    if args.trees is not None:
        result = evaluate_open_development(
            trees_path=args.trees.resolve(),
            output_path=args.output.resolve(),
        )
    else:
        result = run_confirmation(
            output_dir=args.output.resolve(),
            run_id=args.run_id,
        )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
