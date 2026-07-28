#!/usr/bin/env python3
"""Audit the two-query endpoint effect of Number Game first retention."""

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
from scripts.number_game_depth_three_development import (
    retain_parent_hypotheses,
)
from scripts.number_game_generator_aware_bed import (
    choose_predictive_bayes_risk_root,
    evaluate_policy_root,
    predictive_bayes_risk_scores,
)
from scripts.number_game_predictive_risk_replication import (
    cluster_bootstrap_interval,
)
from scripts.number_game_ranking_fidelity_audit import (
    spearman_correlation,
)
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-first-retention-two-query-audit-1"
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 28_320


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _endpoint_summary(endpoint: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in endpoint.items()
        if key != "targets"
    }


def audit_tree(tree: dict[str, Any]) -> dict[str, Any]:
    initial = [_rule(item) for item in tree["initial"]]
    roots = [int(root) for root in tree["roots"]]
    generated_key = (
        "generated_first_branches"
        if "generated_first_branches" in tree
        else "first_branches"
    )
    generated = _first_branches(tree, key=generated_key)
    retained = {
        key: retain_parent_hypotheses(
            parent_support=initial,
            generated_support=branch,
            query=key[0],
            label=key[1],
        )[0]
        for key, branch in generated.items()
    }
    targets = {
        f"target_{index:02d}": _rule(item)
        for index, item in enumerate(tree["targets"])
    }
    generated_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=generated,
    )
    retained_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=retained,
    )
    generated_root = choose_predictive_bayes_risk_root(
        generated_scores
    )
    retained_root = choose_predictive_bayes_risk_root(retained_scores)
    generated_policy = evaluate_policy_root(
        policy="generated_first_support",
        root=generated_root,
        targets=targets,
        branches=generated,
    )
    retained_policy = evaluate_policy_root(
        policy="retained_first_support",
        root=retained_root,
        targets=targets,
        branches=retained,
    )
    generated_root_on_retained = evaluate_policy_root(
        policy="generated_root_on_retained_support",
        root=generated_root,
        targets=targets,
        branches=retained,
    )
    retained_root_endpoints = {
        root: evaluate_policy_root(
            policy=f"retained_root_{root}",
            root=root,
            targets=targets,
            branches=retained,
        )
        for root in roots
    }
    target_brier = [
        retained_root_endpoints[root][
            "mean_posterior_predictive_brier"
        ]
        for root in roots
    ]
    generated_source_brier = [
        generated_scores[root]["mean_posterior_predictive_brier"]
        for root in roots
    ]
    retained_source_brier = [
        retained_scores[root]["mean_posterior_predictive_brier"]
        for root in roots
    ]

    def differences(
        candidate: dict[str, Any],
        baseline: dict[str, Any],
    ) -> dict[str, float]:
        return {
            "candidate_minus_baseline_brier": (
                candidate["mean_posterior_predictive_brier"]
                - baseline["mean_posterior_predictive_brier"]
            ),
            "candidate_minus_baseline_hamming": (
                candidate["mean_best_hamming_error"]
                - baseline["mean_best_hamming_error"]
            ),
            "coverage_difference": (
                candidate["truth_extension_coverage_rate"]
                - baseline["truth_extension_coverage_rate"]
            ),
        }

    return {
        "tree_index": tree["tree_index"],
        "tree_seed": tree["tree_seed"],
        "target_seed": tree["target_seed"],
        "generated_root": generated_root,
        "retained_root": retained_root,
        "root_changed": generated_root != retained_root,
        "endpoint": {
            "generated_policy": _endpoint_summary(generated_policy),
            "retained_policy": _endpoint_summary(retained_policy),
            "generated_root_on_retained_support": _endpoint_summary(
                generated_root_on_retained
            ),
        },
        "complete_policy_difference": differences(
            retained_policy,
            generated_policy,
        ),
        "selection_only_difference": differences(
            retained_policy,
            generated_root_on_retained,
        ),
        "ranking": {
            "generated_source_risk_spearman_target_brier": (
                spearman_correlation(
                    generated_source_brier,
                    target_brier,
                )
            ),
            "retained_source_risk_spearman_target_brier": (
                spearman_correlation(
                    retained_source_brier,
                    target_brier,
                )
            ),
        },
    }


def _aggregate_difference(
    trees: Sequence[dict[str, Any]],
    *,
    key: str,
    candidate_endpoint: str,
    baseline_endpoint: str,
    seed: int,
) -> dict[str, Any]:
    differences = [
        tree[key]["candidate_minus_baseline_brier"]
        for tree in trees
    ]
    hamming = [
        tree[key]["candidate_minus_baseline_hamming"]
        for tree in trees
    ]
    coverage = [
        tree[key]["coverage_difference"] for tree in trees
    ]
    candidate_brier = _mean(
        [
            tree["endpoint"][candidate_endpoint][
                "mean_posterior_predictive_brier"
            ]
            for tree in trees
        ]
    )
    baseline_brier = _mean(
        [
            tree["endpoint"][baseline_endpoint][
                "mean_posterior_predictive_brier"
            ]
            for tree in trees
        ]
    )
    return {
        "candidate_mean_brier": candidate_brier,
        "baseline_mean_brier": baseline_brier,
        "relative_brier_reduction": (
            (baseline_brier - candidate_brier) / baseline_brier
        ),
        "mean_candidate_minus_baseline_brier": _mean(differences),
        "tree_bootstrap_brier_difference_95pct": (
            cluster_bootstrap_interval(
                differences,
                samples=BOOTSTRAP_SAMPLES,
                seed=seed,
            )
        ),
        "brier_tree_wins": sum(value < 0.0 for value in differences),
        "brier_tree_losses": sum(value > 0.0 for value in differences),
        "brier_tree_ties": sum(value == 0.0 for value in differences),
        "mean_candidate_minus_baseline_hamming": _mean(hamming),
        "tree_bootstrap_hamming_difference_95pct": (
            cluster_bootstrap_interval(
                hamming,
                samples=BOOTSTRAP_SAMPLES,
                seed=seed + 1,
            )
        ),
        "mean_coverage_difference": _mean(coverage),
        "tree_bootstrap_coverage_difference_95pct": (
            cluster_bootstrap_interval(
                coverage,
                samples=BOOTSTRAP_SAMPLES,
                seed=seed + 2,
            )
        ),
    }


def aggregate_trees(
    trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    generated_rho = [
        tree["ranking"][
            "generated_source_risk_spearman_target_brier"
        ]
        for tree in trees
    ]
    retained_rho = [
        tree["ranking"][
            "retained_source_risk_spearman_target_brier"
        ]
        for tree in trees
    ]
    return {
        "tree_count": len(trees),
        "roots_changed": sum(tree["root_changed"] for tree in trees),
        "complete_policy": _aggregate_difference(
            trees,
            key="complete_policy_difference",
            candidate_endpoint="retained_policy",
            baseline_endpoint="generated_policy",
            seed=BOOTSTRAP_SEED,
        ),
        "selection_only": _aggregate_difference(
            trees,
            key="selection_only_difference",
            candidate_endpoint="retained_policy",
            baseline_endpoint="generated_root_on_retained_support",
            seed=BOOTSTRAP_SEED + 10,
        ),
        "ranking": {
            "generated_source_risk_spearman_target_brier_mean": (
                _mean(generated_rho)
            ),
            "generated_source_risk_spearman_target_brier_95pct": (
                cluster_bootstrap_interval(
                    generated_rho,
                    samples=BOOTSTRAP_SAMPLES,
                    seed=BOOTSTRAP_SEED + 20,
                )
            ),
            "retained_source_risk_spearman_target_brier_mean": (
                _mean(retained_rho)
            ),
            "retained_source_risk_spearman_target_brier_95pct": (
                cluster_bootstrap_interval(
                    retained_rho,
                    samples=BOOTSTRAP_SAMPLES,
                    seed=BOOTSTRAP_SEED + 21,
                )
            ),
        },
    }


def audit_source(
    *,
    label: str,
    path: Path,
) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    trees = [audit_tree(tree) for tree in payload["trees"]]
    try:
        displayed_path = str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        displayed_path = str(path)
    return {
        "label": label,
        "path": displayed_path,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "aggregate": aggregate_trees(trees),
        "trees": trees,
    }


def run_audit(
    *,
    inputs: Sequence[tuple[str, Path]],
    output_path: Path,
) -> dict[str, Any]:
    sources = [
        audit_source(label=label, path=path)
        for label, path in inputs
    ]
    combined_trees = [
        tree for source in sources for tree in source["trees"]
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "completed_posthoc_audit",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model_calls": 0,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "endpoint": (
                "two queries with independent target rules"
            ),
            "complete_policy_comparison": (
                "retained-selected root on retained first support versus "
                "generated-selected root on generated first support"
            ),
            "selection_only_comparison": (
                "retained-selected versus generated-selected root, both "
                "deployed on retained first support"
            ),
        },
        "aggregate": aggregate_trees(combined_trees),
        "sources": sources,
    }
    checkpoint(output_path, result)
    return result


def _input(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label or not raw_path:
        raise argparse.ArgumentTypeError(
            "input must have the form LABEL=PATH"
        )
    path = Path(raw_path).resolve()
    if not path.is_file():
        raise argparse.ArgumentTypeError(f"input does not exist: {path}")
    return label, path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=_input,
        action="append",
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(
        inputs=args.input,
        output_path=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
