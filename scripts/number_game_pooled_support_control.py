#!/usr/bin/env python3
"""Evaluate an outcome-agnostic pooled-support Number Game control."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    choose_predictive_bayes_risk_root,
    compile_expression,
    evaluate_policy_root,
    predictive_bayes_risk_scores,
)
from scripts.number_game_predictive_risk_holdout import policy_comparison
from scripts.number_game_predictive_risk_replication import (
    aggregate_tree_comparisons,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-pooled-support-control-1"


def _rule(item: dict[str, Any]) -> RuleHypothesis:
    return RuleHypothesis(
        name=item["name"],
        expression=item["expression"],
        extension=compile_expression(item["expression"]),
    )


def dedupe_rules(
    hypotheses: Iterable[RuleHypothesis],
) -> list[RuleHypothesis]:
    unique = []
    seen = set()
    for hypothesis in hypotheses:
        if hypothesis.extension not in seen:
            unique.append(hypothesis)
            seen.add(hypothesis.extension)
    return unique


def pooled_static_branches(
    *,
    roots: Sequence[int],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
) -> tuple[
    dict[tuple[int, bool], list[RuleHypothesis]],
    dict[int, int],
]:
    pooled_branches = {}
    pool_sizes = {}
    for root in roots:
        pool = dedupe_rules(
            [
                *branches[(root, False)],
                *branches[(root, True)],
            ]
        )
        pool_sizes[root] = len(pool)
        for label in (False, True):
            pooled_branches[(root, label)] = [
                hypothesis
                for hypothesis in pool
                if hypothesis.extension[root] == label
            ]
    return pooled_branches, pool_sizes


def global_pooled_static_branches(
    *,
    roots: Sequence[int],
    branches: dict[tuple[int, bool], Sequence[RuleHypothesis]],
) -> tuple[
    dict[tuple[int, bool], list[RuleHypothesis]],
    int,
]:
    pool = dedupe_rules(
        hypothesis
        for root in roots
        for label in (False, True)
        for hypothesis in branches[(root, label)]
    )
    return (
        {
            (root, label): [
                hypothesis
                for hypothesis in pool
                if hypothesis.extension[root] == label
            ]
            for root in roots
            for label in (False, True)
        },
        len(pool),
    )


def score_public_tree(tree: dict[str, Any]) -> dict[str, Any]:
    initial = [_rule(item) for item in tree["initial"]]
    roots = [int(root) for root in tree["roots"]]
    branches = {}
    for key, items in tree["branches"].items():
        root, label = key.split(":")
        branches[(int(root), bool(int(label)))] = [
            _rule(item) for item in items
        ]
    targets_list = [_rule(item) for item in tree["targets"]]
    targets = {
        f"target_{index:02d}_{hypothesis.name}": hypothesis
        for index, hypothesis in enumerate(targets_list)
    }
    initial_extensions = {hypothesis.extension for hypothesis in initial}
    novel_targets = {
        name: hypothesis
        for name, hypothesis in targets.items()
        if hypothesis.extension not in initial_extensions
    }

    actual_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=branches,
    )
    predictive_root = choose_predictive_bayes_risk_root(actual_scores)
    root_pooled_branches, root_pool_sizes = pooled_static_branches(
        roots=roots,
        branches=branches,
    )
    root_pooled_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=root_pooled_branches,
    )
    root_pooled_root = choose_predictive_bayes_risk_root(root_pooled_scores)
    global_pooled_branches, global_pool_size = (
        global_pooled_static_branches(
            roots=roots,
            branches=branches,
        )
    )
    global_pooled_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=global_pooled_branches,
    )
    global_pooled_root = choose_predictive_bayes_risk_root(
        global_pooled_scores
    )

    endpoint = {
        "predictive_bayes_risk": evaluate_policy_root(
            policy="predictive_bayes_risk",
            root=predictive_root,
            targets=targets,
            branches=branches,
        ),
        "global_pooled_support_depth_two": evaluate_policy_root(
            policy="global_pooled_support_depth_two",
            root=global_pooled_root,
            targets=targets,
            branches=branches,
        ),
    }
    novel_endpoint = {
        policy: evaluate_policy_root(
            policy=policy,
            root=result["root"],
            targets=novel_targets,
            branches=branches,
        )
        for policy, result in endpoint.items()
    }
    comparison = policy_comparison(
        endpoint["predictive_bayes_risk"],
        endpoint["global_pooled_support_depth_two"],
    )
    novel_comparison = policy_comparison(
        novel_endpoint["predictive_bayes_risk"],
        novel_endpoint["global_pooled_support_depth_two"],
    )
    return {
        "tree_index": tree["tree_index"],
        "tree_seed": tree["tree_seed"],
        "target_seed": tree["target_seed"],
        "selection": {
            "predictive_bayes_risk_root": predictive_root,
            "root_pooled_support_depth_two_root": root_pooled_root,
            "global_pooled_support_depth_two_root": global_pooled_root,
            "root_pooled_identity": predictive_root == root_pooled_root,
            "roots_differ": predictive_root != global_pooled_root,
        },
        "root_pool_sizes": {
            str(root): root_pool_sizes[root] for root in roots
        },
        "global_pool_size": global_pool_size,
        "endpoint": endpoint,
        "comparisons": {
            "global_pooled_support_depth_two": comparison
        },
        "novel_comparison_vs_pooled": novel_comparison,
    }


def evaluate_sources(
    *,
    tree_paths: Sequence[Path],
    output_path: Path,
) -> dict[str, Any]:
    sources = {}
    for tree_path in tree_paths:
        document = json.loads(tree_path.read_text())
        trees = [score_public_tree(tree) for tree in document["trees"]]
        aggregate = aggregate_tree_comparisons(
            trees,
            baseline="global_pooled_support_depth_two",
        )
        novel_brier = [
            tree["novel_comparison_vs_pooled"][
                "candidate_minus_baseline_brier"
            ]
            for tree in trees
        ]
        novel_hamming = [
            tree["novel_comparison_vs_pooled"][
                "candidate_minus_baseline_hamming"
            ]
            for tree in trees
        ]
        sources[tree_path.parent.name] = {
            "source_sha256": hashlib.sha256(
                tree_path.read_bytes()
            ).hexdigest(),
            "source_protocol": document["protocol"],
            "num_trees": len(trees),
            "roots_differ": sum(
                tree["selection"]["roots_differ"] for tree in trees
            ),
            "root_pooled_identity_count": sum(
                tree["selection"]["root_pooled_identity"]
                for tree in trees
            ),
            "aggregate": aggregate,
            "novel_target_mean_differences": {
                "candidate_minus_pooled_brier": sum(novel_brier)
                / len(novel_brier),
                "candidate_minus_pooled_hamming": sum(novel_hamming)
                / len(novel_hamming),
                "brier_tree_wins": sum(value < 0.0 for value in novel_brier),
                "hamming_tree_wins": sum(
                    value < 0.0 for value in novel_hamming
                ),
            },
            "trees": trees,
        }
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "description": (
                "Pool all generated supports across roots and answers, "
                "dedupe, then filter one global support by the query answer."
            ),
            "llm_calls": 0,
        },
        "sources": sources,
    }
    checkpoint(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate_sources(
        tree_paths=[path.resolve() for path in args.trees],
        output_path=args.output.resolve(),
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
