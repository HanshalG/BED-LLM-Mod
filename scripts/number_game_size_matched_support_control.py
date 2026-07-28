#!/usr/bin/env python3
"""Evaluate a size-matched static-support control on Number Game trees."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    RuleHypothesis,
    binary_entropy,
    choose_predictive_bayes_risk_root,
    compile_expression,
    evaluate_policy_root,
    predictive_bayes_risk_scores,
)
from scripts.number_game_pooled_support_control import dedupe_rules
from scripts.number_game_predictive_risk_holdout import policy_comparison
from scripts.number_game_predictive_risk_replication import (
    aggregate_tree_comparisons,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-size-matched-support-control-1"
SAMPLE_COUNT = 32
SAMPLE_SEED_NAMESPACE = "size-matched-global-support-v1"


def _rule(item: dict[str, Any]) -> RuleHypothesis:
    return RuleHypothesis(
        name=item["name"],
        expression=item["expression"],
        extension=compile_expression(item["expression"]),
    )


def _stable_seed(
    *,
    tree_seed: int,
    sample_index: int,
    root: int,
    label: bool,
) -> int:
    payload = (
        f"{SAMPLE_SEED_NAMESPACE}:{tree_seed}:{sample_index}:"
        f"{root}:{int(label)}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big")


def size_matched_static_branches(
    *,
    global_pool: Sequence[RuleHypothesis],
    actual_branches: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    roots: Sequence[int],
    tree_seed: int,
    sample_index: int,
) -> dict[tuple[int, bool], list[RuleHypothesis]]:
    sampled = {}
    for root in roots:
        for label in (False, True):
            eligible = [
                hypothesis
                for hypothesis in global_pool
                if hypothesis.extension[root] == label
            ]
            branch_size = len(actual_branches[(root, label)])
            if len(eligible) < branch_size:
                raise ValueError("global pool cannot match actual branch size")
            rng = random.Random(
                _stable_seed(
                    tree_seed=tree_seed,
                    sample_index=sample_index,
                    root=root,
                    label=label,
                )
            )
            sampled[(root, label)] = rng.sample(
                eligible,
                branch_size,
            )
    return sampled


def _best_query_index(
    branch: np.ndarray,
    *,
    excluded: set[int],
) -> int:
    counts = branch.sum(axis=0)
    denominator = len(branch)
    scores = [
        (
            -math.inf
            if query in excluded
            else binary_entropy(int(counts[query]) / denominator)
        )
        for query in DOMAIN
    ]
    return max(DOMAIN, key=lambda query: (scores[query], -query))


def fast_predictive_bayes_risk_root(
    *,
    support: Sequence[RuleHypothesis],
    roots: Sequence[int],
    branches: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
) -> int:
    truths = np.asarray(
        [hypothesis.extension for hypothesis in support],
        dtype=np.bool_,
    )
    root_scores = {}
    for root in roots:
        briers = []
        hammings = []
        covered = []
        branch_arrays = {
            label: np.asarray(
                [
                    hypothesis.extension
                    for hypothesis in branches[(root, label)]
                ],
                dtype=np.bool_,
            )
            for label in (False, True)
        }
        for truth in truths:
            label = bool(truth[root])
            branch = branch_arrays[label]
            if len(branch) == 0:
                briers.append(1.0)
                hammings.append(1.0)
                covered.append(False)
                continue
            second_query = _best_query_index(
                branch,
                excluded={root},
            )
            survivors = branch[
                branch[:, second_query] == truth[second_query]
            ]
            if len(survivors) == 0:
                briers.append(1.0)
                hammings.append(1.0)
                covered.append(False)
                continue
            probabilities = survivors.mean(axis=0)
            mask = np.ones(len(DOMAIN), dtype=np.bool_)
            mask[[root, second_query]] = False
            briers.append(
                float(np.mean((probabilities[mask] - truth[mask]) ** 2))
            )
            hammings.append(
                float(np.min(np.mean(survivors != truth, axis=1)))
            )
            covered.append(bool(np.any(np.all(survivors == truth, axis=1))))
        root_scores[root] = (
            sum(briers) / len(briers),
            sum(hammings) / len(hammings),
            -(sum(covered) / len(covered)),
            root,
        )
    return min(root_scores, key=root_scores.__getitem__)


def average_endpoint_mixture(
    endpoints: Sequence[dict[str, Any]],
    *,
    policy: str,
) -> dict[str, Any]:
    if not endpoints:
        raise ValueError("endpoint mixture is empty")
    rows_by_endpoint = [
        {row["target"]: row for row in endpoint["targets"]}
        for endpoint in endpoints
    ]
    target_names = list(rows_by_endpoint[0])
    if any(set(rows) != set(target_names) for rows in rows_by_endpoint):
        raise ValueError("endpoint target sets differ")
    rows = []
    for target in target_names:
        target_rows = [mapping[target] for mapping in rows_by_endpoint]
        rows.append(
            {
                "target": target,
                "posterior_predictive_brier": sum(
                    row["posterior_predictive_brier"]
                    for row in target_rows
                )
                / len(target_rows),
                "best_hamming_error": sum(
                    row["best_hamming_error"] for row in target_rows
                )
                / len(target_rows),
                "truth_extension_covered": sum(
                    row["truth_extension_covered"]
                    for row in target_rows
                )
                / len(target_rows),
            }
        )
    return {
        "policy": policy,
        "root": None,
        "mean_posterior_predictive_brier": sum(
            row["posterior_predictive_brier"] for row in rows
        )
        / len(rows),
        "mean_best_hamming_error": sum(
            row["best_hamming_error"] for row in rows
        )
        / len(rows),
        "truth_extension_coverage_rate": sum(
            row["truth_extension_covered"] for row in rows
        )
        / len(rows),
        "targets": rows,
    }


def score_public_tree(
    tree: dict[str, Any],
    *,
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
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
    global_pool = dedupe_rules(
        hypothesis
        for root in roots
        for label in (False, True)
        for hypothesis in branches[(root, label)]
    )

    reference_scores = predictive_bayes_risk_scores(
        support=initial,
        roots=roots,
        branches=branches,
    )
    predictive_root = choose_predictive_bayes_risk_root(reference_scores)
    fast_root = fast_predictive_bayes_risk_root(
        support=initial,
        roots=roots,
        branches=branches,
    )
    if predictive_root != fast_root:
        raise ValueError("fast and reference root selectors disagree")

    per_root = {
        root: evaluate_policy_root(
            policy=f"root_{root}",
            root=root,
            targets=targets,
            branches=branches,
        )
        for root in roots
    }
    novel_per_root = {
        root: evaluate_policy_root(
            policy=f"novel_root_{root}",
            root=root,
            targets=novel_targets,
            branches=branches,
        )
        for root in roots
    }
    sampled_roots = []
    for sample_index in range(sample_count):
        sampled_branches = size_matched_static_branches(
            global_pool=global_pool,
            actual_branches=branches,
            roots=roots,
            tree_seed=int(tree["tree_seed"]),
            sample_index=sample_index,
        )
        sampled_roots.append(
            fast_predictive_bayes_risk_root(
                support=initial,
                roots=roots,
                branches=sampled_branches,
            )
        )
    root_counts = {
        str(root): sampled_roots.count(root) for root in roots
    }
    modal_sampled_root = min(
        roots,
        key=lambda root: (-root_counts[str(root)], root),
    )
    endpoint = {
        "predictive_bayes_risk": per_root[predictive_root]
        | {"policy": "predictive_bayes_risk"},
        "size_matched_global_support": average_endpoint_mixture(
            [per_root[root] for root in sampled_roots],
            policy="size_matched_global_support",
        ),
    }
    novel_endpoint = {
        "predictive_bayes_risk": novel_per_root[predictive_root]
        | {"policy": "predictive_bayes_risk"},
        "size_matched_global_support": average_endpoint_mixture(
            [novel_per_root[root] for root in sampled_roots],
            policy="size_matched_global_support",
        ),
    }
    return {
        "tree_index": tree["tree_index"],
        "tree_seed": tree["tree_seed"],
        "target_seed": tree["target_seed"],
        "selection": {
            "predictive_bayes_risk_root": predictive_root,
            "sampled_root_counts": root_counts,
            "modal_sampled_root": modal_sampled_root,
            "candidate_differs_from_modal": (
                predictive_root != modal_sampled_root
            ),
            "candidate_selection_frequency": (
                root_counts[str(predictive_root)] / sample_count
            ),
        },
        "sample_count": sample_count,
        "global_pool_size": len(global_pool),
        "actual_branch_sizes": {
            f"{root}:{int(label)}": len(branches[(root, label)])
            for root in roots
            for label in (False, True)
        },
        "endpoint": endpoint,
        "comparisons": {
            "size_matched_global_support": policy_comparison(
                endpoint["predictive_bayes_risk"],
                endpoint["size_matched_global_support"],
            )
        },
        "novel_comparison_vs_size_matched": policy_comparison(
            novel_endpoint["predictive_bayes_risk"],
            novel_endpoint["size_matched_global_support"],
        ),
    }


def evaluate_sources(
    *,
    tree_paths: Sequence[Path],
    output_path: Path,
    sample_count: int = SAMPLE_COUNT,
) -> dict[str, Any]:
    sources = {}
    all_trees = []
    for tree_path in tree_paths:
        document = json.loads(tree_path.read_text())
        trees = [
            score_public_tree(tree, sample_count=sample_count)
            for tree in document["trees"]
        ]
        all_trees.extend(trees)
        aggregate = aggregate_tree_comparisons(
            trees,
            baseline="size_matched_global_support",
        )
        novel_brier = [
            tree["novel_comparison_vs_size_matched"][
                "candidate_minus_baseline_brier"
            ]
            for tree in trees
        ]
        novel_hamming = [
            tree["novel_comparison_vs_size_matched"][
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
            "candidate_differs_from_modal": sum(
                tree["selection"]["candidate_differs_from_modal"]
                for tree in trees
            ),
            "mean_candidate_selection_frequency": sum(
                tree["selection"]["candidate_selection_frequency"]
                for tree in trees
            )
            / len(trees),
            "aggregate": aggregate,
            "novel_target_mean_differences": {
                "candidate_minus_size_matched_brier": sum(novel_brier)
                / len(novel_brier),
                "candidate_minus_size_matched_hamming": sum(novel_hamming)
                / len(novel_hamming),
                "brier_tree_wins": sum(value < 0.0 for value in novel_brier),
                "hamming_tree_wins": sum(
                    value < 0.0 for value in novel_hamming
                ),
            },
            "trees": trees,
        }
    combined_novel_brier = [
        tree["novel_comparison_vs_size_matched"][
            "candidate_minus_baseline_brier"
        ]
        for tree in all_trees
    ]
    combined_novel_hamming = [
        tree["novel_comparison_vs_size_matched"][
            "candidate_minus_baseline_hamming"
        ]
        for tree in all_trees
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "description": (
                "Sample each static global-pool branch without replacement "
                "to exactly match its root-conditioned branch size."
            ),
            "sample_count_per_tree": sample_count,
            "sample_seed_namespace": SAMPLE_SEED_NAMESPACE,
            "llm_calls": 0,
        },
        "sources": sources,
        "combined": {
            "num_trees": len(all_trees),
            "candidate_differs_from_modal": sum(
                tree["selection"]["candidate_differs_from_modal"]
                for tree in all_trees
            ),
            "mean_candidate_selection_frequency": sum(
                tree["selection"]["candidate_selection_frequency"]
                for tree in all_trees
            )
            / len(all_trees),
            "aggregate": aggregate_tree_comparisons(
                all_trees,
                baseline="size_matched_global_support",
            ),
            "novel_target_mean_differences": {
                "candidate_minus_size_matched_brier": sum(
                    combined_novel_brier
                )
                / len(combined_novel_brier),
                "candidate_minus_size_matched_hamming": sum(
                    combined_novel_hamming
                )
                / len(combined_novel_hamming),
                "brier_tree_wins": sum(
                    value < 0.0 for value in combined_novel_brier
                ),
                "hamming_tree_wins": sum(
                    value < 0.0 for value in combined_novel_hamming
                ),
            },
        },
    }
    checkpoint(output_path, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trees", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-count", type=int, default=SAMPLE_COUNT)
    args = parser.parse_args()
    if args.sample_count <= 0:
        raise ValueError("sample count must be positive")
    result = evaluate_sources(
        tree_paths=[path.resolve() for path in args.trees],
        output_path=args.output.resolve(),
        sample_count=args.sample_count,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
