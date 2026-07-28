#!/usr/bin/env python3
"""Audit first-refresh retention and induced Number Game query paths."""

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
    best_query,
    query_eig,
)
from scripts.number_game_predictive_risk_replication import (
    cluster_bootstrap_interval,
)
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-first-retention-path-audit-1"
BOOTSTRAP_SAMPLES = 50_000
BOOTSTRAP_SEED = 28_300


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _actual_second_queries(
    tree: dict[str, Any],
) -> dict[tuple[int, bool], int]:
    queries: dict[tuple[int, bool], int] = {}
    for branch_key in tree["second_branches"]:
        root, first_label, second_query, _ = (
            int(value) for value in branch_key.split(":")
        )
        key = (root, bool(first_label))
        previous = queries.setdefault(key, second_query)
        if previous != second_query:
            raise ValueError(
                f"multiple second queries recorded for branch {key}"
            )
    return queries


def audit_tree(tree: dict[str, Any]) -> dict[str, Any]:
    initial = [_rule(item) for item in tree["initial"]]
    generated_key = (
        "generated_first_branches"
        if "generated_first_branches" in tree
        else "first_branches"
    )
    generated = _first_branches(tree, key=generated_key)
    targets = [_rule(item) for item in tree["targets"]]
    actual_queries = _actual_second_queries(tree)
    if set(generated) != set(actual_queries):
        raise ValueError("first and second branch keys do not agree")

    branches = []
    for root, label in sorted(generated):
        branch = generated[(root, label)]
        retained, diagnostics = retain_parent_hypotheses(
            parent_support=initial,
            generated_support=branch,
            query=root,
            label=label,
        )
        reproduced_query, generated_best_eig = best_query(
            branch,
            excluded=(root,),
        )
        recorded_query = actual_queries[(root, label)]
        retained_query, retained_best_eig = best_query(
            retained,
            excluded=(root,),
        )
        generated_extensions = {
            hypothesis.extension for hypothesis in branch
        }
        retained_extensions = {
            hypothesis.extension for hypothesis in retained
        }
        branch_targets = [
            target
            for target in targets
            if target.extension[root] == label
        ]
        generated_covered = sum(
            target.extension in generated_extensions
            for target in branch_targets
        )
        retained_covered = sum(
            target.extension in retained_extensions
            for target in branch_targets
        )
        branches.append(
            {
                "root": root,
                "first_label": label,
                "recorded_second_query": recorded_query,
                "reproduced_second_query": reproduced_query,
                "retained_second_query": retained_query,
                "recorded_query_reproduced": (
                    recorded_query == reproduced_query
                ),
                "second_query_changed": (
                    retained_query != recorded_query
                ),
                "old_query_eig_on_generated_nats": (
                    generated_best_eig
                ),
                "old_query_eig_on_retained_nats": query_eig(
                    retained,
                    recorded_query,
                ),
                "new_query_eig_on_retained_nats": retained_best_eig,
                "retained_query_eig_gain_nats": (
                    retained_best_eig
                    - query_eig(retained, recorded_query)
                ),
                "generated_support_size": len(branch),
                "retained_support_size": len(retained),
                "retained_initial_consistent_count": diagnostics[
                    "retained_parent_consistent_count"
                ],
                "retained_initial_novel_count": diagnostics[
                    "retained_parent_novel_count"
                ],
                "target_paths": len(branch_targets),
                "generated_target_paths_covered": generated_covered,
                "retained_target_paths_covered": retained_covered,
                "target_paths_recovered": (
                    retained_covered - generated_covered
                ),
                "target_paths_lost": sum(
                    target.extension in generated_extensions
                    and target.extension not in retained_extensions
                    for target in branch_targets
                ),
            }
        )

    target_paths = sum(branch["target_paths"] for branch in branches)
    generated_covered = sum(
        branch["generated_target_paths_covered"]
        for branch in branches
    )
    retained_covered = sum(
        branch["retained_target_paths_covered"]
        for branch in branches
    )
    changed = sum(
        branch["second_query_changed"] for branch in branches
    )
    return {
        "tree_index": tree["tree_index"],
        "tree_seed": tree["tree_seed"],
        "target_seed": tree["target_seed"],
        "branches": branches,
        "summary": {
            "branch_count": len(branches),
            "recorded_second_queries_reproduced": sum(
                branch["recorded_query_reproduced"]
                for branch in branches
            ),
            "second_queries_changed": changed,
            "second_query_change_rate": changed / len(branches),
            "exact_old_response_reuse_rate": (
                (len(branches) - changed) / len(branches)
            ),
            "mean_generated_support_size": _mean(
                [
                    branch["generated_support_size"]
                    for branch in branches
                ]
            ),
            "mean_retained_support_size": _mean(
                [
                    branch["retained_support_size"]
                    for branch in branches
                ]
            ),
            "minimum_generated_support_size": min(
                branch["generated_support_size"]
                for branch in branches
            ),
            "minimum_retained_support_size": min(
                branch["retained_support_size"]
                for branch in branches
            ),
            "mean_retained_query_eig_gain_nats": _mean(
                [
                    branch["retained_query_eig_gain_nats"]
                    for branch in branches
                ]
            ),
            "target_paths": target_paths,
            "generated_target_paths_covered": generated_covered,
            "retained_target_paths_covered": retained_covered,
            "generated_target_coverage_rate": (
                generated_covered / target_paths
            ),
            "retained_target_coverage_rate": (
                retained_covered / target_paths
            ),
            "target_coverage_difference": (
                (retained_covered - generated_covered) / target_paths
            ),
            "target_paths_recovered": (
                retained_covered - generated_covered
            ),
            "target_paths_lost": sum(
                branch["target_paths_lost"] for branch in branches
            ),
        },
    }


def aggregate_trees(
    trees: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    summaries = [tree["summary"] for tree in trees]
    branch_count = sum(item["branch_count"] for item in summaries)
    changed = sum(
        item["second_queries_changed"] for item in summaries
    )
    target_paths = sum(item["target_paths"] for item in summaries)
    recovered = sum(
        item["target_paths_recovered"] for item in summaries
    )
    generated_covered = sum(
        item["generated_target_paths_covered"]
        for item in summaries
    )
    retained_covered = sum(
        item["retained_target_paths_covered"]
        for item in summaries
    )
    branches = [
        branch for tree in trees for branch in tree["branches"]
    ]
    change_rates = [
        item["second_query_change_rate"] for item in summaries
    ]
    coverage_differences = [
        item["target_coverage_difference"] for item in summaries
    ]
    return {
        "tree_count": len(trees),
        "branch_count": branch_count,
        "all_recorded_second_queries_reproduced": all(
            item["recorded_second_queries_reproduced"]
            == item["branch_count"]
            for item in summaries
        ),
        "trees_with_changed_second_query": sum(
            item["second_queries_changed"] > 0
            for item in summaries
        ),
        "second_queries_changed": changed,
        "second_query_change_rate": changed / branch_count,
        "tree_mean_second_query_change_rate": _mean(change_rates),
        "tree_bootstrap_second_query_change_rate_95pct": (
            cluster_bootstrap_interval(
                change_rates,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED,
            )
        ),
        "exact_old_response_reuse_rate": (
            (branch_count - changed) / branch_count
        ),
        "mean_generated_support_size": _mean(
            [branch["generated_support_size"] for branch in branches]
        ),
        "mean_retained_support_size": _mean(
            [branch["retained_support_size"] for branch in branches]
        ),
        "minimum_generated_support_size": min(
            branch["generated_support_size"] for branch in branches
        ),
        "minimum_retained_support_size": min(
            branch["retained_support_size"] for branch in branches
        ),
        "mean_retained_query_eig_gain_nats": _mean(
            [
                branch["retained_query_eig_gain_nats"]
                for branch in branches
            ]
        ),
        "target_paths": target_paths,
        "generated_target_coverage_rate": (
            generated_covered / target_paths
        ),
        "retained_target_coverage_rate": (
            retained_covered / target_paths
        ),
        "target_coverage_difference": recovered / target_paths,
        "tree_mean_target_coverage_difference": _mean(
            coverage_differences
        ),
        "tree_bootstrap_target_coverage_difference_95pct": (
            cluster_bootstrap_interval(
                coverage_differences,
                samples=BOOTSTRAP_SAMPLES,
                seed=BOOTSTRAP_SEED + 1,
            )
        ),
        "target_paths_recovered": recovered,
        "target_paths_lost": sum(
            item["target_paths_lost"] for item in summaries
        ),
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
    aggregate = aggregate_trees(combined_trees)
    gates = {
        "all_recorded_second_queries_reproduced": aggregate[
            "all_recorded_second_queries_reproduced"
        ],
        "every_tree_has_path_dependence": (
            aggregate["trees_with_changed_second_query"]
            == aggregate["tree_count"]
        ),
        "retention_never_loses_target_coverage": (
            aggregate["target_paths_lost"] == 0
        ),
        "every_source_recovers_target_coverage": all(
            source["aggregate"]["target_coverage_difference"] > 0.0
            for source in sources
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model_calls": 0,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "support_operation": (
                "deduplicated union of generated first support and "
                "consistent initial particles"
            ),
        },
        "gates": gates,
        "aggregate": aggregate,
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
