#!/usr/bin/env python3
"""Replay frozen Number Game policies on the canonical 33-concept bank."""

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
from scripts.number_game_crossfit_depth_three_confirmation import (
    aggregate_scored_trees,
)
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree
from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    compile_expression,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-external-canonical-replay-1"
SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_planner_replay31"
    / "number-game-qwen-planner-replay31-20260728"
)
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_RESULT_SHA256 = (
    "a645c92126c9b5fd69e28532fb621028781dcf19d8ec890cd7758e6d1df710f2"
)
SOURCE_TREES_SHA256 = (
    "7abae080ec286b1991a941bf1de0ceb4ea0d3997a648a43ae4bd6ceeaa6c151a"
)
TREE_COUNT = 31
TARGET_COUNT = 33
SOURCE_DOI = "10.1017/S0140525X01000061"
BOOTSTRAP_NOTE = (
    "The imported aggregate uses the source evaluator's frozen 20,000-draw "
    "whole-tree bootstrap."
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _power_expression(base: int) -> str:
    if base == 2:
        return "is_power_of_two(n)"
    values = []
    value = 1
    while value <= 100:
        values.append(value)
        value *= base
    return " or ".join(f"n == {value}" for value in values)


def canonical_target_expressions() -> dict[str, str]:
    """Return the paper's 33 predicates, naturally extended to include zero."""
    expressions = {
        "even_numbers": "divisible(n, 2)",
        "odd_numbers": "n % 2 == 1",
        "prime_numbers": "is_prime(n)",
        "perfect_squares": "is_square(n)",
        "perfect_cubes": (
            "n == 0 or n == 1 or n == 8 or n == 27 or n == 64"
        ),
    }
    expressions.update(
        {
            f"multiples_of_{divisor}": f"divisible(n, {divisor})"
            for divisor in range(3, 11)
        }
    )
    expressions.update(
        {
            f"powers_of_{base}": _power_expression(base)
            for base in range(2, 11)
        }
    )
    expressions.update(
        {
            f"numbers_ending_in_{digit}": f"ends_with(n, {digit})"
            for digit in range(1, 10)
        }
    )
    expressions["both_digits_equal"] = (
        "n >= 11 and n <= 99 and n // 10 == n % 10"
    )
    expressions["numbers_less_than_100"] = "n < 100"
    if len(expressions) != TARGET_COUNT:
        raise AssertionError("canonical target specification is not size 33")
    return expressions


def canonical_targets() -> list[RuleHypothesis]:
    targets = [
        RuleHypothesis(
            name=name,
            expression=expression,
            extension=compile_expression(expression),
        )
        for name, expression in canonical_target_expressions().items()
    ]
    extensions = {target.extension for target in targets}
    if len(extensions) != TARGET_COUNT:
        raise ValueError("canonical target bank contains duplicate extensions")
    return targets


def replay_gates(
    *,
    aggregate: dict[str, Any],
    scored_trees: Sequence[dict[str, Any]],
    targets: Sequence[RuleHypothesis],
) -> dict[str, bool]:
    primary = aggregate["comparisons"]["crossfit_depth_two"]
    myopic = aggregate["comparisons"]["myopic_eig"]
    fixed = aggregate["comparisons"]["fixed_support_depth_three"]
    pts = aggregate["comparisons"]["positive_test_strategy"]
    ranking = aggregate["ranking"]
    return {
        "exactly_31_hash_bound_source_trees": len(scored_trees) == TREE_COUNT,
        "exactly_33_unique_canonical_targets": (
            len(targets) == TARGET_COUNT
            and len({target.extension for target in targets}) == TARGET_COUNT
        ),
        "one_external_endpoint_draw_per_tree": all(
            tree["mechanics"]["endpoint_draw_count"] == 1
            and tree["mechanics"]["minimum_endpoint_support_valid"]
            == TARGET_COUNT
            for tree in scored_trees
        ),
        "depth_three_brier_gain_at_least_one_percent": (
            primary["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_brier_ci_below_zero": (
            primary["tree_cluster_brier_difference_95pct_bootstrap"][1] < 0.0
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
        "depth_three_beats_myopic_by_five_percent_with_ci": (
            myopic["relative_brier_reduction"] >= 0.05
            and myopic["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_fixed_by_five_percent_with_ci": (
            fixed["relative_brier_reduction"] >= 0.05
            and fixed["tree_cluster_brier_difference_95pct_bootstrap"][1]
            < 0.0
        ),
        "depth_three_beats_pts_by_three_percent_with_ci": (
            pts["relative_brier_reduction"] >= 0.03
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


def run_replay(
    *,
    source_result_path: Path,
    source_trees_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source RESULT.json hash changed")
    if sha256_file(source_trees_path) != SOURCE_TREES_SHA256:
        raise ValueError("source TREES.json hash changed")
    source_result = json.loads(source_result_path.read_text())
    source_trees = json.loads(source_trees_path.read_text())
    if len(source_result["trees"]) != TREE_COUNT:
        raise ValueError("source result does not contain exactly 31 trees")
    if len(source_trees["trees"]) != TREE_COUNT:
        raise ValueError("source tree document does not contain exactly 31 trees")

    targets = canonical_targets()
    public_targets = [target.public_dict() for target in targets]
    output_dir.mkdir(parents=True, exist_ok=True)
    targets_document = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "source_doi": SOURCE_DOI,
        "paper_domain": [1, 100],
        "evaluation_domain": [0, 100],
        "domain_adaptation": (
            "Natural predicate extension to n=0; no target is omitted or "
            "reweighted."
        ),
        "target_weighting": "equal weight over all 33 concepts",
        "targets": public_targets,
    }
    checkpoint(output_dir / "TARGETS.json", targets_document)

    scored_trees = [
        score_fixed_tree(
            source_tree,
            source_metrics,
            [public_targets],
        )
        for source_tree, source_metrics in zip(
            source_trees["trees"],
            source_result["trees"],
            strict=True,
        )
    ]
    aggregate = aggregate_scored_trees(scored_trees)
    gates = replay_gates(
        aggregate=aggregate,
        scored_trees=scored_trees,
        targets=targets,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if all(gates.values()) else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "source_doi": SOURCE_DOI,
            "tree_count": TREE_COUNT,
            "target_count": TARGET_COUNT,
            "target_weighting": "equal concept weight",
            "tree_weighting": "equal tree weight",
            "model_calls": 0,
            "cost_usd": 0.0,
            "bootstrap_note": BOOTSTRAP_NOTE,
        },
        "aggregate": aggregate,
        "gates": gates,
        "trees": scored_trees,
        "targets_sha256": sha256_file(output_dir / "TARGETS.json"),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-result", type=Path, default=SOURCE_RESULT)
    parser.add_argument("--source-trees", type=Path, default=SOURCE_TREES)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_replay(
        source_result_path=args.source_result,
        source_trees_path=args.source_trees,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
