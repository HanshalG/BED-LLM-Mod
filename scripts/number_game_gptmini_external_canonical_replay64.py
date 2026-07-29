#!/usr/bin/env python3
"""Replay 64 fixed GPT-5.4 Mini Number Game policies on canonical targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import (
    aggregate_scored_trees,
)
from scripts.number_game_crossfit_depth_three_pooled_audit import (
    audit_pooled,
)
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree
from scripts.number_game_external_canonical_replay import canonical_targets


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-gptmini-external-canonical-replay64-1"
TARGET_COUNT = 33
TREE_COUNT_PER_STUDY = 32
TOTAL_TREE_COUNT = 64
SOURCE_DOI = "10.1017/S0140525X01000061"
STUDIES = (
    {
        "name": "confirmation",
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_confirmation"
            / "number-game-crossfit-depth-three-confirmation-20260728"
            / "RESULT.json"
        ),
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_confirmation"
            / "number-game-crossfit-depth-three-confirmation-20260728"
            / "TREES.json"
        ),
        "result_sha256": (
            "1081da1e8381b88f7cd3fcd905b5ed539047863bcc087d686e509ed8f82d794a"
        ),
        "trees_sha256": (
            "cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9"
        ),
    },
    {
        "name": "fresh_replication",
        "result": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "RESULT.json"
        ),
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "TREES.json"
        ),
        "result_sha256": (
            "25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8"
        ),
        "trees_sha256": (
            "197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_source(
    *,
    result_path: Path,
    trees_path: Path,
    expected_result_sha256: str,
    expected_trees_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(result_path) != expected_result_sha256:
        raise ValueError("source result hash changed")
    if sha256_file(trees_path) != expected_trees_sha256:
        raise ValueError("source trees hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trees = json.loads(trees_path.read_text(encoding="utf-8"))
    if len(result["trees"]) != TREE_COUNT_PER_STUDY:
        raise ValueError("source result must contain exactly 32 trees")
    if len(trees["trees"]) != TREE_COUNT_PER_STUDY:
        raise ValueError("source tree document must contain exactly 32 trees")
    result_seeds = [int(row["tree_seed"]) for row in result["trees"]]
    tree_seeds = [int(row["tree_seed"]) for row in trees["trees"]]
    if result_seeds != tree_seeds or len(set(tree_seeds)) != len(tree_seeds):
        raise ValueError("source tree seeds do not align uniquely")
    return result, trees


def score_study(
    *,
    source_result: dict[str, Any],
    source_trees: dict[str, Any],
    public_targets: list[dict[str, Any]],
) -> dict[str, Any]:
    scored_trees = [
        score_fixed_tree(tree, metrics, [public_targets])
        for tree, metrics in zip(
            source_trees["trees"],
            source_result["trees"],
            strict=True,
        )
    ]
    return {
        "aggregate": aggregate_scored_trees(scored_trees),
        "trees": scored_trees,
    }


def primary_gates(
    *,
    pooled: dict[str, Any],
    study_results: Sequence[dict[str, Any]],
) -> dict[str, bool]:
    depth_two = pooled["comparisons"]["crossfit_depth_two"]
    myopic = pooled["comparisons"]["myopic_eig"]
    study_depth_differences = [
        result["aggregate"]["comparisons"]["crossfit_depth_two"][
            "mean_candidate_minus_baseline_brier"
        ]
        for result in study_results
    ]
    root_differences = sum(
        result["aggregate"]["root_differences"]["crossfit_depth_two"]
        for result in study_results
    )
    return {
        "exactly_64_hash_bound_trees": (
            sum(len(result["trees"]) for result in study_results)
            == TOTAL_TREE_COUNT
        ),
        "crossfit_depth_roots_differ_on_at_least_24_trees": (
            root_differences >= 24
        ),
        "both_independent_studies_directionally_favor_depth_three": all(
            difference <= 0.0 for difference in study_depth_differences
        ),
        "depth_three_beats_depth_two_by_one_percent": (
            depth_two["relative_brier_reduction"] >= 0.01
        ),
        "depth_three_vs_depth_two_ci_below_zero": (
            depth_two[
                "stratified_tree_bootstrap_brier_difference_95pct"
            ][1]
            < 0.0
        ),
        "depth_three_wins_at_least_24_trees_vs_depth_two": (
            depth_two["wins"] >= 24
        ),
        "depth_three_beats_myopic_by_five_percent": (
            myopic["relative_brier_reduction"] >= 0.05
        ),
        "depth_three_vs_myopic_ci_below_zero": (
            myopic[
                "stratified_tree_bootstrap_brier_difference_95pct"
            ][1]
            < 0.0
        ),
        "depth_three_wins_at_least_32_trees_vs_myopic": (
            myopic["wins"] >= 32
        ),
    }


def diagnostic_gates(pooled: dict[str, Any]) -> dict[str, bool]:
    comparisons = pooled["comparisons"]
    depth_two = comparisons["crossfit_depth_two"]
    fixed = comparisons["fixed_support_depth_three"]
    pts = comparisons["positive_test_strategy"]
    random_root = comparisons["uniform_random_candidate_root"]
    ranking = pooled["ranking"]
    return {
        "no_hamming_regression_vs_depth_two": (
            depth_two["mean_hamming_difference"] <= 0.0
        ),
        "no_coverage_regression_vs_depth_two": (
            depth_two["mean_coverage_difference"] >= 0.0
        ),
        "depth_three_directionally_beats_fixed_support": (
            fixed["mean_brier_difference"] <= 0.0
        ),
        "depth_three_directionally_beats_pts": (
            pts["mean_brier_difference"] <= 0.0
        ),
        "depth_three_beats_random_with_ci": (
            random_root[
                "stratified_tree_bootstrap_brier_difference_95pct"
            ][1]
            < 0.0
        ),
        "depth_three_rank_rho_exceeds_depth_two": (
            ranking["crossfit_depth_three_spearman_brier"]["mean"]
            > ranking["crossfit_depth_two_spearman_brier"]["mean"]
        ),
    }


def run_replay(output_dir: Path) -> dict[str, Any]:
    loaded = [
        validate_source(
            result_path=study["result"],
            trees_path=study["trees"],
            expected_result_sha256=study["result_sha256"],
            expected_trees_sha256=study["trees_sha256"],
        )
        for study in STUDIES
    ]
    seed_sets = [
        {int(row["tree_seed"]) for row in trees["trees"]}
        for _, trees in loaded
    ]
    if seed_sets[0] & seed_sets[1]:
        raise ValueError("source studies share planning-tree seeds")

    targets = canonical_targets()
    if len(targets) != TARGET_COUNT:
        raise ValueError("canonical target bank has wrong cardinality")
    if len({target.extension for target in targets}) != TARGET_COUNT:
        raise ValueError("canonical target bank has duplicate extensions")
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

    scored_studies = [
        score_study(
            source_result=result,
            source_trees=trees,
            public_targets=public_targets,
        )
        for result, trees in loaded
    ]
    pooled = audit_pooled(scored_studies)
    primary = primary_gates(
        pooled=pooled,
        study_results=scored_studies,
    )
    diagnostics = diagnostic_gates(pooled)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if all(primary.values()) else "gated_null",
        "protocol": {
            "source_doi": SOURCE_DOI,
            "model_calls": 0,
            "cost_usd": 0.0,
            "tree_count": TOTAL_TREE_COUNT,
            "study_count": len(STUDIES),
            "target_count": TARGET_COUNT,
            "target_weighting": "equal concept weight",
            "tree_weighting": "equal within and across two 32-tree studies",
            "endpoint_draws_per_tree": 1,
            "endpoint_is_external_and_deterministic": True,
            "policies_were_fixed_before_canonical_bank_scoring": True,
            "source_hashes": {
                study["name"]: {
                    "result_sha256": study["result_sha256"],
                    "trees_sha256": study["trees_sha256"],
                }
                for study in STUDIES
            },
        },
        "studies": [
            {
                "name": study["name"],
                "aggregate": scored["aggregate"],
                "trees": scored["trees"],
            }
            for study, scored in zip(STUDIES, scored_studies, strict=True)
        ],
        "pooled": pooled,
        "primary_gates": primary,
        "diagnostic_gates": diagnostics,
        "all_primary_gates_pass": all(primary.values()),
        "targets_sha256": sha256_file(output_dir / "TARGETS.json"),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_replay(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
