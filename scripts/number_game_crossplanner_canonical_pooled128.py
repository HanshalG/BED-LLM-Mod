#!/usr/bin/env python3
"""Pool four exact-canonical Number Game cohorts across two planners."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_pooled_audit import (
    BASELINES,
    exact_one_sided_sign_pvalue,
    stratified_bootstrap_interval,
)
from scripts.number_game_qwen_external_canonical_pooled64 import (
    STUDIES as QWEN_STUDIES,
    validate_source as validate_qwen_source,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-crossplanner-canonical-pooled128-1"
TREE_COUNT_PER_BLOCK = 32
BLOCK_COUNT = 4
TOTAL_TREE_COUNT = 128
TARGET_COUNT = 33
BOOTSTRAP_SEED = 57_000
GPTMINI_REPLAY = (
    REPO_ROOT
    / "results/nonmyopic/number_game_gptmini_external_canonical_replay64"
    / "number-game-gptmini-external-canonical-replay64-20260729T061141Z"
    / "RESULT.json"
)
GPTMINI_REPLAY_SHA256 = (
    "33263d27fa0f3fb19b44908070df1a21fb014ffbf1bc1cd715d362d3ac531427"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_gptmini_blocks() -> list[dict[str, Any]]:
    if sha256_file(GPTMINI_REPLAY) != GPTMINI_REPLAY_SHA256:
        raise ValueError("GPT-Mini replay hash changed")
    result = json.loads(GPTMINI_REPLAY.read_text(encoding="utf-8"))
    if result["protocol"]["model_calls"] != 0:
        raise ValueError("GPT-Mini replay unexpectedly used model calls")
    if result["protocol"]["tree_count"] != 64:
        raise ValueError("GPT-Mini replay tree count changed")
    if result["protocol"]["target_count"] != TARGET_COUNT:
        raise ValueError("GPT-Mini replay target count changed")
    blocks = result["studies"]
    if len(blocks) != 2:
        raise ValueError("GPT-Mini replay must contain two source blocks")
    for block in blocks:
        if len(block["trees"]) != TREE_COUNT_PER_BLOCK:
            raise ValueError("GPT-Mini block must contain 32 trees")
    return blocks


def load_blocks() -> list[dict[str, Any]]:
    qwen_sources = [validate_qwen_source(study) for study in QWEN_STUDIES]
    blocks = [
        {
            "name": f"qwen_{study['name']}",
            "family": "qwen/qwen3.7-plus",
            "aggregate": source["aggregate"],
            "trees": source["trees"],
        }
        for study, source in zip(QWEN_STUDIES, qwen_sources, strict=True)
    ]
    blocks.extend(
        {
            "name": f"gptmini_{block['name']}",
            "family": "openai/gpt-5.4-mini",
            "aggregate": block["aggregate"],
            "trees": block["trees"],
        }
        for block in validate_gptmini_blocks()
    )
    seed_sets = [
        {int(tree["tree_seed"]) for tree in block["trees"]}
        for block in blocks
    ]
    if any(
        seed_sets[left] & seed_sets[right]
        for left in range(len(seed_sets))
        for right in range(left + 1, len(seed_sets))
    ):
        raise ValueError("source blocks share planning-tree seeds")
    return blocks


def summarize_block(block: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": block["name"],
        "family": block["family"],
        "tree_count": len(block["trees"]),
        "myopic_comparison": block["aggregate"]["comparisons"]["myopic_eig"],
        "depth_two_comparison": block["aggregate"]["comparisons"][
            "crossfit_depth_two"
        ],
    }


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def audit_blocks(blocks: list[dict[str, Any]]) -> dict[str, Any]:
    if not blocks or any(
        len(block["trees"]) != TREE_COUNT_PER_BLOCK for block in blocks
    ):
        raise ValueError("every pooled block must contain 32 trees")
    comparisons = {}
    for baseline_index, baseline in enumerate(BASELINES):
        brier_groups = [
            [
                tree["comparisons"][baseline][
                    "candidate_minus_baseline_brier"
                ]
                for tree in block["trees"]
            ]
            for block in blocks
        ]
        hamming_groups = [
            [
                tree["comparisons"][baseline][
                    "candidate_minus_baseline_hamming"
                ]
                for tree in block["trees"]
            ]
            for block in blocks
        ]
        coverage_groups = [
            [
                tree["comparisons"][baseline]["coverage_difference"]
                for tree in block["trees"]
            ]
            for block in blocks
        ]
        candidate_brier = _mean(
            [
                tree["endpoint"]["crossfit_depth_three"][
                    "mean_posterior_predictive_brier"
                ]
                for block in blocks
                for tree in block["trees"]
            ]
        )
        baseline_brier = _mean(
            [
                tree["endpoint"][baseline][
                    "mean_posterior_predictive_brier"
                ]
                for block in blocks
                for tree in block["trees"]
            ]
        )
        differences = [value for group in brier_groups for value in group]
        wins = sum(value < -1e-15 for value in differences)
        losses = sum(value > 1e-15 for value in differences)
        comparisons[baseline] = {
            "candidate_mean_brier": candidate_brier,
            "baseline_mean_brier": baseline_brier,
            "relative_brier_reduction": (
                baseline_brier - candidate_brier
            )
            / baseline_brier,
            "mean_brier_difference": _mean(differences),
            "stratified_tree_bootstrap_brier_difference_95pct": (
                stratified_bootstrap_interval(
                    brier_groups,
                    seed=BOOTSTRAP_SEED + baseline_index * 3,
                )
            ),
            "wins": wins,
            "ties": len(differences) - wins - losses,
            "losses": losses,
            "one_sided_exact_sign_pvalue_excluding_ties": (
                exact_one_sided_sign_pvalue(wins, losses)
            ),
            "mean_hamming_difference": _mean(
                [value for group in hamming_groups for value in group]
            ),
            "stratified_tree_bootstrap_hamming_difference_95pct": (
                stratified_bootstrap_interval(
                    hamming_groups,
                    seed=BOOTSTRAP_SEED + baseline_index * 3 + 1,
                )
            ),
            "mean_coverage_difference": _mean(
                [value for group in coverage_groups for value in group]
            ),
            "stratified_tree_bootstrap_coverage_difference_95pct": (
                stratified_bootstrap_interval(
                    coverage_groups,
                    seed=BOOTSTRAP_SEED + baseline_index * 3 + 2,
                )
            ),
        }

    ranking = {}
    ranking_keys = (
        "crossfit_depth_three_spearman_brier",
        "crossfit_depth_two_spearman_brier",
        "crossfit_depth_three_pairwise_concordance",
        "crossfit_depth_two_pairwise_concordance",
    )
    for offset, key in enumerate(ranking_keys):
        groups = [
            [tree["ranking"][key] for tree in block["trees"]]
            for block in blocks
        ]
        ranking[key] = {
            "mean": _mean([value for group in groups for value in group]),
            "stratified_tree_bootstrap_95pct": (
                stratified_bootstrap_interval(
                    groups,
                    seed=BOOTSTRAP_SEED + 100 + offset,
                )
            ),
        }

    novel = {}
    novel_keys = (
        "candidate_minus_baseline_brier",
        "candidate_minus_baseline_hamming",
        "coverage_difference",
    )
    for offset, key in enumerate(novel_keys):
        groups = [
            [
                tree[
                    "novel_comparison_crossfit_depth_three_vs_depth_two"
                ][key]
                for tree in block["trees"]
            ]
            for block in blocks
        ]
        novel[key] = {
            "mean": _mean([value for group in groups for value in group]),
            "stratified_tree_bootstrap_95pct": (
                stratified_bootstrap_interval(
                    groups,
                    seed=BOOTSTRAP_SEED + 200 + offset,
                )
            ),
        }
    return {
        "protocol": {
            "model_calls": 0,
            "block_count": len(blocks),
            "tree_count": sum(len(block["trees"]) for block in blocks),
            "bootstrap_samples": 20_000,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "comparisons": comparisons,
        "ranking": ranking,
        "novel_targets": novel,
    }


def robustness_checks(
    *,
    blocks: list[dict[str, Any]],
    families: dict[str, dict[str, Any]],
    pooled: dict[str, Any],
) -> dict[str, bool]:
    block_myopic = [
        block["aggregate"]["comparisons"]["myopic_eig"] for block in blocks
    ]
    pooled_myopic = pooled["comparisons"]["myopic_eig"]
    return {
        "exactly_four_disjoint_32_tree_blocks": (
            len(blocks) == BLOCK_COUNT
            and all(
                len(block["trees"]) == TREE_COUNT_PER_BLOCK
                for block in blocks
            )
        ),
        "all_blocks_myopic_gain_at_least_eight_percent": all(
            comparison["relative_brier_reduction"] >= 0.08
            for comparison in block_myopic
        ),
        "all_blocks_myopic_ci_below_zero": all(
            comparison[
                "tree_cluster_brier_difference_95pct_bootstrap"
            ][1]
            < 0.0
            for comparison in block_myopic
        ),
        "all_blocks_myopic_wins_at_least_twenty": all(
            comparison["brier_tree_wins"] >= 20
            for comparison in block_myopic
        ),
        "both_families_myopic_gain_at_least_ten_percent": all(
            family["comparisons"]["myopic_eig"][
                "relative_brier_reduction"
            ]
            >= 0.10
            for family in families.values()
        ),
        "both_families_myopic_ci_below_zero": all(
            family["comparisons"]["myopic_eig"][
                "stratified_tree_bootstrap_brier_difference_95pct"
            ][1]
            < 0.0
            for family in families.values()
        ),
        "pooled_myopic_gain_at_least_ten_percent": (
            pooled_myopic["relative_brier_reduction"] >= 0.10
        ),
        "pooled_myopic_ci_below_zero": (
            pooled_myopic[
                "stratified_tree_bootstrap_brier_difference_95pct"
            ][1]
            < 0.0
        ),
        "pooled_myopic_wins_at_least_eighty": (
            pooled_myopic["wins"] >= 80
        ),
    }


def run_synthesis(output_dir: Path) -> dict[str, Any]:
    blocks = load_blocks()
    pooled = audit_blocks(blocks)
    family_names = sorted({block["family"] for block in blocks})
    families = {
        family: audit_blocks(
            [block for block in blocks if block["family"] == family]
        )
        for family in family_names
    }
    checks = robustness_checks(
        blocks=blocks,
        families=families,
        pooled=pooled,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "retrospective_crossplanner_robustness_positive"
            if all(checks.values())
            else "retrospective_crossplanner_robustness_mixed"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "cannot_rescue_source_statuses": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "planner_family_count": len(families),
            "block_count": len(blocks),
            "tree_count": TOTAL_TREE_COUNT,
            "target_count": TARGET_COUNT,
            "bootstrap": (
                "resample 32 trees independently within each of four blocks"
            ),
            "source_hashes": {
                **{
                    f"qwen_{study['name']}": study["sha256"]
                    for study in QWEN_STUDIES
                },
                "gptmini_replay64": GPTMINI_REPLAY_SHA256,
            },
        },
        "blocks": [summarize_block(block) for block in blocks],
        "families": families,
        "pooled": pooled,
        "robustness_checks": checks,
        "depth_interpretation": (
            "Any pooled depth-three versus depth-two result is descriptive; "
            "block-level depth effects remain heterogeneous."
        ),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {args.output_dir}")
    result = run_synthesis(args.output_dir)
    print(
        json.dumps(
            {
                "status": result["status"],
                "robustness_checks": result["robustness_checks"],
                "families": {
                    family: aggregate["comparisons"]["myopic_eig"]
                    for family, aggregate in result["families"].items()
                },
                "pooled_myopic": result["pooled"]["comparisons"]["myopic_eig"],
                "pooled_depth_two": result["pooled"]["comparisons"][
                    "crossfit_depth_two"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
