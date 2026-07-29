#!/usr/bin/env python3
"""Audit the first-link mechanism across four Number Game planner blocks."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossplanner_canonical_pooled128 import load_blocks
from scripts.number_game_qwen_first_link_mechanism64 import (
    _interval,
    _mean,
    _summary,
)
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-crossplanner-first-link-mechanism128-1"
BOOTSTRAP_SEED = 58_000
BOOTSTRAP_SAMPLES = 20_000
BASELINES = {
    "myopic_eig": "myopic_root",
    "fixed_support_depth_three": "fixed_support_depth_three_root",
    "crossfit_depth_two": "crossfit_depth_two_root",
}


def rows_for_baseline(
    blocks: Sequence[dict[str, Any]],
    *,
    root_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for block_index, block in enumerate(blocks):
        for tree in block["trees"]:
            selection = tree["selection"]
            candidate_root = str(selection["crossfit_depth_three_root"])
            baseline_root = str(selection[root_key])
            predicted = selection["crossfit_depth_three_brier"]
            realized = tree["per_root_endpoint_brier"]
            rows.append(
                {
                    "block_index": block_index,
                    "block_name": block["name"],
                    "family": block["family"],
                    "tree_seed": int(tree["tree_seed"]),
                    "roots_differ": candidate_root != baseline_root,
                    "predicted_advantage": (
                        predicted[baseline_root] - predicted[candidate_root]
                    ),
                    "realized_advantage": (
                        realized[baseline_root] - realized[candidate_root]
                    ),
                }
            )
    return rows


def stratified_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    block_indices: Sequence[int],
    seed: int,
) -> dict[str, list[float]]:
    groups = [
        [row for row in rows if row["block_index"] == block_index]
        for block_index in block_indices
    ]
    if any(not group for group in groups):
        raise ValueError("bootstrap received an empty source block")
    rng = random.Random(seed)
    divergent_mean_values = []
    divergent_rho_values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            rng.choice(group)
            for group in groups
            for _ in range(len(group))
        ]
        divergent = [row for row in sample if row["roots_differ"]]
        if not divergent:
            raise ValueError("bootstrap sample has no changed roots")
        divergent_mean_values.append(
            _mean([row["realized_advantage"] for row in divergent])
        )
        divergent_rho_values.append(
            spearman_correlation(
                [row["predicted_advantage"] for row in divergent],
                [row["realized_advantage"] for row in divergent],
            )
        )
    return {
        "divergent_mean_realized_advantage_95pct": _interval(
            divergent_mean_values
        ),
        "divergent_score_to_realized_spearman_95pct": _interval(
            divergent_rho_values
        ),
    }


def summarize_subset(
    rows: Sequence[dict[str, Any]],
    *,
    block_indices: Sequence[int],
    seed: int,
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["block_index"] in set(block_indices)
    ]
    divergent = [row for row in selected if row["roots_differ"]]
    return {
        "tree_count": len(selected),
        "root_differences": len(divergent),
        "all_trees": _summary(selected),
        "divergent_trees": _summary(divergent),
        "stratified_bootstrap": stratified_bootstrap(
            selected,
            block_indices=block_indices,
            seed=seed,
        ),
    }


def analyze_baseline(
    blocks: Sequence[dict[str, Any]],
    *,
    baseline: str,
    root_key: str,
    seed: int,
) -> dict[str, Any]:
    rows = rows_for_baseline(blocks, root_key=root_key)
    family_names = sorted({block["family"] for block in blocks})
    family_indices = {
        family: [
            index
            for index, block in enumerate(blocks)
            if block["family"] == family
        ]
        for family in family_names
    }
    return {
        "baseline": baseline,
        "pooled": summarize_subset(
            rows,
            block_indices=list(range(len(blocks))),
            seed=seed,
        ),
        "by_family": {
            family: summarize_subset(
                rows,
                block_indices=indices,
                seed=seed + 100 + family_offset,
            )
            for family_offset, (family, indices) in enumerate(
                family_indices.items()
            )
        },
        "by_block": [
            {
                "block": block["name"],
                "family": block["family"],
                **summarize_subset(
                    rows,
                    block_indices=[block_index],
                    seed=seed + 200 + block_index,
                ),
            }
            for block_index, block in enumerate(blocks)
        ],
    }


def run_analysis(output_dir: Path) -> dict[str, Any]:
    blocks = load_blocks()
    analyses = {
        baseline: analyze_baseline(
            blocks,
            baseline=baseline,
            root_key=root_key,
            seed=BOOTSTRAP_SEED + offset * 1_000,
        )
        for offset, (baseline, root_key) in enumerate(BASELINES.items())
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_crossplanner_first_link_summary",
        "protocol": {
            "analysis_is_retrospective": True,
            "cannot_rescue_source_statuses": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "planner_family_count": len(
                {block["family"] for block in blocks}
            ),
            "block_count": len(blocks),
            "tree_count": sum(len(block["trees"]) for block in blocks),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap": (
                "resample 32 trees independently within every source block"
            ),
            "advantage_sign": (
                "positive means crossfit depth three has lower Brier"
            ),
            "blocks": [
                {
                    "name": block["name"],
                    "family": block["family"],
                    "tree_count": len(block["trees"]),
                }
                for block in blocks
            ],
        },
        "analyses": analyses,
        "interpretation": {
            "first_link": (
                "simulated risk advantage versus a competing root is compared "
                "with exact-canonical realized Brier advantage"
            ),
            "second_link_not_tested": (
                "no online execution or noisy observation update is introduced"
            ),
            "formal_status_unchanged": True,
        },
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
    result = run_analysis(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
