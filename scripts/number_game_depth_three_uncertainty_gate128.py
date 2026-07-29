#!/usr/bin/env python3
"""Audit a target-blind uncertainty gate for Number Game depth three."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import (
    _target_mapping,
)
from scripts.number_game_crossfit_depth_three_pooled_audit import (
    stratified_bootstrap_interval,
)
from scripts.number_game_crossplanner_canonical_pooled128 import load_blocks
from scripts.number_game_ranking_fidelity_audit import spearman_correlation
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
    retained_second_branches,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-depth-three-uncertainty-gate128-1"
BOOTSTRAP_SEED = 59_000
BOOTSTRAP_SAMPLES = 20_000
MIN_POSITIVE_DRAWS = 6
LOWER_STANDARD_ERROR_MULTIPLIER = 1.0
SOURCE_SPECS = (
    {
        "name": "qwen_confirmation_v1",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_qwen_external_canonical_confirmation"
            / "number-game-qwen-external-canonical-confirmation-20260729"
            / "TREES.json"
        ),
        "sha256": (
            "39b79f391ae3b613d15794c9dd6c86ef02eb96907fbaa33591b157b2ac19cc63"
        ),
    },
    {
        "name": "qwen_replication_v2",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic"
            / "number_game_qwen_external_canonical_replication_v2"
            / "number-game-qwen-external-canonical-replication-v2-20260729T063429Z"
            / "TREES.json"
        ),
        "sha256": (
            "f1ccd0b9a5f0e40786e8673a30400492626ae7a3f26de4dea42ac4316dc3d38c"
        ),
    },
    {
        "name": "gptmini_confirmation",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_confirmation"
            / "number-game-crossfit-depth-three-confirmation-20260728"
            / "TREES.json"
        ),
        "sha256": (
            "cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9"
        ),
    },
    {
        "name": "gptmini_fresh_replication",
        "trees": (
            REPO_ROOT
            / "results/nonmyopic/number_game_crossfit_depth_three_fresh_replication"
            / "number-game-crossfit-depth-three-fresh-replication-20260728"
            / "TREES.json"
        ),
        "sha256": (
            "197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_source_blocks() -> list[dict[str, Any]]:
    scored_blocks = load_blocks()
    if [block["name"] for block in scored_blocks] != [
        spec["name"] for spec in SOURCE_SPECS
    ]:
        raise ValueError("source block order changed")
    blocks = []
    for block_index, (scored, spec) in enumerate(
        zip(scored_blocks, SOURCE_SPECS, strict=True)
    ):
        path = Path(spec["trees"])
        if sha256_file(path) != spec["sha256"]:
            raise ValueError(f"{spec['name']} source tree hash changed")
        raw = json.loads(path.read_text(encoding="utf-8"))
        raw_trees = raw["trees"]
        scored_trees = scored["trees"]
        raw_seeds = [int(tree["tree_seed"]) for tree in raw_trees]
        scored_seeds = [int(tree["tree_seed"]) for tree in scored_trees]
        if len(raw_trees) != 32 or raw_seeds != scored_seeds:
            raise ValueError(f"{spec['name']} trees do not align")
        if any(len(tree["validation_supports"]) != 8 for tree in raw_trees):
            raise ValueError(f"{spec['name']} validation draw count changed")
        blocks.append(
            {
                "block_index": block_index,
                "name": scored["name"],
                "family": scored["family"],
                "raw_trees": raw_trees,
                "scored_trees": scored_trees,
                "trees_sha256": spec["sha256"],
            }
        )
    return blocks


def uncertainty_gate(advantages: Sequence[float]) -> dict[str, Any]:
    if len(advantages) != 8:
        raise ValueError("uncertainty gate requires exactly eight draws")
    mean = statistics.fmean(advantages)
    standard_error = statistics.stdev(advantages) / math.sqrt(len(advantages))
    positive_draws = sum(value > 0.0 for value in advantages)
    lower_bound = mean - LOWER_STANDARD_ERROR_MULTIPLIER * standard_error
    accepted = positive_draws >= MIN_POSITIVE_DRAWS and lower_bound > 0.0
    return {
        "accepted": accepted,
        "mean_advantage": mean,
        "standard_error": standard_error,
        "one_se_lower_bound": lower_bound,
        "positive_draws": positive_draws,
    }


def depth_three_draw_risks(raw_tree: dict[str, Any]) -> list[dict[int, float]]:
    roots = [int(root) for root in raw_tree["roots"]]
    first = _first_branches(raw_tree)
    generated_second = _second_branches(
        raw_tree,
        key="generated_second_branches",
    )
    second, _, _ = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated_second,
    )
    rows = []
    for draw_index, support in enumerate(raw_tree["validation_supports"]):
        targets = _target_mapping(
            [_rule(item) for item in support],
            draw_index=draw_index,
        )
        rows.append(
            {
                root: depth.evaluate_policy_root_depth_three(
                    policy=f"uncertainty_gate_root_{root}",
                    root=root,
                    targets=targets,
                    first_branches=first,
                    second_branches=second,
                )["mean_posterior_predictive_brier"]
                for root in roots
            }
        )
    return rows


def score_tree(
    *,
    raw_tree: dict[str, Any],
    scored_tree: dict[str, Any],
    block_index: int,
    block_name: str,
    family: str,
) -> dict[str, Any]:
    selection = scored_tree["selection"]
    depth_three_root = int(selection["crossfit_depth_three_root"])
    depth_two_root = int(selection["crossfit_depth_two_root"])
    if depth_three_root == depth_two_root:
        gate = {
            "accepted": True,
            "mean_advantage": 0.0,
            "standard_error": 0.0,
            "one_se_lower_bound": 0.0,
            "positive_draws": 8,
            "shared_root": True,
        }
        selected_root = depth_three_root
    else:
        draw_risks = depth_three_draw_risks(raw_tree)
        advantages = [
            row[depth_two_root] - row[depth_three_root]
            for row in draw_risks
        ]
        gate = uncertainty_gate(advantages) | {"shared_root": False}
        selected_root = (
            depth_three_root if gate["accepted"] else depth_two_root
        )

    endpoint = {
        int(root): float(value)
        for root, value in scored_tree["per_root_endpoint_brier"].items()
    }
    myopic_root = int(selection["myopic_root"])
    return {
        "block_index": block_index,
        "block_name": block_name,
        "family": family,
        "tree_seed": int(scored_tree["tree_seed"]),
        "depth_three_root": depth_three_root,
        "depth_two_root": depth_two_root,
        "myopic_root": myopic_root,
        "selected_root": selected_root,
        "gate": gate,
        "endpoint_brier": {
            "gated": endpoint[selected_root],
            "depth_three": endpoint[depth_three_root],
            "depth_two": endpoint[depth_two_root],
            "myopic": endpoint[myopic_root],
        },
    }


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def summarize(
    rows: Sequence[dict[str, Any]],
    *,
    block_indices: Sequence[int],
    baseline: str,
    seed: int,
) -> dict[str, Any]:
    selected = [
        row for row in rows if row["block_index"] in set(block_indices)
    ]
    differences = [
        row["endpoint_brier"]["gated"] - row["endpoint_brier"][baseline]
        for row in selected
    ]
    groups = [
        [
            row["endpoint_brier"]["gated"]
            - row["endpoint_brier"][baseline]
            for row in selected
            if row["block_index"] == block_index
        ]
        for block_index in block_indices
    ]
    wins = sum(value < -1e-15 for value in differences)
    losses = sum(value > 1e-15 for value in differences)
    baseline_mean = _mean(
        [row["endpoint_brier"][baseline] for row in selected]
    )
    switched = [
        row
        for row in selected
        if row["selected_root"]
        != (row["depth_two_root"] if baseline == "depth_two" else row["myopic_root"])
    ]
    accepted_changes = [
        row
        for row in selected
        if row["depth_three_root"] != row["depth_two_root"]
        and row["gate"]["accepted"]
    ]
    predicted = [
        row["gate"]["mean_advantage"] for row in accepted_changes
    ]
    realized = [
        row["endpoint_brier"]["depth_two"]
        - row["endpoint_brier"]["gated"]
        for row in accepted_changes
    ]
    return {
        "tree_count": len(selected),
        "candidate_mean_brier": _mean(
            [row["endpoint_brier"]["gated"] for row in selected]
        ),
        "baseline_mean_brier": baseline_mean,
        "mean_brier_difference": _mean(differences),
        "relative_brier_reduction": -_mean(differences) / baseline_mean,
        "stratified_tree_bootstrap_brier_difference_95pct": (
            stratified_bootstrap_interval(groups, seed=seed)
        ),
        "wins": wins,
        "ties": len(differences) - wins - losses,
        "losses": losses,
        "root_differences": len(switched),
        "accepted_depth_three_changes": len(accepted_changes),
        "accepted_score_to_realized_advantage_spearman": (
            spearman_correlation(predicted, realized)
            if len(accepted_changes) >= 2
            else 0.0
        ),
    }


def run_analysis(output_dir: Path) -> dict[str, Any]:
    blocks = load_source_blocks()
    rows = [
        score_tree(
            raw_tree=raw_tree,
            scored_tree=scored_tree,
            block_index=block["block_index"],
            block_name=block["name"],
            family=block["family"],
        )
        for block in blocks
        for raw_tree, scored_tree in zip(
            block["raw_trees"],
            block["scored_trees"],
            strict=True,
        )
    ]
    family_names = sorted({block["family"] for block in blocks})
    family_indices = {
        family: [
            block["block_index"]
            for block in blocks
            if block["family"] == family
        ]
        for family in family_names
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_uncertainty_gate_summary",
        "protocol": {
            "analysis_is_retrospective": True,
            "gate_is_target_blind": True,
            "cannot_rescue_source_statuses": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "tree_count": len(rows),
            "block_count": len(blocks),
            "planner_family_count": len(family_names),
            "validation_draws_per_tree": 8,
            "minimum_positive_draws": MIN_POSITIVE_DRAWS,
            "lower_standard_error_multiplier": (
                LOWER_STANDARD_ERROR_MULTIPLIER
            ),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "source_blocks": [
                {
                    "name": block["name"],
                    "family": block["family"],
                    "tree_count": len(block["scored_trees"]),
                    "trees_sha256": block["trees_sha256"],
                }
                for block in blocks
            ],
        },
        "gate_counts": {
            "shared_roots": sum(
                row["depth_three_root"] == row["depth_two_root"]
                for row in rows
            ),
            "candidate_changes": sum(
                row["depth_three_root"] != row["depth_two_root"]
                for row in rows
            ),
            "accepted_changes": sum(
                row["depth_three_root"] != row["depth_two_root"]
                and row["gate"]["accepted"]
                for row in rows
            ),
            "fallbacks": sum(
                row["depth_three_root"] != row["depth_two_root"]
                and not row["gate"]["accepted"]
                for row in rows
            ),
        },
        "comparisons": {
            "depth_two": {
                "pooled": summarize(
                    rows,
                    block_indices=list(range(len(blocks))),
                    baseline="depth_two",
                    seed=BOOTSTRAP_SEED,
                ),
                "by_family": {
                    family: summarize(
                        rows,
                        block_indices=indices,
                        baseline="depth_two",
                        seed=BOOTSTRAP_SEED + 100 + offset,
                    )
                    for offset, (family, indices) in enumerate(
                        family_indices.items()
                    )
                },
                "by_block": [
                    {
                        "name": block["name"],
                        "family": block["family"],
                        **summarize(
                            rows,
                            block_indices=[block["block_index"]],
                            baseline="depth_two",
                            seed=BOOTSTRAP_SEED + 200 + block["block_index"],
                        ),
                    }
                    for block in blocks
                ],
            },
            "myopic": {
                "pooled": summarize(
                    rows,
                    block_indices=list(range(len(blocks))),
                    baseline="myopic",
                    seed=BOOTSTRAP_SEED + 500,
                )
            },
        },
        "rows": rows,
        "interpretation": {
            "formal_status_unchanged": True,
            "single_fixed_gate_no_threshold_sweep": True,
            "positive_result_scope": (
                "retrospective evidence for target-blind uncertainty-aware "
                "deployment, not a monotonic-depth confirmation"
            ),
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
