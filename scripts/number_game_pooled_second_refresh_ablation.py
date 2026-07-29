#!/usr/bin/env python3
"""Ablate second-step LLM regeneration in the fresh pooled Number Game trees."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_depth_three_development as depth
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_crossfit_depth_three_confirmation import (
    _mean,
    _target_mapping,
)
from scripts.number_game_depth_three_crossfit_audit import (
    select_minimum_risk_root,
)
from scripts.number_game_qwen_first_link_mechanism64 import _interval
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
    retained_second_branches,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-pooled-second-refresh-ablation-1"
SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_pooled_first_link_confirmation32"
    / "number-game-qwen-pooled-first-link-confirmation32-20260729T094455Z"
)
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_RESULT_SHA256 = (
    "cef6ded08050e6df07de62bbbf548635f229a8bf73fac149b4d0bd960f7b0253"
)
SOURCE_TREES_SHA256 = (
    "78bbc36bab604c78e31a68e50d31a1f9d92bea02bd295b1f983fe2f2a96a4d68"
)
TREE_COUNT = 32
VALIDATION_DRAWS = 8
BOOTSTRAP_SEED = 63_800
BOOTSTRAP_SAMPLES = 20_000
VARIANTS = (
    "merged_retained_generated",
    "parent_only",
    "generated_only",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def variant_root_selections(tree: dict[str, Any]) -> dict[str, Any]:
    roots = [int(root) for root in tree["roots"]]
    first = _first_branches(tree)
    generated_second = _second_branches(
        tree,
        key="generated_second_branches",
    )
    merged_second, parent_only_second, _ = retained_second_branches(
        first_branches=first,
        generated_second_branches=generated_second,
    )
    second_by_variant = {
        "merged_retained_generated": merged_second,
        "parent_only": parent_only_second,
        "generated_only": generated_second,
    }
    validation_supports = [
        [_rule(item) for item in support]
        for support in tree["validation_supports"]
    ]
    risk_by_variant = {}
    root_by_variant = {}
    for variant, second in second_by_variant.items():
        draw_risks = []
        for draw_index, support in enumerate(validation_supports):
            targets = _target_mapping(support, draw_index=draw_index)
            draw_risks.append(
                {
                    root: depth.evaluate_policy_root_depth_three(
                        policy=f"{variant}_root_{root}",
                        root=root,
                        targets=targets,
                        first_branches=first,
                        second_branches=second,
                    )["mean_posterior_predictive_brier"]
                    for root in roots
                }
            )
        risks = {
            root: _mean([draw[root] for draw in draw_risks])
            for root in roots
        }
        risk_by_variant[variant] = risks
        root_by_variant[variant] = select_minimum_risk_root(roots, risks)
    return {
        "roots": root_by_variant,
        "risks": risk_by_variant,
    }


def comparison_summary(
    rows: Sequence[dict[str, Any]],
    *,
    baseline: str,
    bootstrap_indices: Sequence[Sequence[int]],
) -> dict[str, Any]:
    candidate = [
        row["endpoint_brier"]["merged_retained_generated"] for row in rows
    ]
    baseline_values = [row["endpoint_brier"][baseline] for row in rows]
    differences = [
        candidate_value - baseline_value
        for candidate_value, baseline_value in zip(
            candidate,
            baseline_values,
            strict=True,
        )
    ]
    candidate_mean = _mean(candidate)
    baseline_mean = _mean(baseline_values)
    bootstrap = [
        _mean([differences[index] for index in indices])
        for indices in bootstrap_indices
    ]
    wins = sum(value < -1e-15 for value in differences)
    losses = sum(value > 1e-15 for value in differences)
    return {
        "candidate_mean_brier": candidate_mean,
        "baseline_mean_brier": baseline_mean,
        "mean_candidate_minus_baseline_brier": _mean(differences),
        "relative_brier_reduction": (
            (baseline_mean - candidate_mean) / baseline_mean
        ),
        "tree_bootstrap_brier_difference_95pct": _interval(bootstrap),
        "root_differences": sum(
            row["selected_roots"]["merged_retained_generated"]
            != row["selected_roots"][baseline]
            for row in rows
        ),
        "wins": wins,
        "ties": len(rows) - wins - losses,
        "losses": losses,
        "wins_minus_losses": wins - losses,
    }


def run_analysis(
    *,
    output_dir: Path,
    source_result_path: Path = SOURCE_RESULT,
    source_trees_path: Path = SOURCE_TREES,
) -> dict[str, Any]:
    if sha256_file(source_result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source RESULT hash changed")
    if sha256_file(source_trees_path) != SOURCE_TREES_SHA256:
        raise ValueError("source TREES hash changed")
    source_result = json.loads(
        source_result_path.read_text(encoding="utf-8")
    )
    source_trees = json.loads(
        source_trees_path.read_text(encoding="utf-8")
    )
    result_by_seed = {
        int(tree["tree_seed"]): tree for tree in source_result["trees"]
    }
    public_trees = source_trees["trees"]
    if len(public_trees) != TREE_COUNT or len(result_by_seed) != TREE_COUNT:
        raise ValueError("source tree count changed")

    rows = []
    for tree in public_trees:
        tree_seed = int(tree["tree_seed"])
        source_row = result_by_seed[tree_seed]
        selection = variant_root_selections(tree)
        selected_roots = selection["roots"]
        registered_root = int(
            source_row["selection"]["crossfit_depth_three_root"]
        )
        if selected_roots["merged_retained_generated"] != registered_root:
            raise ValueError(
                f"tree {tree_seed} does not reproduce registered root"
            )
        per_root = {
            int(root): float(value)
            for root, value in source_row["per_root_endpoint_brier"].items()
        }
        rows.append(
            {
                "tree_seed": tree_seed,
                "selected_roots": selected_roots,
                "crossfit_risks": selection["risks"],
                "endpoint_brier": {
                    variant: per_root[selected_roots[variant]]
                    for variant in VARIANTS
                },
            }
        )

    rng = random.Random(BOOTSTRAP_SEED)
    bootstrap_indices = [
        [rng.randrange(TREE_COUNT) for _ in range(TREE_COUNT)]
        for _ in range(BOOTSTRAP_SAMPLES)
    ]
    comparisons = {
        baseline: comparison_summary(
            rows,
            baseline=baseline,
            bootstrap_indices=bootstrap_indices,
        )
        for baseline in ("parent_only", "generated_only")
    }
    parent = comparisons["parent_only"]
    gates = {
        "merged_and_parent_only_roots_differ_on_at_least_twelve_trees": (
            parent["root_differences"] >= 12
        ),
        "merged_beats_parent_only_by_two_percent": (
            parent["relative_brier_reduction"] >= 0.02
        ),
        "merged_vs_parent_only_interval_below_zero": (
            parent["tree_bootstrap_brier_difference_95pct"][1] < 0.0
        ),
        "merged_wins_minus_losses_at_least_eight": (
            parent["wins_minus_losses"] >= 8
        ),
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "retrospective_second_refresh_positive"
            if all(gates.values())
            else "retrospective_second_refresh_null"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "analysis_is_retrospective": True,
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "tree_count": TREE_COUNT,
            "validation_draws": VALIDATION_DRAWS,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "primary_baseline": "parent_only",
            "generated_only_is_diagnostic": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "cannot_rescue_source_status": True,
        },
        "gates": gates,
        "comparisons": comparisons,
        "rows": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-result", type=Path, default=SOURCE_RESULT)
    parser.add_argument("--source-trees", type=Path, default=SOURCE_TREES)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    result = run_analysis(
        output_dir=args.output_dir,
        source_result_path=args.source_result,
        source_trees_path=args.source_trees,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "comparisons": result["comparisons"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
