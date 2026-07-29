#!/usr/bin/env python3
"""Audit score-to-realized-advantage linkage on 64 fixed Qwen trees."""

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
from scripts.number_game_qwen_external_canonical_pooled64 import (
    STUDIES,
    validate_source,
)
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-first-link-mechanism64-1"
BOOTSTRAP_SEED = 56_000
BOOTSTRAP_SAMPLES = 20_000
BASELINES = {
    "myopic_eig": "myopic_root",
    "fixed_support_depth_three": "fixed_support_depth_three_root",
    "crossfit_depth_two": "crossfit_depth_two_root",
}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def rows_for_baseline(
    sources: Sequence[dict[str, Any]],
    *,
    root_key: str,
) -> list[dict[str, Any]]:
    rows = []
    for study_index, source in enumerate(sources):
        for tree in source["trees"]:
            selection = tree["selection"]
            candidate_root = str(selection["crossfit_depth_three_root"])
            baseline_root = str(selection[root_key])
            predicted = selection["crossfit_depth_three_brier"]
            realized = tree["per_root_endpoint_brier"]
            rows.append(
                {
                    "study_index": study_index,
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


def _summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    predicted = [float(row["predicted_advantage"]) for row in rows]
    realized = [float(row["realized_advantage"]) for row in rows]
    return {
        "tree_count": len(rows),
        "mean_predicted_advantage": _mean(predicted),
        "mean_realized_advantage": _mean(realized),
        "score_to_realized_advantage_spearman": spearman_correlation(
            predicted,
            realized,
        ),
        "wins": sum(value > 1e-15 for value in realized),
        "ties": sum(abs(value) <= 1e-15 for value in realized),
        "losses": sum(value < -1e-15 for value in realized),
    }


def stratified_bootstrap(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
) -> dict[str, list[float]]:
    by_study = [
        [row for row in rows if row["study_index"] == study_index]
        for study_index in range(len(STUDIES))
    ]
    rng = random.Random(seed)
    mean_values = []
    divergent_mean_values = []
    divergent_rho_values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            rng.choice(study_rows)
            for study_rows in by_study
            for _ in range(len(study_rows))
        ]
        mean_values.append(
            _mean([row["realized_advantage"] for row in sample])
        )
        divergent = [row for row in sample if row["roots_differ"]]
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
        "mean_realized_advantage_95pct": _interval(mean_values),
        "divergent_mean_realized_advantage_95pct": _interval(
            divergent_mean_values
        ),
        "divergent_score_to_realized_spearman_95pct": _interval(
            divergent_rho_values
        ),
    }


def analyze_baseline(
    sources: Sequence[dict[str, Any]],
    *,
    baseline: str,
    root_key: str,
    seed: int,
) -> dict[str, Any]:
    rows = rows_for_baseline(sources, root_key=root_key)
    divergent = [row for row in rows if row["roots_differ"]]
    return {
        "baseline": baseline,
        "root_differences": len(divergent),
        "all_trees": _summary(rows),
        "divergent_trees": _summary(divergent),
        "by_study": [
            {
                "study": STUDIES[study_index]["name"],
                "all_trees": _summary(
                    [
                        row
                        for row in rows
                        if row["study_index"] == study_index
                    ]
                ),
                "divergent_trees": _summary(
                    [
                        row
                        for row in divergent
                        if row["study_index"] == study_index
                    ]
                ),
            }
            for study_index in range(len(STUDIES))
        ],
        "stratified_bootstrap": stratified_bootstrap(rows, seed=seed),
    }


def run_analysis(output_dir: Path) -> dict[str, Any]:
    sources = [validate_source(study) for study in STUDIES]
    analyses = {
        baseline: analyze_baseline(
            sources,
            baseline=baseline,
            root_key=root_key,
            seed=BOOTSTRAP_SEED + offset,
        )
        for offset, (baseline, root_key) in enumerate(BASELINES.items())
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_first_link_summary",
        "protocol": {
            "analysis_is_retrospective": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "tree_count": 64,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "advantage_sign": (
                "positive means crossfit depth three has lower Brier"
            ),
            "source_hashes": {
                study["name"]: study["sha256"] for study in STUDIES
            },
        },
        "analyses": analyses,
        "interpretation": {
            "first_link": (
                "cross-fitted simulated risk advantage versus a competing "
                "root predicts exact-canonical realized Brier advantage"
            ),
            "second_link_not_tested": (
                "no realized online observation trajectory or posterior "
                "execution noise is introduced"
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
