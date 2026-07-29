#!/usr/bin/env python3
"""Pool dynamic-support versus fixed-support depth-three Number Game results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_pooled_replication_synthesis64 as pooled


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-pooled-dynamic-vs-fixed64-1"
BOOTSTRAP_SEED = 64_700
BOOTSTRAP_SAMPLES = 20_000


def comparison_rows(
    first: dict[str, Any],
    second: dict[str, Any],
) -> list[list[dict[str, Any]]]:
    cohorts = []
    for source in (first, second):
        rows = []
        for tree in source["trees"]:
            candidate = tree["endpoint"]["crossfit_depth_three"]
            baseline = tree["endpoint"]["fixed_support_depth_three"]
            rows.append(
                {
                    "tree_seed": int(tree["tree_seed"]),
                    "comparison": {
                        "candidate_brier": float(
                            candidate["mean_posterior_predictive_brier"]
                        ),
                        "baseline_brier": float(
                            baseline["mean_posterior_predictive_brier"]
                        ),
                        "candidate_hamming": float(
                            candidate["mean_best_hamming_error"]
                        ),
                        "baseline_hamming": float(
                            baseline["mean_best_hamming_error"]
                        ),
                        "candidate_coverage": float(
                            candidate["truth_extension_coverage_rate"]
                        ),
                        "baseline_coverage": float(
                            baseline["truth_extension_coverage_rate"]
                        ),
                    },
                }
            )
        cohorts.append(rows)

    first_seeds = {row["tree_seed"] for row in cohorts[0]}
    second_seeds = {row["tree_seed"] for row in cohorts[1]}
    if len(first_seeds) != 32 or len(second_seeds) != 32:
        raise ValueError("source tree seeds are not unique")
    if first_seeds & second_seeds:
        raise ValueError("source cohorts share tree seeds")
    return cohorts


def source_mean_difference(
    rows: list[dict[str, Any]],
    *,
    metric: str,
) -> float:
    return sum(
        row["comparison"][f"candidate_{metric}"]
        - row["comparison"][f"baseline_{metric}"]
        for row in rows
    ) / len(rows)


def validate_source_reproduction(
    cohorts: list[list[dict[str, Any]]],
    sources: tuple[dict[str, Any], dict[str, Any]],
) -> None:
    for rows, source in zip(cohorts, sources, strict=True):
        published = source["aggregate"]["comparisons"][
            "fixed_support_depth_three"
        ]
        if abs(
            source_mean_difference(rows, metric="brier")
            - published["mean_candidate_minus_baseline_brier"]
        ) > 1e-12:
            raise ValueError("source Brier comparison does not reproduce")
        if abs(
            source_mean_difference(rows, metric="hamming")
            - published["mean_candidate_minus_baseline_hamming"]
        ) > 1e-12:
            raise ValueError("source Hamming comparison does not reproduce")
        if abs(
            source_mean_difference(rows, metric="coverage")
            - published["mean_coverage_difference"]
        ) > 1e-12:
            raise ValueError("source coverage comparison does not reproduce")


def run_analysis(output_dir: Path) -> dict[str, Any]:
    first, _, second = pooled.load_sources()
    cohorts = comparison_rows(first, second)
    validate_source_reproduction(cohorts, (first, second))
    bootstrap = pooled.stratified_bootstrap_indices(
        cohort_sizes=[len(cohort) for cohort in cohorts],
        seed=BOOTSTRAP_SEED,
        samples=BOOTSTRAP_SAMPLES,
    )
    summaries = {
        metric: pooled.summarize_difference(
            cohorts,
            candidate_path=("comparison", f"candidate_{metric}"),
            baseline_path=("comparison", f"baseline_{metric}"),
            bootstrap_indices=bootstrap,
        )
        for metric in ("brier", "hamming", "coverage")
    }
    brier = summaries["brier"]
    gates = {
        "both_source_mean_brier_differences_below_zero": all(
            value < 0.0 for value in brier["source_mean_differences"]
        ),
        "pooled_relative_brier_reduction_at_least_three_percent": (
            brier["relative_reduction"] >= 0.03
        ),
        "pooled_brier_interval_below_zero": (
            brier["stratified_bootstrap_difference_95pct"][1] < 0.0
        ),
        "pooled_brier_wins_at_least_twenty_eight": brier["wins"] >= 28,
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "retrospective_dynamic_support_positive"
            if all(gates.values())
            else "retrospective_dynamic_support_null"
        ),
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "analysis_is_retrospective": True,
            "cannot_rescue_or_reclassify_sources": True,
            "fresh_confirmation_required": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "cohort_count": 2,
            "tree_count": 64,
            "trees_per_cohort": 32,
            "target_count": 33,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "candidate": "crossfit_depth_three",
            "baseline": "fixed_support_depth_three",
            "source_hashes": {
                "cohort_one": pooled.FIRST_POLICY_SHA256,
                "cohort_two": pooled.SECOND_SHA256,
            },
        },
        "sources": [
            {
                "name": name,
                "status": source["status"],
                "comparison": source["aggregate"]["comparisons"][
                    "fixed_support_depth_three"
                ],
            }
            for name, source in (
                ("pooled_qwen_cohort_one", first),
                ("pooled_qwen_cohort_two", second),
            )
        ],
        "pooled": summaries,
        "gates": gates,
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "gates": result["gates"],
                "pooled": result["pooled"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
