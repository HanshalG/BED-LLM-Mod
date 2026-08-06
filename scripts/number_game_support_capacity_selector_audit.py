#!/usr/bin/env python3
"""Retrospectively audit future support capacity as a Number Game selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_crossfit_endpoint_precision as precision
from scripts import number_game_two_draw_diversity_bonus_audit as diversity
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-support-capacity-selector-audit-1"
CAPACITY_COEFFICIENT = -2.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 113_000
COEFFICIENT_GRID = (-2.0, -1.0, -0.5, -0.25, -0.125, 0.0)
TARGET_HASHES = {
    "development96": (
        "2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6"
    ),
    "later_fresh32": (
        "b799a5d6609f5e8088e2f0115ec1eaa3c4520cebcd187283daf679714cdc5e2b"
    ),
}
POLICIES = {
    "capacity": "capacity_root",
    "unadjusted_dynamic_d3": "original_root",
    "jaccard_dynamic_d3": "bonus_root",
    "dynamic_d2": "depth_two_root",
    "fixed_d3": "fixed_depth_three_root",
    "myopic": "myopic_root",
}


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def select_capacity_root(
    root_rows: Sequence[dict[str, Any]],
    *,
    coefficient: float = CAPACITY_COEFFICIENT,
) -> tuple[int, dict[int, float]]:
    predicted = {
        int(row["root"]): float(row["predicted_brier"])
        for row in root_rows
    }
    capacity = {
        int(row["root"]): float(row["mean_union_size"])
        for row in root_rows
    }
    risk_z = diversity._standardized(predicted)
    capacity_z = diversity._standardized(capacity)
    scores = {
        root: risk_z[root] + coefficient * capacity_z[root]
        for root in predicted
    }
    return min(scores, key=lambda root: (scores[root], root)), scores


def _attach_endpoint_coverage(
    rows: list[dict[str, Any]],
    *,
    source_dir: Path,
) -> None:
    public_trees = _load(source_dir / "TREES.json")["trees"]
    scored_trees = _load(source_dir / "RESULT.json")["trees"]
    targets = _load(source_dir / "TARGETS.json")["targets"]
    coverage_by_seed = {}
    for public_tree, scored_tree in zip(
        public_trees, scored_trees, strict=True
    ):
        replay = precision.score_fixed_tree(
            public_tree,
            scored_tree,
            [targets],
        )
        coverage_by_seed[int(scored_tree["tree_seed"])] = {
            int(root): float(value)
            for root, value in replay["per_root_endpoint_coverage"].items()
        }
    for row in rows:
        coverage = coverage_by_seed[int(row["tree_seed"])]
        for root_row in row["root_rows"]:
            root_row["realized_coverage"] = coverage[int(root_row["root"])]


def _bootstrap_interval(
    values: Sequence[float],
    *,
    seed: int,
    strata: Sequence[str] | None = None,
) -> list[float]:
    if not values:
        raise ValueError("cannot bootstrap an empty comparison")
    rng = random.Random(seed)
    grouped: list[list[float]]
    if strata is None:
        grouped = [list(values)]
    else:
        if len(strata) != len(values):
            raise ValueError("bootstrap strata do not align with values")
        grouped = []
        for name in sorted(set(strata)):
            grouped.append(
                [value for value, source in zip(values, strata) if source == name]
            )
    samples = []
    for _ in range(BOOTSTRAP_SAMPLES):
        draw = [
            rng.choice(group)
            for group in grouped
            for _ in range(len(group))
        ]
        samples.append(statistics.fmean(draw))
    return [diversity._quantile(samples, 0.025), diversity._quantile(samples, 0.975)]


def _comparison_rows(
    rows: Sequence[dict[str, Any]],
    *,
    candidate_key: str,
    baseline_key: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        root_rows = {
            int(item["root"]): item for item in row["root_rows"]
        }
        candidate_root = int(row[candidate_key])
        baseline_root = int(row[baseline_key])
        candidate = root_rows[candidate_root]
        baseline = root_rows[baseline_root]
        output.append(
            {
                "source": str(row["source"]),
                "tree_seed": int(row["tree_seed"]),
                "candidate_root": candidate_root,
                "baseline_root": baseline_root,
                "brier_difference": (
                    float(candidate["realized_brier"])
                    - float(baseline["realized_brier"])
                ),
                "coverage_difference": (
                    float(candidate["realized_coverage"])
                    - float(baseline["realized_coverage"])
                ),
            }
        )
    return output


def summarize_comparison(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    stratified: bool,
) -> dict[str, Any]:
    brier = [float(row["brier_difference"]) for row in rows]
    coverage = [float(row["coverage_difference"]) for row in rows]
    strata = [str(row["source"]) for row in rows] if stratified else None
    return {
        "tree_count": len(rows),
        "changed_roots": sum(
            int(row["candidate_root"]) != int(row["baseline_root"])
            for row in rows
        ),
        "mean_candidate_minus_baseline_brier": statistics.fmean(brier),
        "brier_tree_bootstrap_95pct": _bootstrap_interval(
            brier, seed=seed, strata=strata
        ),
        "brier_wins_ties_losses": {
            "wins": sum(value < -1e-15 for value in brier),
            "ties": sum(abs(value) <= 1e-15 for value in brier),
            "losses": sum(value > 1e-15 for value in brier),
        },
        "mean_candidate_minus_baseline_coverage": statistics.fmean(coverage),
        "coverage_tree_bootstrap_95pct": _bootstrap_interval(
            coverage, seed=seed + 1, strata=strata
        ),
        "coverage_wins_ties_losses": {
            "wins": sum(value > 1e-15 for value in coverage),
            "ties": sum(abs(value) <= 1e-15 for value in coverage),
            "losses": sum(value < -1e-15 for value in coverage),
        },
    }


def _source_rows(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source_spec = dict(spec)
    source_spec["targets_sha256"] = TARGET_HASHES[str(spec["name"])]
    source_dir = Path(source_spec["directory"])
    if diversity.sha256_file(source_dir / "TARGETS.json") != source_spec[
        "targets_sha256"
    ]:
        raise ValueError(f"{spec['name']} targets hash changed")
    rows = diversity.load_source(
        source_spec,
        coefficients=(0.0, diversity.DIVERSITY_COEFFICIENT),
    )
    _attach_endpoint_coverage(rows, source_dir=source_dir)
    for row in rows:
        selected, scores = select_capacity_root(row["root_rows"])
        row["capacity_root"] = selected
        row["capacity_scores"] = scores
    return rows


def _proxy_correlations(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    values = {
        "negative_predicted_brier": [],
        "mean_jaccard_distance": [],
        "mean_union_size": [],
        "mean_second_draw_union_fraction": [],
    }
    for row in rows:
        roots = sorted(int(item["root"]) for item in row["root_rows"])
        root_rows = {int(item["root"]): item for item in row["root_rows"]}
        coverage = [
            float(root_rows[root]["realized_coverage"]) for root in roots
        ]
        values["negative_predicted_brier"].append(
            spearman_correlation(
                [-float(root_rows[root]["predicted_brier"]) for root in roots],
                coverage,
            )
        )
        for key in (
            "mean_jaccard_distance",
            "mean_union_size",
            "mean_second_draw_union_fraction",
        ):
            values[key].append(
                spearman_correlation(
                    [float(root_rows[root][key]) for root in roots],
                    coverage,
                )
            )
    return {key: statistics.fmean(items) for key, items in values.items()}


def _coefficient_grid(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    output = {}
    for coefficient in COEFFICIENT_GRID:
        adjusted = []
        for row in rows:
            selected, _ = select_capacity_root(
                row["root_rows"], coefficient=coefficient
            )
            item = dict(row)
            item["grid_root"] = selected
            adjusted.append(item)
        comparison = _comparison_rows(
            adjusted,
            candidate_key="grid_root",
            baseline_key="original_root",
        )
        output[str(coefficient)] = {
            "changed_roots": sum(
                int(row["candidate_root"]) != int(row["baseline_root"])
                for row in comparison
            ),
            "mean_candidate_minus_unadjusted_brier": statistics.fmean(
                float(row["brier_difference"]) for row in comparison
            ),
            "mean_candidate_minus_unadjusted_coverage": statistics.fmean(
                float(row["coverage_difference"]) for row in comparison
            ),
        }
    return output


def run_audit(output_dir: Path) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    rows_by_source = {
        str(spec["name"]): _source_rows(spec)
        for spec in diversity.SOURCE_SPECS
    }
    all_rows = [row for rows in rows_by_source.values() for row in rows]
    source_results = {}
    comparisons = {
        name: ("capacity_root", key)
        for name, key in POLICIES.items()
        if name != "capacity"
    }
    for source_index, (source, rows) in enumerate(rows_by_source.items()):
        source_results[source] = {
            "tree_count": len(rows),
            "proxy_correlations": _proxy_correlations(rows),
            "coefficient_grid": _coefficient_grid(rows),
            "comparisons": {
                name: summarize_comparison(
                    _comparison_rows(
                        rows,
                        candidate_key=candidate,
                        baseline_key=baseline,
                    ),
                    seed=BOOTSTRAP_SEED + 100 * source_index + index * 2,
                    stratified=False,
                )
                for index, (name, (candidate, baseline)) in enumerate(
                    comparisons.items()
                )
            },
        }
    pooled = {
        name: summarize_comparison(
            _comparison_rows(
                all_rows,
                candidate_key=candidate,
                baseline_key=baseline,
            ),
            seed=BOOTSTRAP_SEED + 1_000 + index * 2,
            stratified=True,
        )
        for index, (name, (candidate, baseline)) in enumerate(
            comparisons.items()
        )
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_candidate_audit",
        "decision": (
            "candidate_for_separately_preregistered_fresh_confirmation_only"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "selector": (
                "within-tree z(predicted Brier) - 2 * "
                "z(mean future generated-support union size)"
            ),
            "capacity_coefficient": CAPACITY_COEFFICIENT,
            "coefficient_selected_after_inspecting_both_open_cohorts": True,
            "external_canonical_targets_used_for_evaluation_only": True,
            "future_fresh_data_required_for_any_claim": True,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "source_artifacts": {
                str(spec["name"]): {
                    key: (
                        TARGET_HASHES[str(spec["name"])]
                        if key == "targets_sha256"
                        else str(spec[key])
                    )
                    for key in (
                        "result_sha256",
                        "trees_sha256",
                        "raw_sha256",
                        "targets_sha256",
                    )
                }
                for spec in (
                    dict(item, targets_sha256=TARGET_HASHES[str(item["name"])])
                    for item in diversity.SOURCE_SPECS
                )
            },
        },
        "sources": source_results,
        "pooled_stratified": pooled,
        "public_rows": [
            {
                "source": str(row["source"]),
                "tree_seed": int(row["tree_seed"]),
                **{
                    key: int(row[value]) for key, value in POLICIES.items()
                },
            }
            for row in all_rows
        ],
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = run_audit(args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
