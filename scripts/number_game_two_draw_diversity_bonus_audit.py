#!/usr/bin/env python3
"""Audit whether independent support-draw diversity improves root selection."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_item_isolated_codec import (
    parse_proposals_item_isolated,
)
from scripts.number_game_pooled_support import POOLED_FIELD, POOL_SIZE
from scripts.number_game_ranking_fidelity_audit import spearman_correlation


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-two-draw-diversity-bonus-audit-1"
BOOTSTRAP_SEED = 108_600
BOOTSTRAP_SAMPLES = 20_000
DIVERSITY_COEFFICIENT = -0.5
COEFFICIENT_GRID = (-2.0, -1.0, -0.5, -0.25, -0.125, 0.0, 0.125,
                    0.25, 0.5, 1.0, 2.0)

SOURCE_SPECS = (
    {
        "name": "development96",
        "role": "retrospective_development",
        "directory": REPO_ROOT / (
            "results/nonmyopic/"
            "number_game_qwen_dynamic_vs_fixed_resilient96_v2/"
            "number-game-qwen-dynamic-vs-fixed-resilient96-v2-"
            "20260730T023000Z"
        ),
        "tree_count": 96,
        "result_sha256": (
            "04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e"
        ),
        "trees_sha256": (
            "8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82"
        ),
        "raw_sha256": (
            "c0a95d6c43ccc95e2a07a275e7800b451d653092f82d33c8dfcd8ddbbed6ac2d"
        ),
    },
    {
        "name": "later_fresh32",
        "role": "retrospective_later_cohort_replication",
        "directory": REPO_ROOT / (
            "results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/"
            "number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/"
            "source"
        ),
        "tree_count": 32,
        "result_sha256": (
            "13fd3361a8ef8f525f68733182a9bdb30151e37a4a5e14dfb35dde78540ab523"
        ),
        "trees_sha256": (
            "f812e4a356f5129a6f2f22d5f0f995b76b4ac624a0f312154064ff8c51f2c7b0"
        ),
        "raw_sha256": (
            "6f08534d51992e1350ab3c9670ff84cd0a34b872c75b999d18ddd4ae49e30cf9"
        ),
    },
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    return statistics.fmean(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _extension_sha256(extension: Sequence[bool]) -> str:
    return hashlib.sha256(bytes(extension)).hexdigest()


def pooled_draw_extensions(
    response: str,
    *,
    observations: Sequence[tuple[int, bool]],
) -> tuple[set[tuple[bool, ...]], set[tuple[bool, ...]]]:
    value = json.loads(response)
    if (
        not isinstance(value, dict)
        or set(value) != {POOLED_FIELD}
        or not isinstance(value[POOLED_FIELD], list)
        or len(value[POOLED_FIELD]) != POOL_SIZE
        or not all(isinstance(item, str) for item in value[POOLED_FIELD])
    ):
        raise ValueError("expected an exact two-draw pooled response")
    draws = []
    for raw_response in value[POOLED_FIELD]:
        support, _ = parse_proposals_item_isolated(
            raw_response,
            observations=observations,
        )
        draws.append({hypothesis.extension for hypothesis in support})
    return draws[0], draws[1]


def draw_disagreement(
    first: set[tuple[bool, ...]],
    second: set[tuple[bool, ...]],
) -> dict[str, float]:
    union = first | second
    intersection = first & second
    return {
        "jaccard_distance": (
            1.0 - len(intersection) / len(union) if union else 0.0
        ),
        "union_size": float(len(union)),
        "intersection_size": float(len(intersection)),
        "first_size": float(len(first)),
        "second_size": float(len(second)),
        "second_draw_union_fraction": (
            len(second - first) / len(union) if union else 0.0
        ),
    }


def _validate_public_generated_support(
    union: Iterable[tuple[bool, ...]],
    public_support: Sequence[dict[str, Any]],
    *,
    label: str,
) -> None:
    reconstructed = {_extension_sha256(extension) for extension in union}
    expected = {str(item["extension_sha256"]) for item in public_support}
    if reconstructed != expected or len(reconstructed) != len(public_support):
        raise ValueError(f"{label} does not reconstruct public support")


def _raw_rows(rows: Sequence[dict[str, Any]]) -> dict[tuple[Any, ...], str]:
    return {tuple(row["key"]): str(row["response"]) for row in rows}


def _standardized(values: dict[int, float]) -> dict[int, float]:
    mean = _mean(list(values.values()))
    scale = statistics.pstdev(values.values())
    if not math.isfinite(scale) or scale <= 0.0:
        return {key: 0.0 for key in values}
    return {key: (value - mean) / scale for key, value in values.items()}


def adjusted_root(
    predicted_risk: dict[int, float],
    diversity: dict[int, float],
    *,
    coefficient: float = DIVERSITY_COEFFICIENT,
) -> tuple[int, dict[int, float]]:
    if set(predicted_risk) != set(diversity):
        raise ValueError("risk and diversity roots differ")
    risk_z = _standardized(predicted_risk)
    diversity_z = _standardized(diversity)
    adjusted = {
        root: risk_z[root] + coefficient * diversity_z[root]
        for root in predicted_risk
    }
    selected = min(adjusted, key=lambda root: (adjusted[root], root))
    return selected, adjusted


def _root_rows(
    public_tree: dict[str, Any],
    raw_tree: dict[str, Any],
    scored_tree: dict[str, Any],
) -> list[dict[str, Any]]:
    roots = [int(root) for root in public_tree["roots"]]
    metrics = {root: [] for root in roots}
    raw_first = _raw_rows(raw_tree["first_responses"])
    raw_second = _raw_rows(raw_tree["second_responses"])

    for key, response in raw_first.items():
        root, label = key
        observations = ((int(root), bool(label)),)
        first, second = pooled_draw_extensions(
            response,
            observations=observations,
        )
        public_key = f"{root}:{int(label)}"
        _validate_public_generated_support(
            first | second,
            public_tree["generated_first_branches"][public_key],
            label=f"first branch {public_key}",
        )
        metrics[int(root)].append(draw_disagreement(first, second))

    for key, response in raw_second.items():
        root, first_label, second_query, second_label = key
        observations = (
            (int(root), bool(first_label)),
            (int(second_query), bool(second_label)),
        )
        first, second = pooled_draw_extensions(
            response,
            observations=observations,
        )
        public_key = (
            f"{root}:{int(first_label)}:{second_query}:{int(second_label)}"
        )
        _validate_public_generated_support(
            first | second,
            public_tree["generated_second_branches"][public_key],
            label=f"second branch {public_key}",
        )
        metrics[int(root)].append(draw_disagreement(first, second))

    predicted = {
        int(root): float(value)
        for root, value in scored_tree["selection"][
            "crossfit_depth_three_brier"
        ].items()
    }
    realized = {
        int(root): float(value)
        for root, value in scored_tree["per_root_endpoint_brier"].items()
    }
    if set(predicted) != set(roots) or set(realized) != set(roots):
        raise ValueError("candidate roots changed")
    if any(len(metrics[root]) != 6 for root in roots):
        raise ValueError("each root must have six generated future branches")

    return [
        {
            "root": root,
            "predicted_brier": predicted[root],
            "realized_brier": realized[root],
            "prediction_error": realized[root] - predicted[root],
            "absolute_prediction_error": abs(realized[root] - predicted[root]),
            "mean_jaccard_distance": _mean(
                [item["jaccard_distance"] for item in metrics[root]]
            ),
            "mean_union_size": _mean(
                [item["union_size"] for item in metrics[root]]
            ),
            "mean_second_draw_union_fraction": _mean(
                [
                    item["second_draw_union_fraction"]
                    for item in metrics[root]
                ]
            ),
        }
        for root in roots
    ]


def load_source(
    spec: dict[str, Any],
    *,
    coefficients: Sequence[float] = COEFFICIENT_GRID,
) -> list[dict[str, Any]]:
    if 0.0 not in coefficients:
        raise ValueError("coefficient set must reproduce the original selector")
    if DIVERSITY_COEFFICIENT not in coefficients:
        raise ValueError("coefficient set must contain the diversity bonus")
    if len(set(coefficients)) != len(coefficients):
        raise ValueError("coefficient set contains duplicates")
    directory = Path(spec["directory"])
    paths = {
        "result": directory / "RESULT.json",
        "trees": directory / "TREES.json",
        "raw": directory / "private" / "RAW_RESPONSES.json",
    }
    for key, path in paths.items():
        if sha256_file(path) != spec[f"{key}_sha256"]:
            raise ValueError(f"{spec['name']} {key} hash changed")
    result = json.loads(paths["result"].read_text(encoding="utf-8"))
    trees = json.loads(paths["trees"].read_text(encoding="utf-8"))["trees"]
    raw = json.loads(paths["raw"].read_text(encoding="utf-8"))["trees"]
    if not (len(result["trees"]) == len(trees) == len(raw) == spec["tree_count"]):
        raise ValueError(f"{spec['name']} tree count changed")

    rows = []
    for scored_tree, public_tree, raw_tree in zip(
        result["trees"], trees, raw, strict=True
    ):
        if not (
            int(scored_tree["tree_seed"])
            == int(public_tree["tree_seed"])
        ):
            raise ValueError("scored and public tree seeds do not align")
        root_rows = _root_rows(public_tree, raw_tree, scored_tree)
        predicted = {
            row["root"]: row["predicted_brier"] for row in root_rows
        }
        diversity = {
            row["root"]: row["mean_jaccard_distance"] for row in root_rows
        }
        selections = {}
        adjusted_scores = {}
        for coefficient in coefficients:
            selected, adjusted = adjusted_root(
                predicted,
                diversity,
                coefficient=coefficient,
            )
            selections[str(coefficient)] = selected
            adjusted_scores[str(coefficient)] = adjusted
        original_root = int(
            scored_tree["selection"]["crossfit_depth_three_root"]
        )
        if selections["0.0"] != original_root:
            raise ValueError("zero coefficient does not reproduce selection")
        rows.append(
            {
                "source": spec["name"],
                "role": spec["role"],
                "tree_index": int(scored_tree["tree_index"]),
                "tree_seed": int(scored_tree["tree_seed"]),
                "original_root": original_root,
                "bonus_root": selections[str(DIVERSITY_COEFFICIENT)],
                "myopic_root": int(scored_tree["selection"]["myopic_root"]),
                "fixed_depth_three_root": int(
                    scored_tree["selection"][
                        "fixed_support_depth_three_root"
                    ]
                ),
                "depth_two_root": int(
                    scored_tree["selection"]["crossfit_depth_two_root"]
                ),
                "pts_roots": [
                    int(root)
                    for root in scored_tree["selection"]["pts_roots"]
                ],
                "random_roots": [
                    int(root)
                    for root in scored_tree["selection"]["random_roots"]
                ],
                "coefficient_grid_roots": selections,
                "root_rows": root_rows,
                "adjusted_scores": adjusted_scores[
                    str(DIVERSITY_COEFFICIENT)
                ],
            }
        )
    return rows


def _root_value(row: dict[str, Any], root: int, key: str) -> float:
    return next(
        float(item[key])
        for item in row["root_rows"]
        if int(item["root"]) == root
    )


def _comparison_rows(
    rows: Sequence[dict[str, Any]],
    *,
    candidate_key: str,
    baseline_key: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        candidate_root = int(row[candidate_key])
        baseline_root = int(row[baseline_key])
        candidate = _root_value(row, candidate_root, "realized_brier")
        baseline = _root_value(row, baseline_root, "realized_brier")
        output.append(
            {
                "source": row["source"],
                "tree_seed": row["tree_seed"],
                "candidate_root": candidate_root,
                "baseline_root": baseline_root,
                "candidate_brier": candidate,
                "baseline_brier": baseline,
                "difference": candidate - baseline,
            }
        )
    return output


def _multi_root_comparison_rows(
    rows: Sequence[dict[str, Any]],
    *,
    candidate_key: str,
    baseline_key: str,
) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        candidate_root = int(row[candidate_key])
        baseline_roots = [int(root) for root in row[baseline_key]]
        if not baseline_roots:
            raise ValueError(f"empty multi-root baseline: {baseline_key}")
        candidate = _root_value(row, candidate_root, "realized_brier")
        baseline = _mean(
            [
                _root_value(row, root, "realized_brier")
                for root in baseline_roots
            ]
        )
        output.append(
            {
                "source": row["source"],
                "tree_seed": row["tree_seed"],
                "candidate_root": candidate_root,
                "baseline_roots": baseline_roots,
                "candidate_brier": candidate,
                "baseline_brier": baseline,
                "difference": candidate - baseline,
            }
        )
    return output


def comparison_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    differences = [float(row["difference"]) for row in rows]
    candidate = [float(row["candidate_brier"]) for row in rows]
    baseline = [float(row["baseline_brier"]) for row in rows]
    baseline_mean = _mean(baseline)
    sample_sd = lambda values: (
        statistics.stdev(values) if len(values) > 1 else 0.0
    )
    return {
        "tree_count": len(rows),
        "candidate_mean_brier": _mean(candidate),
        "baseline_mean_brier": baseline_mean,
        "mean_candidate_minus_baseline_brier": _mean(differences),
        "candidate_brier_sample_sd": sample_sd(candidate),
        "baseline_brier_sample_sd": sample_sd(baseline),
        "paired_difference_sample_sd": sample_sd(differences),
        "relative_brier_reduction": (
            (baseline_mean - _mean(candidate)) / baseline_mean
        ),
        "changed_roots": sum(
            (
                row["candidate_root"] not in row["baseline_roots"]
                if "baseline_roots" in row
                else row["candidate_root"] != row["baseline_root"]
            )
            for row in rows
        ),
        "wins": sum(value < -1e-15 for value in differences),
        "ties": sum(abs(value) <= 1e-15 for value in differences),
        "losses": sum(value > 1e-15 for value in differences),
    }


def bootstrap_comparison(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    stratified: bool,
) -> list[float]:
    rng = random.Random(seed)
    groups = []
    if stratified:
        for source in dict.fromkeys(row["source"] for row in rows):
            groups.append([row for row in rows if row["source"] == source])
    else:
        groups = [list(rows)]
    values = []
    for _ in range(BOOTSTRAP_SAMPLES):
        sample = [
            rng.choice(group)
            for group in groups
            for _ in range(len(group))
        ]
        values.append(_mean([row["difference"] for row in sample]))
    return _interval(values)


def _centered_values(
    rows: Sequence[dict[str, Any]], key: str
) -> list[float]:
    values = []
    for row in rows:
        mean = _mean([float(item[key]) for item in row["root_rows"]])
        values.extend(float(item[key]) - mean for item in row["root_rows"])
    return values


def source_summary(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    include_coefficient_grid: bool = True,
) -> dict[str, Any]:
    original = _comparison_rows(
        rows, candidate_key="bonus_root", baseline_key="original_root"
    )
    comparisons = {
        "original_depth_three": comparison_summary(original)
        | {
            "tree_bootstrap_95pct": bootstrap_comparison(
                original, seed=seed, stratified=False
            )
        }
    }
    for offset, (name, key) in enumerate(
        (("myopic_eig", "myopic_root"),
         ("fixed_support_depth_three", "fixed_depth_three_root"),
         ("crossfit_depth_two", "depth_two_root")),
        start=1,
    ):
        comparison = _comparison_rows(
            rows, candidate_key="bonus_root", baseline_key=key
        )
        comparisons[name] = comparison_summary(comparison) | {
            "tree_bootstrap_95pct": bootstrap_comparison(
                comparison, seed=seed + offset, stratified=False
            )
        }
    for offset, (name, key) in enumerate(
        (
            ("positive_test_strategy", "pts_roots"),
            ("uniform_random_candidate_root", "random_roots"),
        ),
        start=4,
    ):
        if not all(key in row for row in rows):
            continue
        comparison = _multi_root_comparison_rows(
            rows,
            candidate_key="bonus_root",
            baseline_key=key,
        )
        comparisons[name] = comparison_summary(comparison) | {
            "baseline_is_mean_of_two_roots_per_tree": True,
            "tree_bootstrap_95pct": bootstrap_comparison(
                comparison, seed=seed + offset, stratified=False
            ),
        }

    risk = _centered_values(rows, "predicted_brier")
    realized = _centered_values(rows, "realized_brier")
    diversity = _centered_values(rows, "mean_jaccard_distance")
    absolute_error = _centered_values(rows, "absolute_prediction_error")
    summary = {
        "source": rows[0]["source"],
        "role": rows[0]["role"],
        "tree_count": len(rows),
        "mean_branch_jaccard_distance": _mean(
            [
                float(item["mean_jaccard_distance"])
                for row in rows
                for item in row["root_rows"]
            ]
        ),
        "within_tree_centered_correlations": {
            "diversity_to_realized_brier_spearman": spearman_correlation(
                diversity, realized
            ),
            "diversity_to_absolute_prediction_error_spearman": (
                spearman_correlation(diversity, absolute_error)
            ),
            "predicted_to_realized_brier_spearman": spearman_correlation(
                risk, realized
            ),
        },
        "comparisons": comparisons,
    }
    if include_coefficient_grid:
        summary["coefficient_grid"] = {
            str(coefficient): comparison_summary(
                _comparison_rows(
                    [
                        row
                        | {
                            "grid_root": row["coefficient_grid_roots"][
                                str(coefficient)
                            ]
                        }
                        for row in rows
                    ],
                    candidate_key="grid_root",
                    baseline_key="original_root",
                )
            )
            for coefficient in COEFFICIENT_GRID
        }
    return summary


def run_analysis(output_dir: Path) -> dict[str, Any]:
    rows_by_source = [load_source(spec) for spec in SOURCE_SPECS]
    summaries = [
        source_summary(rows, seed=BOOTSTRAP_SEED + index * 10)
        for index, rows in enumerate(rows_by_source)
    ]
    all_rows = [row for rows in rows_by_source for row in rows]
    combined = _comparison_rows(
        all_rows, candidate_key="bonus_root", baseline_key="original_root"
    )
    combined_summary = comparison_summary(combined) | {
        "source_stratified_bootstrap_95pct": bootstrap_comparison(
            combined,
            seed=BOOTSTRAP_SEED + 100,
            stratified=True,
        )
    }
    replicated_direction = all(
        summary["comparisons"]["original_depth_three"][
            "mean_candidate_minus_baseline_brier"
        ] < 0.0
        and summary["comparisons"]["original_depth_three"]["wins"]
        > summary["comparisons"]["original_depth_three"]["losses"]
        for summary in summaries
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "retrospective_diversity_bonus_replication_positive"
            if replicated_direction
            else "retrospective_diversity_bonus_null"
        ),
        "protocol": {
            "analysis_is_retrospective": True,
            "later_cohort_is_not_claimed_as_prospectively_held_out": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "diversity_metric": (
                "mean Jaccard distance between the two independent valid "
                "extension sets across each root's six generated future "
                "branch contexts"
            ),
            "initial_draw_excluded_because_it_is_shared_by_all_roots": True,
            "adjusted_score": (
                "within-tree z(predicted Brier) - 0.5 * "
                "within-tree z(mean branch draw Jaccard distance)"
            ),
            "coefficient": DIVERSITY_COEFFICIENT,
            "coefficient_was_selected_retrospectively": True,
            "coefficient_grid": list(COEFFICIENT_GRID),
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": BOOTSTRAP_SAMPLES,
            "private_raw_text_is_not_published": True,
            "source_hashes": {
                spec["name"]: {
                    "result_sha256": spec["result_sha256"],
                    "trees_sha256": spec["trees_sha256"],
                    "raw_sha256": spec["raw_sha256"],
                }
                for spec in SOURCE_SPECS
            },
        },
        "sources": summaries,
        "combined_bonus_vs_original_depth_three": combined_summary,
        "replicated_direction": replicated_direction,
        "rows": all_rows,
        "interpretation": {
            "disagreement_penalty": (
                "unsupported: positive coefficients worsen selection in "
                "both cohorts"
            ),
            "diversity_bonus": (
                "promising retrospective selector that requires a new "
                "prospective fresh-tree confirmation"
            ),
            "prior_claims_unchanged": True,
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
    print(
        json.dumps(
            {
                "status": result["status"],
                "sources": result["sources"],
                "combined": result[
                    "combined_bonus_vs_original_depth_three"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
