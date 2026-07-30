#!/usr/bin/env python3
"""Audit the first-link mechanism in the fresh dynamic-support96 result."""

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
from scripts.number_game_generator_aware_bed import (
    choose_predictive_bayes_risk_root,
)
from scripts.number_game_ranking_fidelity_audit import (
    pairwise_concordance,
    spearman_correlation,
)
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-dynamic-support-mechanism96-1"
SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_dynamic_vs_fixed_resilient96_v2"
    / "number-game-qwen-dynamic-vs-fixed-resilient96-v2-20260730T023000Z"
)
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_RESULT_SHA256 = (
    "04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e"
)
SOURCE_TREES_SHA256 = (
    "8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82"
)
TREE_COUNT = 96
ROOTS_PER_TREE = 8
BOOTSTRAP_SEED = 86_000
BOOTSTRAP_SAMPLES = 20_000
TIE_TOLERANCE = 1e-15


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _mean(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("cannot average an empty sequence")
    return sum(values) / len(values)


def _quantile(values: Sequence[float], probability: float) -> float:
    if not values:
        raise ValueError("cannot take a quantile of an empty sequence")
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _interval(values: Sequence[float]) -> list[float]:
    return [_quantile(values, 0.025), _quantile(values, 0.975)]


def _mean_field(rows: Sequence[dict[str, Any]], key: str) -> float:
    return _mean([float(row[key]) for row in rows])


def _fixed_risk_map(
    public_tree: dict[str, Any],
) -> tuple[dict[str, float], int]:
    initial = [_rule(item) for item in public_tree["initial"]]
    roots = [int(root) for root in public_tree["roots"]]
    first, second = depth.static_depth_three_branches(
        support=initial,
        roots=roots,
    )
    scores = depth.depth_three_scores(
        support=initial,
        roots=roots,
        first_branches=first,
        second_branches=second,
    )
    selected = choose_predictive_bayes_risk_root(scores)
    return (
        {
            str(root): float(
                scores[root]["mean_posterior_predictive_brier"]
            )
            for root in roots
        },
        selected,
    )


def _novelty_row(
    *,
    stage: str,
    root: int,
    parent_extensions: set[tuple[bool, ...]],
    generated_extensions: set[tuple[bool, ...]],
) -> dict[str, Any]:
    novel = generated_extensions - parent_extensions
    generated_count = len(generated_extensions)
    return {
        "stage": stage,
        "root": root,
        "generated_count": generated_count,
        "parent_consistent_count": len(parent_extensions),
        "novel_count": len(novel),
        "novel_fraction": (
            len(novel) / generated_count if generated_count else 0.0
        ),
    }


def support_novelty(
    public_tree: dict[str, Any],
) -> dict[str, Any]:
    initial = [_rule(item) for item in public_tree["initial"]]
    first = _first_branches(public_tree)
    generated_first = _first_branches(
        public_tree,
        key="generated_first_branches",
    )
    generated_second = _second_branches(
        public_tree,
        key="generated_second_branches",
    )
    rows = []
    for (root, label), generated in sorted(generated_first.items()):
        parent = {
            hypothesis.extension
            for hypothesis in initial
            if hypothesis.extension[root] == label
        }
        rows.append(
            _novelty_row(
                stage="first",
                root=root,
                parent_extensions=parent,
                generated_extensions={
                    hypothesis.extension for hypothesis in generated
                },
            )
        )
    for (
        root,
        first_label,
        second_query,
        second_label,
    ), generated in sorted(generated_second.items()):
        parent = {
            hypothesis.extension
            for hypothesis in first[(root, first_label)]
            if hypothesis.extension[second_query] == second_label
        }
        rows.append(
            _novelty_row(
                stage="second",
                root=root,
                parent_extensions=parent,
                generated_extensions={
                    hypothesis.extension for hypothesis in generated
                },
            )
        )

    roots = [int(root) for root in public_tree["roots"]]
    by_root = {}
    for root in roots:
        root_rows = [row for row in rows if row["root"] == root]
        first_rows = [row for row in root_rows if row["stage"] == "first"]
        second_rows = [row for row in root_rows if row["stage"] == "second"]
        by_root[str(root)] = {
            "first_mean_novel_count": _mean_field(
                first_rows, "novel_count"
            ),
            "first_mean_novel_fraction": _mean_field(
                first_rows, "novel_fraction"
            ),
            "second_mean_novel_count": _mean_field(
                second_rows, "novel_count"
            ),
            "second_mean_novel_fraction": _mean_field(
                second_rows, "novel_fraction"
            ),
            "combined_mean_novel_count": _mean_field(
                root_rows, "novel_count"
            ),
            "combined_mean_novel_fraction": _mean_field(
                root_rows, "novel_fraction"
            ),
        }
    return {"branches": rows, "by_root": by_root}


def _rank_metrics(
    estimated: dict[str, float],
    realized: dict[str, float],
    roots: Sequence[str],
) -> dict[str, float]:
    estimates = [float(estimated[root]) for root in roots]
    outcomes = [float(realized[root]) for root in roots]
    return {
        "spearman": spearman_correlation(estimates, outcomes),
        "pairwise_concordance": pairwise_concordance(
            estimates,
            outcomes,
        ),
    }


def analyze_tree(
    public_tree: dict[str, Any],
    scored_tree: dict[str, Any],
) -> dict[str, Any]:
    tree_seed = int(public_tree["tree_seed"])
    if tree_seed != int(scored_tree["tree_seed"]):
        raise ValueError("public and scored tree seeds differ")
    roots = [str(int(root)) for root in public_tree["roots"]]
    if len(roots) != ROOTS_PER_TREE or len(set(roots)) != ROOTS_PER_TREE:
        raise ValueError(f"tree {tree_seed} does not have eight unique roots")

    selection = scored_tree["selection"]
    dynamic_risk = {
        str(root): float(value)
        for root, value in selection["crossfit_depth_three_brier"].items()
    }
    realized = {
        str(root): float(value)
        for root, value in scored_tree["per_root_endpoint_brier"].items()
    }
    fixed_risk, recomputed_fixed_root = _fixed_risk_map(public_tree)
    if set(dynamic_risk) != set(roots):
        raise ValueError(f"tree {tree_seed} dynamic-risk roots changed")
    if set(fixed_risk) != set(roots) or set(realized) != set(roots):
        raise ValueError(f"tree {tree_seed} fixed or endpoint roots changed")

    dynamic_root = str(int(selection["crossfit_depth_three_root"]))
    fixed_root = str(int(selection["fixed_support_depth_three_root"]))
    if int(fixed_root) != recomputed_fixed_root:
        raise ValueError(f"tree {tree_seed} fixed root did not reproduce")
    expected_dynamic_root = min(
        roots,
        key=lambda root: (dynamic_risk[root], int(root)),
    )
    if dynamic_root != expected_dynamic_root:
        raise ValueError(f"tree {tree_seed} dynamic root did not reproduce")

    novelty = support_novelty(public_tree)
    dynamic_novelty = novelty["by_root"][dynamic_root]
    fixed_novelty = novelty["by_root"][fixed_root]
    oracle_root = min(roots, key=lambda root: (realized[root], int(root)))
    roots_differ = dynamic_root != fixed_root
    row = {
        "tree_index": int(public_tree["tree_index"]),
        "tree_seed": tree_seed,
        "roots_differ": roots_differ,
        "dynamic_root": int(dynamic_root),
        "fixed_root": int(fixed_root),
        "oracle_root": int(oracle_root),
        "dynamic_rank": _rank_metrics(dynamic_risk, realized, roots),
        "fixed_rank": _rank_metrics(fixed_risk, realized, roots),
        "dynamic_predicted_advantage": (
            dynamic_risk[fixed_root] - dynamic_risk[dynamic_root]
        ),
        "fixed_counter_advantage": (
            fixed_risk[dynamic_root] - fixed_risk[fixed_root]
        ),
        "score_reversal_margin": (
            dynamic_risk[fixed_root]
            - dynamic_risk[dynamic_root]
            + fixed_risk[dynamic_root]
            - fixed_risk[fixed_root]
        ),
        "realized_advantage": (
            realized[fixed_root] - realized[dynamic_root]
        ),
        "dynamic_oracle_regret": (
            realized[dynamic_root] - realized[oracle_root]
        ),
        "fixed_oracle_regret": (
            realized[fixed_root] - realized[oracle_root]
        ),
        "dynamic_minus_fixed_oracle_regret": (
            realized[dynamic_root] - realized[fixed_root]
        ),
        "dynamic_selected_novelty": dynamic_novelty,
        "fixed_selected_novelty": fixed_novelty,
        "novelty_differences": {
            key: float(dynamic_novelty[key]) - float(fixed_novelty[key])
            for key in dynamic_novelty
        },
        "all_branch_novelty": novelty["branches"],
        "fixed_risk": fixed_risk,
    }
    if row["dynamic_predicted_advantage"] < -TIE_TOLERANCE:
        raise ValueError(f"tree {tree_seed} has negative dynamic advantage")
    if row["fixed_counter_advantage"] < -TIE_TOLERANCE:
        raise ValueError(f"tree {tree_seed} has negative fixed advantage")
    return row


def _ranking_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    dynamic_spearman = [
        float(row["dynamic_rank"]["spearman"]) for row in rows
    ]
    fixed_spearman = [
        float(row["fixed_rank"]["spearman"]) for row in rows
    ]
    dynamic_concordance = [
        float(row["dynamic_rank"]["pairwise_concordance"]) for row in rows
    ]
    fixed_concordance = [
        float(row["fixed_rank"]["pairwise_concordance"]) for row in rows
    ]
    return {
        "dynamic_mean_spearman": _mean(dynamic_spearman),
        "fixed_mean_spearman": _mean(fixed_spearman),
        "dynamic_minus_fixed_mean_spearman": _mean(
            [
                dynamic - fixed
                for dynamic, fixed in zip(
                    dynamic_spearman,
                    fixed_spearman,
                    strict=True,
                )
            ]
        ),
        "dynamic_mean_pairwise_concordance": _mean(dynamic_concordance),
        "fixed_mean_pairwise_concordance": _mean(fixed_concordance),
        "dynamic_minus_fixed_mean_pairwise_concordance": _mean(
            [
                dynamic - fixed
                for dynamic, fixed in zip(
                    dynamic_concordance,
                    fixed_concordance,
                    strict=True,
                )
            ]
        ),
    }


def _changed_root_summary(
    rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    changed = [row for row in rows if row["roots_differ"]]
    predicted = [
        float(row["dynamic_predicted_advantage"]) for row in changed
    ]
    fixed = [float(row["fixed_counter_advantage"]) for row in changed]
    reversal = [float(row["score_reversal_margin"]) for row in changed]
    realized = [float(row["realized_advantage"]) for row in changed]
    return {
        "tree_count": len(changed),
        "mean_dynamic_predicted_advantage": _mean(predicted),
        "mean_fixed_counter_advantage": _mean(fixed),
        "mean_score_reversal_margin": _mean(reversal),
        "mean_realized_advantage": _mean(realized),
        "wins": sum(value > TIE_TOLERANCE for value in realized),
        "ties": sum(abs(value) <= TIE_TOLERANCE for value in realized),
        "losses": sum(value < -TIE_TOLERANCE for value in realized),
        "dynamic_margin_to_realized_spearman": spearman_correlation(
            predicted,
            realized,
        ),
        "fixed_margin_to_realized_spearman": spearman_correlation(
            fixed,
            realized,
        ),
        "reversal_margin_to_realized_spearman": spearman_correlation(
            reversal,
            realized,
        ),
    }


def _regret_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        "mean_dynamic_oracle_regret": _mean_field(
            rows, "dynamic_oracle_regret"
        ),
        "mean_fixed_oracle_regret": _mean_field(
            rows, "fixed_oracle_regret"
        ),
        "mean_dynamic_minus_fixed_oracle_regret": _mean_field(
            rows, "dynamic_minus_fixed_oracle_regret"
        ),
        "dynamic_oracle_root_selections": sum(
            row["dynamic_root"] == row["oracle_root"] for row in rows
        ),
        "fixed_oracle_root_selections": sum(
            row["fixed_root"] == row["oracle_root"] for row in rows
        ),
    }


def _novelty_summary(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    branch_rows = [
        branch
        for row in rows
        for branch in row["all_branch_novelty"]
    ]
    changed = [row for row in rows if row["roots_differ"]]
    selected = {}
    for stage in ("first", "second", "combined"):
        for measure in ("novel_count", "novel_fraction"):
            key = f"{stage}_mean_{measure}"
            selected[f"dynamic_mean_{key}"] = _mean(
                [
                    float(row["dynamic_selected_novelty"][key])
                    for row in rows
                ]
            )
            selected[f"fixed_mean_{key}"] = _mean(
                [
                    float(row["fixed_selected_novelty"][key])
                    for row in rows
                ]
            )
            difference_key = f"{key}"
            selected[f"dynamic_minus_fixed_mean_{key}"] = _mean(
                [
                    float(row["novelty_differences"][difference_key])
                    for row in rows
                ]
            )
    correlations = {}
    realized = [float(row["realized_advantage"]) for row in changed]
    for stage in ("first", "second", "combined"):
        for measure in ("novel_count", "novel_fraction"):
            key = f"{stage}_mean_{measure}"
            correlations[
                f"{key}_difference_to_realized_spearman"
            ] = spearman_correlation(
                [
                    float(row["novelty_differences"][key])
                    for row in changed
                ],
                realized,
            )
    return {
        "branch_count": len(branch_rows),
        "first_branch_count": sum(
            row["stage"] == "first" for row in branch_rows
        ),
        "second_branch_count": sum(
            row["stage"] == "second" for row in branch_rows
        ),
        "all_branches_mean_novel_count": _mean_field(
            branch_rows, "novel_count"
        ),
        "all_branches_mean_novel_fraction": _mean_field(
            branch_rows, "novel_fraction"
        ),
        "selected_roots": selected,
        "changed_root_correlations": correlations,
    }


def bootstrap_analysis(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int = BOOTSTRAP_SEED,
    samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, list[float]]:
    rng = random.Random(seed)
    values: dict[str, list[float]] = {
        "dynamic_minus_fixed_mean_spearman": [],
        "dynamic_minus_fixed_mean_pairwise_concordance": [],
        "changed_root_mean_realized_advantage": [],
        "dynamic_margin_to_realized_spearman": [],
        "fixed_margin_to_realized_spearman": [],
        "reversal_margin_to_realized_spearman": [],
        "mean_dynamic_minus_fixed_oracle_regret": [],
    }
    for _ in range(samples):
        sample = [rng.choice(rows) for _ in rows]
        ranking = _ranking_summary(sample)
        changed = _changed_root_summary(sample)
        regret = _regret_summary(sample)
        values["dynamic_minus_fixed_mean_spearman"].append(
            ranking["dynamic_minus_fixed_mean_spearman"]
        )
        values[
            "dynamic_minus_fixed_mean_pairwise_concordance"
        ].append(
            ranking["dynamic_minus_fixed_mean_pairwise_concordance"]
        )
        values["changed_root_mean_realized_advantage"].append(
            changed["mean_realized_advantage"]
        )
        for key in (
            "dynamic_margin_to_realized_spearman",
            "fixed_margin_to_realized_spearman",
            "reversal_margin_to_realized_spearman",
        ):
            values[key].append(changed[key])
        values["mean_dynamic_minus_fixed_oracle_regret"].append(
            regret["mean_dynamic_minus_fixed_oracle_regret"]
        )
    return {f"{key}_95pct": _interval(value) for key, value in values.items()}


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int = BOOTSTRAP_SEED,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if len(rows) != TREE_COUNT:
        raise ValueError(f"expected {TREE_COUNT} rows, received {len(rows)}")
    ranking = _ranking_summary(rows)
    changed = _changed_root_summary(rows)
    regret = _regret_summary(rows)
    bootstrap = bootstrap_analysis(
        rows,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    coherent = {
        "changed_root_realized_advantage_ci_above_zero": (
            bootstrap[
                "changed_root_mean_realized_advantage_95pct"
            ][0]
            > 0.0
        ),
        "dynamic_ranking_exceeds_fixed_without_reversal": (
            (
                ranking["dynamic_minus_fixed_mean_spearman"] > 0.0
                or ranking[
                    "dynamic_minus_fixed_mean_pairwise_concordance"
                ]
                > 0.0
            )
            and ranking["dynamic_minus_fixed_mean_spearman"] >= 0.0
            and ranking[
                "dynamic_minus_fixed_mean_pairwise_concordance"
            ]
            >= 0.0
        ),
        "dynamic_oracle_regret_ci_below_fixed": (
            bootstrap[
                "mean_dynamic_minus_fixed_oracle_regret_95pct"
            ][1]
            < 0.0
        ),
    }
    return {
        "candidate_ranking": ranking,
        "changed_root_first_link": changed,
        "candidate_set_oracle_regret": regret,
        "support_regeneration": _novelty_summary(rows),
        "bootstrap": bootstrap,
        "directional_coherence_conditions": coherent,
        "directionally_coherent": all(coherent.values()),
    }


def load_and_validate_sources(
    result_path: Path = SOURCE_RESULT,
    trees_path: Path = SOURCE_TREES,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if sha256_file(result_path) != SOURCE_RESULT_SHA256:
        raise ValueError("source RESULT.json hash changed")
    if sha256_file(trees_path) != SOURCE_TREES_SHA256:
        raise ValueError("source TREES.json hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trees = json.loads(trees_path.read_text(encoding="utf-8"))
    if len(result.get("trees") or []) != TREE_COUNT:
        raise ValueError("source result tree count changed")
    if len(trees.get("trees") or []) != TREE_COUNT:
        raise ValueError("source public tree count changed")
    result_seeds = [int(tree["tree_seed"]) for tree in result["trees"]]
    public_seeds = [int(tree["tree_seed"]) for tree in trees["trees"]]
    if result_seeds != public_seeds or len(set(result_seeds)) != TREE_COUNT:
        raise ValueError("source tree seeds changed or are not unique")
    return result, trees


def run_analysis(
    output_dir: Path,
    *,
    result_path: Path = SOURCE_RESULT,
    trees_path: Path = SOURCE_TREES,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    source_result, source_trees = load_and_validate_sources(
        result_path,
        trees_path,
    )
    rows = [
        analyze_tree(public_tree, scored_tree)
        for public_tree, scored_tree in zip(
            source_trees["trees"],
            source_result["trees"],
            strict=True,
        )
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_mechanism_audit",
        "protocol": {
            "analysis_is_retrospective": True,
            "source_status_unchanged": True,
            "cannot_rescue_or_relabel_source": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "tree_count": TREE_COUNT,
            "candidate_roots_per_tree": ROOTS_PER_TREE,
            "endpoint": "exact 33-concept canonical bank",
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": bootstrap_samples,
            "tie_tolerance": TIE_TOLERANCE,
        },
        "analysis": summarize_rows(
            rows,
            bootstrap_seed=BOOTSTRAP_SEED,
            bootstrap_samples=bootstrap_samples,
        ),
        "trees": rows,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    result = run_analysis(args.output_dir)
    print(json.dumps(result["analysis"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
