#!/usr/bin/env python3
"""Audit whether routed Number Game support approximates canonical belief."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_generator_aware_bed import (
    DOMAIN,
    RuleHypothesis,
    best_query,
)
from scripts.number_game_ranking_fidelity_audit import spearman_correlation
from scripts.number_game_retained_depth_three import (
    _first_branches,
    _rule,
    _second_branches,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-dynamic-support-quality96-1"
SOURCE_DIR = (
    REPO_ROOT
    / "results/nonmyopic/number_game_qwen_dynamic_vs_fixed_resilient96_v2"
    / "number-game-qwen-dynamic-vs-fixed-resilient96-v2-20260730T023000Z"
)
SOURCE_RESULT = SOURCE_DIR / "RESULT.json"
SOURCE_TREES = SOURCE_DIR / "TREES.json"
SOURCE_TARGETS = SOURCE_DIR / "TARGETS.json"
SOURCE_RESULT_SHA256 = (
    "04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e"
)
SOURCE_TREES_SHA256 = (
    "8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82"
)
SOURCE_TARGETS_SHA256 = (
    "2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6"
)
TREE_COUNT = 96
ROOTS_PER_TREE = 8
TARGET_COUNT = 33
BOOTSTRAP_SEED = 87_000
BOOTSTRAP_SAMPLES = 20_000
SUPPORT_NAMES = ("fixed", "dynamic", "blind_pool")
STAGE_NAMES = ("first", "second")


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


def _dedupe(
    hypotheses: Iterable[RuleHypothesis],
) -> list[RuleHypothesis]:
    by_extension = {}
    for hypothesis in hypotheses:
        by_extension.setdefault(hypothesis.extension, hypothesis)
    return list(by_extension.values())


def _consistent(
    hypotheses: Iterable[RuleHypothesis],
    observations: Sequence[tuple[int, bool]],
) -> list[RuleHypothesis]:
    return _dedupe(
        hypothesis
        for hypothesis in hypotheses
        if all(
            hypothesis.extension[query] == label
            for query, label in observations
        )
    )


def support_metric(
    *,
    support: Sequence[RuleHypothesis],
    exact_support: Sequence[RuleHypothesis],
    truth: RuleHypothesis,
    queried: Sequence[int],
) -> dict[str, float]:
    if not exact_support:
        raise ValueError("exact canonical posterior is empty")
    excluded = set(queried)
    prediction_domain = [number for number in DOMAIN if number not in excluded]
    if not prediction_domain:
        raise ValueError("query history exhausts the evaluation domain")

    def probabilities(
        hypotheses: Sequence[RuleHypothesis],
    ) -> list[float]:
        return [
            sum(
                float(hypothesis.extension[number])
                for hypothesis in hypotheses
            )
            / len(hypotheses)
            for number in prediction_domain
        ]

    exact_probabilities = probabilities(exact_support)
    if support:
        approximate_probabilities = probabilities(support)
        mse = _mean(
            [
                (approximate - exact) ** 2
                for approximate, exact in zip(
                    approximate_probabilities,
                    exact_probabilities,
                    strict=True,
                )
            ]
        )
        extensions = {hypothesis.extension for hypothesis in support}
    else:
        mse = 1.0
        extensions = set()
    return {
        "posterior_predictive_mse": mse,
        "truth_extension_coverage": float(truth.extension in extensions),
        "support_size": float(len(support)),
    }


def _average_metrics(
    rows: Sequence[dict[str, float]],
) -> dict[str, float]:
    return {
        metric: _mean([float(row[metric]) for row in rows])
        for metric in (
            "posterior_predictive_mse",
            "truth_extension_coverage",
            "support_size",
        )
    }


def _pooled_first(
    generated_first: dict[tuple[int, bool], Sequence[RuleHypothesis]],
    root: int,
) -> list[RuleHypothesis]:
    return _dedupe(
        hypothesis
        for label in (False, True)
        for hypothesis in generated_first[(root, label)]
    )


def _pooled_second(
    generated_second: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
    root: int,
) -> list[RuleHypothesis]:
    return _dedupe(
        hypothesis
        for branch, support in generated_second.items()
        if branch[0] == root
        for hypothesis in support
    )


def evaluate_root(
    *,
    root: int,
    initial: Sequence[RuleHypothesis],
    canonical_targets: Sequence[RuleHypothesis],
    dynamic_first: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    dynamic_second: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
    generated_first: dict[
        tuple[int, bool], Sequence[RuleHypothesis]
    ],
    generated_second: dict[
        tuple[int, bool, int, bool], Sequence[RuleHypothesis]
    ],
) -> dict[str, Any]:
    pooled_first = _pooled_first(generated_first, root)
    pooled_second = _pooled_second(generated_second, root)
    target_rows = []
    for truth in canonical_targets:
        first_label = truth.extension[root]
        first_observations = ((root, first_label),)
        fixed_first = _consistent(initial, first_observations)
        routed_first = _dedupe(dynamic_first[(root, first_label)])
        blind_first = _consistent(
            [*initial, *pooled_first],
            first_observations,
        )
        exact_first = _consistent(canonical_targets, first_observations)
        if not {
            hypothesis.extension for hypothesis in routed_first
        }.issubset(
            {hypothesis.extension for hypothesis in blind_first}
        ):
            raise ValueError("routed first support is not in blind pool")

        second_query, _ = best_query(routed_first, excluded=(root,))
        second_label = truth.extension[second_query]
        second_observations = (
            (root, first_label),
            (second_query, second_label),
        )
        branch = (root, first_label, second_query, second_label)
        if branch not in dynamic_second:
            raise ValueError(f"missing stored dynamic branch {branch}")
        fixed_second = _consistent(initial, second_observations)
        routed_second = _dedupe(dynamic_second[branch])
        blind_second = _consistent(
            [*initial, *pooled_first, *pooled_second],
            second_observations,
        )
        exact_second = _consistent(canonical_targets, second_observations)
        if not {
            hypothesis.extension for hypothesis in routed_second
        }.issubset(
            {hypothesis.extension for hypothesis in blind_second}
        ):
            raise ValueError("routed second support is not in blind pool")

        target_rows.append(
            {
                "target": truth.name,
                "first_label": first_label,
                "second_query": second_query,
                "second_label": second_label,
                "first": {
                    "fixed": support_metric(
                        support=fixed_first,
                        exact_support=exact_first,
                        truth=truth,
                        queried=(root,),
                    ),
                    "dynamic": support_metric(
                        support=routed_first,
                        exact_support=exact_first,
                        truth=truth,
                        queried=(root,),
                    ),
                    "blind_pool": support_metric(
                        support=blind_first,
                        exact_support=exact_first,
                        truth=truth,
                        queried=(root,),
                    ),
                },
                "second": {
                    "fixed": support_metric(
                        support=fixed_second,
                        exact_support=exact_second,
                        truth=truth,
                        queried=(root, second_query),
                    ),
                    "dynamic": support_metric(
                        support=routed_second,
                        exact_support=exact_second,
                        truth=truth,
                        queried=(root, second_query),
                    ),
                    "blind_pool": support_metric(
                        support=blind_second,
                        exact_support=exact_second,
                        truth=truth,
                        queried=(root, second_query),
                    ),
                },
            }
        )
    stages = {
        stage: {
            support: _average_metrics(
                [row[stage][support] for row in target_rows]
            )
            for support in SUPPORT_NAMES
        }
        for stage in STAGE_NAMES
    }
    for stage in STAGE_NAMES:
        dynamic_mse = stages[stage]["dynamic"][
            "posterior_predictive_mse"
        ]
        stages[stage]["differences"] = {
            "dynamic_minus_fixed_predictive_mse": (
                dynamic_mse
                - stages[stage]["fixed"]["posterior_predictive_mse"]
            ),
            "dynamic_minus_blind_predictive_mse": (
                dynamic_mse
                - stages[stage]["blind_pool"][
                    "posterior_predictive_mse"
                ]
            ),
            "dynamic_minus_fixed_truth_coverage": (
                stages[stage]["dynamic"]["truth_extension_coverage"]
                - stages[stage]["fixed"]["truth_extension_coverage"]
            ),
            "dynamic_minus_blind_truth_coverage": (
                stages[stage]["dynamic"]["truth_extension_coverage"]
                - stages[stage]["blind_pool"]["truth_extension_coverage"]
            ),
        }
    return {
        "root": root,
        "stages": stages,
        "canonical_target_path_count": len(target_rows),
    }


def analyze_tree(
    public_tree: dict[str, Any],
    scored_tree: dict[str, Any],
    canonical_targets: Sequence[RuleHypothesis],
) -> dict[str, Any]:
    tree_seed = int(public_tree["tree_seed"])
    if tree_seed != int(scored_tree["tree_seed"]):
        raise ValueError("public and scored tree seeds differ")
    roots = [int(root) for root in public_tree["roots"]]
    if len(roots) != ROOTS_PER_TREE or len(set(roots)) != ROOTS_PER_TREE:
        raise ValueError(f"tree {tree_seed} does not have eight unique roots")
    selection = scored_tree["selection"]
    dynamic_root = int(selection["crossfit_depth_three_root"])
    fixed_root = int(selection["fixed_support_depth_three_root"])
    if dynamic_root not in roots or fixed_root not in roots:
        raise ValueError(f"tree {tree_seed} selected root is not a candidate")

    initial = [_rule(item) for item in public_tree["initial"]]
    dynamic_first = _first_branches(public_tree)
    dynamic_second = _second_branches(public_tree, key="second_branches")
    generated_first = _first_branches(
        public_tree,
        key="generated_first_branches",
    )
    generated_second = _second_branches(
        public_tree,
        key="generated_second_branches",
    )
    root_rows = {
        root: evaluate_root(
            root=root,
            initial=initial,
            canonical_targets=canonical_targets,
            dynamic_first=dynamic_first,
            dynamic_second=dynamic_second,
            generated_first=generated_first,
            generated_second=generated_second,
        )
        for root in roots
    }
    tree_stages = {}
    for stage in STAGE_NAMES:
        tree_stages[stage] = {
            support: {
                metric: _mean(
                    [
                        root_rows[root]["stages"][stage][support][metric]
                        for root in roots
                    ]
                )
                for metric in (
                    "posterior_predictive_mse",
                    "truth_extension_coverage",
                    "support_size",
                )
            }
            for support in SUPPORT_NAMES
        }
        tree_stages[stage]["differences"] = {
            key: _mean(
                [
                    root_rows[root]["stages"][stage]["differences"][key]
                    for root in roots
                ]
            )
            for key in root_rows[roots[0]]["stages"][stage][
                "differences"
            ]
        }

    def quality_gain(root: int) -> float:
        stage = root_rows[root]["stages"]["second"]
        return (
            stage["fixed"]["posterior_predictive_mse"]
            - stage["dynamic"]["posterior_predictive_mse"]
        )

    realized = {
        int(root): float(value)
        for root, value in scored_tree["per_root_endpoint_brier"].items()
    }
    if set(realized) != set(roots):
        raise ValueError(f"tree {tree_seed} endpoint roots changed")
    roots_differ = dynamic_root != fixed_root
    selected = {
        "dynamic_root": dynamic_root,
        "fixed_root": fixed_root,
        "roots_differ": roots_differ,
        "dynamic_root_refresh_quality_gain": quality_gain(dynamic_root),
        "fixed_root_refresh_quality_gain": quality_gain(fixed_root),
        "dynamic_minus_fixed_root_refresh_quality_gain": (
            quality_gain(dynamic_root) - quality_gain(fixed_root)
        ),
        "dynamic_root_dynamic_minus_blind_predictive_mse": (
            root_rows[dynamic_root]["stages"]["second"]["differences"][
                "dynamic_minus_blind_predictive_mse"
            ]
        ),
        "fixed_root_dynamic_minus_blind_predictive_mse": (
            root_rows[fixed_root]["stages"]["second"]["differences"][
                "dynamic_minus_blind_predictive_mse"
            ]
        ),
        "realized_advantage": (
            realized[fixed_root] - realized[dynamic_root]
            if roots_differ
            else 0.0
        ),
    }
    return {
        "tree_index": int(public_tree["tree_index"]),
        "tree_seed": tree_seed,
        "roots": roots,
        "tree_mean": tree_stages,
        "selected_roots": selected,
        "root_rows": {str(root): root_rows[root] for root in roots},
    }


def _bootstrap_summary(
    rows: Sequence[dict[str, Any]],
    *,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    rng = random.Random(seed)
    changed = [
        row for row in rows if row["selected_roots"]["roots_differ"]
    ]
    if not changed:
        raise ValueError("no changed-root trees")
    replicate_values: dict[str, list[float]] = {
        f"{stage}_{comparison}_95pct": []
        for stage in STAGE_NAMES
        for comparison in (
            "dynamic_minus_fixed_predictive_mse",
            "dynamic_minus_blind_predictive_mse",
        )
    }
    replicate_values.update(
        {
            "changed_root_quality_gain_contrast_95pct": [],
            "quality_gain_contrast_to_realized_spearman_95pct": [],
        }
    )
    for _ in range(samples):
        sampled = [rng.choice(rows) for _ in rows]
        sampled_changed = [rng.choice(changed) for _ in changed]
        for stage in STAGE_NAMES:
            for comparison in (
                "dynamic_minus_fixed_predictive_mse",
                "dynamic_minus_blind_predictive_mse",
            ):
                replicate_values[f"{stage}_{comparison}_95pct"].append(
                    _mean(
                        [
                            row["tree_mean"][stage]["differences"][
                                comparison
                            ]
                            for row in sampled
                        ]
                    )
                )
        contrasts = [
            row["selected_roots"][
                "dynamic_minus_fixed_root_refresh_quality_gain"
            ]
            for row in sampled_changed
        ]
        advantages = [
            row["selected_roots"]["realized_advantage"]
            for row in sampled_changed
        ]
        replicate_values[
            "changed_root_quality_gain_contrast_95pct"
        ].append(_mean(contrasts))
        replicate_values[
            "quality_gain_contrast_to_realized_spearman_95pct"
        ].append(spearman_correlation(contrasts, advantages))
    return {
        key: _interval(values)
        for key, values in replicate_values.items()
    }


def summarize_rows(
    rows: Sequence[dict[str, Any]],
    *,
    bootstrap_seed: int,
    bootstrap_samples: int,
) -> dict[str, Any]:
    if len(rows) != TREE_COUNT:
        raise ValueError(f"expected {TREE_COUNT} rows")
    stages = {}
    for stage in STAGE_NAMES:
        stages[stage] = {
            support: {
                metric: _mean(
                    [
                        row["tree_mean"][stage][support][metric]
                        for row in rows
                    ]
                )
                for metric in (
                    "posterior_predictive_mse",
                    "truth_extension_coverage",
                    "support_size",
                )
            }
            for support in SUPPORT_NAMES
        }
        stages[stage]["differences"] = {
            key: _mean(
                [
                    row["tree_mean"][stage]["differences"][key]
                    for row in rows
                ]
            )
            for key in rows[0]["tree_mean"][stage]["differences"]
        }
    changed = [
        row for row in rows if row["selected_roots"]["roots_differ"]
    ]
    contrasts = [
        row["selected_roots"][
            "dynamic_minus_fixed_root_refresh_quality_gain"
        ]
        for row in changed
    ]
    advantages = [
        row["selected_roots"]["realized_advantage"] for row in changed
    ]
    selected = {
        "changed_root_tree_count": len(changed),
        "mean_dynamic_root_refresh_quality_gain": _mean(
            [
                row["selected_roots"][
                    "dynamic_root_refresh_quality_gain"
                ]
                for row in changed
            ]
        ),
        "mean_fixed_root_refresh_quality_gain": _mean(
            [
                row["selected_roots"]["fixed_root_refresh_quality_gain"]
                for row in changed
            ]
        ),
        "mean_dynamic_minus_fixed_root_refresh_quality_gain": _mean(
            contrasts
        ),
        "quality_gain_contrast_to_realized_advantage_spearman": (
            spearman_correlation(contrasts, advantages)
        ),
        "mean_dynamic_root_dynamic_minus_blind_predictive_mse": _mean(
            [
                row["selected_roots"][
                    "dynamic_root_dynamic_minus_blind_predictive_mse"
                ]
                for row in rows
            ]
        ),
        "mean_fixed_root_dynamic_minus_blind_predictive_mse": _mean(
            [
                row["selected_roots"][
                    "fixed_root_dynamic_minus_blind_predictive_mse"
                ]
                for row in rows
            ]
        ),
    }
    bootstrap = _bootstrap_summary(
        rows,
        seed=bootstrap_seed,
        samples=bootstrap_samples,
    )
    conditions = {
        "second_stage_dynamic_mse_ci_below_fixed": (
            bootstrap[
                "second_dynamic_minus_fixed_predictive_mse_95pct"
            ][1]
            < 0.0
        ),
        "second_stage_dynamic_mse_ci_below_history_blind": (
            bootstrap[
                "second_dynamic_minus_blind_predictive_mse_95pct"
            ][1]
            < 0.0
        ),
        "changed_root_refresh_quality_gain_contrast_ci_above_zero": (
            bootstrap[
                "changed_root_quality_gain_contrast_95pct"
            ][0]
            > 0.0
        ),
    }
    blind_pool_exactly_matches_dynamic = all(
        row["tree_mean"][stage]["differences"][
            "dynamic_minus_blind_predictive_mse"
        ]
        == 0.0
        and row["tree_mean"][stage]["differences"][
            "dynamic_minus_blind_truth_coverage"
        ]
        == 0.0
        and row["tree_mean"][stage]["dynamic"]["support_size"]
        == row["tree_mean"][stage]["blind_pool"]["support_size"]
        for row in rows
        for stage in STAGE_NAMES
    )
    return {
        "stages": stages,
        "selected_root_quality": selected,
        "history_blind_control_diagnostic": {
            "exactly_matches_dynamic_support": (
                blind_pool_exactly_matches_dynamic
            ),
            "interpretation": (
                "branch-consistency parsing makes counterfactual-branch "
                "hypotheses inconsistent with the realized history, so "
                "post-pooling consistency filtering recovers routed support"
            ),
        },
        "bootstrap": bootstrap,
        "directional_coherence_conditions": conditions,
        "directionally_coherent": all(conditions.values()),
    }


def load_and_validate_sources(
    *,
    result_path: Path = SOURCE_RESULT,
    trees_path: Path = SOURCE_TREES,
    targets_path: Path = SOURCE_TARGETS,
) -> tuple[dict[str, Any], dict[str, Any], list[RuleHypothesis]]:
    for path, expected, label in (
        (result_path, SOURCE_RESULT_SHA256, "RESULT"),
        (trees_path, SOURCE_TREES_SHA256, "TREES"),
        (targets_path, SOURCE_TARGETS_SHA256, "TARGETS"),
    ):
        if sha256_file(path) != expected:
            raise ValueError(f"source {label}.json hash changed")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    trees = json.loads(trees_path.read_text(encoding="utf-8"))
    targets_document = json.loads(targets_path.read_text(encoding="utf-8"))
    if len(result.get("trees") or []) != TREE_COUNT:
        raise ValueError("source result tree count changed")
    if len(trees.get("trees") or []) != TREE_COUNT:
        raise ValueError("source public tree count changed")
    result_seeds = [int(tree["tree_seed"]) for tree in result["trees"]]
    public_seeds = [int(tree["tree_seed"]) for tree in trees["trees"]]
    if result_seeds != public_seeds or len(set(result_seeds)) != TREE_COUNT:
        raise ValueError("source tree seeds changed or are not unique")
    canonical_targets = [
        _rule(item) for item in targets_document.get("targets") or []
    ]
    if (
        len(canonical_targets) != TARGET_COUNT
        or len(
            {target.extension for target in canonical_targets}
        )
        != TARGET_COUNT
    ):
        raise ValueError("canonical target count or uniqueness changed")
    for item, target in zip(
        targets_document["targets"],
        canonical_targets,
        strict=True,
    ):
        if target.public_dict()["extension_sha256"] != item[
            "extension_sha256"
        ]:
            raise ValueError("canonical target extension hash changed")
    return result, trees, canonical_targets


def run_analysis(
    output_dir: Path,
    *,
    result_path: Path = SOURCE_RESULT,
    trees_path: Path = SOURCE_TREES,
    targets_path: Path = SOURCE_TARGETS,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    source_result, source_trees, canonical_targets = (
        load_and_validate_sources(
            result_path=result_path,
            trees_path=trees_path,
            targets_path=targets_path,
        )
    )
    rows = [
        analyze_tree(public_tree, scored_tree, canonical_targets)
        for public_tree, scored_tree in zip(
            source_trees["trees"],
            source_result["trees"],
            strict=True,
        )
    ]
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "retrospective_support_quality_audit",
        "protocol": {
            "analysis_is_retrospective": True,
            "source_status_unchanged": True,
            "cannot_rescue_or_relabel_source": True,
            "model_calls": 0,
            "cost_usd": 0.0,
            "tree_count": TREE_COUNT,
            "candidate_roots_per_tree": ROOTS_PER_TREE,
            "canonical_target_count": TARGET_COUNT,
            "canonical_target_weighting": "uniform",
            "source_result_sha256": SOURCE_RESULT_SHA256,
            "source_trees_sha256": SOURCE_TREES_SHA256,
            "source_targets_sha256": SOURCE_TARGETS_SHA256,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_samples": bootstrap_samples,
            "history_blind_control": (
                "all generated hypotheses under the root pooled before "
                "filtering by the realized dynamic-policy history"
            ),
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
