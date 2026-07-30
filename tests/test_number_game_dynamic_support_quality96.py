from __future__ import annotations

from copy import deepcopy

import pytest

from scripts import number_game_dynamic_support_quality96 as quality
from scripts.number_game_generator_aware_bed import RuleHypothesis


def _hypothesis(name: str, positives: set[int]) -> RuleHypothesis:
    return RuleHypothesis(
        name=name,
        expression="n == n",
        extension=tuple(number in positives for number in quality.DOMAIN),
    )


def _stage(
    dynamic_minus_fixed: float,
    dynamic_minus_blind: float,
) -> dict:
    return {
        "fixed": {
            "posterior_predictive_mse": 0.2,
            "truth_extension_coverage": 0.4,
            "support_size": 4.0,
        },
        "dynamic": {
            "posterior_predictive_mse": 0.2 + dynamic_minus_fixed,
            "truth_extension_coverage": 0.6,
            "support_size": 6.0,
        },
        "blind_pool": {
            "posterior_predictive_mse": (
                0.2 + dynamic_minus_fixed - dynamic_minus_blind
            ),
            "truth_extension_coverage": 0.8,
            "support_size": 8.0,
        },
        "differences": {
            "dynamic_minus_fixed_predictive_mse": dynamic_minus_fixed,
            "dynamic_minus_blind_predictive_mse": dynamic_minus_blind,
            "dynamic_minus_fixed_truth_coverage": 0.2,
            "dynamic_minus_blind_truth_coverage": -0.2,
        },
    }


def _row(
    *,
    contrast: float = 0.02,
    realized: float = 0.01,
) -> dict:
    stage = _stage(-0.03, -0.01)
    return {
        "tree_index": 0,
        "tree_seed": 80000,
        "tree_mean": {
            "first": deepcopy(stage),
            "second": deepcopy(stage),
        },
        "selected_roots": {
            "dynamic_root": 1,
            "fixed_root": 2,
            "roots_differ": True,
            "dynamic_root_refresh_quality_gain": 0.04,
            "fixed_root_refresh_quality_gain": 0.04 - contrast,
            "dynamic_minus_fixed_root_refresh_quality_gain": contrast,
            "dynamic_root_dynamic_minus_blind_predictive_mse": -0.01,
            "fixed_root_dynamic_minus_blind_predictive_mse": -0.005,
            "realized_advantage": realized,
        },
    }


def test_support_metric_matches_hand_computed_predictive_error() -> None:
    truth = _hypothesis("truth", {0, 2})
    other = _hypothesis("other", {0, 3})
    approximation = _hypothesis("approximation", {0, 2, 3})

    metric = quality.support_metric(
        support=[truth, approximation],
        exact_support=[truth, other],
        truth=truth,
        queried=(0,),
    )

    # Only n=2 differs, by 0.5, over 100 unqueried values.
    assert metric["posterior_predictive_mse"] == pytest.approx(0.0025)
    assert metric["truth_extension_coverage"] == 1.0
    assert metric["support_size"] == 2.0


def test_history_blind_pool_contains_routed_support_after_filtering() -> None:
    initial = [_hypothesis("initial", {0, 1})]
    routed = _hypothesis("routed", {0, 2})
    other_branch_compatible = _hypothesis("other", {0, 3})
    generated = {
        (0, True): [routed],
        (0, False): [other_branch_compatible],
    }

    pooled = quality._consistent(
        [*initial, *quality._pooled_first(generated, 0)],
        ((0, True),),
    )

    extensions = {hypothesis.extension for hypothesis in pooled}
    assert routed.extension in extensions
    assert other_branch_compatible.extension in extensions


def test_bootstrap_and_summary_follow_frozen_signs() -> None:
    rows = [
        _row(
            contrast=0.01 + index / 100000,
            realized=0.02 + index / 100000,
        )
        for index in range(quality.TREE_COUNT)
    ]

    first = quality.summarize_rows(
        rows,
        bootstrap_seed=7,
        bootstrap_samples=100,
    )
    second = quality.summarize_rows(
        rows,
        bootstrap_seed=7,
        bootstrap_samples=100,
    )

    assert first == second
    assert first["directionally_coherent"] is True
    assert (
        first["bootstrap"][
            "second_dynamic_minus_fixed_predictive_mse_95pct"
        ][1]
        < 0.0
    )
    assert (
        first["bootstrap"][
            "changed_root_quality_gain_contrast_95pct"
        ][0]
        > 0.0
    )


def test_summarize_rows_requires_complete_source_cohort() -> None:
    with pytest.raises(ValueError, match="expected 96 rows"):
        quality.summarize_rows(
            [_row()],
            bootstrap_seed=7,
            bootstrap_samples=10,
        )
