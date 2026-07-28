from __future__ import annotations

import pytest

from scripts.number_game_generator_aware_bed import (
    RuleHypothesis,
    compile_expression,
)
from scripts.number_game_multidraw_prior_development import (
    EXTRA_PRIOR_SEEDS,
    aggregate_comparisons,
    dedupe_support,
)


def _rule(name: str, expression: str) -> RuleHypothesis:
    return RuleHypothesis(name, expression, compile_expression(expression))


def test_extra_prior_seed_set_is_frozen_and_disjoint():
    assert EXTRA_PRIOR_SEEDS == tuple(range(26300, 26316))
    assert not set(EXTRA_PRIOR_SEEDS) & set(range(26080, 26088))


def test_dedupe_support_preserves_first_extension():
    even = _rule("even", "divisible(n, 2)")
    duplicate = _rule("also even", "n % 2 == 0")
    three = _rule("three", "divisible(n, 3)")

    result = dedupe_support([[even], [duplicate, three]])

    assert result == [even, three]


def _tree(candidate: float, baseline: float) -> dict:
    return {
        "endpoint": {
            "multidraw_predictive_risk": {
                "mean_posterior_predictive_brier": candidate,
            },
            "control": {
                "mean_posterior_predictive_brier": baseline,
            },
        },
        "comparisons": {
            "control": {
                "candidate_minus_baseline_brier": candidate - baseline,
                "coverage_difference": 0.1,
            }
        },
    }


def test_aggregate_comparisons_weights_trees_equally():
    result = aggregate_comparisons(
        [_tree(0.1, 0.2), _tree(0.2, 0.4)],
        "control",
    )

    assert result["candidate_mean_brier"] == pytest.approx(0.15)
    assert result["baseline_mean_brier"] == pytest.approx(0.3)
    assert result["relative_brier_reduction"] == pytest.approx(0.5)
    assert result["strict_tree_wins"] == 2
