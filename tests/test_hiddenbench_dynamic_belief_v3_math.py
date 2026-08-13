from __future__ import annotations

import pytest

from scripts.hiddenbench_dynamic_belief_v3_math import (
    CHANNEL_IDS,
    QUERY_IDS,
    branch_diagnostics,
    dynamic_depth_two,
    endpoint_metrics,
    fixed_depth_two,
    update,
)


def matrices():
    return {
        "Q1": [[0.70, 0.20, 0.10], [0.20, 0.60, 0.20], [0.10, 0.25, 0.65]],
        "Q2": [[0.60, 0.30, 0.10], [0.15, 0.70, 0.15], [0.20, 0.20, 0.60]],
        "Q3": [[0.55, 0.25, 0.20], [0.25, 0.55, 0.20], [0.20, 0.20, 0.60]],
        "Q4": [[0.65, 0.20, 0.15], [0.25, 0.60, 0.15], [0.15, 0.25, 0.60]],
    }


def refreshed(prior, values):
    result = {}
    for query_id in QUERY_IDS:
        result[query_id] = {}
        for channel_index, channel_id in enumerate(CHANNEL_IDS):
            exact = update(prior, values[query_id], channel_index)
            # Small deterministic deviation from fixed Bayes.
            shifted = [0.98 * value + 0.02 / len(exact) for value in exact]
            result[query_id][channel_id] = shifted
    return result


def test_dynamic_and_fixed_planners_are_finite() -> None:
    prior = [1 / 3] * 3
    values = matrices()
    dynamic = dynamic_depth_two(prior, values, refreshed(prior, values))
    fixed = fixed_depth_two(prior, values)
    assert dynamic["first_query_id"] in QUERY_IDS
    assert fixed["first_query_id"] in QUERY_IDS
    assert dynamic["margin"] >= 0
    assert set(dynamic["second_actions"]) == set(QUERY_IDS)


def test_branch_diagnostics_and_endpoint_definition() -> None:
    prior = [1 / 3] * 3
    diagnostics = branch_diagnostics(prior, matrices(), refreshed(prior, matrices()))
    assert diagnostics["obedient_branch_count"] == 12
    assert diagnostics["mean_exact_bayes_tv"] > 0
    metrics = endpoint_metrics([0.2, 0.7, 0.1], 1)
    assert metrics["brier"] == pytest.approx(0.14)
    assert metrics["correct_probability"] == 0.7
    with pytest.raises(ValueError, match="saturated"):
        endpoint_metrics([1.0, 0.0, 0.0], 0)
