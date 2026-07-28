from __future__ import annotations

import pytest

from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
    aggregate_tree_comparisons,
    cluster_bootstrap_interval,
)
from scripts.discoverphysics_dark_matter_structured_replication_v3 import (
    DefaultRoutingStructuredAdapter,
)


def test_seeded_adapter_adds_seed_without_changing_structured_route(
    monkeypatch,
):
    adapter = object.__new__(SeededStructuredAdapter)
    adapter.request_seed = 123

    def parent_payload(*args, **kwargs):
        del args, kwargs
        return {
            "provider": {"require_parameters": False},
            "reasoning": {"enabled": False, "exclude": True},
        }

    monkeypatch.setattr(
        DefaultRoutingStructuredAdapter,
        "_payload",
        parent_payload,
    )
    payload = adapter._payload([], 0.7, 1, response_format={"x": 1})

    assert payload["seed"] == 123
    assert payload["provider"] == {"require_parameters": False}
    assert payload["reasoning"]["enabled"] is False


def test_tree_cluster_bootstrap_is_reproducible():
    first = cluster_bootstrap_interval([-0.1, -0.2, -0.3], samples=200)
    second = cluster_bootstrap_interval([-0.1, -0.2, -0.3], samples=200)

    assert first == second
    assert first[1] < 0.0


def _tree(candidate: float, baseline: float) -> dict:
    comparison = {
        "candidate_minus_baseline_brier": candidate - baseline,
        "candidate_minus_baseline_hamming": candidate - baseline,
        "coverage_difference": 0.1,
    }
    return {
        "endpoint": {
            "predictive_bayes_risk": {
                "mean_posterior_predictive_brier": candidate,
                "mean_best_hamming_error": candidate,
            },
            "myopic_eig": {
                "mean_posterior_predictive_brier": baseline,
                "mean_best_hamming_error": baseline,
            },
        },
        "comparisons": {"myopic_eig": comparison},
    }


def test_aggregate_tree_comparisons_equal_weights_trees():
    result = aggregate_tree_comparisons(
        [_tree(0.1, 0.2), _tree(0.2, 0.4)],
        baseline="myopic_eig",
    )

    assert result["candidate_mean_brier"] == pytest.approx(0.15)
    assert result["baseline_mean_brier"] == pytest.approx(0.3)
    assert result["relative_brier_reduction"] == pytest.approx(0.5)
    assert result["brier_tree_wins"] == 2
