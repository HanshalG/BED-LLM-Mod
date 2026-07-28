from __future__ import annotations

import pytest

from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
    TARGET_SEEDS,
    TREE_SEEDS,
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


def test_v2_uses_entirely_fresh_seed_ranges():
    assert TREE_SEEDS == tuple(range(26080, 26088))
    assert TARGET_SEEDS == tuple(range(26180, 26188))
    assert not set(TREE_SEEDS) & set(range(26070, 26078))
    assert not set(TARGET_SEEDS) & set(range(26170, 26178))


def test_zero_cost_provider_error_is_retried(monkeypatch):
    adapter = object.__new__(SeededStructuredAdapter)
    adapter.max_retries = 2
    adapter.backoff_seconds = 0.0
    adapter.retry_count = 0
    adapter.provider_error_retries = 0
    import threading

    adapter._usage_lock = threading.Lock()
    responses = iter(
        [
            {
                "choices": [{"finish_reason": "error"}],
                "usage": {"cost": 0.0},
            },
            {
                "choices": [{"finish_reason": "stop"}],
                "usage": {"cost": 0.1},
            },
        ]
    )
    monkeypatch.setattr(
        DefaultRoutingStructuredAdapter,
        "_post",
        lambda self, payload: next(responses),
    )

    result = adapter._post({"test": True})

    assert result["choices"][0]["finish_reason"] == "stop"
    assert adapter.retry_count == 1
    assert adapter.provider_error_retries == 1


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


def test_aggregate_tree_comparisons_accepts_candidate_key():
    tree = _tree(0.1, 0.2)
    tree["endpoint"]["depth_three"] = tree["endpoint"].pop(
        "predictive_bayes_risk"
    )

    result = aggregate_tree_comparisons(
        [tree],
        baseline="myopic_eig",
        candidate="depth_three",
    )

    assert result["candidate_mean_brier"] == pytest.approx(0.1)
