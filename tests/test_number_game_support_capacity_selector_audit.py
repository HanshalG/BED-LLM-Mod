from __future__ import annotations

import pytest

from scripts import number_game_support_capacity_selector_audit as audit


def _root(root: int, predicted: float, capacity: float) -> dict[str, float]:
    return {
        "root": root,
        "predicted_brier": predicted,
        "mean_union_size": capacity,
    }


def test_capacity_selector_rewards_larger_future_support() -> None:
    rows = [
        _root(0, 0.10, 10.0),
        _root(1, 0.11, 30.0),
        _root(2, 0.20, 20.0),
    ]
    selected, scores = audit.select_capacity_root(rows, coefficient=-2.0)
    assert selected == 1
    assert scores[1] < scores[0]


def test_zero_capacity_coefficient_reproduces_risk_selection() -> None:
    rows = [
        _root(0, 0.10, 10.0),
        _root(1, 0.11, 30.0),
    ]
    selected, _ = audit.select_capacity_root(rows, coefficient=0.0)
    assert selected == 0


def test_comparison_summary_uses_opposite_brier_and_coverage_directions(
    monkeypatch,
) -> None:
    monkeypatch.setattr(audit, "BOOTSTRAP_SAMPLES", 200)
    rows = [
        {
            "source": "a",
            "candidate_root": 1,
            "baseline_root": 0,
            "brier_difference": -0.02,
            "coverage_difference": 0.10,
        },
        {
            "source": "a",
            "candidate_root": 1,
            "baseline_root": 0,
            "brier_difference": -0.01,
            "coverage_difference": 0.05,
        },
    ]
    summary = audit.summarize_comparison(
        rows,
        seed=7,
        stratified=False,
    )
    assert summary["mean_candidate_minus_baseline_brier"] == pytest.approx(
        -0.015
    )
    assert summary["mean_candidate_minus_baseline_coverage"] == pytest.approx(
        0.075
    )
    assert summary["brier_wins_ties_losses"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
    }
    assert summary["coverage_wins_ties_losses"] == {
        "wins": 2,
        "ties": 0,
        "losses": 0,
    }
