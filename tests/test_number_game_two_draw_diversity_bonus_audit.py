from __future__ import annotations

import pytest

from scripts.number_game_two_draw_diversity_bonus_audit import (
    adjusted_root,
    comparison_summary,
    draw_disagreement,
)


def _extension(*positive: int) -> tuple[bool, ...]:
    return tuple(index in positive for index in range(5))


def test_draw_disagreement_uses_extension_jaccard_distance() -> None:
    shared = _extension(1)
    first_only = _extension(2)
    second_only = _extension(3)
    metrics = draw_disagreement(
        {shared, first_only},
        {shared, second_only},
    )

    assert metrics["jaccard_distance"] == pytest.approx(2.0 / 3.0)
    assert metrics["union_size"] == 3.0
    assert metrics["intersection_size"] == 1.0
    assert metrics["second_draw_union_fraction"] == pytest.approx(1.0 / 3.0)


def test_negative_coefficient_rewards_diversity() -> None:
    predicted = {1: 0.10, 2: 0.1005, 3: 0.12}
    diversity = {1: 0.10, 2: 0.90, 3: 0.20}

    original, _ = adjusted_root(predicted, diversity, coefficient=0.0)
    rewarded, scores = adjusted_root(predicted, diversity, coefficient=-0.5)

    assert original == 1
    assert rewarded == 2
    assert scores[2] < scores[1]


def test_comparison_summary_uses_lower_brier_as_a_win() -> None:
    rows = [
        {
            "candidate_root": 1,
            "baseline_root": 2,
            "candidate_brier": 0.08,
            "baseline_brier": 0.10,
            "difference": -0.02,
        },
        {
            "candidate_root": 3,
            "baseline_root": 3,
            "candidate_brier": 0.11,
            "baseline_brier": 0.11,
            "difference": 0.0,
        },
    ]

    summary = comparison_summary(rows)

    assert summary["mean_candidate_minus_baseline_brier"] == pytest.approx(-0.01)
    assert summary["relative_brier_reduction"] == pytest.approx(0.01 / 0.105)
    assert summary["changed_roots"] == 1
    assert (summary["wins"], summary["ties"], summary["losses"]) == (1, 1, 0)
