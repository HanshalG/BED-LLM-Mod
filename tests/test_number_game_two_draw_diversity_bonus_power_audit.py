from __future__ import annotations

import pytest

from scripts.number_game_two_draw_diversity_bonus_power_audit import (
    gate_values,
    sample_statistics,
)


def _row(*, depth_two_difference: float, original_difference: float, changed: bool):
    return {
        "bonus_brier": 0.10 + depth_two_difference,
        "depth_two_brier": 0.10,
        "original_brier": 0.10 + depth_two_difference - original_difference,
        "bonus_minus_depth_two": depth_two_difference,
        "bonus_minus_original": original_difference,
        "bonus_changes_original": changed,
    }


def test_sample_statistics_preserve_paired_signs() -> None:
    sample = [
        _row(depth_two_difference=-0.01, original_difference=-0.002, changed=True),
        _row(depth_two_difference=-0.02, original_difference=0.0, changed=False),
        _row(depth_two_difference=0.0, original_difference=-0.001, changed=True),
        _row(depth_two_difference=-0.01, original_difference=0.0, changed=False),
    ]
    stats = sample_statistics(sample)

    assert stats["mean_bonus_minus_depth_two"] == pytest.approx(-0.01)
    assert stats["relative_reduction_vs_depth_two"] == pytest.approx(0.10)
    assert stats["depth_two_wins"] == 3
    assert stats["depth_two_losses"] == 0
    assert stats["changed_original_roots"] == 2
    assert stats["original_changed_root_wins"] == 2


def test_primary_gate_does_not_require_redundant_changed_root_win_count() -> None:
    stats = {
        "relative_reduction_vs_depth_two": 0.05,
        "normal_approximation_upper_95pct": -0.001,
        "depth_two_wins": 18,
        "depth_two_losses": 8,
        "changed_original_roots": 10,
        "mean_bonus_minus_original": -0.001,
        "original_changed_root_wins": 4,
        "original_changed_root_losses": 6,
    }
    gates = gate_values(stats, sample_size=32)

    assert gates["primary_only"]
    assert gates["primary_plus_nonworsening"]
    assert not gates["original_redundant_gate"]
