from __future__ import annotations

import pytest

from scripts.number_game_predictive_risk_holdout import (
    aggregate_random_root_control,
    paired_bootstrap_interval,
    policy_comparison,
)


def _result(root: int, values: list[tuple[str, float, float, bool]]):
    rows = [
        {
            "target": target,
            "posterior_predictive_brier": brier,
            "best_hamming_error": hamming,
            "truth_extension_covered": covered,
        }
        for target, brier, hamming, covered in values
    ]
    return {
        "policy": f"root_{root}",
        "root": root,
        "mean_posterior_predictive_brier": sum(
            row["posterior_predictive_brier"] for row in rows
        )
        / len(rows),
        "mean_best_hamming_error": sum(
            row["best_hamming_error"] for row in rows
        )
        / len(rows),
        "truth_extension_coverage_rate": sum(
            row["truth_extension_covered"] for row in rows
        )
        / len(rows),
        "targets": rows,
    }


def test_paired_bootstrap_is_reproducible_and_brackets_constant():
    assert paired_bootstrap_interval([-0.1] * 5, samples=100) == pytest.approx(
        [-0.1, -0.1]
    )


def test_policy_comparison_reports_candidate_gain():
    candidate = _result(
        1,
        [("a", 0.1, 0.05, True), ("b", 0.2, 0.1, False)],
    )
    baseline = _result(
        2,
        [("a", 0.2, 0.1, True), ("b", 0.3, 0.2, False)],
    )

    comparison = policy_comparison(candidate, baseline)

    assert comparison["relative_brier_reduction"] == pytest.approx(0.4)
    assert comparison["relative_hamming_reduction"] == pytest.approx(0.5)
    assert comparison["coverage_difference"] == 0.0


def test_random_control_averages_each_target_over_roots():
    roots = {
        1: _result(
            1,
            [("a", 0.1, 0.2, True), ("b", 0.3, 0.4, False)],
        ),
        2: _result(
            2,
            [("a", 0.3, 0.4, False), ("b", 0.5, 0.6, True)],
        ),
    }

    result = aggregate_random_root_control(roots)

    assert result["mean_posterior_predictive_brier"] == pytest.approx(0.3)
    assert result["mean_best_hamming_error"] == pytest.approx(0.4)
    assert result["truth_extension_coverage_rate"] == pytest.approx(0.5)
