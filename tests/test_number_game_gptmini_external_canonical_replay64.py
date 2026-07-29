from __future__ import annotations

from scripts.number_game_gptmini_external_canonical_replay64 import (
    diagnostic_gates,
    primary_gates,
)


def _comparison(
    *,
    reduction: float,
    upper: float,
    wins: int,
    brier: float = -0.01,
    hamming: float = -0.01,
    coverage: float = 0.01,
) -> dict:
    return {
        "relative_brier_reduction": reduction,
        "stratified_tree_bootstrap_brier_difference_95pct": [
            -0.02,
            upper,
        ],
        "wins": wins,
        "mean_brier_difference": brier,
        "mean_hamming_difference": hamming,
        "mean_coverage_difference": coverage,
    }


def _pooled() -> dict:
    return {
        "comparisons": {
            "crossfit_depth_two": _comparison(
                reduction=0.02,
                upper=-0.001,
                wins=24,
            ),
            "myopic_eig": _comparison(
                reduction=0.08,
                upper=-0.01,
                wins=40,
            ),
            "fixed_support_depth_three": _comparison(
                reduction=0.04,
                upper=0.001,
                wins=30,
            ),
            "positive_test_strategy": _comparison(
                reduction=0.03,
                upper=0.001,
                wins=30,
            ),
            "uniform_random_candidate_root": _comparison(
                reduction=0.05,
                upper=-0.001,
                wins=35,
            ),
        },
        "ranking": {
            "crossfit_depth_three_spearman_brier": {"mean": 0.4},
            "crossfit_depth_two_spearman_brier": {"mean": 0.2},
        },
    }


def _studies() -> list[dict]:
    return [
        {
            "trees": [{}] * 32,
            "aggregate": {
                "root_differences": {"crossfit_depth_two": 14},
                "comparisons": {
                    "crossfit_depth_two": {
                        "mean_candidate_minus_baseline_brier": -0.01
                    }
                },
            },
        },
        {
            "trees": [{}] * 32,
            "aggregate": {
                "root_differences": {"crossfit_depth_two": 13},
                "comparisons": {
                    "crossfit_depth_two": {
                        "mean_candidate_minus_baseline_brier": -0.005
                    }
                },
            },
        },
    ]


def test_primary_gates_pass_frozen_positive_fixture() -> None:
    assert all(primary_gates(pooled=_pooled(), study_results=_studies()).values())


def test_primary_gates_keep_each_study_directional() -> None:
    studies = _studies()
    studies[1]["aggregate"]["comparisons"]["crossfit_depth_two"][
        "mean_candidate_minus_baseline_brier"
    ] = 0.001
    gates = primary_gates(pooled=_pooled(), study_results=studies)
    assert not gates[
        "both_independent_studies_directionally_favor_depth_three"
    ]


def test_diagnostic_gates_are_reported_separately() -> None:
    gates = diagnostic_gates(_pooled())
    assert all(gates.values())
