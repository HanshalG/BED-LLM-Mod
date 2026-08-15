from __future__ import annotations

import numpy as np

from scripts.chembench_adaptive_smc_calibration import (
    _paired_counts,
    _rankdata,
    _spearman,
    summarize_cases,
)


def _case(index: int, first: float, second: float) -> dict[str, object]:
    return {
        "num_parameters": 2,
        "outside_prior_coordinates": index % 2,
        "truth_inside_prior": index % 2 == 0,
        "mse": {
            "static_16": first + 1.0,
            "static_100": first,
            "smc_1": second,
            "smc_2": second,
            "smc_mean": second,
        },
        "smc": {
            "bank_1": {
                "log_evidence": float(index),
                "num_rungs": 3,
                "accepted": 10,
                "proposals": 20,
                "invalid_proposal_rate": 0.1,
            },
            "bank_2": {
                "log_evidence": float(index) + 0.1,
                "num_rungs": 4,
                "accepted": 12,
                "proposals": 20,
                "invalid_proposal_rate": 0.2,
            },
        },
    }


def test_rank_and_spearman_handle_ties() -> None:
    np.testing.assert_array_equal(_rankdata((3.0, 1.0, 1.0)), (2.0, 0.5, 0.5))
    assert np.isclose(_spearman((1.0, 2.0, 3.0), (10.0, 20.0, 30.0)), 1.0)


def test_paired_counts_use_practical_tolerance() -> None:
    assert _paired_counts((0.9, 1.0, 1.2), (1.0, 1.0 + 1e-10, 1.0)) == {
        "wins": 1,
        "ties": 1,
        "losses": 1,
    }


def test_summary_reports_predictive_and_smc_diagnostics() -> None:
    summary = summarize_cases((_case(0, 2.0, 1.0), _case(1, 4.0, 2.0)))
    assert summary["num_cases"] == 2
    assert summary["mse"]["smc_mean"] == 1.5
    assert summary["paired_vs_static_16"] == {"wins": 2, "ties": 0, "losses": 0}
    assert np.isclose(summary["log_evidence_spearman"], 1.0)
    assert np.isclose(summary["outside_prior_coordinate_fraction"], 0.25)
    assert summary["smc_banks"]["bank_1"]["max_num_rungs"] == 3
