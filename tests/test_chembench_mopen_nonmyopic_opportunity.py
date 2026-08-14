from __future__ import annotations

import math

import numpy as np
import pytest

from scripts.chembench_mopen_nonmyopic_opportunity import (
    ExactPlanner,
    VALIDATION_SLICES,
    apply_gate,
    categorical_likelihoods,
    frozen_assays,
    query_assays,
)


def test_frozen_protocol_dimensions_and_query_sampling() -> None:
    assays = frozen_assays()
    assert len(assays) == 18
    assert len({assay.name for assay in assays}) == 18
    assert len(VALIDATION_SLICES) == 8

    bounds = {
        "C_A": (0.01, 100.0),
        "C_I": (0.0, 50.0),
        "C_B": (0.01, 100.0),
        "C_P": (0.0, 20.0),
        "Enz": (0.01, 10.0),
        "T": (278.0, 368.0),
        "pH": (4.0, 10.0),
    }
    first = query_assays(7, bounds, {"C_A", "C_B", "Enz"}, count=5)
    second = query_assays(7, bounds, {"C_A", "C_B", "Enz"}, count=5)
    assert np.array_equal(first, second)
    assert first.shape == (5, 7)
    assert np.all(first[:, 0] >= 0.01)
    assert np.all(first[:, 0] <= 100.0)


def test_categorical_likelihoods_are_normalized_and_ordered() -> None:
    means = np.array(
        [
            [1.0, 4.0],
            [2.0, 3.0],
            [3.0, 2.0],
            [4.0, 1.0],
        ]
    )
    likelihoods = categorical_likelihoods(means, noise_level=0.05)
    assert likelihoods.shape == (4, 2, 3)
    assert np.allclose(likelihoods.sum(axis=2), 1.0)
    assert np.argmax(likelihoods[0, 0]) == 0
    assert np.argmax(likelihoods[-1, 0]) == 2
    assert np.argmax(likelihoods[0, 1]) == 2
    assert np.argmax(likelihoods[-1, 1]) == 0


def test_exact_planner_truth_replay_matches_prior_risk() -> None:
    means = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
        ]
    )
    planner = ExactPlanner(categorical_likelihoods(means, noise_level=0.1), features)
    for horizon in (1, 2, 3):
        result = planner.evaluate_horizon(horizon, budget=2)
        assert math.isfinite(result["expected_terminal_mse"])
        assert result["expected_terminal_mse"] == pytest.approx(
            np.mean(result["truth_losses"]), abs=1e-9
        )
        assert result["root_action_index"] in range(3)


def _slice_result(name: str, d1: float, d2: float, d3: float) -> dict:
    return {
        "slice": name,
        "horizons": {
            "d1": {"expected_terminal_mse": d1, "truth_losses": [d1] * 57},
            "d2": {"expected_terminal_mse": d2, "truth_losses": [d2] * 57},
            "d3": {"expected_terminal_mse": d3, "truth_losses": [d3] * 57},
        },
    }


def test_frozen_gate_is_conjunctive() -> None:
    passing = [_slice_result(item.name, 1.0, 0.9, 0.8) for item in VALIDATION_SLICES]
    gate = apply_gate(passing)
    assert gate["passed"] is True
    assert all(gate["conditions"].values())

    failing = list(passing)
    for index in (-1, -2, -3):
        failing[index] = _slice_result(VALIDATION_SLICES[index].name, 1.0, 0.9, 0.91)
    gate = apply_gate(failing)
    assert gate["passed"] is False
    assert gate["conditions"]["d3_beats_d1_on_all_8_slices"] is True
    assert gate["conditions"]["d3_slice_wins_at_least_6_of_8"] is False
    assert gate["conditions"]["d3_truth_cell_majority"] is True

    failing = [_slice_result(item.name, 1.0, 0.96, 0.92) for item in VALIDATION_SLICES]
    gate = apply_gate(failing)
    assert gate["passed"] is False
    assert gate["conditions"]["d2_mean_reduction_at_least_5pct"] is False
