from __future__ import annotations

import numpy as np

from scripts.chembench_rqmc_sampling_fidelity import (
    MAX_SAMPLES,
    evaluate_rqmc_case,
    predictive_outcomes_from_coordinates,
    rqmc_coordinates,
)


def test_rqmc_coordinates_are_deterministic_and_nested() -> None:
    first = rqmc_coordinates(123, MAX_SAMPLES)
    second = rqmc_coordinates(123, MAX_SAMPLES)
    assert np.array_equal(first, second)
    assert np.array_equal(first[:32], rqmc_coordinates(123, 32))
    assert np.all((first >= 0) & (first < 1))


def test_predictive_outcomes_share_particle_and_noise_coordinates() -> None:
    coordinates = np.asarray([[0.1, 0.5], [0.9, 0.5]])
    weights = np.asarray([0.5, 0.5])
    first = predictive_outcomes_from_coordinates(
        coordinates,
        np.asarray([1.0, 2.0]),
        np.asarray([0.5, 0.5]),
        weights,
    )
    second = predictive_outcomes_from_coordinates(
        coordinates,
        np.asarray([11.0, 12.0]),
        np.asarray([0.5, 0.5]),
        weights,
    )
    assert np.allclose(first, [1.0, 2.0])
    assert np.allclose(second - first, 10.0)


def test_rqmc_case_uses_nested_replicate_prefixes() -> None:
    risks = np.empty((4, 3, MAX_SAMPLES), dtype=float)
    for replicate in range(4):
        for action in range(3):
            risks[replicate, action] = (
                replicate * 1_000 + action * 10_000 + np.arange(MAX_SAMPLES)
            )
    result = evaluate_rqmc_case(
        reference_action_risks=np.asarray([0.0, 1.0, 2.0]),
        replicate_outcome_risks=risks,
        action_indices=(1, 2, 3),
        root_risk=10.0,
        component_action_risks={
            "bank_1": np.asarray([0.0, 1.0, 2.0]),
            "bank_2": np.asarray([0.0, 2.0, 1.0]),
        },
        component_root_risks={"bank_1": 10.0, "bank_2": 10.0},
    )
    assert result["estimates"]["32"][0]["estimated_action_risks"][0] == 15.5
    assert result["estimates"]["32"][1]["estimated_action_risks"][0] == 1015.5
    assert result["ensemble_1024"]["estimated_action_risks"][0] == 1627.5
