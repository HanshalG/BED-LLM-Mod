from __future__ import annotations

import math

import numpy as np

from scripts.discoverphysics_extra_dimensions_confirmation import (
    FROZEN_CANDIDATE,
    build_hypotheses,
    heldout_features,
    numpy_displacements,
    raw_force,
    transformed_means,
)
from scripts.discoverphysics_extra_dimensions_opportunity import (
    evaluate_scalar_policy,
)


def test_frozen_hypotheses_are_calibrated_and_prior_is_valid() -> None:
    hypotheses, prior = build_hypotheses()

    assert len(hypotheses) == 18
    assert prior.shape == (18,)
    assert np.isclose(prior.sum(), 1.0)
    for group_index, start in enumerate((0, 6, 12)):
        calibration_radius = FROZEN_CANDIDATE.calibration_radii[group_index]
        expected = (
            FROZEN_CANDIDATE.group_offsets[group_index]
            / (2.0 * math.pi * calibration_radius)
        )
        values = [
            spec.strength
            * raw_force(
                spec.family,
                spec.parameter,
                np.array([calibration_radius]),
            )[0]
            for spec in hypotheses[start : start + 6]
        ]
        assert np.allclose(values, expected)


def test_frozen_numpy_mechanics_has_adaptive_opportunity() -> None:
    hypotheses, prior = build_hypotheses()
    means = transformed_means(numpy_displacements(hypotheses))
    values = evaluate_scalar_policy(
        means,
        prior,
        quadrature_points=8,
        heldout_features=heldout_features(hypotheses),
    )
    myopic_index = int(np.argmax(values["immediate_eig_nats"]))
    depth_two_index = int(np.argmax(values["total_eig_nats"]))
    immediate_sacrifice = (
        values["immediate_eig_nats"][myopic_index]
        - values["immediate_eig_nats"][depth_two_index]
    )
    total_gain = (
        values["total_eig_nats"][depth_two_index]
        - values["total_eig_nats"][myopic_index]
    )
    risks = values["heldout_prediction_mse"]
    risk_reduction = (
        risks[myopic_index] - risks[depth_two_index]
    ) / risks[myopic_index]

    assert FROZEN_CANDIDATE.action_radii[myopic_index] == 5.5
    assert FROZEN_CANDIDATE.action_radii[depth_two_index] == 2.4
    assert immediate_sacrifice > 0.10
    assert total_gain > 0.05
    assert risk_reduction > 0.20
