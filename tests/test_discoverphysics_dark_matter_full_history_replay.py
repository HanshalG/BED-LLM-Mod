from __future__ import annotations

import numpy as np

from scripts.discoverphysics_dark_matter_full_history_replay import (
    corrected_full_history_posterior,
)


def test_correction_reduces_to_branch_prior_at_representative_without_new_info():
    branch_prior = np.array([0.7, 0.3])
    representative = np.array([0.0, 0.0])
    root_means = np.array([[-1.0, 0.0], [1.0, 0.0]])
    continuation_means = np.array([[0.0, 0.0], [0.0, 0.0]])

    posterior = corrected_full_history_posterior(
        branch_prior=branch_prior,
        representative_observation=representative,
        actual_root_observation=representative,
        root_means=root_means,
        continuation_observations=np.array([[0.0, 0.0]]),
        continuation_means=continuation_means,
    )

    assert np.allclose(posterior[0], branch_prior)


def test_correction_retains_off_centroid_root_information():
    branch_prior = np.array([0.5, 0.5])
    representative = np.array([0.0, 0.0])
    root_means = np.array([[-1.0, 0.0], [1.0, 0.0]])
    continuation_means = np.array([[0.0, 0.0], [0.0, 0.0]])

    posterior = corrected_full_history_posterior(
        branch_prior=branch_prior,
        representative_observation=representative,
        actual_root_observation=np.array([0.9, 0.0]),
        root_means=root_means,
        continuation_observations=np.array([[0.0, 0.0]]),
        continuation_means=continuation_means,
    )

    assert posterior[0, 1] > 0.99
    assert np.isclose(posterior.sum(), 1.0)
