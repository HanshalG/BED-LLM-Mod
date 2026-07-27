from __future__ import annotations

import numpy as np

from scripts.discoverphysics_dark_matter_retained_support_replay import (
    retained_support_full_history_posterior,
)


def test_representative_preserves_equal_component_mass_without_new_info():
    representative = np.array([0.0, 0.0])
    initial_prior = np.array([0.7, 0.3])
    refresh_prior = np.array([0.2, 0.5, 0.3])
    initial_root_means = np.zeros((2, 2))
    refresh_root_means = np.zeros((3, 2))
    initial_continuation_means = np.zeros((2, 2))
    refresh_continuation_means = np.zeros((3, 2))

    posterior = retained_support_full_history_posterior(
        initial_branch_prior=initial_prior,
        refresh_branch_prior=refresh_prior,
        representative_observation=representative,
        actual_root_observation=representative,
        initial_root_means=initial_root_means,
        refresh_root_means=refresh_root_means,
        continuation_observations=np.array([[0.0, 0.0]]),
        initial_continuation_means=initial_continuation_means,
        refresh_continuation_means=refresh_continuation_means,
    )[0]

    assert np.allclose(posterior[:2], 0.5 * initial_prior)
    assert np.allclose(posterior[2:], 0.5 * refresh_prior)
    assert np.isclose(posterior.sum(), 1.0)


def test_exact_root_observation_updates_across_both_components():
    representative = np.array([0.0, 0.0])
    posterior = retained_support_full_history_posterior(
        initial_branch_prior=np.array([1.0]),
        refresh_branch_prior=np.array([1.0]),
        representative_observation=representative,
        actual_root_observation=np.array([0.9, 0.0]),
        initial_root_means=np.array([[-1.0, 0.0]]),
        refresh_root_means=np.array([[1.0, 0.0]]),
        continuation_observations=np.array([[0.0, 0.0]]),
        initial_continuation_means=np.array([[0.0, 0.0]]),
        refresh_continuation_means=np.array([[0.0, 0.0]]),
    )[0]

    assert posterior[1] > 0.99
    assert np.isclose(posterior.sum(), 1.0)


def test_zero_refresh_mass_recovers_initial_component():
    posterior = retained_support_full_history_posterior(
        initial_branch_prior=np.array([0.6, 0.4]),
        refresh_branch_prior=np.array([1.0]),
        representative_observation=np.array([0.0, 0.0]),
        actual_root_observation=np.array([0.0, 0.0]),
        initial_root_means=np.zeros((2, 2)),
        refresh_root_means=np.zeros((1, 2)),
        continuation_observations=np.array([[0.0, 0.0]]),
        initial_continuation_means=np.zeros((2, 2)),
        refresh_continuation_means=np.zeros((1, 2)),
        initial_component_mass=1.0,
        refresh_component_mass=0.0,
    )[0]

    assert np.allclose(posterior[:2], [0.6, 0.4])
    assert posterior[2] == 0.0
