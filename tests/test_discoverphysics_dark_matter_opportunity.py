from __future__ import annotations

import numpy as np

from scripts.discoverphysics_dark_matter_opportunity import (
    NUM_HYPOTHESES,
    active_probe_actions,
    entropy,
    hidden_halo_family,
    posterior_batch,
    summarize_evaluation,
)


def test_frozen_halo_family_and_action_bank_are_well_formed():
    halos, halo_labels = hidden_halo_family()
    actions, action_labels = active_probe_actions()

    assert halos.shape == (NUM_HYPOTHESES, 10, 2)
    assert len(set(halo_labels)) == NUM_HYPOTHESES
    assert actions.shape == (25, 2)
    assert action_labels[0] == "center"
    assert len(set(action_labels)) == len(action_labels)
    assert np.allclose(halos, hidden_halo_family()[0])


def test_posterior_batch_matches_symmetric_tiny_case():
    priors = np.array([[0.5, 0.5]])
    means = np.array([[-1.0], [1.0]])
    posterior = posterior_batch(
        priors,
        observations=np.array([[0.0]]),
        means=means,
        noise_std=1.0,
    )

    assert posterior.shape == (1, 1, 2)
    assert np.allclose(posterior[0, 0], [0.5, 0.5])
    assert np.isclose(entropy(posterior[0, 0]), np.log(2.0))


def test_summary_uses_information_selected_roots_and_behavioral_risk():
    values = {
        "immediate_eig_nats": np.array([0.7, 0.5]),
        "total_eig_nats": np.array([0.9, 1.1]),
        "heldout_trajectory_mse": np.array([0.4, 0.2]),
    }

    summary = summarize_evaluation(values, ["myopic", "lookahead"])

    assert summary["myopic_root"] == "myopic"
    assert summary["depth_two_root"] == "lookahead"
    assert np.isclose(summary["immediate_sacrifice_nats"], 0.2)
    assert np.isclose(summary["total_eig_gain_nats"], 0.2)
    assert np.isclose(summary["heldout_risk_reduction"], 0.5)
