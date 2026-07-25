from __future__ import annotations

import numpy as np

from scripts.newtonbench_sound_speed_opportunity_audit import (
    ACTION_BANK_SHA256,
    HYPOTHESIS_IDS,
    compact_json_sha256,
    entropy,
    expected_entropies_for_priors,
    generate_action_bank,
    posterior_probabilities,
    quadrature_likelihood_tables,
)


def test_action_bank_matches_preregistered_hash_and_ranges() -> None:
    actions = generate_action_bank()

    assert len(actions) == 32
    assert compact_json_sha256(actions) == ACTION_BANK_SHA256
    assert all(1.3 <= action["adiabatic_index"] <= 1.7 for action in actions)
    assert all(10.0 <= action["temperature"] <= 1000.0 for action in actions)
    assert all(0.001 <= action["molar_mass"] <= 0.1 for action in actions)


def test_posterior_probabilities_normalize_for_batched_observations() -> None:
    prior = np.asarray([0.25, 0.75])
    log_likelihoods = np.log(
        np.asarray(
            [
                [0.8, 0.2],
                [0.1, 0.9],
            ]
        )
    )

    posterior = posterior_probabilities(prior, log_likelihoods)

    np.testing.assert_allclose(posterior.sum(axis=-1), 1.0)
    np.testing.assert_allclose(posterior[0], [4.0 / 7.0, 3.0 / 7.0])


def test_quadrature_expected_entropy_is_bounded_by_prior_entropy() -> None:
    means = np.asarray(
        [
            [1.0, 1.0],
            [1.0, 3.0],
        ]
    )
    tables, source_indices, mixture_weights = quadrature_likelihood_tables(
        means, noise_level=0.1, quadrature_order=9
    )
    priors = np.asarray([[0.5, 0.5], [0.2, 0.8]])

    expected = expected_entropies_for_priors(
        priors,
        tables[1],
        source_indices,
        mixture_weights,
    )

    assert np.all(expected >= 0.0)
    assert np.all(expected <= entropy(priors) + 1e-12)
    assert expected[0] < 2e-6


def test_hypothesis_order_is_complete_cross_product() -> None:
    assert HYPOTHESIS_IDS == (
        "easy:v0",
        "easy:v1",
        "easy:v2",
        "medium:v0",
        "medium:v1",
        "medium:v2",
        "hard:v0",
        "hard:v1",
        "hard:v2",
    )
