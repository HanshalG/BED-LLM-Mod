from __future__ import annotations

import math

import numpy as np

from environments.chembench_mopen.smc import (
    TransformedParameterPrior,
    adaptive_tempered_smc,
    static_importance_sample,
    systematic_resample,
)


def test_transformed_prior_round_trips_and_enforces_order() -> None:
    states = (
        {"kcat": 1.0, "n": 1.0, "pKa1": 5.0, "pKa2": 8.0},
        {"kcat": 2.0, "n": 2.0, "pKa1": 6.0, "pKa2": 9.0},
    )
    prior = TransformedParameterPrior.from_parameter_states(states)
    particles = prior.sample(np.random.default_rng(17), 100)
    assert particles.shape == (100, 4)
    assert prior.valid_rows(particles).all()
    decoded = prior.decode(particles[0])
    np.testing.assert_allclose(prior.encode(decoded), particles[0])
    assert decoded["kcat"] > 0
    assert decoded["pKa1"] < decoded["pKa2"]
    assert prior.outside_coordinate_count(states[0]) == 0


def test_scrambled_sobol_prior_is_reproducible_and_well_spread() -> None:
    prior = TransformedParameterPrior.from_parameter_states(
        ({"kcat": 1.0, "pKa1": 5.0, "pKa2": 7.0}, {"kcat": 4.0, "pKa1": 6.0, "pKa2": 9.0})
    )
    first = prior.sample_sobol(23, 256)
    second = prior.sample_sobol(23, 256)
    np.testing.assert_array_equal(first, second)
    assert prior.valid_rows(first).all()
    scaled = (first - prior.lower) / prior.width
    assert np.all(np.mean(scaled, axis=0) > 0.4)
    assert np.all(np.mean(scaled, axis=0) < 0.6)


def test_static_importance_sampling_has_normalized_finite_weights() -> None:
    prior = TransformedParameterPrior.from_parameter_states(
        ({"alpha": -2.0}, {"alpha": 2.0}),
        expansion_factor=0.0,
        clamp_identity_positive=False,
    )

    def log_likelihood(particles: np.ndarray) -> np.ndarray:
        return -0.5 * np.square((particles[:, 0] - 0.5) / 0.25)

    result = static_importance_sample(prior, log_likelihood, num_particles=100, seed=3)
    assert np.isclose(result.weights.sum(), 1.0)
    assert np.isfinite(result.log_evidence)
    assert 1.0 <= result.effective_sample_size <= 100.0


def test_systematic_resampling_is_deterministic_and_respects_mass() -> None:
    weights = np.array([0.0, 0.2, 0.8])
    first = systematic_resample(weights, np.random.default_rng(11))
    second = systematic_resample(weights, np.random.default_rng(11))
    np.testing.assert_array_equal(first, second)
    assert np.count_nonzero(first == 2) >= np.count_nonzero(first == 1)
    assert not np.any(first == 0)


def test_adaptive_smc_matches_a_gaussian_posterior_and_evidence() -> None:
    prior = TransformedParameterPrior.from_parameter_states(
        ({"alpha": -5.0}, {"alpha": 5.0}),
        expansion_factor=0.0,
        clamp_identity_positive=False,
    )
    observation = 1.0
    sigma = 0.5

    def log_likelihood(particles: np.ndarray) -> np.ndarray:
        residual = (observation - particles[:, 0]) / sigma
        return -0.5 * residual**2 - math.log(sigma * math.sqrt(2.0 * math.pi))

    result = adaptive_tempered_smc(
        prior,
        log_likelihood,
        num_particles=500,
        seed=29,
        target_ess_fraction=0.6,
        rejuvenation_moves=3,
    )
    posterior_mean = float(np.dot(result.weights, result.particles[:, 0]))
    posterior_variance = float(
        np.dot(result.weights, np.square(result.particles[:, 0] - posterior_mean))
    )
    assert abs(posterior_mean - observation) < 0.08
    assert abs(posterior_variance - sigma**2) < 0.06
    assert abs(result.log_evidence - math.log(0.1)) < 0.15
    assert result.diagnostics.temperatures[-1] == 1.0
    assert result.diagnostics.num_rungs <= 80
    assert 0.0 < result.diagnostics.aggregate_acceptance_rate < 1.0


def test_adaptive_smc_is_reproducible_from_seed() -> None:
    prior = TransformedParameterPrior.from_parameter_states(
        ({"alpha": -3.0}, {"alpha": 3.0}),
        expansion_factor=0.0,
        clamp_identity_positive=False,
    )

    def log_likelihood(particles: np.ndarray) -> np.ndarray:
        return -np.square(particles[:, 0] - 1.5)

    first = adaptive_tempered_smc(prior, log_likelihood, num_particles=100, seed=41)
    second = adaptive_tempered_smc(prior, log_likelihood, num_particles=100, seed=41)
    np.testing.assert_array_equal(first.particles, second.particles)
    np.testing.assert_array_equal(first.weights, second.weights)
    assert first.log_evidence == second.log_evidence
    assert first.diagnostics == second.diagnostics


def test_sobol_full_covariance_smc_is_finite_and_reproducible() -> None:
    prior = TransformedParameterPrior.from_parameter_states(
        (
            {"alpha": -3.0, "beta": -2.0},
            {"alpha": 3.0, "beta": 2.0},
        ),
        expansion_factor=0.0,
        clamp_identity_positive=False,
    )

    def log_likelihood(particles: np.ndarray) -> np.ndarray:
        residual = particles[:, 0] + 0.8 * particles[:, 1] - 1.0
        return -0.5 * np.square(residual / 0.2)

    kwargs = dict(
        num_particles=256,
        seed=53,
        initialization="sobol",
        proposal_geometry="full",
    )
    first = adaptive_tempered_smc(prior, log_likelihood, **kwargs)
    second = adaptive_tempered_smc(prior, log_likelihood, **kwargs)
    np.testing.assert_array_equal(first.particles, second.particles)
    assert first.log_evidence == second.log_evidence
    assert first.diagnostics.temperatures[-1] == 1.0
    assert 0.0 < first.diagnostics.aggregate_acceptance_rate < 1.0
