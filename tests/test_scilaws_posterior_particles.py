import numpy as np
import pytest

from environments.scilaws.posterior_particles import sample_posterior
from scripts.scilaws_mixed_refinement_audit import fixture


def test_family_mass_noise_and_reproducibility():
    m, state = fixture(4, ((0, .7),))
    a = sample_posterior(m, state, particles_per_family=16, rng=np.random.default_rng(41))
    b = sample_posterior(m, state, particles_per_family=16, rng=np.random.default_rng(41))
    np.testing.assert_array_equal(a.model.means, b.model.means)
    for i, mass in enumerate(np.exp(state.log_weights)):
        assert np.exp(a.model.initial_state)[a.family_indices == i].sum() == pytest.approx(mass)
    assert np.ptp(a.noise_variances) > 0
    np.testing.assert_array_equal(a.model.sigmas[:, 0]**2, a.model.sigmas[:, 1]**2)
    np.testing.assert_allclose(a.model.target_conditional_variances[:, 0], a.noise_variances)
    assert not a.noise_variances.flags.writeable


def test_correlated_precision_sampling_formula():
    from environments.scilaws.horizon_control_variate import HorizonControlVariateMixture
    from environments.scilaws.regression_belief import RegressionBelief
    precision = np.array([[3., 1.], [1., 2.]])
    belief = RegressionBelief([.2, -.4], precision, 4., .6)
    m = HorizonControlVariateMixture([np.eye(2)], [np.eye(2)], [belief], [1.], target_weights=[.5, .5])
    r = sample_posterior(m, m.initial_state, particles_per_family=8, rng=np.random.default_rng(17))
    rng = np.random.default_rng(17)
    variance = .6/rng.gamma(4., 1., 8)
    z = rng.standard_normal((8, 2))
    expected = [.2, -.4] + np.sqrt(variance)[:, None]*np.linalg.solve(np.linalg.cholesky(precision).T, z.T).T
    np.testing.assert_allclose(r.coefficients, expected)
    assert not np.any(r.model.target_conditional_variances)


def test_particle_update_is_full_gaussian_likelihood_not_second_initialization():
    m, state = fixture(4, ((0, .7),))
    r = sample_posterior(m, state, particles_per_family=32, rng=np.random.default_rng(8))
    p = r.model
    joint = np.asarray(p.initial_state)+p.log_likelihood(1, -.3)
    expected = np.exp(joint-np.max(joint))
    expected /= expected.sum()
    np.testing.assert_allclose(np.exp(p.condition(p.initial_state, 1, -.3)), expected)


@pytest.mark.parametrize('bad', [0, 4097, True])
def test_bad_counts_fail(bad):
    m, state = fixture(4, ())
    with pytest.raises(ValueError):
        sample_posterior(m, state, particles_per_family=bad, rng=np.random.default_rng(1))
