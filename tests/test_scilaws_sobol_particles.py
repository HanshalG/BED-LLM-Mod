import numpy as np
import pytest
from scipy.special import gammaincinv, ndtri
from scipy.stats import qmc

from environments.scilaws.posterior_particles import sample_posterior
from scripts.scilaws_mixed_refinement_audit import fixture


def test_joint_sobol_transform_and_reproducibility():
    m, state = fixture(4, ((0, .7),))
    result = sample_posterior(m, state, particles_per_family=32, rng=np.random.default_rng(12), sampling='sobol')
    again = sample_posterior(m, state, particles_per_family=32, rng=np.random.default_rng(12), sampling='sobol')
    np.testing.assert_array_equal(result.model.means, again.model.means)
    rng = np.random.default_rng(12)
    for i, b in enumerate(state.components):
        u = qmc.Sobol(len(b.mean)+1, scramble=True, bits=52, seed=int(rng.integers(2**32))).random_base2(5)
        variance = b.scale/gammaincinv(b.shape, u[:, 0])
        beta = np.asarray(b.mean)+np.sqrt(variance)[:, None]*np.linalg.solve(np.linalg.cholesky(b.precision).T, ndtri(u[:, 1:]).T).T
        select = result.family_indices == i
        np.testing.assert_allclose(result.noise_variances[select], variance)
        np.testing.assert_allclose(np.asarray(result.coefficients)[select], beta)
        assert np.exp(result.model.initial_state)[select].sum() == pytest.approx(np.exp(state.log_weights[i]))


def test_nested_samples_keep_family_prefix():
    m, s = fixture(4, ())
    a = sample_posterior(m, s, particles_per_family=16, rng=np.random.default_rng(1), sampling='sobol')
    b = sample_posterior(m, s, particles_per_family=32, rng=np.random.default_rng(1), sampling='sobol')
    for i in range(2):
        np.testing.assert_array_equal(a.noise_variances[a.family_indices==i], b.noise_variances[b.family_indices==i][:16])


@pytest.mark.parametrize('n,mode', [(3, 'sobol'), (4, 'unknown')])
def test_bad_mode_or_count(n, mode):
    m, s = fixture(4, ())
    with pytest.raises(ValueError):
        sample_posterior(m, s, particles_per_family=n, rng=np.random.default_rng(1), sampling=mode)
