import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.particle_linear_correction import ParticleLinearCorrection
from environments.scilaws.particle_reference import ParticleReference


def model(offset=0.):
    return QuantileGaussianModel([[-1., .2], [1., -.5]], [[.7, 1.], [1.1, .8]],
        np.array([[0., 2.], [1., 0.]])+offset, [.3, .7], target_weights=[.25, .75],
        target_conditional_variances=[[.1], [.8]], branch_count=128)


@pytest.mark.parametrize('conditioned', [False, True])
def test_independent_integral_and_linear_identity(conditioned):
    m = model()
    state = m.condition(m.initial_state, 0, .4) if conditioned else m.initial_state
    reference = ParticleReference(m, state)
    corrected = ParticleLinearCorrection(m, state)
    for a in range(2):
        row = corrected.action(a)
        truth = reference.action(a)['value']
        w = np.exp(state)
        mu = float(w @ m.means[:, a])
        target_mean = w @ m.targets
        covariance = (w * (m.means[:, a] - mu)) @ (m.targets - target_mean)
        variance = float(w @ ((m.means[:, a] - mu)**2 + m.sigmas[:, a]**2))

        def integrand(y):
            likelihood = (-.5*((y-m.means[:, a])/m.sigmas[:, a])**2
                          -np.log(m.sigmas[:, a])-.5*np.log(2*np.pi))
            joint = state+likelihood
            density_log = logsumexp(joint)
            residual = np.exp(joint-density_log) @ m.targets - target_mean - covariance/variance*(y-mu)
            return np.exp(density_log) * float(residual**2 @ m.target_weights)

        advantage, error = quad(integrand, -np.inf, np.inf, epsabs=1e-10)
        assert error < 1e-8
        assert row['linear_risk'] - advantage == pytest.approx(truth, abs=1e-8)
        # Finite quantile integration has a separate, looser decision gate.
        assert row['value'] == pytest.approx(truth, abs=1e-4)
        assert row['advantage'] >= 0
        assert row['linear_risk'] >= row['value']


def test_uninformative_observation_with_unequal_target_noise():
    m = QuantileGaussianModel([[0.], [0.]], .3, [[1.], [2.]], [.2, .8],
                             target_conditional_variances=[[.1], [.9]], branch_count=4)
    row = ParticleLinearCorrection(m, m.initial_state).action(0)
    assert row['value'] == pytest.approx(.16 + .2*.1 + .8*.9)
    assert row['advantage'] == pytest.approx(0., abs=1e-20)


def test_target_translation_and_shared_caps():
    a, b = model(), model(1e10)
    assert ParticleLinearCorrection(a, a.initial_state).action(0)['value'] == pytest.approx(
        ParticleLinearCorrection(b, b.initial_state).action(0)['value'], abs=1e-12)
    evaluator = ParticleLinearCorrection(a, a.initial_state, max_states=128)
    evaluator.action(0)
    with pytest.raises(SearchLimitExceeded):
        evaluator.action(1)
    with pytest.raises(SearchLimitExceeded):
        ParticleLinearCorrection(a, a.initial_state, max_workspace_bytes=1)
