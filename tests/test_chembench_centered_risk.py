import numpy as np
import pytest

from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel


@pytest.mark.parametrize('depth', [0, 1, 2, 3])
@pytest.mark.parametrize('mode', ['adaptive', 'open_loop'])
def test_full_root_equivalence(depth, mode):
    m = QuantileGaussianModel([[-1., .2], [1., -.5]], [[.7, 1.], [1.1, .8]],
        [[0., 2.], [1., 0.]], [.3, .7], target_weights=[.25, .75], branch_count=4,
        target_conditional_variances=[[.1], [.8]])
    old = plan_batched(m, m.initial_state, depth, mode=mode, allow_repeats=True)
    new = plan_batched(m, m.initial_state, depth, mode=mode, allow_repeats=True, risk_backend='centered')
    assert new.value == pytest.approx(old.value, abs=1e-12)
    assert dict(new.root_values) == pytest.approx(dict(old.root_values), abs=1e-12)
    assert new.action == old.action


@pytest.mark.parametrize('weights', [[.2, .3, .5], [0., 0., 1.], [1e-15, 0., 1-1e-15]])
def test_large_offset_and_concentrated_weights(weights):
    targets = np.array([[1e12, 1e12], [1e12+2, 1e12-3], [1e12+8, 1e12+5]])
    m = QuantileGaussianModel([[0.], [1.], [2.]], 1., targets, weights, target_weights=[.4, .6])
    w = np.array(weights)
    expected = sum(w[i]*w[j]*np.sum((targets[i]-targets[j])**2*m.target_weights)
                   for i in range(3) for j in range(i))
    assert CenteredTargetRisk(m)(w[None, :])[0] == pytest.approx(expected, abs=1e-12)


def test_invalid_backend():
    m = QuantileGaussianModel([[0.]], 1., [[0.]], [1.])
    with pytest.raises(ValueError, match='backend'):
        plan_batched(m, m.initial_state, 0, risk_backend='invalid')


def test_calibrated_count_memory_preflight_without_opening_branches(monkeypatch):
    from environments.chembench_mopen import batch_horizon
    from environments.chembench_mopen.horizon import SearchLimitExceeded
    m = QuantileGaussianModel(np.zeros((2048, 8)), 1., np.zeros((2048, 64)),
                             np.full(2048, 1/2048), branch_count=16,
                             target_conditional_variances=1.)
    def boundary(*args, **kwargs):
        raise RuntimeError('branch boundary reached')
    monkeypatch.setattr(batch_horizon, 'posterior_branches_many', boundary)
    with pytest.raises(SearchLimitExceeded, match='workspace'):
        plan_batched(m, m.initial_state, 3, allow_repeats=True)
    with pytest.raises(RuntimeError, match='branch boundary reached'):
        plan_batched(m, m.initial_state, 3, allow_repeats=True, risk_backend='centered')
