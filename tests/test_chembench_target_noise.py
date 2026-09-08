import numpy as np
import pytest

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.horizon import HorizonPlanner
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel


def model(noise=0.):
    return QuantileGaussianModel([[-1., .2], [1., -.5]], [[.7, 1.], [1.1, .8]],
        [[0., 2.], [1., 0.]], [.3, .7], target_weights=[.25, .75], branch_count=4,
        target_conditional_variances=noise)


def test_total_variance_and_posterior_update():
    v = np.array([[.1, .4], [.8, .2]])
    m = model(v)
    zero = model()
    states = [m.initial_state, m.condition(m.initial_state, 0, .6)]
    for state in states:
        w = np.exp(state)
        expected = zero.risk(state) + w @ (v @ m.target_weights)
        assert m.risk(state) == pytest.approx(expected)
        assert plan_batched(m, state, 0).value == pytest.approx(expected)
        np.testing.assert_array_equal(m.forecast(state), zero.forecast(state))
    assert m.condition(m.initial_state, 0, .6) == zero.condition(zero.initial_state, 0, .6)
    v[:] = 99
    assert m.target_conditional_variances[0, 0] == .1
    assert not m.target_conditional_variances.flags.writeable


@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('mode', ['adaptive', 'open_loop'])
def test_noisy_scalar_and_batched_root_agreement(depth, mode):
    m = model([[.1], [.8]])
    scalar = HorizonPlanner(m).plan(m.initial_state, depth, mode=mode, allow_repeats=True)
    batch = plan_batched(m, m.initial_state, depth, mode=mode, allow_repeats=True)
    assert dict(batch.root_values) == pytest.approx(dict(scalar.root_action_values), abs=1e-9)
    assert batch.value == pytest.approx(scalar.root.expected_risk, abs=1e-9)


def test_common_target_noise_adds_constant_without_action_change():
    old, new = model(), model(.7)
    a = plan_batched(old, old.initial_state, 2, allow_repeats=True)
    b = plan_batched(new, new.initial_state, 2, allow_repeats=True)
    assert b.value-a.value == pytest.approx(.7, abs=1e-12)
    assert b.action == a.action


@pytest.mark.parametrize('bad', [-1., float('nan'), float('inf'), np.ones((3, 2))])
def test_invalid_variances_rejected(bad):
    with pytest.raises(ValueError):
        model(bad)


def test_zero_mean_variance_does_not_erase_target_noise_in_raw_reference():
    from environments.chembench_mopen.raw_horizon import plan_raw_horizon
    m = QuantileGaussianModel([[-1.], [1.]], 1., [[0.], [0.]], [.5, .5],
                             target_conditional_variances=[[.4], [.8]])
    result = plan_raw_horizon(m, m.initial_state, 1, tolerance=1e-7)
    assert result.value == pytest.approx(.6, abs=1e-7)


def test_myopic_policy_evaluator_uses_same_loss():
    from environments.chembench_mopen.policy_value import evaluate_myopic_policy
    m = model([[.1], [.8]])
    for budget in (0, 1):
        value = evaluate_myopic_policy(m, m.initial_state, budget).value
        assert value == pytest.approx(plan_batched(m, m.initial_state, budget).value, abs=1e-12)
