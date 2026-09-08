import pytest

from environments.scilaws.particle_reference import ParticleReference
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.raw_horizon import plan_raw_horizon


def test_independent_reference_agreement():
    m = QuantileGaussianModel([[-1., .2], [1., -.5]], [[.7, 1.], [1.1, .8]],
        [[0., 2.], [1., 0.]], [.3, .7], target_weights=[.25, .75],
        target_conditional_variances=[[.1], [.8]])
    old = plan_raw_horizon(m, m.initial_state, 1, tolerance=1e-7)
    new = ParticleReference(m, m.initial_state)
    for a, value in old.root_values:
        r = new.action(a)
        assert r['value'] == pytest.approx(value, abs=1e-7)
        assert r['mass_error'] <= 1e-8
        assert r['tail_bound'] <= 1e-9


def test_single_particle_noisy_target():
    m = QuantileGaussianModel([[1.]], .2, [[4.]], [1.], target_conditional_variances=.3)
    assert ParticleReference(m, m.initial_state).action(0)['value'] == pytest.approx(.3, abs=1e-9)


def test_shared_budget_is_not_reset_per_action():
    m = QuantileGaussianModel([[0., 1.]], 1., [[0.]], [1.], target_conditional_variances=1.)
    ref = ParticleReference(m, m.initial_state, max_evaluations=1)
    with pytest.raises(SearchLimitExceeded):
        ref.action(0)
    before = ref.evaluations
    with pytest.raises(SearchLimitExceeded):
        ref.action(1)
    assert ref.evaluations > before
