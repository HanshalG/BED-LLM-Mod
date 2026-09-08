import pytest

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.scilaws.particle_reference import ParticleReference
from environments.scilaws.particle_residual_reference import ParticleResidualReference


def test_independent_risk_agreement_and_tail_accounting():
    m = QuantileGaussianModel([[-1.], [2.]], [[.4], [1.2]], [[0.], [2.]], [.3, .7],
                             target_conditional_variances=[[.1], [.3]], branch_count=32)
    r = ParticleResidualReference(m, m.initial_state).action(0)
    assert r['value'] == pytest.approx(ParticleReference(m, m.initial_state).action(0)['value'], abs=1e-7)
    assert 0 < r['outer_advantage'] < r['advantage']
    assert r['error_estimate']+r['tail_bound'] <= 1e-7
    assert r['mass_error'] <= 1e-8


def test_zero_residual_and_shared_cap():
    m = QuantileGaussianModel([[1.]], .3, [[2.]], [1.], target_conditional_variances=.5)
    r = ParticleResidualReference(m, m.initial_state).action(0)
    assert r['value'] == pytest.approx(.5)
    assert r['advantage'] == pytest.approx(0.)
    limited = ParticleResidualReference(m, m.initial_state, max_evaluations=1)
    with pytest.raises(SearchLimitExceeded):
        limited.action(0)
