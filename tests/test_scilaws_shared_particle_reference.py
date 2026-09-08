import pytest

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.scilaws.shared_particle_reference import SharedParticleReference


def test_budget_does_not_reset_after_failure():
    m = QuantileGaussianModel([[0.]], 1., [[1.]], [1.], target_conditional_variances=.5)
    budget = SharedParticleReference(max_evaluations=1)
    with pytest.raises(SearchLimitExceeded):
        budget.action(m, m.initial_state, 0)
    count = budget.evaluations
    assert count > 1
    with pytest.raises(SearchLimitExceeded):
        budget.action(m, m.initial_state, 0)
    assert budget.evaluations == count


def test_predecessor_work_is_charged():
    budget = SharedParticleReference(max_evaluations=32)
    with pytest.raises(SearchLimitExceeded):
        budget.charge(32)
