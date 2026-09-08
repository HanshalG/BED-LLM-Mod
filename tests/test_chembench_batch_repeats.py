import pytest

from environments.chembench_mopen.batch_horizon import plan_batched
from environments.chembench_mopen.horizon import HorizonPlanner
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel


@pytest.mark.parametrize('mode', ['adaptive', 'open_loop'])
@pytest.mark.parametrize('depth', [1, 2, 3])
@pytest.mark.parametrize('actions', [1, 2])
def test_repeated_measurements_match_scalar_tree(mode, depth, actions):
    m = QuantileGaussianModel([[-1., .3][:actions], [1., -.8][:actions]],
                             [[.8, 1.1][:actions], [1.2, .7][:actions]],
                             [[0.], [1.]], [.4, .6], branch_count=4)
    old = plan_batched(m, m.initial_state, depth, mode=mode)
    assert old.effective_horizon == min(depth, actions)
    new = plan_batched(m, m.initial_state, depth, mode=mode, allow_repeats=True)
    scalar = HorizonPlanner(m).plan(m.initial_state, depth, mode=mode, allow_repeats=True)
    assert new.effective_horizon == depth
    assert new.value == pytest.approx(scalar.root.expected_risk, abs=1e-10)
    assert dict(new.root_values) == pytest.approx(dict(scalar.root_action_values), abs=1e-10)
    if mode == 'open_loop':
        assert len(new.fixed_sequence) == depth
        if actions == 1:
            assert new.fixed_sequence == (0,)*depth


def test_repeat_validation_and_empty_menu():
    m = QuantileGaussianModel([[-1.], [1.]], 1., [[0.], [1.]], [.5, .5], branch_count=4)
    with pytest.raises(ValueError, match='boolean'):
        plan_batched(m, m.initial_state, 2, allow_repeats=1)
    assert plan_batched(m, m.initial_state, 2, available=[], allow_repeats=True).effective_horizon == 0
