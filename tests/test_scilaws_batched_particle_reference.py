import pytest

from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.particle_reference import ParticleReference
from environments.scilaws.batched_particle_reference import integrate_actions


@pytest.mark.parametrize('conditioned', [False, True])
def test_full_menu_matches_scalar_integrals(conditioned):
    m = QuantileGaussianModel([[-1., .2], [2., -.5]], [[.3, 1.], [1., .8]],
        [[0., 1.], [1., 0.]], [.3, .7], target_conditional_variances=[[.1], [.8]])
    state = m.condition(m.initial_state, 0, 2.) if conditioned else m.initial_state
    result = integrate_actions(m, state)
    reference = ParticleReference(m, state)
    for a, row in enumerate(result['roots']):
        assert row['value'] == pytest.approx(reference.action(a)['value'], abs=1e-7)
        assert row['mass_error'] <= 1e-8
    assert result['evaluations'] == result['callbacks']*2


def test_noisy_and_zero_degenerate_cases_and_guards():
    m = QuantileGaussianModel([[0., 2.]], 1., [[1.]], [1.], target_conditional_variances=.2)
    assert [r['value'] for r in integrate_actions(m, m.initial_state)['roots']] == pytest.approx([.2, .2])
    with pytest.raises(SearchLimitExceeded):
        integrate_actions(m, m.initial_state, max_evaluations=1)
    with pytest.raises(SearchLimitExceeded):
        integrate_actions(m, m.initial_state, max_workspace_bytes=1)
    m = QuantileGaussianModel([[0.]], 1., [[1.]], [1.])
    assert integrate_actions(m, m.initial_state)['evaluations'] == 0


def test_unresolved_narrow_modes_fail_closed():
    m = QuantileGaussianModel([[0.], [1000.]], .001, [[0.], [1.]], [.5, .5],
                             target_conditional_variances=.2)
    with pytest.raises(ValueError, match='integration/mass'):
        integrate_actions(m, m.initial_state)
