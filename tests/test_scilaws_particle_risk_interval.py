import numpy as np
import pytest

from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.scilaws.particle_risk_interval import ParticleRiskIntervals
from environments.scilaws.particle_reference import ParticleReference
from environments.scilaws.weighted_intervals import weighted_min_interval


@pytest.mark.parametrize('conditioned', [False, True])
def test_bounds_contain_independent_risks(conditioned):
    m = QuantileGaussianModel([[-2., .2], [1., -.4]], [[.4, 1.], [1.2, .8]],
        [[0., 2.], [2., 0.]], [.3, .7], target_weights=[.2, .8],
        target_conditional_variances=[[.1], [.3]])
    state = m.condition(m.initial_state, 0, 3.) if conditioned else m.initial_state
    rows = ParticleRiskIntervals(m).actions(state)
    reference = ParticleReference(m, state)
    for action, row in enumerate(rows):
        value = reference.action(action)['value']
        assert row['lower'] <= value <= row['upper']


def test_known_particle_and_translation():
    m = QuantileGaussianModel([[1., 2.]], .3, [[1e10, -1e10]], [1.],
                             target_conditional_variances=.5)
    for row in ParticleRiskIntervals(m).actions(m.initial_state):
        assert row['lower'] <= .5 <= row['upper']
        assert row['upper']-row['lower'] < 3e-12


def test_probability_weighted_uncertainty_is_not_local_accuracy():
    result = weighted_min_interval([1e-6, 1-1e-6], [[(0., 1.)], [(0., 0.)]])
    assert result['midpoint_terminal_error_bound'] == pytest.approx(5e-7)
    assert result['within_terminal_budget']
    assert not result['outer_error_bounded']


def test_target_shift_invariance():
    rows = []
    for offset in (0., 1e10):
        m = QuantileGaussianModel([[-1.], [2.]], .5,
                                  np.array([[0.], [2.]])+offset, [.2, .8])
        rows.append(ParticleRiskIntervals(m).actions(m.initial_state))
    assert rows[0][0]['upper'] == pytest.approx(rows[1][0]['upper'], abs=1e-12)
