import numpy as np
import pytest
from scipy.special import gamma

from environments.scilaws.switch_reference import SwitchReference
from environments.scilaws.adaptive_reference import AdaptiveReference
from scripts.scilaws_mixed_refinement_audit import fixture


def test_partition_jacobian_and_full_domain():
    m, _ = fixture(8, ())
    from scipy.stats import norm
    ref = AdaptiveReference(m)
    v, _ = ref.integrate(lambda y: abs(y-5)*norm.pdf(y, loc=5, scale=2),
                         center=5, scale=2, points=[5])
    assert v == pytest.approx(2*np.sqrt(2/np.pi), abs=1e-8)


def test_switch_reference_analytic_minimum():
    class Model:
        num_actions = 2
        def _state(self, state):
            return np.array([0.0])
        def condition(self, state, action, y):
            return y
        def state_risk_lower_bound(self, *args):
            return 0.0
        def action_risk_lower_bound(self, *args):
            return 0.0

    class Reference(SwitchReference):
        def density_parameters(self, *args):
            return np.array([[6.0], [0.0], [1.0]])
        def terminal(self, state, action):
            self.check()
            return (state - (1 if action == 0 else -1))**2, 0.0

    ref = Reference(Model(), predictive_coordinates=True)
    value, error = ref.action(0.0, 0, 2)
    absolute_t_mean = 2*np.sqrt(6)*gamma(3.5)/(np.sqrt(np.pi)*5*gamma(3))
    assert value == pytest.approx(2.5-2*absolute_t_mean, abs=1e-8)
    assert ref.switches[0]['points'] == [0.0]
    assert error < 1e-8
    assert ref.evaluations > 17*2
