import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import logsumexp

from environments.scilaws.linear_risk_interval import terminal_risk_interval
from environments.scilaws.adaptive_reference import AdaptiveReference
from scripts.scilaws_mixed_refinement_audit import fixture


@pytest.mark.parametrize('action', [0, 1])
def test_interval_contains_adaptive_risk_and_linear_rule_loss(action):
    m, state = fixture(8, ((0, .7),))
    interval = terminal_risk_interval(m, state, action)
    optimal, _ = AdaptiveReference(m).terminal(state, action)
    assert interval['lower'] <= optimal <= interval['upper']

    def integrand(y):
        density = np.exp(logsumexp([
            w+b.log_predictive(x[action], y) for w,b,x in
            zip(state.log_weights, state.components, m.action_features, strict=True)]))
        child = m.condition(state, action, y)
        prediction = (np.asarray(interval['target_mean'])
                      +np.asarray(interval['slope'])*(y-interval['observation_mean']))
        return density*(m.risk(child)+m.target_weights @ (m.forecast(child)-prediction)**2)

    actual, _ = quad(integrand, -np.inf, np.inf, epsabs=1e-8)
    assert actual == pytest.approx(interval['linear_risk'], abs=1e-7)


def test_single_family_bounds_coincide():
    from environments.scilaws.horizon_control_variate import HorizonControlVariateMixture
    from environments.scilaws.regression_belief import RegressionBelief
    m = HorizonControlVariateMixture([[[1.0], [2.0]]], [[[1.0]]],
        [RegressionBelief([.3], [[1.0]], 3, .2)], [1.0], target_weights=[1.0],
        quadrature_order=4, include_observation_noise=True)
    interval = terminal_risk_interval(m, m.initial_state, 0)
    assert interval['width'] < 1e-10
