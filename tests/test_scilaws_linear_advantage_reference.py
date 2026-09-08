import pytest

from environments.scilaws.adaptive_reference import AdaptiveReference
from environments.scilaws.linear_advantage_reference import LinearAdvantageReference
from scripts.scilaws_mixed_refinement_audit import fixture, HISTORIES


@pytest.mark.parametrize('history', HISTORIES)
def test_independent_residual_agreement(history):
    model, state = fixture(4, history)
    old = AdaptiveReference(model, predictive_coordinates=True)
    new = LinearAdvantageReference(model, predictive_coordinates=True)
    for a in range(2):
        v, e = old.terminal(state, a)
        w, f = new.terminal(state, a)
        assert w == pytest.approx(v, abs=1e-7, rel=0)
        assert e <= 1e-7 and f <= 1e-7


def test_single_family_exact_linear_predictor():
    from environments.scilaws.horizon_control_variate import HorizonControlVariateMixture
    from environments.scilaws.regression_belief import RegressionBelief
    model = HorizonControlVariateMixture([[[1.], [2.]]], [[[1.], [3.]]],
        [RegressionBelief([.7], [[2.]], 3., .4)], [1.], target_weights=[.25, .75],
        include_observation_noise=True)
    ref = LinearAdvantageReference(model, predictive_coordinates=True)
    for a in range(2):
        value, error = ref.terminal(model.initial_state, a)
        assert value == pytest.approx(model.action_risk_lower_bound(model.initial_state, a, 1), abs=1e-12)
        assert error <= 1e-12
