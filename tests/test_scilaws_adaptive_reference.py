import pytest
from scipy.integrate import quad
from scipy.special import logsumexp
import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded
from environments.scilaws.adaptive_reference import AdaptiveReference
from scripts.scilaws_mixed_refinement_audit import fixture


def test_terminal_matches_direct_full_risk_integration():
    m, state = fixture(8, ((0, 0.7),))
    ref = AdaptiveReference(m)
    value, error = ref.terminal(state, 1)

    def integrand(y):
        logp = logsumexp([
            w + b.log_predictive(x[1], y)
            for w, b, x in zip(state.log_weights, state.components,
                                m.action_features, strict=True)
        ])
        return np.exp(logp) * m.risk(m.condition(state, 1, y))

    direct, _ = quad(integrand, -np.inf, np.inf, epsabs=1e-8, epsrel=1e-8)
    assert value == pytest.approx(direct, abs=1e-7)
    assert error < 1e-7


def test_caps_and_invalid_depth():
    m, state = fixture(4, ())
    with pytest.raises(SearchLimitExceeded):
        AdaptiveReference(m, max_evaluations=1).terminal(state, 0)
    with pytest.raises(ValueError):
        AdaptiveReference(m).action(state, 0, 3)
