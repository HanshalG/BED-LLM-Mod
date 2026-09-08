"""Analytic within-family chance-node control variate; explicit alternative.

The corrected integral is E[U(next)] + Q[V(next)-U(next)], where U is the
within-family target variance (plus observation noise when requested). Remaining
nonlinear integration error is not bounded or removed by this identity.
"""

from functools import lru_cache

from .regression_mixture import RegressionMixture


class ControlVariateMixture(RegressionMixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_terms = lru_cache(maxsize=256)(self._terminal_risk_terms)

    def chance_risk_correction(self, state, action):
        _, _, sampled, analytic = self._cached_terms(state, action)
        return analytic - sampled

    def expected_terminal_risk(self, state, action):
        value, count, sampled, analytic = self._cached_terms(state, action)
        return value - sampled + analytic, count
