"""Independent adaptive residual integrals, diagnostic only (depths one/two)."""

import math
from time import monotonic

import numpy as np
from scipy.integrate import quad
from scipy.special import logsumexp
from scipy.stats import t

from environments.chembench_mopen.horizon import SearchLimitExceeded


class AdaptiveReference:
    def __init__(self, model, *, tolerance=1e-8, max_seconds=5.0, max_evaluations=100000):
        if not math.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("invalid tolerance")
        if not math.isfinite(max_seconds) or max_seconds <= 0 or max_evaluations <= 0:
            raise ValueError("invalid reference limits")
        self.model = model
        self.tolerance = tolerance
        self.max_seconds = max_seconds
        self.max_evaluations = max_evaluations
        self.start = monotonic()
        self.evaluations = 0
        self.max_inner_error = 0.0

    def check(self):
        self.evaluations += 1
        if self.evaluations > self.max_evaluations:
            raise SearchLimitExceeded("adaptive reference exceeded evaluations")
        if monotonic() - self.start > self.max_seconds:
            raise SearchLimitExceeded("adaptive reference exceeded seconds")

    def integrate(self, fn):
        def bounded(y):
            self.check()
            value = float(fn(y))
            if not math.isfinite(value):
                raise ValueError("nonfinite adaptive integrand")
            return value

        result = quad(bounded, -np.inf, np.inf, epsabs=self.tolerance,
                      epsrel=self.tolerance, limit=100, full_output=1)
        if len(result) != 3:
            raise ValueError("adaptive integrator did not converge: " + result[3])
        value, error, _ = result
        if not math.isfinite(value) or not math.isfinite(error):
            raise ValueError("nonfinite adaptive integral")
        return value, error

    def density_parameters(self, state, action):
        return np.asarray([
            b.predictive(x[action]) for b, x in zip(
                state.components, self.model.action_features, strict=True)
        ]).T

    def terminal(self, state, action):
        m = self.model
        df, loc, scale2 = self.density_parameters(state, action)
        logs = m._state(state)
        predictions, slopes = [], []
        for b, x, targets in zip(state.components, m.action_features,
                                 m.target_features, strict=True):
            phi = x[action]
            solve = np.linalg.solve(np.asarray(b.precision), phi)
            gain = solve / (1 + phi @ solve)
            predictions.append(targets @ np.asarray(b.mean))
            slopes.append(targets @ gain)
        predictions, slopes = np.asarray(predictions), np.asarray(slopes)

        def residual(y):
            joint = logs + t.logpdf(y, df, loc=loc, scale=np.sqrt(scale2))
            total = logsumexp(joint)
            weights = np.exp(joint - total)
            means = predictions + slopes * (y - loc)[:, None]
            mean = weights @ means
            between = weights @ ((means - mean) ** 2 @ m.target_weights)
            return np.exp(total) * between

        value, error = self.integrate(residual)
        self.max_inner_error = max(self.max_inner_error, error)
        return m.action_risk_lower_bound(state, action, 1) + value, error

    def action(self, state, action, depth):
        if depth == 1:
            return self.terminal(state, action)
        if depth != 2:
            raise ValueError("adaptive diagnostic supports only depth one/two")
        m = self.model
        df, loc, scale2 = self.density_parameters(state, action)
        logs = m._state(state)

        def residual(y):
            probability = np.exp(logsumexp(
                logs + t.logpdf(y, df, loc=loc, scale=np.sqrt(scale2))))
            child = m.condition(state, action, y)
            value = min(self.terminal(child, a)[0] for a in range(m.num_actions))
            return probability * (value - m.state_risk_lower_bound(child, 1))

        value, error = self.integrate(residual)
        return m.action_risk_lower_bound(state, action, 2) + value, error
