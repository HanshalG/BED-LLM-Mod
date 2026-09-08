"""Counted generic switch partition for a depth-two diagnostic reference."""

from functools import lru_cache
from itertools import combinations

import numpy as np
from scipy.optimize import brentq

from .adaptive_reference import AdaptiveReference, student_log_density


class SwitchReference(AdaptiveReference):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.switches = []

    def action(self, state, action, depth):
        if depth != 2:
            return super().action(state, action, depth)
        m = self.model
        df, loc, scale2 = self.density_parameters(state, action)
        logs = m._state(state)
        density = student_log_density(df, loc, scale2)
        coordinates = self.coordinates(df, loc, scale2, logs)
        center, scale = coordinates['center'], coordinates['scale']

        @lru_cache(maxsize=256)
        def values(y):
            # Count scan/root-solver states as well as all nested integrands.
            self.check()
            child = m.condition(state, action, y)
            scores = tuple(self.terminal(child, a)[0] for a in range(m.num_actions))
            return scores, m.state_risk_lower_bound(child, 1)

        def residual(y):
            scores, lower = values(float(y))
            return np.exp(np.logaddexp.reduce(logs + density(y))) * (min(scores)-lower)

        try:
            grid = tuple(float(center + scale*z) for z in np.linspace(-8, 8, 17))
            sampled = [values(y)[0] for y in grid]
            points = set()
            for a, b in combinations(range(m.num_actions), 2):
                gaps = [s[a]-s[b] for s in sampled]
                for i, y in enumerate(grid[:-1]):
                    if gaps[i] == 0:
                        points.add(y)
                    if gaps[i]*gaps[i+1] < 0:
                        def gap(x):
                            scores, _ = values(float(x))
                            return scores[a]-scores[b]
                        points.add(float(brentq(gap, y, grid[i+1], xtol=1e-8,
                                                 maxiter=32)))
            self.switches.append(dict(action=action, points=sorted(points)))
            result, error = self.integrate(residual, points=points, **coordinates)
            return m.action_risk_lower_bound(state, action, 2)+result, error
        finally:
            values.cache_clear()
