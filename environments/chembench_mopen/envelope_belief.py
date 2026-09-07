"""Equal-noise upper-envelope splits, with the complete predictive mixture.

This reduces integration knots, not particles. General unequal-noise cases use
the previously qualified all-crossing rule. Approximation still needs refinement.
"""

import math
import numpy as np
from scipy.special import ndtr

from .crossing_belief import CrossingGaussianModel, _gauss
from .horizon import SearchLimitExceeded


class EnvelopeGaussianModel(CrossingGaussianModel):
    def quadrature_rule(self, state, action):
        logs = self._logs(state)
        action = self._action(action)
        weights = np.exp(logs)
        active = weights > 0
        mu = self.means[active, action]
        sigma = self.sigmas[active, action]
        if not np.all(sigma == sigma[0]):
            return super().quadrature_rule(state, action)
        variance = sigma[0] ** 2
        starts = self._envelope_starts(mu, variance, logs[active])
        cuts = [0.0]
        for x in starts[1:]:
            u = float(weights[active] @ ndtr((x - mu) / sigma))
            if u - cuts[-1] > 1e-14 and 1 - u > 1e-14:
                cuts.append(u)
        cuts.append(1.0)
        intervals = len(cuts) - 1
        if 2 * intervals > self.branch_count:
            raise SearchLimitExceeded("density envelope exceeds quadrature budget")
        orders = np.full(intervals, self.branch_count // intervals, dtype=int)
        orders[: self.branch_count % intervals] += 1
        us, ws = [], []
        for low, high, n in zip(cuts[:-1], cuts[1:], orders):
            nodes, masses = _gauss(int(n))
            us.extend(low + (high - low) * (nodes + 1) / 2)
            ws.extend((high - low) * masses / 2)
        return np.array(us), np.array(ws)

    @staticmethod
    def _envelope_starts(mu, variance, logs):
        # Subtracting the common -y^2/(2*sigma^2) term makes log densities
        # straight lines. The upper hull has at most one segment per particle.
        lines = sorted(zip(mu / variance, logs - 0.5 * mu**2 / variance))
        if not all(math.isfinite(v) for line in lines for v in line):
            raise ValueError("unrepresentable density envelope")
        unique = []
        for slope, intercept in lines:
            if unique and slope == unique[-1][0]:
                unique[-1] = (slope, max(intercept, unique[-1][1]))
            else:
                unique.append((slope, intercept))
        hull, starts = [], []
        for slope, intercept in unique:
            start = -math.inf
            while hull:
                start = (hull[-1][1] - intercept) / (slope - hull[-1][0])
                if start > starts[-1]:
                    break
                hull.pop()
                starts.pop()
            hull.append((slope, intercept))
            starts.append(start if len(hull) > 1 else -math.inf)
        return starts
