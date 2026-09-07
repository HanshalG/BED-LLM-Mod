"""Composite quadrature split at Gaussian posterior-density crossings."""

from functools import lru_cache
import math
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr

from .batch_horizon import posterior_branches_many
from .horizon import BeliefBranch, SearchLimitExceeded
from .quantile_belief import QuantileGaussianModel


@lru_cache(maxsize=128)
def _gauss(order):
    nodes, weights = leggauss(order)
    nodes.setflags(write=False)
    weights.setflags(write=False)
    return nodes, weights


class CrossingGaussianModel(QuantileGaussianModel):
    """Splits guide integration, never change likelihoods or prune particles.

    Too many crossings fail closed. This remains a numerical approximation.
    """

    def quadrature_rule(self, state, action):
        logs = self._logs(state)
        action = self._action(action)
        active = np.flatnonzero(np.exp(logs) > 0)
        mu, sigma = self.means[:, action], self.sigmas[:, action]
        weights = np.exp(logs)
        cuts = {0.0, 1.0}
        for offset, i in enumerate(active):
            for j in active[offset + 1 :]:
                a = 0.5 * (1 / sigma[j] ** 2 - 1 / sigma[i] ** 2)
                b = mu[i] / sigma[i] ** 2 - mu[j] / sigma[j] ** 2
                c = (
                    logs[i]
                    - logs[j]
                    - math.log(sigma[i] / sigma[j])
                    - 0.5 * (mu[i] / sigma[i]) ** 2
                    + 0.5 * (mu[j] / sigma[j]) ** 2
                )
                if not all(math.isfinite(v) for v in (a, b, c)):
                    raise ValueError("unrepresentable crossing polynomial")
                if a == 0:
                    roots = [] if b == 0 else [-c / b]
                else:
                    scale = max(abs(a), abs(b), abs(c))
                    aa, bb, cc = a / scale, b / scale, c / scale
                    disc = bb * bb - 4 * aa * cc
                    if disc < 0:
                        roots = []
                    else:
                        q = -0.5 * (bb + math.copysign(math.sqrt(disc), bb))
                        roots = [-bb / (2 * aa)] if q == 0 else [q / aa, cc / q]
                for y in roots:
                    if math.isfinite(y):
                        u = float(weights @ ndtr((y - mu) / sigma))
                        if 0 < u < 1:
                            cuts.add(u)
        knots = [0.0]
        for cut in sorted(cuts)[1:]:
            # Coalesce only numerically tiny quadrature intervals; no change to
            # physical likelihoods, support, or real posterior conditioning.
            if cut - knots[-1] > 1e-14:
                knots.append(cut)
            elif cut == 1:
                knots[-1] = 1.0
        intervals = len(knots) - 1
        if 2 * intervals > self.branch_count:
            raise SearchLimitExceeded("crossings exceed quadrature budget")
        orders = np.full(intervals, self.branch_count // intervals, dtype=int)
        orders[: self.branch_count % intervals] += 1
        us, ws = [], []
        for low, high, order in zip(knots[:-1], knots[1:], orders):
            nodes, masses = _gauss(int(order))
            us.extend(low + (high - low) * (nodes + 1) / 2)
            ws.extend((high - low) * masses / 2)
        return np.array(us), np.array(ws)

    def branches(self, state, action):
        observations, posterior, weights = posterior_branches_many(
            self, np.asarray(state)[None, :], action, return_weights=True
        )
        masses, states = {}, {}
        for y, p, logs in zip(observations[0], weights[0], posterior[0]):
            y = float(y)
            masses[y] = masses.get(y, 0) + float(p)
            states[y] = tuple(float(v) for v in logs)
        return tuple(BeliefBranch(y, p, states[y]) for y, p in sorted(masses.items()))
