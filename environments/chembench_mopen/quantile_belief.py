"""Fixed-size predictive-quantile quadrature for bounded contingent planning.

This changes numerical integration only. Real and simulated posterior updates
still use the full scalar Gaussian likelihood. It is an approximation, with no
intrinsic error certificate; decisions require separate refinement checks.
"""

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import ndtr, ndtri

from .horizon import BeliefBranch, _integer
from .raw_belief import GaussianParticleModel


class QuantileGaussianModel(GaussianParticleModel):
    def __init__(self, *args, branch_count=16, **kwargs):
        branch_count = _integer(branch_count, "branch_count", minimum=2)
        if branch_count > 128:
            raise ValueError("branch_count must be <=128")
        super().__init__(*args, **kwargs)
        nodes, weights = leggauss(branch_count)
        self._quantiles = (nodes + 1) / 2
        self._quadrature_weights = weights / 2
        self.branch_count = branch_count

    def branches(self, state, action):
        action = self._action(action)
        weights = np.exp(self._logs(state))
        active = weights > 0
        weights = weights[active]
        means = self.means[active, action]
        sigmas = self.sigmas[active, action]
        component_quantiles = means[:, None] + sigmas[:, None] * ndtri(self._quantiles)
        low = component_quantiles.min(axis=0)
        high = component_quantiles.max(axis=0)
        if not np.isfinite(low).all() or not np.isfinite(high).all():
            raise ValueError("predictive quantiles exceed numerical range")
        # Component quantiles bracket the mixture quantile. Vectorized bisection
        # uses bounded workspace O(particles*branches), independent of horizon.
        for _ in range(64):
            middle = low / 2 + high / 2
            cdf = weights @ ndtr((middle[None, :] - means[:, None]) / sigmas[:, None])
            low = np.where(cdf < self._quantiles, middle, low)
            high = np.where(cdf >= self._quantiles, middle, high)
        observations = low / 2 + high / 2
        cdf = weights @ ndtr((observations[None, :] - means[:, None]) / sigmas[:, None])
        if np.max(np.abs(cdf - self._quantiles)) > 1e-8:
            raise ArithmeticError("predictive quantile inversion failed")
        masses = {}
        for observation, probability in zip(observations, self._quadrature_weights):
            value = float(observation)
            masses[value] = masses.get(value, 0.0) + float(probability)
        return tuple(
            BeliefBranch(y, p, self.condition(state, action, y))
            for y, p in sorted(masses.items())
        )
