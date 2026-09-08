"""Remaining-horizon potential and bounds for its corrected numerical objective."""

from functools import lru_cache
from itertools import combinations_with_replacement

import numpy as np
from scipy.special import logsumexp
from scipy.stats import t

from environments.chembench_mopen.horizon import _integer
from .control_variate import ControlVariateMixture


class HorizonControlVariateMixture(ControlVariateMixture):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._coefficient = lru_cache(maxsize=1024)(self._coefficient_uncached)
        self._moment_error = lru_cache(maxsize=256)(self._moment_error_uncached)

    def _coefficient_uncached(self, component, precision, steps):
        best = np.inf
        features = self.action_features[component]
        for sequence in combinations_with_replacement(range(self.num_actions), steps):
            updated = np.asarray(precision).copy()
            for action in sequence:
                updated += np.outer(features[action], features[action])
            value = float(
                np.trace(np.linalg.solve(updated, self._target_grams[component]))
            )
            best = min(best, value)
        return best + int(self.include_observation_noise)

    def _moment_error_uncached(self, state, action):
        logs = self._state(state)
        nodes, masses = np.asarray(self._quadrature(state, action)).T
        densities, noise, exact = [], [], []
        for b, features in zip(state.components, self.action_features, strict=True):
            phi = features[action]
            df, mean, scale2 = b.predictive(phi)
            densities.append(t.logpdf(nodes, df, loc=mean, scale=np.sqrt(scale2)))
            denominator = scale2 * b.shape / b.scale
            noise.append(
                (b.scale + 0.5 * (nodes - mean) ** 2 / denominator) / (b.shape - 0.5)
            )
            exact.append(b.noise_variance)
        posterior = np.asarray(densities).T + logs
        posterior = np.exp(posterior - logsumexp(posterior, axis=1)[:, None])
        error = np.exp(logs) * exact - masses @ (posterior * np.asarray(noise).T)
        if not np.isfinite(error).all():
            raise ValueError("invalid noise-moment error")
        return tuple(error)

    def horizon_chance_risk_correction(self, state, action, depth):
        depth = _integer(depth, "depth", minimum=1)
        if depth > 3:
            raise ValueError("horizon control variate limited to depth three")
        action = self._action(action)
        self._state(state)
        if depth == 1:
            return self.chance_risk_correction(state, action)
        coefficients = []
        for i, (b, features) in enumerate(
            zip(state.components, self.action_features, strict=True)
        ):
            phi = features[action]
            precision = tuple(
                tuple(row) for row in np.asarray(b.precision) + np.outer(phi, phi)
            )
            coefficients.append(self._coefficient(i, precision, depth - 1))
        value = float(np.dot(self._moment_error(state, action), coefficients))
        if not np.isfinite(value):
            raise ValueError("invalid horizon correction")
        return value

    def action_risk_lower_bound(self, state, action, depth):
        """Family-revelation relaxation, valid for the positive corrected operator."""
        depth = _integer(depth, "depth", minimum=1)
        if depth > 3:
            raise ValueError("horizon bound limited to depth three")
        action = self._action(action)
        weights = np.exp(self._state(state))
        values = []
        for i, (b, features) in enumerate(
            zip(state.components, self.action_features, strict=True)
        ):
            phi = features[action]
            precision = tuple(
                tuple(row) for row in np.asarray(b.precision) + np.outer(phi, phi)
            )
            values.append(b.noise_variance * self._coefficient(i, precision, depth - 1))
        value = float(np.dot(weights, values))
        if not np.isfinite(value) or value < 0:
            raise ValueError("invalid family action bound")
        return value
