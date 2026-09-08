"""Fixed-structure predictive mixture for the ordinary horizon planner.

Continuous observations use deterministic Student-t Jacobi quadrature, not
categorical likelihoods. Parameter and model updates retain the raw observation.
Quadrature order must be qualified separately before scientific use.
"""

from dataclasses import dataclass
from functools import lru_cache
import math

import numpy as np
from scipy.special import gammaln, logsumexp, roots_jacobi

from environments.chembench_mopen.horizon import BeliefBranch, _integer
from .regression_belief import RegressionBelief, _feature_solve


@dataclass(frozen=True)
class MixtureState:
    components: tuple[RegressionBelief, ...]
    log_weights: tuple[float, ...]


class RegressionMixture:
    def __init__(
        self,
        action_features,
        target_features,
        components,
        prior,
        *,
        target_weights,
        quadrature_order=8,
        include_observation_noise=False,
    ):
        self.components = tuple(components)
        count = len(self.components)
        if not 1 <= count <= 16 or not all(
            isinstance(b, RegressionBelief) for b in self.components
        ):
            raise ValueError("one to sixteen regression components required")
        if len(action_features) != count or len(target_features) != count:
            raise ValueError("features must cover every component")

        def features(values):
            rows = []
            for values_i, b in zip(values, self.components, strict=True):
                x = np.array(values_i, dtype=float, copy=True)
                if (
                    x.ndim != 2
                    or not 1 <= len(x) <= 1024
                    or x.shape[1] != len(b.mean)
                    or not np.isfinite(x).all()
                ):
                    raise ValueError("invalid feature matrix")
                x.setflags(write=False)
                rows.append(x)
            if len({len(x) for x in rows}) != 1:
                raise ValueError("feature row identities must align across models")
            return tuple(rows)

        self.action_features = features(action_features)
        self.target_features = features(target_features)
        self.num_actions = len(self.action_features[0])
        if self.num_actions > 8:
            raise ValueError("at most eight actions")
        self.quadrature_order = _integer(
            quadrature_order, "quadrature_order", minimum=2
        )
        if self.quadrature_order > 128:
            raise ValueError("quadrature order exceeds cap")
        if type(include_observation_noise) is not bool:
            raise ValueError("noise inclusion must be explicit boolean")
        self.include_observation_noise = include_observation_noise

        def weights(values, n):
            w = np.array(values, dtype=float, copy=True)
            if (
                w.shape != (n,)
                or not np.isfinite(w).all()
                or np.any(w < 0)
                or not math.isclose(float(w.sum()), 1, abs_tol=1e-12, rel_tol=0)
            ):
                raise ValueError("normalized nonnegative weights required")
            w.setflags(write=False)
            return w

        self.target_weights = weights(target_weights, len(self.target_features[0]))
        self._target_grams = tuple(
            x.T @ (self.target_weights[:, None] * x) for x in self.target_features
        )
        for gram in self._target_grams:
            gram.setflags(write=False)

        @lru_cache(maxsize=256)
        def target_leverage(component, precision):
            return float(
                np.trace(np.linalg.solve(precision, self._target_grams[component]))
            )

        self._target_leverage = target_leverage
        p = weights(prior, count)
        with np.errstate(divide="ignore"):
            self.initial_state = MixtureState(self.components, tuple(np.log(p)))

    def _state(self, state):
        if not isinstance(state, MixtureState) or len(state.components) != len(
            self.components
        ):
            raise ValueError("invalid mixture state")
        logs = np.asarray(state.log_weights, dtype=float)
        if (
            logs.shape != (len(self.components),)
            or np.isnan(logs).any()
            or np.isposinf(logs).any()
            or not math.isclose(float(logsumexp(logs)), 0, abs_tol=1e-12)
        ):
            raise ValueError("normalized log weights required")
        if any(
            not isinstance(b, RegressionBelief) or len(b.mean) != len(a.mean)
            for a, b in zip(self.components, state.components, strict=True)
        ):
            raise ValueError("component dimensions changed")
        return logs

    def _action(self, action):
        action = _integer(action, "action")
        if action >= self.num_actions:
            raise ValueError("invalid action")
        return action

    def moments(self, state):
        w = np.exp(self._state(state))
        predictions, variances = [], []
        for b, x in zip(state.components, self.target_features, strict=True):
            mean, variance = b.target_moments(x)
            predictions.append(mean)
            variances.append(
                variance + (b.noise_variance if self.include_observation_noise else 0)
            )
        predictions = np.asarray(predictions)
        mean = w @ predictions
        variance = w @ (np.asarray(variances) + (predictions - mean) ** 2)
        if not np.isfinite(mean).all() or not np.isfinite(variance).all():
            raise ValueError("mixture moments exceed numeric range")
        return mean, variance

    def forecast(self, state):
        return self.moments(state)[0]

    def risk(self, state):
        weights = np.exp(self._state(state))
        predictions, within = [], []
        for i, (belief, x) in enumerate(
            zip(state.components, self.target_features, strict=True)
        ):
            predictions.append(x @ belief.mean)
            within.append(
                belief.noise_variance
                * (
                    self._target_leverage(i, belief.precision)
                    + int(self.include_observation_noise)
                )
            )
        predictions = np.asarray(predictions)
        mean = weights @ predictions
        between = (predictions - mean) ** 2 @ self.target_weights
        result = float(weights @ (np.asarray(within) + between))
        if not math.isfinite(result) or result < 0:
            raise ValueError("mixture risk exceeds numeric range")
        return result

    def condition(self, state, action, observation):
        logs = self._state(state).copy()
        action = self._action(action)
        updated = []
        for i, (b, x) in enumerate(
            zip(state.components, self.action_features, strict=True)
        ):
            next_b, density = b.condition(x[action], observation)
            updated.append(next_b)
            logs[i] += density
        logs -= logsumexp(logs)
        return MixtureState(tuple(updated), tuple(float(v) for v in logs))

    def _quadrature(self, state, action):
        w = np.exp(self._state(state))
        action = self._action(action)
        masses = {}
        for weight, b, x in zip(w, state.components, self.action_features, strict=True):
            if weight == 0:
                continue
            df, loc, scale2 = b.predictive(x[action])
            # z = sqrt(df) u / sqrt(1-u^2) maps Student-t mass to
            # a normalized Jacobi weight (1-u^2)^(df/2-1) on (-1, 1).
            u, q = roots_jacobi(self.quadrature_order, df / 2 - 1, df / 2 - 1)
            q = q / q.sum()
            values = loc + np.sqrt(scale2 * df) * u / np.sqrt(1 - u * u)
            for y, probability in zip(values, weight * q, strict=True):
                y = float(y)
                masses[y] = masses.get(y, 0.0) + float(probability)
        total = math.fsum(masses.values())
        return tuple((y, p / total) for y, p in sorted(masses.items()) if p > 0)

    def branches(self, state, action):
        return tuple(
            BeliefBranch(y, p, self.condition(state, action, y))
            for y, p in self._quadrature(state, action)
        )

    def expected_terminal_risk(self, state, action):
        value, count, _, _ = self._terminal_risk_terms(state, action)
        return value, count

    def _terminal_risk_terms(self, state, action):
        """Same terminal nodes/likelihoods, with vectorized conjugate updates."""
        logs = self._state(state)
        action = self._action(action)
        rows = self._quadrature(state, action)
        y, masses = np.asarray(rows).T
        count, models, targets = len(y), len(state.components), len(self.target_weights)
        if count * models * (targets + 32) * 8 * 5 > 64 * 1024**2:
            raise ValueError("terminal batch workspace cap exceeded")
        predictions, within, densities, exact_within = [], [], [], []
        for i, (b, x, target) in enumerate(
            zip(
                state.components,
                self.action_features,
                self.target_features,
                strict=True,
            )
        ):
            phi = x[action]
            solved = _feature_solve(b.precision, tuple(phi))
            denominator = 1 + phi @ solved
            residual = y - phi @ b.mean
            df = 2 * b.shape
            scale2 = b.scale / b.shape * denominator
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                densities.append(
                    gammaln((df + 1) / 2)
                    - gammaln(df / 2)
                    - 0.5 * math.log(df * math.pi * scale2)
                    - (df + 1) / 2 * np.log1p(residual**2 / scale2 / df)
                )
                means = np.asarray(b.mean) + residual[:, None] * (solved / denominator)
                noise = (b.scale + 0.5 * residual**2 / denominator) / (b.shape - 0.5)
            precision = tuple(
                tuple(row) for row in np.asarray(b.precision) + np.outer(phi, phi)
            )
            within.append(
                noise
                * (
                    self._target_leverage(i, precision)
                    + int(self.include_observation_noise)
                )
            )
            predictions.append(means @ target.T)
            exact_within.append(
                b.noise_variance
                * (
                    self._target_leverage(i, precision)
                    + int(self.include_observation_noise)
                )
            )
        posterior = np.asarray(densities).T + logs
        posterior = np.exp(posterior - logsumexp(posterior, axis=1)[:, None])
        predictions = np.asarray(predictions).transpose(1, 0, 2)
        average = np.sum(posterior[:, :, None] * predictions, axis=1)
        between = (predictions - average[:, None, :]) ** 2 @ self.target_weights
        risks = np.sum(posterior * (np.asarray(within).T + between), axis=1)
        value = float(masses @ risks)
        if (
            not np.isfinite(risks).all()
            or np.any(risks < 0)
            or not math.isfinite(value)
        ):
            raise ValueError("terminal risk exceeds numeric range")
        sampled_within = float(
            masses @ np.sum(posterior * np.asarray(within).T, axis=1)
        )
        analytic_within = float(np.exp(logs) @ exact_within)
        return value, len(rows), sampled_within, analytic_within
