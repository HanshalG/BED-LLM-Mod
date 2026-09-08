"""Fixed-structure predictive mixture for the ordinary horizon planner.

Continuous observations use deterministic Student-t Jacobi quadrature, not
categorical likelihoods. Parameter and model updates retain the raw observation.
Quadrature order must be qualified separately before scientific use.
"""

from dataclasses import dataclass
import math

import numpy as np
from scipy.special import logsumexp, roots_jacobi

from environments.chembench_mopen.horizon import BeliefBranch, _integer
from .regression_belief import RegressionBelief


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
        return float(self.moments(state)[1] @ self.target_weights)

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

    def branches(self, state, action):
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
        return tuple(
            BeliefBranch(y, p / total, self.condition(state, action, y))
            for y, p in sorted(masses.items())
            if p > 0
        )
