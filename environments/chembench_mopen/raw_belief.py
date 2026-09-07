"""Raw Gaussian measurements over a fixed, truth-independent particle prior.

Componentwise Gauss-Hermite quadrature approximates predictive integrals, not
the observation likelihood. Its particle-count times order branching factor
makes this a bounded reference adapter, not yet a scalable chemistry planner.
"""

from __future__ import annotations

import math
from typing import Hashable

import numpy as np
from numpy.polynomial.hermite import hermgauss

from .horizon import BeliefBranch, FiniteBeliefModel, _integer


class GaussianParticleModel:
    """Independent scalar Gaussian measurements conditional on each particle.

    States are normalized log weights. Zero prior support stays zero, but a
    tiny nonzero posterior is not permanently lost through exponentiation.
    Targets and their loss weights remain fixed across measurement choices.
    No parameter rejuvenation, hypothesis generation, or truth access occurs.
    """

    def __init__(
        self,
        means: np.ndarray,
        sigmas: np.ndarray,
        targets: np.ndarray,
        prior: np.ndarray,
        *,
        quadrature_order: int = 9,
        target_weights: np.ndarray | None = None,
    ) -> None:
        means = np.array(means, dtype=float, copy=True)
        if means.ndim != 2 or min(means.shape) == 0 or not np.isfinite(means).all():
            raise ValueError("means must be finite (particles, actions)")
        sigmas = np.broadcast_to(np.asarray(sigmas, dtype=float), means.shape).copy()
        if not np.isfinite(sigmas).all() or np.any(sigmas <= 0):
            raise ValueError("sigmas must be finite and positive")
        # Share target/prior validation and squared-error risk conventions.
        validated = FiniteBeliefModel(
            np.ones((*means.shape, 1)), targets, prior, target_weights=target_weights
        )
        self.num_particles, self.num_actions = means.shape
        self.targets = validated.targets
        self.target_weights = validated.target_weights
        self.quadrature_order = _integer(
            quadrature_order, "quadrature_order", minimum=1
        )
        if self.quadrature_order > 128:
            raise ValueError("quadrature_order must be <= 128")
        nodes, weights = hermgauss(self.quadrature_order)
        self._nodes = nodes * math.sqrt(2)
        self._masses = weights / math.sqrt(math.pi)
        means.setflags(write=False)
        sigmas.setflags(write=False)
        self.means, self.sigmas = means, sigmas
        with np.errstate(divide="ignore"):
            self.initial_state = tuple(np.log(validated.initial_state))

    def _logs(self, state: Hashable) -> np.ndarray:
        logs = np.asarray(state, dtype=float)
        if (
            logs.shape != (self.num_particles,)
            or np.isnan(logs).any()
            or np.isposinf(logs).any()
            or not math.isclose(float(np.logaddexp.reduce(logs)), 0, abs_tol=1e-12)
        ):
            raise ValueError("state must be normalized log weights")
        return logs

    def _action(self, action: int) -> int:
        action = _integer(action, "action")
        if action >= self.num_actions:
            raise ValueError("action is out of range")
        return action

    def forecast(self, state: Hashable) -> np.ndarray:
        return np.exp(self._logs(state)) @ self.targets

    def risk(self, state: Hashable) -> float:
        weights = np.exp(self._logs(state))
        mean = weights @ self.targets
        return float(weights @ ((self.targets - mean) ** 2 @ self.target_weights))

    def log_likelihood(self, action: int, observation: float) -> np.ndarray:
        action = self._action(action)
        if not math.isfinite(observation):
            raise ValueError("observation must be finite")
        with np.errstate(over="ignore", invalid="ignore"):
            residual = (observation - self.means[:, action]) / self.sigmas[:, action]
            values = (
                -0.5 * residual**2
                - np.log(self.sigmas[:, action])
                - 0.5 * math.log(2 * math.pi)
            )
        if not np.isfinite(values).all():
            raise ValueError("measurement exceeds representable likelihood range")
        return values

    def condition(self, state: Hashable, action: int, observation: float) -> tuple:
        logs = self._logs(state) + self.log_likelihood(action, observation)
        logs -= np.logaddexp.reduce(logs)
        return tuple(float(value) for value in logs)

    def branches(self, state: Hashable, action: int) -> tuple[BeliefBranch, ...]:
        action = self._action(action)
        weights = np.exp(self._logs(state))
        # Merge identical nodes (e.g. identical predictive components) exactly.
        masses: dict[float, float] = {}
        for particle, weight in enumerate(weights):
            for node, mass in zip(self._nodes, self._masses):
                probability = float(weight * mass)
                if probability == 0:
                    continue
                observation = float(
                    self.means[particle, action] + self.sigmas[particle, action] * node
                )
                masses[observation] = masses.get(observation, 0.0) + probability
        total = math.fsum(masses.values())
        return tuple(
            BeliefBranch(value, mass / total, self.condition(state, action, value))
            for value, mass in sorted(masses.items())
        )
