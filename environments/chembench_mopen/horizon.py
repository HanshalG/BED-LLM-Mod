"""Bounded, ordinary-horizon reference planning on a fixed predictive model.

This module performs no proposal generation and has no access to an episode's
realized latent world. Finite observations are exact; other branch providers must
state their integration approximation explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations, product
import math
from numbers import Integral
from time import monotonic
from typing import Hashable, Literal, Protocol

import numpy as np


Observation = int | float | str | None
Mode = Literal["adaptive", "open_loop"]


@dataclass(frozen=True)
class BeliefBranch:
    observation: Observation
    probability: float
    state: Hashable


class PredictiveModel(Protocol):
    """Risk and branches must be deterministic for a state during one plan.

    Numerical branch approximations must use a frozen rule or state-keyed seed;
    resampling on each invocation would change the objective during optimization.

    Optional chance_risk_correction(state, action) adds a signed integration
    control variate, recorded explicitly on each returned PolicyNode. An optional
    expected_terminal_risk(state, action) batch hook must already include that
    correction and return (value, evaluated_leaf_count).
    horizon_chance_risk_correction(state, action, depth), when supplied, takes
    precedence over the depth-independent correction hook.
    Optional action_risk_lower_bound(state, action, depth) must bound this
    model's numerical objective for every allowed menu (including repeats).
    It is only used with explicit adaptive use_action_bounds=True. Providers
    are responsible for its validity; evaluated violations fail closed.
    """

    num_actions: int

    def risk(self, state: Hashable) -> float: ...

    def branches(self, state: Hashable, action: int) -> tuple[BeliefBranch, ...]: ...


class SearchLimitExceeded(RuntimeError):
    """No completed plan is returned when the reference search exceeds its cap."""


def _integer(value: int, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


@dataclass(frozen=True)
class SearchLimits:
    max_nodes: int = 50_000
    max_seconds: float = 30.0
    cache_size: int = 4_096
    max_depth: int = 8

    def __post_init__(self) -> None:
        for name in ("max_nodes", "cache_size", "max_depth"):
            _integer(getattr(self, name), name, minimum=1)
        if not math.isfinite(self.max_seconds) or self.max_seconds <= 0:
            raise ValueError("max_seconds must be finite and positive")


@dataclass(frozen=True)
class PolicyEdge:
    observation: Observation
    probability: float
    child: PolicyNode


@dataclass(frozen=True)
class PolicyNode:
    action: int | None
    expected_risk: float
    remaining_depth: int
    branches: tuple[PolicyEdge, ...] = ()
    quadrature_correction: float = 0.0


@dataclass(frozen=True)
class HorizonPlan:
    mode: Mode
    requested_horizon: int
    effective_horizon: int
    root: PolicyNode
    fixed_sequence: tuple[int, ...] | None
    root_action_values: tuple[tuple[int, float], ...]
    expanded_nodes: int
    elapsed_seconds: float
    root_pruned_lower_bounds: tuple[tuple[int, float], ...] = ()
    pruned_actions: int = 0


class FiniteBeliefModel:
    """Static latent particles, conditionally independent categorical measurements.

    Targets have fixed weights, independent of selected measurement actions.
    Each particle can represent a structure AND a parameter setting. The prior
    and likelihoods specify the whole reference model, not an oracle truth index.
    """

    def __init__(
        self,
        likelihoods: np.ndarray,
        targets: np.ndarray,
        prior: np.ndarray,
        *,
        target_weights: np.ndarray | None = None,
    ) -> None:
        likelihoods = np.array(likelihoods, dtype=float, copy=True)
        targets = np.array(targets, dtype=float, copy=True)
        prior = np.array(prior, dtype=float, copy=True)
        if likelihoods.ndim != 3 or min(likelihoods.shape) == 0:
            raise ValueError(
                "likelihoods must have shape (particles, actions, outcomes)"
            )
        if (
            not np.isfinite(likelihoods).all()
            or np.any(likelihoods < 0)
            or not np.allclose(likelihoods.sum(axis=2), 1.0, atol=1e-12, rtol=0)
        ):
            raise ValueError("likelihood rows must be finite probability distributions")
        if (
            targets.ndim != 2
            or targets.shape[0] != likelihoods.shape[0]
            or targets.shape[1] == 0
            or not np.isfinite(targets).all()
        ):
            raise ValueError("targets must be finite with shape (particles, targets)")
        if target_weights is None:
            target_weights = np.full(targets.shape[1], 1.0 / targets.shape[1])
        weights = np.array(target_weights, dtype=float, copy=True)
        self.num_particles, self.num_actions, self.num_outcomes = likelihoods.shape
        self._validate_weights(weights, targets.shape[1])
        self._validate_weights(prior, self.num_particles)
        for array in (likelihoods, targets, weights):
            array.setflags(write=False)
        self.likelihoods = likelihoods
        self.targets = targets
        self.target_weights = weights
        self.initial_state = tuple(float(value) for value in prior)

    @staticmethod
    def _validate_weights(weights: np.ndarray, count: int) -> None:
        if (
            weights.shape != (count,)
            or not np.isfinite(weights).all()
            or np.any(weights < 0)
            or not math.isclose(float(weights.sum()), 1.0, abs_tol=1e-12, rel_tol=0)
        ):
            raise ValueError("weights must be a normalized nonnegative vector")

    def _weights(self, state: Hashable) -> np.ndarray:
        weights = np.asarray(state, dtype=float)
        self._validate_weights(weights, self.num_particles)
        return weights

    def _action(self, action: int) -> int:
        action = _integer(action, "action")
        if action >= self.num_actions:
            raise ValueError("action is out of range")
        return action

    def forecast(self, state: Hashable) -> np.ndarray:
        return self._weights(state) @ self.targets

    def risk(self, state: Hashable) -> float:
        weights = self._weights(state)
        mean = weights @ self.targets
        return float(weights @ ((self.targets - mean) ** 2 @ self.target_weights))

    def condition(
        self, state: Hashable, action: int, observation: int
    ) -> tuple[float, ...]:
        action = self._action(action)
        observation = _integer(observation, "observation")
        if observation >= self.num_outcomes:
            raise ValueError("observation is out of range")
        masses = self._weights(state) * self.likelihoods[:, action, observation]
        probability = float(masses.sum())
        if probability <= 0:
            raise ValueError("observation has zero predictive probability")
        return tuple(float(value) for value in masses / probability)

    def branches(self, state: Hashable, action: int) -> tuple[BeliefBranch, ...]:
        action = self._action(action)
        probabilities = self._weights(state) @ self.likelihoods[:, action, :]
        return tuple(
            BeliefBranch(
                outcome, float(probability), self.condition(state, action, outcome)
            )
            for outcome, probability in enumerate(probabilities)
            if probability > 0
        )


class HorizonPlanner:
    """Optimize all contingent decisions, or one precommitted action sequence.

    Caches live for one planning call only. Time caps are checked between model
    operations; an individual provider operation must itself be bounded.
    """

    def __init__(self, model: PredictiveModel, *, limits: SearchLimits | None = None):
        self.model = model
        self.num_actions = _integer(model.num_actions, "num_actions", minimum=1)
        self.limits = limits or SearchLimits()

    def plan(
        self,
        state: Hashable,
        horizon: int,
        *,
        available: tuple[int, ...] | None = None,
        mode: Mode = "adaptive",
        allow_repeats: bool = False,
        use_action_bounds: bool = False,
    ) -> HorizonPlan:
        horizon = _integer(horizon, "horizon")
        if mode not in ("adaptive", "open_loop"):
            raise ValueError("mode must be adaptive or open_loop")
        if not isinstance(allow_repeats, bool):
            raise ValueError("allow_repeats must be boolean")
        if not isinstance(use_action_bounds, bool):
            raise ValueError("use_action_bounds must be boolean")
        bound_hook = getattr(self.model, "action_risk_lower_bound", None)
        if use_action_bounds and (mode != "adaptive" or not callable(bound_hook)):
            raise ValueError("action bounds require adaptive mode and a model bound hook")
        if available is None:
            available = tuple(range(self.num_actions))
        actions = tuple(_integer(action, "action") for action in available)
        if len(set(actions)) != len(actions) or any(
            a >= self.num_actions for a in actions
        ):
            raise ValueError("available actions must be unique and in range")
        actions = tuple(sorted(actions))
        effective = (
            (horizon if allow_repeats else min(horizon, len(actions))) if actions else 0
        )
        if effective > self.limits.max_depth:
            raise SearchLimitExceeded("requested effective horizon exceeds max_depth")
        hash(state)
        start = monotonic()
        nodes = 0
        pruned_actions = 0

        def check(*, expand: bool = False) -> None:
            nonlocal nodes
            nodes += int(expand)
            if nodes > self.limits.max_nodes:
                raise SearchLimitExceeded("reference search exceeded max_nodes")
            if monotonic() - start > self.limits.max_seconds:
                raise SearchLimitExceeded("reference search exceeded max_seconds")

        def risk(belief: Hashable) -> float:
            value = float(self.model.risk(belief))
            if not math.isfinite(value) or value < 0:
                raise ValueError("model risk must be finite and nonnegative")
            return value

        def branches(belief: Hashable, action: int) -> tuple[BeliefBranch, ...]:
            rows = tuple(self.model.branches(belief, action))
            if (
                not rows
                or any(
                    not math.isfinite(row.probability) or row.probability <= 0
                    for row in rows
                )
                or not math.isclose(
                    math.fsum(row.probability for row in rows),
                    1,
                    abs_tol=1e-12,
                    rel_tol=0,
                )
                or len({row.observation for row in rows}) != len(rows)
            ):
                raise ValueError(
                    "branches must have distinct observations and normalized positive mass"
                )
            for row in rows:
                hash(row.state)
                if isinstance(row.observation, float) and not math.isfinite(
                    row.observation
                ):
                    raise ValueError("branch observation must be finite")
            check()
            return rows

        def remainder(menu: tuple[int, ...], action: int) -> tuple[int, ...]:
            return menu if allow_repeats else tuple(a for a in menu if a != action)

        def correction(belief: Hashable, action: int, depth: int) -> float:
            horizon_hook = getattr(self.model, "horizon_chance_risk_correction", None)
            hook = getattr(self.model, "chance_risk_correction", None)
            if callable(horizon_hook):
                value = float(horizon_hook(belief, action, depth))
            else:
                value = float(hook(belief, action)) if callable(hook) else 0.0
            if not math.isfinite(value):
                raise ValueError("quadrature correction must be finite")
            check()
            return value

        def corrected(value: float, offset: float) -> float:
            value += offset
            if not math.isfinite(value) or value < 0:
                raise ValueError("corrected risk must be finite and nonnegative")
            return value

        def evaluate_actions(belief, menu, depth):
            nonlocal pruned_actions
            ordered = []
            for a in menu:
                lower = float(bound_hook(belief, a, depth)) if use_action_bounds else 0.0
                if not math.isfinite(lower) or lower < 0:
                    raise ValueError("action lower bound must be finite and nonnegative")
                ordered.append((lower, a))
                check()
            values, pruned = {}, {}
            incumbent = math.inf
            for lower, a in sorted(ordered):
                # Strict separation preserves ties and absorbs floating-point noise.
                margin = 1e-10 * max(1.0, abs(lower), abs(incumbent))
                if use_action_bounds and lower > incumbent + margin:
                    pruned[a] = lower
                    pruned_actions += 1
                    continue
                value = action_value(belief, menu, depth, a)
                if use_action_bounds and lower > value + 1e-10 * max(1.0, abs(value), lower):
                    raise ValueError("evaluated risk violates model action lower bound")
                values[a] = value
                incumbent = min(incumbent, value)
            return values, pruned

        @lru_cache(maxsize=self.limits.cache_size)
        def choose(
            belief: Hashable, menu: tuple[int, ...], depth: int
        ) -> tuple[float, int | None]:
            check(expand=True)
            if depth == 0:
                return risk(belief), None
            values, _ = evaluate_actions(belief, menu, depth)
            return min((value, a) for a, value in values.items())

        def action_value(
            belief: Hashable, menu: tuple[int, ...], depth: int, action: int
        ) -> float:
            terminal = getattr(self.model, "expected_terminal_risk", None)
            if depth == 1 and callable(terminal):
                value, leaf_count = terminal(belief, action)
                leaf_count = _integer(leaf_count, "terminal leaf count", minimum=1)
                value = float(value)
                if not math.isfinite(value) or value < 0:
                    raise ValueError("terminal risk must be finite and nonnegative")
                # Charge the evaluated leaves even when their states are batched.
                for _ in range(leaf_count):
                    check(expand=True)
                return value
            next_menu = remainder(menu, action)
            return corrected(
                math.fsum(
                    row.probability * choose(row.state, next_menu, depth - 1)[0]
                    for row in branches(belief, action)
                ),
                correction(belief, action, depth),
            )

        @lru_cache(maxsize=self.limits.cache_size)
        def sequence_value(belief: Hashable, sequence: tuple[int, ...]) -> float:
            check(expand=True)
            if not sequence:
                return risk(belief)
            if len(sequence) == 1:
                return action_value(belief, sequence, 1, sequence[0])
            return corrected(
                math.fsum(
                    row.probability * sequence_value(row.state, sequence[1:])
                    for row in branches(belief, sequence[0])
                ),
                correction(belief, sequence[0], len(sequence)),
            )

        def materialize(
            belief: Hashable,
            menu: tuple[int, ...],
            depth: int,
            sequence: tuple[int, ...] | None,
            known_action: int | None = None,
        ) -> PolicyNode:
            check(expand=True)
            if depth == 0:
                return PolicyNode(None, risk(belief), 0)
            action = known_action
            if action is None:
                action = (
                    sequence[0]
                    if sequence is not None
                    else choose(belief, menu, depth)[1]
                )
            assert action is not None
            children = tuple(
                PolicyEdge(
                    row.observation,
                    row.probability,
                    materialize(
                        row.state,
                        remainder(menu, action),
                        depth - 1,
                        None if sequence is None else sequence[1:],
                    ),
                )
                for row in branches(belief, action)
            )
            value = math.fsum(
                edge.probability * edge.child.expected_risk for edge in children
            )
            offset = correction(belief, action, depth)
            return PolicyNode(action, corrected(value, offset), depth, children, offset)

        try:
            fixed = None
            root_values: dict[int, float] = {}
            root_pruned: dict[int, float] = {}
            if mode == "open_loop":
                sequences = (
                    product(actions, repeat=effective)
                    if allow_repeats
                    else permutations(actions, effective)
                )
                best: tuple[float, tuple[int, ...]] | None = None
                for sequence in sequences:
                    check()
                    value = sequence_value(state, sequence)
                    if sequence:
                        root_values[sequence[0]] = min(
                            root_values.get(sequence[0], math.inf), value
                        )
                    candidate = value, sequence
                    if best is None or candidate < best:
                        best = candidate
                assert best is not None
                fixed = best[1]
            elif effective:
                root_values, root_pruned = evaluate_actions(state, actions, effective)
            # Root values were already computed even when the bounded cache has
            # evicted child states. Do not optimize the root again for display.
            known_action = (
                min((v, a) for a, v in root_values.items())[1]
                if root_values and fixed is None
                else None
            )
            root = materialize(state, actions, effective, fixed, known_action)
            check()
            return HorizonPlan(
                mode,
                horizon,
                effective,
                root,
                fixed,
                tuple(sorted(root_values.items())),
                nodes,
                monotonic() - start,
                tuple(sorted(root_pruned.items())),
                pruned_actions,
            )
        finally:
            choose.cache_clear()
            sequence_value.cache_clear()
