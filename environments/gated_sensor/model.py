"""Exact diagnosis model with setup actions that unlock precise measurements.

The hidden fault is a five-bit code. Weak screens are always available, while a
zero-information panel activation is required before the panel's precise tests
can be used. This makes the value of an activation entirely non-myopic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import itertools
from typing import Final

import numpy as np


EPSILON: Final = 1e-15
OUTCOMES: Final = ("positive", "negative")


@dataclass(frozen=True)
class Predicate:
    name: str
    bit_indices: tuple[int, ...]

    def evaluate(self, hidden_state: tuple[int, ...]) -> bool:
        return bool(sum(hidden_state[index] for index in self.bit_indices) % 2)


@dataclass(frozen=True)
class SensorState:
    active_panel: str | None = None


class GatedSensorModel:
    """Finite exact-belief model for sequential fault diagnosis."""

    def __init__(self, *, screen_accuracy: float = 0.65, precise_accuracy: float = 0.95) -> None:
        if not (0.5 < screen_accuracy < precise_accuracy < 1.0):
            raise ValueError("accuracies must satisfy 0.5 < screen_accuracy < precise_accuracy < 1")
        self.screen_accuracy = screen_accuracy
        self.precise_accuracy = precise_accuracy
        self.predicates = (
            Predicate("bit-0", (0,)),
            Predicate("bit-1", (1,)),
            Predicate("bit-2", (2,)),
            Predicate("bit-3", (3,)),
            Predicate("bit-4", (4,)),
            Predicate("xor-0-1", (0, 1)),
            Predicate("xor-1-2", (1, 2)),
            Predicate("xor-0-2", (0, 2)),
            Predicate("xor-2-3", (2, 3)),
            Predicate("xor-3-4", (3, 4)),
            Predicate("xor-2-4", (2, 4)),
            Predicate("xor-0-3", (0, 3)),
            Predicate("xor-0-4", (0, 4)),
        )
        self.panels: dict[str, tuple[str, ...]] = {
            "A": ("bit-0", "bit-1", "bit-2", "xor-0-1", "xor-1-2", "xor-0-2"),
            "B": ("bit-2", "bit-3", "bit-4", "xor-2-3", "xor-3-4", "xor-2-4"),
            "C": ("bit-0", "bit-3", "bit-4", "xor-0-3", "xor-0-4", "xor-3-4"),
        }
        self._predicate_by_name = {predicate.name: predicate for predicate in self.predicates}

    @cached_property
    def hidden_states(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.product((0, 1), repeat=5))

    @cached_property
    def initial_belief(self) -> np.ndarray:
        return np.full(len(self.hidden_states), 1.0 / len(self.hidden_states), dtype=float)

    @property
    def initial_state(self) -> SensorState:
        return SensorState()

    def legal_actions(self, state: SensorState) -> tuple[str, ...]:
        setup = tuple(f"activate:{panel}" for panel in self.panels if panel != state.active_panel)
        screens = tuple(f"screen:{predicate.name}" for predicate in self.predicates)
        precise = (
            tuple(f"precise:{name}" for name in self.panels[state.active_panel])
            if state.active_panel is not None
            else ()
        )
        return setup + screens + precise

    @staticmethod
    def action_kind(action: str) -> str:
        return action.partition(":")[0]

    @staticmethod
    def action_target(action: str) -> str:
        kind, separator, target = action.partition(":")
        if not separator or kind not in {"activate", "screen", "precise"} or not target:
            raise ValueError(f"invalid gated-sensor action {action!r}")
        return target

    def next_state(self, state: SensorState, action: str) -> SensorState:
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal gated-sensor action {action!r} in state {state}")
        if self.action_kind(action) == "activate":
            return SensorState(active_panel=self.action_target(action))
        return state

    def outcomes(self, action: str) -> tuple[str | None, ...]:
        return (None,) if self.action_kind(action) == "activate" else OUTCOMES

    @lru_cache(maxsize=None)
    def likelihood_vector(self, action: str, outcome: str | None) -> np.ndarray:
        kind = self.action_kind(action)
        if kind == "activate":
            if outcome is not None:
                raise ValueError("activation has only the null observation")
            return np.ones(len(self.hidden_states), dtype=float)
        if outcome not in OUTCOMES:
            raise ValueError(f"invalid measurement outcome {outcome!r}")
        predicate = self._predicate_by_name[self.action_target(action)]
        accuracy = self.screen_accuracy if kind == "screen" else self.precise_accuracy
        truth_values = np.asarray([predicate.evaluate(hidden) for hidden in self.hidden_states])
        reports_positive = outcome == "positive"
        return np.where(truth_values == reports_positive, accuracy, 1.0 - accuracy)

    @staticmethod
    def entropy(belief: np.ndarray) -> float:
        nonzero = belief[belief > 0.0]
        return -float(np.dot(nonzero, np.log(nonzero)))

    def outcome_probability(self, belief: np.ndarray, action: str, outcome: str | None) -> float:
        return float(np.dot(belief, self.likelihood_vector(action, outcome)))

    def posterior(self, belief: np.ndarray, action: str, outcome: str | None) -> np.ndarray:
        posterior = belief * self.likelihood_vector(action, outcome)
        normalizer = float(posterior.sum())
        if normalizer <= EPSILON:
            raise ValueError("cannot condition on an impossible gated-sensor observation")
        return posterior / normalizer

    def expected_information_gain(self, belief: np.ndarray, action: str) -> float:
        expected_entropy = 0.0
        for outcome in self.outcomes(action):
            probability = self.outcome_probability(belief, action, outcome)
            if probability > EPSILON:
                expected_entropy += probability * self.entropy(self.posterior(belief, action, outcome))
        return self.entropy(belief) - expected_entropy

    def decode_map_index(self, belief: np.ndarray) -> int:
        return int(np.argmax(belief))
