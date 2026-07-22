"""Exact finite-population active feature acquisition for mushroom edibility."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Final
import csv
import math

import numpy as np


EPSILON: Final = 1e-15
COLLECT_ACTION: Final = "collect:specimen"
FEATURE_NAMES: Final = (
    "cap-shape",
    "cap-surface",
    "cap-color",
    "bruises",
    "odor",
    "gill-attachment",
    "gill-spacing",
    "gill-size",
    "gill-color",
    "stalk-shape",
    "stalk-root",
    "stalk-surface-above-ring",
    "stalk-surface-below-ring",
    "stalk-color-above-ring",
    "stalk-color-below-ring",
    "veil-type",
    "veil-color",
    "ring-number",
    "ring-type",
    "spore-print-color",
    "population",
    "habitat",
)
FIELD_FEATURES: Final = frozenset(
    {"cap-shape", "cap-surface", "cap-color", "population", "habitat"}
)
FEATURE_DESCRIPTIONS: Final = {
    "cap-shape": "overall cap shape",
    "cap-surface": "cap surface texture",
    "cap-color": "cap color",
    "bruises": "whether handling produces bruising",
    "odor": "specimen odor category",
    "gill-attachment": "how the gills attach to the stalk",
    "gill-spacing": "spacing between gills",
    "gill-size": "gill width category",
    "gill-color": "gill color",
    "stalk-shape": "stalk shape",
    "stalk-root": "stalk root structure",
    "stalk-surface-above-ring": "stalk texture above the ring",
    "stalk-surface-below-ring": "stalk texture below the ring",
    "stalk-color-above-ring": "stalk color above the ring",
    "stalk-color-below-ring": "stalk color below the ring",
    "veil-type": "veil type",
    "veil-color": "veil color",
    "ring-number": "number of rings",
    "ring-type": "ring type",
    "spore-print-color": "spore-print color",
    "population": "observed population pattern",
    "habitat": "habitat category",
}


@dataclass(frozen=True)
class AcquisitionState:
    specimen_collected: bool = False
    observed_features: frozenset[str] = frozenset()


class MushroomFeatureModel:
    """Uniform empirical prior over the UCI Mushroom catalog."""

    def __init__(self, data_path: Path | None = None) -> None:
        if data_path is None:
            data_path = Path(__file__).with_name("data") / "agaricus-lepiota.data"
        rows = [
            row
            for row in csv.reader(data_path.open(encoding="ascii"))
            if row
        ]
        if not rows or any(len(row) != len(FEATURE_NAMES) + 1 for row in rows):
            raise ValueError("mushroom data must contain one class and 22 features per row")
        if any(row[0] not in {"e", "p"} for row in rows):
            raise ValueError("mushroom class labels must be edible=e or poisonous=p")
        self.rows = tuple(tuple(row) for row in rows)
        self.classes = np.asarray([row[0] for row in rows])
        self.feature_values = np.asarray([row[1:] for row in rows])
        self._feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
        self._outcomes = {
            name: tuple(sorted(set(self.feature_values[:, index])))
            for index, name in enumerate(FEATURE_NAMES)
        }

    @cached_property
    def initial_belief(self) -> np.ndarray:
        return np.full(len(self.rows), 1.0 / len(self.rows), dtype=float)

    @property
    def initial_state(self) -> AcquisitionState:
        return AcquisitionState()

    def legal_actions(self, state: AcquisitionState) -> tuple[str, ...]:
        available = FEATURE_NAMES if state.specimen_collected else tuple(
            name for name in FEATURE_NAMES if name in FIELD_FEATURES
        )
        queries = tuple(
            f"query:{name}" for name in available if name not in state.observed_features
        )
        setup = () if state.specimen_collected else (COLLECT_ACTION,)
        return setup + queries

    @staticmethod
    def action_kind(action: str) -> str:
        return action.partition(":")[0]

    def action_feature(self, action: str) -> str:
        kind, separator, feature = action.partition(":")
        if not separator or kind != "query" or feature not in self._feature_index:
            raise ValueError(f"invalid mushroom query action {action!r}")
        return feature

    def next_state(self, state: AcquisitionState, action: str) -> AcquisitionState:
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal mushroom action {action!r} in state {state}")
        if action == COLLECT_ACTION:
            return AcquisitionState(True, state.observed_features)
        return AcquisitionState(
            state.specimen_collected,
            state.observed_features | {self.action_feature(action)},
        )

    def outcomes(self, action: str) -> tuple[str | None, ...]:
        if action == COLLECT_ACTION:
            return (None,)
        return self._outcomes[self.action_feature(action)]

    @lru_cache(maxsize=None)
    def likelihood_vector(self, action: str, outcome: str | None) -> np.ndarray:
        if action == COLLECT_ACTION:
            if outcome is not None:
                raise ValueError("specimen collection has only the null observation")
            return np.ones(len(self.rows), dtype=float)
        feature = self.action_feature(action)
        if outcome not in self._outcomes[feature]:
            raise ValueError(f"invalid outcome {outcome!r} for {feature}")
        index = self._feature_index[feature]
        return (self.feature_values[:, index] == outcome).astype(float)

    @staticmethod
    def _binary_entropy(probability: float) -> float:
        if probability <= EPSILON or probability >= 1.0 - EPSILON:
            return 0.0
        return -probability * math.log(probability) - (1.0 - probability) * math.log(
            1.0 - probability
        )

    def class_probability(self, belief: np.ndarray, label: str) -> float:
        if label not in {"e", "p"}:
            raise ValueError(f"invalid mushroom class {label!r}")
        return float(belief[self.classes == label].sum())

    def target_entropy(self, belief: np.ndarray) -> float:
        return self._binary_entropy(self.class_probability(belief, "p"))

    def outcome_probability(
        self, belief: np.ndarray, action: str, outcome: str | None
    ) -> float:
        return float(np.dot(belief, self.likelihood_vector(action, outcome)))

    def posterior(
        self, belief: np.ndarray, action: str, outcome: str | None
    ) -> np.ndarray:
        posterior = belief * self.likelihood_vector(action, outcome)
        normalizer = float(posterior.sum())
        if normalizer <= EPSILON:
            raise ValueError("cannot condition on an impossible mushroom observation")
        return posterior / normalizer

    def expected_target_entropy(self, belief: np.ndarray, action: str) -> float:
        expected = 0.0
        for outcome in self.outcomes(action):
            probability = self.outcome_probability(belief, action, outcome)
            if probability > EPSILON:
                expected += probability * self.target_entropy(
                    self.posterior(belief, action, outcome)
                )
        return expected

    def expected_information_gain(self, belief: np.ndarray, action: str) -> float:
        return self.target_entropy(belief) - self.expected_target_entropy(belief, action)

    def observation(self, truth_index: int, action: str) -> str | None:
        if not 0 <= truth_index < len(self.rows):
            raise ValueError("truth index is outside the mushroom catalog")
        if action == COLLECT_ACTION:
            return None
        return str(self.feature_values[truth_index, self._feature_index[self.action_feature(action)]])

    def truth_log_probability(self, belief: np.ndarray, truth_index: int) -> float:
        label = str(self.classes[truth_index])
        return math.log(max(self.class_probability(belief, label), np.finfo(float).tiny))

    def decode_map_class(self, belief: np.ndarray) -> str:
        return "p" if self.class_probability(belief, "p") >= 0.5 else "e"
