"""Exact finite-population active testing on the UCI Cleveland heart cohort."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import math
from pathlib import Path
from typing import Final

import numpy as np


EPSILON: Final = 1e-15
ORDER_WORKUP_ACTION: Final = "order:clinical-workup"
FEATURE_NAMES: Final = (
    "age",
    "sex",
    "chest-pain",
    "resting-blood-pressure",
    "serum-cholesterol",
    "fasting-blood-sugar",
    "resting-ecg",
    "max-heart-rate",
    "exercise-angina",
    "st-depression",
    "st-slope",
    "major-vessels",
    "thal",
)
INITIAL_FEATURES: Final = frozenset(
    {"age", "sex", "chest-pain", "resting-blood-pressure", "fasting-blood-sugar"}
)
CONTINUOUS_THRESHOLDS: Final = {
    "age": (51.0, 59.0),
    "resting-blood-pressure": (120.0, 138.0),
    "serum-cholesterol": (226.0, 267.0),
    "max-heart-rate": (142.66666666666666, 162.0),
    "st-depression": (0.1, 1.4),
}
FEATURE_DESCRIPTIONS: Final = {
    "age": "patient age band",
    "sex": "recorded sex",
    "chest-pain": "chest-pain presentation type",
    "resting-blood-pressure": "resting blood-pressure band",
    "serum-cholesterol": "serum cholesterol band",
    "fasting-blood-sugar": "whether fasting blood sugar exceeds 120 mg/dl",
    "resting-ecg": "resting electrocardiogram category",
    "max-heart-rate": "maximum achieved heart-rate band",
    "exercise-angina": "whether exercise induces angina",
    "st-depression": "exercise-induced ST-depression band",
    "st-slope": "slope of the peak exercise ST segment",
    "major-vessels": "number of major vessels colored by fluoroscopy",
    "thal": "thalassemia test category",
}


@dataclass(frozen=True)
class WorkupState:
    workup_ordered: bool = False
    observed_features: frozenset[str] = frozenset()


class HeartWorkupModel:
    """Uniform empirical prior over complete processed Cleveland records."""

    def __init__(self, data_path: Path | None = None) -> None:
        if data_path is None:
            data_path = Path(__file__).with_name("data") / "processed.cleveland.data"
        raw_rows = [line.strip().split(",") for line in data_path.read_text(encoding="ascii").splitlines() if line.strip()]
        complete = [row for row in raw_rows if "?" not in row]
        if len(raw_rows) != 303 or len(complete) != 297 or any(len(row) != 14 for row in complete):
            raise ValueError("Cleveland data must contain 303 rows and 297 complete 14-column rows")
        numeric = np.asarray(complete, dtype=float)
        self.raw_rows = numeric
        self.classes = (numeric[:, 13] > 0).astype(int)
        values = np.empty((len(numeric), len(FEATURE_NAMES)), dtype=object)
        for index, name in enumerate(FEATURE_NAMES):
            column = numeric[:, index]
            if name in CONTINUOUS_THRESHOLDS:
                encoded = np.digitize(column, CONTINUOUS_THRESHOLDS[name], right=True)
                values[:, index] = np.asarray([("low", "mid", "high")[value] for value in encoded])
            else:
                values[:, index] = np.asarray([str(int(value)) for value in column])
        self.feature_values = values.astype(str)
        self.rows = tuple(tuple(row) for row in self.feature_values)
        self._feature_index = {name: index for index, name in enumerate(FEATURE_NAMES)}
        self._outcomes = {
            name: tuple(sorted(set(self.feature_values[:, index])))
            for index, name in enumerate(FEATURE_NAMES)
        }

    @cached_property
    def initial_belief(self) -> np.ndarray:
        return np.full(len(self.rows), 1.0 / len(self.rows), dtype=float)

    @property
    def initial_state(self) -> WorkupState:
        return WorkupState()

    def legal_actions(self, state: WorkupState) -> tuple[str, ...]:
        available = FEATURE_NAMES if state.workup_ordered else tuple(
            name for name in FEATURE_NAMES if name in INITIAL_FEATURES
        )
        queries = tuple(
            f"query:{name}" for name in available if name not in state.observed_features
        )
        setup = () if state.workup_ordered else (ORDER_WORKUP_ACTION,)
        return setup + queries

    def action_feature(self, action: str) -> str:
        kind, separator, feature = action.partition(":")
        if not separator or kind != "query" or feature not in self._feature_index:
            raise ValueError(f"invalid heart-workup query {action!r}")
        return feature

    def next_state(self, state: WorkupState, action: str) -> WorkupState:
        if action not in self.legal_actions(state):
            raise ValueError(f"illegal heart-workup action {action!r}")
        if action == ORDER_WORKUP_ACTION:
            return WorkupState(True, state.observed_features)
        return WorkupState(
            state.workup_ordered,
            state.observed_features | {self.action_feature(action)},
        )

    def outcomes(self, action: str) -> tuple[str | None, ...]:
        if action == ORDER_WORKUP_ACTION:
            return (None,)
        return self._outcomes[self.action_feature(action)]

    @lru_cache(maxsize=None)
    def likelihood_vector(self, action: str, outcome: str | None) -> np.ndarray:
        if action == ORDER_WORKUP_ACTION:
            if outcome is not None:
                raise ValueError("clinical workup has only the null observation")
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
        return -probability * math.log(probability) - (1.0 - probability) * math.log(1.0 - probability)

    def class_probability(self, belief: np.ndarray, label: int) -> float:
        if label not in {0, 1}:
            raise ValueError(f"invalid heart-disease class {label!r}")
        return float(belief[self.classes == label].sum())

    def target_entropy(self, belief: np.ndarray) -> float:
        return self._binary_entropy(self.class_probability(belief, 1))

    def outcome_probability(self, belief: np.ndarray, action: str, outcome: str | None) -> float:
        return float(np.dot(belief, self.likelihood_vector(action, outcome)))

    def posterior(self, belief: np.ndarray, action: str, outcome: str | None) -> np.ndarray:
        posterior = belief * self.likelihood_vector(action, outcome)
        normalizer = float(posterior.sum())
        if normalizer <= EPSILON:
            raise ValueError("cannot condition on an impossible heart-workup observation")
        return posterior / normalizer

    def expected_target_entropy(self, belief: np.ndarray, action: str) -> float:
        expected = 0.0
        for outcome in self.outcomes(action):
            probability = self.outcome_probability(belief, action, outcome)
            if probability > EPSILON:
                expected += probability * self.target_entropy(self.posterior(belief, action, outcome))
        return expected

    def expected_information_gain(self, belief: np.ndarray, action: str) -> float:
        return self.target_entropy(belief) - self.expected_target_entropy(belief, action)

    def observation(self, truth_index: int, action: str) -> str | None:
        if not 0 <= truth_index < len(self.rows):
            raise ValueError("truth index is outside the Cleveland cohort")
        if action == ORDER_WORKUP_ACTION:
            return None
        return str(self.feature_values[truth_index, self._feature_index[self.action_feature(action)]])

    def truth_log_probability(self, belief: np.ndarray, truth_index: int) -> float:
        label = int(self.classes[truth_index])
        return math.log(max(self.class_probability(belief, label), np.finfo(float).tiny))

    def decode_map_class(self, belief: np.ndarray) -> int:
        return int(self.class_probability(belief, 1) >= 0.5)
