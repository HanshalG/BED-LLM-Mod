"""Finite-population posterior for the UCI ann-thyroid cohort."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
from typing import Final

import numpy as np


FEATURE_NAMES: Final = (
    "age",
    "sex",
    "on-thyroxine",
    "query-on-thyroxine",
    "on-antithyroid-medication",
    "sick",
    "pregnant",
    "thyroid-surgery",
    "i131-treatment",
    "query-hypothyroid",
    "query-hyperthyroid",
    "lithium",
    "goitre",
    "tumor",
    "hypopituitary",
    "psych",
    "tsh",
    "t3",
    "tt4",
    "t4u",
    "fti",
)
IMMEDIATE_FEATURES: Final = tuple(range(16))
DELAYED_ASSAYS: Final = (16, 17, 18, 19)
BINNED_FEATURES: Final = (0, 16, 17, 18, 19, 20)
COLLECT_BLOOD_ACTION: Final = "collect:blood-sample"
EPSILON: Final = 1e-15


@dataclass(frozen=True)
class ThyroidWorkupState:
    blood_collected: bool = False
    asked_features: tuple[int, ...] = ()


class ThyroidWorkupModel:
    """Exact empirical prior over all 7,200 released patient rows."""

    def __init__(self, data_dir: Path | None = None) -> None:
        root = Path(__file__).resolve().parents[2]
        self.data_dir = data_dir or root / "data" / "nonmyopic" / "uci_thyroid"
        rows: list[list[float]] = []
        for filename in ("ann-train.data", "ann-test.data"):
            for line in (self.data_dir / filename).read_text(encoding="utf-8").splitlines():
                fields = line.split()
                if fields:
                    rows.append([*map(float, fields[:-1]), float(fields[-1])])
        array = np.asarray(rows, dtype=float)
        if array.shape != (7200, 22):
            raise ValueError(f"unexpected ann-thyroid shape {array.shape}")
        self.raw_features = array[:, :21]
        self.targets = array[:, 21].astype(int)
        self.features = self.raw_features.astype(int)
        self.bin_edges: dict[int, np.ndarray] = {}
        for feature in BINNED_FEATURES:
            edges = np.unique(
                np.quantile(self.raw_features[:, feature], np.linspace(0.0, 1.0, 7)[1:-1])
            )
            self.bin_edges[feature] = edges
            self.features[:, feature] = np.digitize(
                self.raw_features[:, feature], edges, right=True
            )
        self._outcomes = {
            feature: tuple(int(value) for value in np.unique(self.features[:, feature]))
            for feature in range(len(FEATURE_NAMES))
        }
        self._masks = {
            (feature, outcome): self.features[:, feature] == outcome
            for feature in range(len(FEATURE_NAMES))
            for outcome in self._outcomes[feature]
        }

    @property
    def initial_state(self) -> ThyroidWorkupState:
        return ThyroidWorkupState()

    @property
    def initial_belief(self) -> np.ndarray:
        return np.full(len(self.targets), 1.0 / len(self.targets), dtype=float)

    def action_feature(self, action: str) -> int:
        if not action.startswith("query:"):
            raise ValueError(f"not a thyroid query action: {action}")
        name = action.split(":", 1)[1]
        try:
            return FEATURE_NAMES.index(name)
        except ValueError as exc:
            raise ValueError(f"unknown thyroid feature {name}") from exc

    def legal_actions(self, state: ThyroidWorkupState) -> tuple[str, ...]:
        available = (*IMMEDIATE_FEATURES, *(DELAYED_ASSAYS if state.blood_collected else ()))
        queries = tuple(
            f"query:{FEATURE_NAMES[feature]}"
            for feature in available
            if feature not in state.asked_features
        )
        return queries if state.blood_collected else (*queries, COLLECT_BLOOD_ACTION)

    def outcomes(self, action: str) -> tuple[str | None, ...]:
        if action == COLLECT_BLOOD_ACTION:
            return (None,)
        return tuple(str(value) for value in self._outcomes[self.action_feature(action)])

    def observation(self, truth_index: int, action: str) -> str | None:
        if action == COLLECT_BLOOD_ACTION:
            return None
        return str(int(self.features[truth_index, self.action_feature(action)]))

    def outcome_probability(
        self, belief: np.ndarray, action: str, outcome: str | None
    ) -> float:
        if action == COLLECT_BLOOD_ACTION:
            return 1.0 if outcome is None else 0.0
        if outcome is None:
            return 0.0
        mask = self._masks[(self.action_feature(action), int(outcome))]
        return float(belief[mask].sum())

    def posterior(
        self, belief: np.ndarray, action: str, outcome: str | None
    ) -> np.ndarray:
        if action == COLLECT_BLOOD_ACTION:
            if outcome is not None:
                raise ValueError("blood collection has only a none observation")
            return belief.copy()
        if outcome is None:
            raise ValueError("thyroid query needs a categorical observation")
        mask = self._masks[(self.action_feature(action), int(outcome))]
        posterior = belief * mask
        normalizer = float(posterior.sum())
        if normalizer <= 0.0:
            raise ValueError("cannot update on an impossible thyroid outcome")
        return posterior / normalizer

    def next_state(self, state: ThyroidWorkupState, action: str) -> ThyroidWorkupState:
        if action == COLLECT_BLOOD_ACTION:
            return ThyroidWorkupState(True, state.asked_features)
        feature = self.action_feature(action)
        return ThyroidWorkupState(
            state.blood_collected,
            tuple(sorted((*state.asked_features, feature))),
        )

    def class_probability(self, belief: np.ndarray, target_class: int) -> float:
        return float(belief[self.targets == target_class].sum())

    def target_entropy(self, belief: np.ndarray) -> float:
        probabilities = np.asarray(
            [self.class_probability(belief, target_class) for target_class in (1, 2, 3)]
        )
        probabilities = probabilities[probabilities > EPSILON]
        return -float(np.dot(probabilities, np.log(probabilities)))

    def truth_log_probability(self, belief: np.ndarray, truth_index: int) -> float:
        probability = self.class_probability(belief, int(self.targets[truth_index]))
        return math.log(max(probability, np.finfo(float).tiny))

    def expected_target_entropy(self, belief: np.ndarray, action: str) -> float:
        if action == COLLECT_BLOOD_ACTION:
            return self.target_entropy(belief)
        total = 0.0
        for outcome in self.outcomes(action):
            probability = self.outcome_probability(belief, action, outcome)
            if probability <= EPSILON:
                continue
            total += probability * self.target_entropy(
                self.posterior(belief, action, outcome)
            )
        return float(total)

    def expected_information_gain(self, belief: np.ndarray, action: str) -> float:
        return self.target_entropy(belief) - self.expected_target_entropy(belief, action)

    def decode_class(self, belief: np.ndarray) -> int:
        probabilities = [self.class_probability(belief, target_class) for target_class in (1, 2, 3)]
        return (1, 2, 3)[int(np.argmax(probabilities))]
