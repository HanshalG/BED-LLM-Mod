"""Paper-map Rock Diagnosis with exact finite beliefs.

The domain follows Araya-Lopez, Buffet, and Thomas (2013): rocks have fixed
unobserved binary types, rover motion is deterministic, and check observations
become less accurate with distance.  ``pomdp_py`` supplies the RockSample
transition and observation implementations; this wrapper removes destructive
sampling and terminal exits because Rock Diagnosis is information-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property, lru_cache
import itertools
import math
import sys
from typing import Final

import numpy as np

# Some test runners inject a minimal ``torch`` module.  SciPy's optional torch
# detection assumes any loaded module has Tensor; remove only that malformed stub.
if (torch_module := sys.modules.get("torch")) is not None and not hasattr(torch_module, "Tensor"):
    del sys.modules["torch"]

from pomdp_py.problems.rocksample.rocksample_problem import (
    CheckAction,
    MoveEast,
    MoveNorth,
    MoveSouth,
    MoveWest,
    Observation,
    RSObservationModel,
    RSTransitionModel,
    RockType,
    State,
)


EPSILON: Final = 1e-15
PAPER_URL: Final = "https://members.loria.fr/olivier.buffet/papiers/jfpda13-b.pdf"
ROCKSAMPLE_7_8_URL: Final = (
    "https://www.ri.cmu.edu/pub_files/pub4/smith_trey_2004_1/smith_trey_2004_1.pdf"
)
MOVES: Final = (MoveNorth, MoveEast, MoveSouth, MoveWest)


@dataclass(frozen=True)
class RockDiagnosisMap:
    """A fixed paper map plus an explicitly declared rover entry position."""

    name: str
    grid_size: int
    rock_positions: tuple[tuple[int, int], ...]
    start_position: tuple[int, int]
    source_page: int = 10
    source_citation: str = "Araya-Lopez, Buffet, and Thomas (2013)"
    source_url: str = PAPER_URL

    def __post_init__(self) -> None:
        if self.grid_size < 2:
            raise ValueError("grid_size must be at least two")
        if not self.rock_positions:
            raise ValueError("a Rock Diagnosis map needs at least one rock")
        if len(set(self.rock_positions)) != len(self.rock_positions):
            raise ValueError("rock positions must be distinct")
        for position in (*self.rock_positions, self.start_position):
            if not (0 <= position[0] < self.grid_size and 0 <= position[1] < self.grid_size):
                raise ValueError(f"position outside the grid: {position}")


# Figure 4 gives rock layouts but no start state.  We fix the conventional
# left-centre entry before evaluation so the adaptation is reproducible.
PAPER_MAPS: Final[dict[str, RockDiagnosisMap]] = {
    "3-6": RockDiagnosisMap(
        name="3-6",
        grid_size=6,
        rock_positions=((4, 1), (1, 4), (4, 4)),
        start_position=(0, 3),
    ),
    "5-7": RockDiagnosisMap(
        name="5-7",
        grid_size=7,
        rock_positions=((4, 0), (6, 2), (2, 3), (3, 5), (5, 5)),
        start_position=(0, 3),
    ),
    # Smith and Simmons (2004), Figure 4, transcribed with zero-based coordinates.
    "7-8": RockDiagnosisMap(
        name="7-8",
        grid_size=7,
        rock_positions=((1, 0), (5, 1), (2, 2), (3, 2), (6, 3), (0, 5), (3, 5), (2, 6)),
        start_position=(0, 3),
        source_page=5,
        source_citation="Smith and Simmons (2004)",
        source_url=ROCKSAMPLE_7_8_URL,
    ),
}


def get_paper_map(name: str) -> RockDiagnosisMap:
    """Return one of the registered literature benchmark layouts."""

    try:
        return PAPER_MAPS[name]
    except KeyError as exc:
        choices = ", ".join(sorted(PAPER_MAPS))
        raise ValueError(f"unknown Rock Diagnosis map {name!r}; choose one of {choices}") from exc


class RockDiagnosisModel:
    """Finite exact-belief adapter over ``pomdp_py`` RockSample primitives."""

    def __init__(self, map_spec: RockDiagnosisMap, *, half_efficiency_distance: float = math.log(2.0)) -> None:
        if half_efficiency_distance <= 0.0:
            raise ValueError("half_efficiency_distance must be positive")
        self.map_spec = map_spec
        self.half_efficiency_distance = half_efficiency_distance
        self._rock_locations = {position: index for index, position in enumerate(map_spec.rock_positions)}
        self._check_actions = tuple(CheckAction(index) for index in range(len(map_spec.rock_positions)))
        self._actions = {str(action): action for action in (*MOVES, *self._check_actions)}
        self._transition = RSTransitionModel(map_spec.grid_size, self._rock_locations, lambda _position: False)
        self._observation = RSObservationModel(
            self._rock_locations,
            half_efficiency_dist=half_efficiency_distance,
        )

    @cached_property
    def hidden_states(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            tuple(types)
            for types in itertools.product((RockType.BAD, RockType.GOOD), repeat=len(self.map_spec.rock_positions))
        )

    @cached_property
    def initial_belief(self) -> np.ndarray:
        return np.full(len(self.hidden_states), 1.0 / len(self.hidden_states), dtype=float)

    @property
    def num_rocks(self) -> int:
        return len(self.map_spec.rock_positions)

    def action(self, action_name: str):
        try:
            return self._actions[action_name]
        except KeyError as exc:
            raise ValueError(f"unknown Rock Diagnosis action {action_name!r}") from exc

    def is_move(self, action_name: str) -> bool:
        return action_name.startswith("move-")

    def check_id(self, action_name: str) -> int | None:
        action = self.action(action_name)
        return action.rock_id if isinstance(action, CheckAction) else None

    @lru_cache(maxsize=None)
    def legal_actions(self, position: tuple[int, int]) -> tuple[str, ...]:
        legal_moves = tuple(
            action
            for action in MOVES
            if 0 <= position[0] + action.motion[0] < self.map_spec.grid_size
            and 0 <= position[1] + action.motion[1] < self.map_spec.grid_size
        )
        return tuple(str(action) for action in (*legal_moves, *self._check_actions))

    @lru_cache(maxsize=None)
    def next_position(self, position: tuple[int, int], action_name: str) -> tuple[int, int]:
        state = State(position, self.hidden_states[0])
        return self._transition.sample(state, self.action(action_name)).position

    def outcomes(self, action_name: str) -> tuple[str | None, ...]:
        return (RockType.GOOD, RockType.BAD) if self.check_id(action_name) is not None else (None,)

    @lru_cache(maxsize=None)
    def likelihood_vector(
        self,
        position: tuple[int, int],
        action_name: str,
        outcome: str | None,
    ) -> np.ndarray:
        action = self.action(action_name)
        if not isinstance(action, CheckAction):
            # Rock Diagnosis moves have the deterministic ``none`` observation.
            return np.ones(len(self.hidden_states), dtype=float)
        observation = Observation(outcome)
        return np.asarray(
            [
                self._observation.probability(
                    observation,
                    self._transition.sample(State(position, types), action),
                    action,
                )
                for types in self.hidden_states
            ],
            dtype=float,
        )

    @staticmethod
    def entropy(belief: np.ndarray) -> float:
        nonzero = belief[belief > 0.0]
        return -float(np.dot(nonzero, np.log(nonzero)))

    def outcome_probability(
        self,
        position: tuple[int, int],
        belief: np.ndarray,
        action_name: str,
        outcome: str | None,
    ) -> float:
        return float(np.dot(belief, self.likelihood_vector(position, action_name, outcome)))

    def posterior(
        self,
        position: tuple[int, int],
        belief: np.ndarray,
        action_name: str,
        outcome: str | None,
    ) -> np.ndarray:
        if self.check_id(action_name) is None:
            return belief.copy()
        posterior = belief * self.likelihood_vector(position, action_name, outcome)
        normalizer = float(posterior.sum())
        if normalizer <= EPSILON:
            raise ValueError("cannot update on an impossible Rock Diagnosis observation")
        return posterior / normalizer

    def expected_information_gain(self, position: tuple[int, int], belief: np.ndarray, action_name: str) -> float:
        if self.check_id(action_name) is None:
            return 0.0
        expected_entropy = 0.0
        for outcome in self.outcomes(action_name):
            probability = self.outcome_probability(position, belief, action_name, outcome)
            if probability > EPSILON:
                expected_entropy += probability * self.entropy(self.posterior(position, belief, action_name, outcome))
        return self.entropy(belief) - expected_entropy

    def decode_map_index(self, belief: np.ndarray) -> int:
        return int(np.argmax(belief))
