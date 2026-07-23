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

# Some test runners inject a minimal ``torch`` module. SciPy's optional torch
# detection assumes any loaded module has Tensor, while adapter tests still need
# the same stub later in collection. Complete only that missing type surface.
if (torch_module := sys.modules.get("torch")) is not None and not hasattr(torch_module, "Tensor"):
    torch_module.Tensor = type("Tensor", (), {})

from pomdp_py.problems.rocksample.rocksample_problem import (
    CheckAction,
    MoveEast,
    MoveNorth,
    MoveSouth,
    MoveWest,
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
ROCKSAMPLE_11_11_URL: Final = (
    "https://github.com/AdaCompNUS/sarsop/blob/"
    "d9141104392fd0a7b35327fdf7d40ef4b71a13ca/examples/POMDPX/"
    "RockSample_11_11.pomdpx"
)
ROCKSAMPLE_15_15_URL: Final = (
    "https://github.com/taodav/pobax/blob/"
    "a5e1d62d14e4efe783885b9d4f19cffa2a568eec/pobax/envs/jax/rocksample.py"
)
MOVES: Final = (MoveNorth, MoveEast, MoveSouth, MoveWest)


class _FactorizedRockBelief(np.ndarray):
    """Joint vector carrying exact independent per-rock marginals."""

    rock_good_probabilities: np.ndarray | None

    def __new__(
        cls, values: np.ndarray, rock_good_probabilities: np.ndarray
    ) -> "_FactorizedRockBelief":
        belief = np.asarray(values, dtype=float).view(cls)
        belief.rock_good_probabilities = np.asarray(rock_good_probabilities, dtype=float)
        return belief

    def __array_finalize__(self, source: np.ndarray | None) -> None:
        self.rock_good_probabilities = getattr(source, "rock_good_probabilities", None)


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
    # Standard SARSOP POMDPX benchmark instance, using its zero-based coordinates.
    "11-11": RockDiagnosisMap(
        name="11-11",
        grid_size=11,
        rock_positions=(
            (0, 3),
            (0, 7),
            (1, 8),
            (2, 4),
            (3, 3),
            (3, 8),
            (4, 3),
            (5, 8),
            (6, 1),
            (9, 3),
            (9, 9),
        ),
        start_position=(0, 5),
        source_page=1,
        source_citation="SARSOP benchmark repository",
        source_url=ROCKSAMPLE_11_11_URL,
    ),
    # POBAX samples RockSample coordinates from a JAX key. This instance is
    # frozen from PRNG key 24098 before any BED endpoint was evaluated.
    "15-15": RockDiagnosisMap(
        name="15-15",
        grid_size=15,
        rock_positions=(
            (13, 9),
            (4, 2),
            (13, 8),
            (2, 6),
            (2, 10),
            (10, 1),
            (14, 10),
            (9, 5),
            (5, 12),
            (13, 7),
            (4, 5),
            (3, 9),
            (0, 0),
            (14, 2),
            (3, 7),
        ),
        start_position=(0, 7),
        source_page=1,
        source_citation="POBAX RockSample generator (JAX key 24098)",
        source_url=ROCKSAMPLE_15_15_URL,
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
        state_count = 2**self.num_rocks
        return _FactorizedRockBelief(
            np.full(state_count, 1.0 / state_count, dtype=float),
            np.full(self.num_rocks, 0.5, dtype=float),
        )

    @cached_property
    def _good_state_masks(self) -> tuple[np.ndarray, ...]:
        return tuple(
            np.asarray([types[rock_id] == RockType.GOOD for types in self.hidden_states])
            for rock_id in range(self.num_rocks)
        )

    @property
    def num_rocks(self) -> int:
        return len(self.map_spec.rock_positions)

    def rock_good_probability(self, belief: np.ndarray, rock_id: int) -> float:
        if not 0 <= rock_id < self.num_rocks:
            raise ValueError(f"rock_id out of range: {rock_id}")
        if isinstance(belief, _FactorizedRockBelief) and belief.rock_good_probabilities is not None:
            return float(belief.rock_good_probabilities[rock_id])
        return float(np.dot(belief, self._good_state_masks[rock_id]))

    def sensor_accuracy(self, position: tuple[int, int], rock_id: int) -> float:
        if not 0 <= rock_id < self.num_rocks:
            raise ValueError(f"rock_id out of range: {rock_id}")
        distance = math.dist(position, self.map_spec.rock_positions[rock_id])
        return 0.5 * (1.0 + 2.0 ** (-distance / self.half_efficiency_distance))

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
        rock_id = self.check_id(action_name)
        if rock_id is None:
            # Rock Diagnosis moves have the deterministic ``none`` observation.
            return np.ones(len(self.hidden_states), dtype=float)
        if outcome not in (RockType.GOOD, RockType.BAD):
            raise ValueError(f"invalid Rock Diagnosis check outcome: {outcome!r}")
        accuracy = self.sensor_accuracy(position, rock_id)
        reports_good = outcome == RockType.GOOD
        matches_report = self._good_state_masks[rock_id] == reports_good
        return np.where(matches_report, accuracy, 1.0 - accuracy)

    @staticmethod
    def entropy(belief: np.ndarray) -> float:
        if isinstance(belief, _FactorizedRockBelief) and belief.rock_good_probabilities is not None:
            entropy = 0.0
            for probability in belief.rock_good_probabilities:
                if EPSILON < probability < 1.0 - EPSILON:
                    entropy -= probability * math.log(probability)
                    entropy -= (1.0 - probability) * math.log(1.0 - probability)
            return float(entropy)
        nonzero = belief[belief > 0.0]
        return -float(np.dot(nonzero, np.log(nonzero)))

    def outcome_probability(
        self,
        position: tuple[int, int],
        belief: np.ndarray,
        action_name: str,
        outcome: str | None,
    ) -> float:
        # Use the same joint normalizer as posterior(). Computing the complementary
        # binary probability as 1 - p can leave a tiny positive cancellation
        # residue for an observation whose likelihood is exactly zero.
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
        rock_id = self.check_id(action_name)
        assert rock_id is not None
        posterior = belief * self.likelihood_vector(position, action_name, outcome)
        normalizer = float(posterior.sum())
        if not math.isfinite(normalizer) or normalizer <= 0.0:
            raise ValueError("cannot update on an impossible Rock Diagnosis observation")
        normalized = np.asarray(posterior / normalizer)
        if isinstance(belief, _FactorizedRockBelief) and belief.rock_good_probabilities is not None:
            marginals = belief.rock_good_probabilities.copy()
            prior_good = float(marginals[rock_id])
            accuracy = self.sensor_accuracy(position, rock_id)
            good_likelihood = accuracy if outcome == RockType.GOOD else 1.0 - accuracy
            marginals[rock_id] = prior_good * good_likelihood / normalizer
            return _FactorizedRockBelief(normalized, marginals)
        return normalized

    def expected_information_gain(self, position: tuple[int, int], belief: np.ndarray, action_name: str) -> float:
        rock_id = self.check_id(action_name)
        if rock_id is None:
            return 0.0
        p_good = self.rock_good_probability(belief, rock_id)
        accuracy = self.sensor_accuracy(position, rock_id)
        p_observe_good = p_good * accuracy + (1.0 - p_good) * (1.0 - accuracy)

        def binary_entropy(probability: float) -> float:
            if probability <= EPSILON or probability >= 1.0 - EPSILON:
                return 0.0
            return -probability * math.log(probability) - (1.0 - probability) * math.log(
                1.0 - probability
            )

        # Y is conditionally independent of the remaining rock vector given this
        # rock, so I(full state; Y) equals the binary-channel mutual information.
        return max(0.0, binary_entropy(p_observe_good) - binary_entropy(accuracy))

    def decode_map_index(self, belief: np.ndarray) -> int:
        return int(np.argmax(belief))


class RangeGatedRockDiagnosisModel(RockDiagnosisModel):
    """Rock Diagnosis where accurate inspection requires reaching a rock."""

    def __init__(
        self,
        map_spec: RockDiagnosisMap,
        *,
        remote_accuracy: float = 0.55,
        onsite_accuracy: float = 0.95,
    ) -> None:
        if not 0.5 <= remote_accuracy < onsite_accuracy <= 1.0:
            raise ValueError(
                "range-gated accuracies must satisfy 0.5 <= remote < onsite <= 1"
            )
        super().__init__(map_spec)
        self.remote_accuracy = float(remote_accuracy)
        self.onsite_accuracy = float(onsite_accuracy)

    def sensor_accuracy(self, position: tuple[int, int], rock_id: int) -> float:
        if not 0 <= rock_id < self.num_rocks:
            raise ValueError(f"rock_id out of range: {rock_id}")
        return (
            self.onsite_accuracy
            if position == self.map_spec.rock_positions[rock_id]
            else self.remote_accuracy
        )
