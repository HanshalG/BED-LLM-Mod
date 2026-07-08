from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from core.belief import BeliefState as _BeliefState


SourceConfig = tuple[tuple[float, ...], ...]


Location = tuple[float, ...]


@dataclass(frozen=True)
class LocationObservation:
    query: Location
    value: float


@dataclass(frozen=True)
class LocationFindingMetrics:
    source_rmse: list[float]
    top_probability: list[float]
    selected_eig: list[float]
    realized_entropy_drop: list[float] = field(default_factory=list)


@dataclass(frozen=True)
class LocationStrategyEntry:
    strategy: str
    mean_score: float
    score_variance: float
    root_query_fingerprint: str
    round_index: int
    root_query: Location | None = None


class LocationStrategyLibrary:
    def __init__(self) -> None:
        self.entries: list[LocationStrategyEntry] = []

    def __len__(self) -> int:
        return len(self.entries)

    def retrieve_top_m(self, count: int) -> list[LocationStrategyEntry]:
        if count <= 0:
            return []
        ranked = sorted(
            self.entries,
            key=lambda entry: (entry.mean_score, -entry.score_variance),
            reverse=True,
        )
        return ranked[:count]

    def replace_entries(self, entries: list[LocationStrategyEntry]) -> None:
        """Replace all entries with the given list (keeps only the latest round)."""
        self.entries = list(entries)


def normalize_location(raw_location: object, dim: int) -> Location:
    try:
        values = [float(value) for value in raw_location]  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Location must be an iterable of {dim} numeric values") from exc
    if len(values) != dim:
        raise ValueError(f"Location must have length {dim}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Location values must be finite")
    return tuple(values)


def location_max_step_radius(config: Any) -> float | None:
    value = getattr(config, "location_max_step_radius", None)
    if value is None:
        return None
    radius = float(value)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("location_max_step_radius must be a positive number or null")
    return radius


def last_query_from_observations(observations: list[LocationObservation]) -> Location | None:
    return observations[-1].query if observations else None


def project_location_to_step_radius(
    location: Location,
    previous_query: Location | None,
    config: Any,
) -> Location:
    radius = location_max_step_radius(config)
    dim = int(getattr(config, "location_dim", len(location)))
    query = np.asarray(normalize_location(location, dim), dtype=float)
    if radius is None or previous_query is None:
        return tuple(float(value) for value in query)

    previous = np.asarray(normalize_location(previous_query, dim), dtype=float)
    delta = query - previous
    distance = float(np.linalg.norm(delta))
    if distance <= radius or distance <= 0.0:
        return tuple(float(value) for value in query)
    projected = previous + delta * (radius / distance)
    return tuple(float(value) for value in projected)


def normalize_source_config(raw_config: object, num_sources: int, dim: int) -> SourceConfig:
    try:
        source_rows = list(raw_config)  # type: ignore[arg-type]
    except TypeError as exc:
        raise ValueError("Source configuration must be an iterable of source coordinates") from exc
    if len(source_rows) != num_sources:
        raise ValueError(f"Source configuration must contain exactly {num_sources} sources")
    sources = [normalize_location(row, dim) for row in source_rows]
    if len(set(sources)) != num_sources:
        raise ValueError("Source configuration must contain distinct source coordinates")
    return tuple(sorted(sources))


def _dedupe_source_configs(configs: list[SourceConfig]) -> list[SourceConfig]:
    deduped: list[SourceConfig] = []
    seen: set[tuple[tuple[float, ...], ...]] = set()
    for config in configs:
        key = tuple(tuple(round(value, 6) for value in source) for source in config)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(config)
    return deduped


class LocationFindingEnv:
    def __init__(
        self,
        num_sources: int = 3,
        dim: int = 2,
        b: float = 0.1,
        m: float = 1e-4,
        alpha: float = 1.0,
        noise_sd: float = 0.5,
        signal_model: str = "inverse_square",
        signal_lengthscale: float = 0.75,
        signal_amplitude: float = 5.0,
        source_prior: str = "normal",
        source_radius: float = 1.0,
        true_theta: np.ndarray | None = None,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.num_sources = num_sources
        self.dim = dim
        self.b = b
        self.m = m
        self.alpha = alpha
        self.noise_sd = noise_sd
        self.signal_model = signal_model
        self.signal_lengthscale = signal_lengthscale
        self.signal_amplitude = signal_amplitude
        self.source_prior = source_prior
        self.source_radius = source_radius
        self.rng = rng or np.random.default_rng()
        self.observed_data: list[LocationObservation] = []
        self.true_theta = np.zeros((self.num_sources, self.dim), dtype=float)
        self.reset(true_theta=true_theta)

    def reset(self, true_theta: np.ndarray | None = None) -> None:
        self.observed_data = []
        if true_theta is None:
            from .physics import sample_source_configs_from_prior

            self.true_theta = sample_source_configs_from_prior(
                self.rng,
                count=1,
                num_sources=self.num_sources,
                dim=self.dim,
                source_prior=self.source_prior,
                source_radius=self.source_radius,
            )[0]
        else:
            theta = np.asarray(true_theta, dtype=float)
            if theta.shape != (self.num_sources, self.dim):
                raise ValueError(
                    f"true_theta must have shape {(self.num_sources, self.dim)}, received {theta.shape}"
                )
            self.true_theta = theta

    def signal_intensity(self, query: Location | np.ndarray) -> float:
        query_arr = np.asarray(query, dtype=float)
        if query_arr.shape != (self.dim,):
            raise ValueError(f"query must have shape {(self.dim,)}, received {query_arr.shape}")
        from .physics import signal_intensities_from_distances

        distances_squared = np.sum((self.true_theta - query_arr) ** 2, axis=1)
        return float(
            signal_intensities_from_distances(
                distances_squared,
                b=self.b,
                m=self.m,
                alpha=self.alpha,
                signal_model=self.signal_model,
                signal_lengthscale=self.signal_lengthscale,
                signal_amplitude=self.signal_amplitude,
            )
        )

    def step(self, query: Location | np.ndarray) -> float:
        intensity = self.signal_intensity(query)
        from .physics import sample_observation

        return sample_observation(intensity, self.noise_sd, self.rng)

    def run_experiment(self, query: Location | np.ndarray) -> LocationObservation:
        query_tuple = normalize_location(query, self.dim)
        from .physics import round_positive_observation

        observation = round_positive_observation(self.step(query_tuple), 2)
        result = LocationObservation(query=query_tuple, value=float(observation))
        self.observed_data.append(result)
        return result

    def sample_random_input(self, bounds: tuple[float, float] = (-2.0, 2.0)) -> Location:
        low, high = bounds
        return tuple(float(value) for value in self.rng.uniform(low, high, size=self.dim))


@dataclass
class _LocationTrialState:
    trial_idx: int
    env: "LocationFindingEnv"
    observations: list[LocationObservation]
    rng: np.random.Generator
    belief_state: _BeliefState[SourceConfig] | None = None
    strategy_library: "LocationStrategyLibrary | None" = None
    final_estimate: SourceConfig | None = None
    final_rmse: float = float("inf")


@dataclass(frozen=True)
class _StrategyLocationRequest:
    strategy: str
    belief_state: _BeliefState[SourceConfig]
    observations: list[LocationObservation]


@dataclass(frozen=True)
class LocationStrategyCandidate:
    strategy: str
    root_query: Location | None = None


@dataclass
class _StrategyRollout:
    request_index: int
    strategy_index: int
    strategy: str
    truth: SourceConfig
    start_probability: float
    start_belief_state: _BeliefState[SourceConfig]
    belief_state: _BeliefState[SourceConfig]
    particle_support: list[SourceConfig] = field(default_factory=list)
    fixed_common_support: list[SourceConfig] = field(default_factory=list)
    generated_hypotheses: list[SourceConfig] = field(default_factory=list)
    final_generated_hypotheses: list[SourceConfig] = field(default_factory=list)
    final_scoring_belief_state: _BeliefState[SourceConfig] | None = None
    simulated_observations: list[LocationObservation] = field(default_factory=list)
    simulated_belief_states: list[_BeliefState[SourceConfig]] = field(default_factory=list)
    simulated_supports: list[list[SourceConfig]] = field(default_factory=list)
    root_query: Location | None = None
    scoring_seed: int | None = None
    noise_zs: tuple[float, ...] = field(default_factory=tuple)
    rollout_index: int = 0


@dataclass(frozen=True)
class _StrategyEvaluationRequest:
    strategies: list[str]
    belief_state: _BeliefState[SourceConfig]
    observations: list[LocationObservation]
    rng: np.random.Generator
    root_queries: list[Location | None] | None = None


@dataclass(frozen=True)
class LocationStrategyEvaluation:
    strategy: str
    mean_score: float
    score_variance: float
    root_query_fingerprint: str
    rollout_scores: list[float]
    root_query: Location | None = None
