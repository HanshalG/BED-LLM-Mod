from __future__ import annotations

import math
from itertools import permutations
from typing import Any

import numpy as np

from core import BeliefState
from .types import Location, SourceConfig


def branch_decoy_endpoints(radius: float) -> np.ndarray:
    return np.asarray(
        [
            [-0.65 * radius, 0.0],
            [radius, 0.8 * radius],
            [radius, -0.8 * radius],
        ],
        dtype=float,
    )


def sample_source_configs_from_prior(
    rng: np.random.Generator,
    *,
    count: int,
    num_sources: int,
    dim: int,
    source_prior: str = "normal",
    source_radius: float = 1.0,
) -> np.ndarray:
    if source_prior == "normal":
        return rng.normal(0.0, 1.0, size=(count, num_sources, dim))
    if source_prior == "branch_decoy":
        if dim != 2:
            raise ValueError("branch_decoy source prior currently supports location_dim=2")
        endpoint_indices = rng.integers(0, 3, size=(count, num_sources))
        endpoints = branch_decoy_endpoints(float(source_radius))
        jitter_scales = np.asarray([0.05, 0.12, 0.12], dtype=float)
        jitter = rng.normal(0.0, jitter_scales[endpoint_indices][:, :, None], size=(count, num_sources, dim))
        return endpoints[endpoint_indices] + jitter
    raise ValueError("location_source_prior must be one of: normal, branch_decoy")


def location_signal_model(config: Any | None) -> str:
    return str(getattr(config, "location_signal_model", "inverse_square"))


def _local_bump_signal(distances_squared: np.ndarray, *, b: float, lengthscale: float, amplitude: float) -> np.ndarray:
    lengthscale_sq = max(float(lengthscale), 1e-12) ** 2
    return b + amplitude * np.sum(np.exp(-0.5 * distances_squared / lengthscale_sq), axis=-1)


def signal_intensities_from_distances(
    distances_squared: np.ndarray,
    *,
    b: float = 0.1,
    m: float = 1e-4,
    alpha: float = 1.0,
    signal_model: str = "inverse_square",
    signal_lengthscale: float = 0.75,
    signal_amplitude: float = 5.0,
) -> np.ndarray:
    if signal_model == "inverse_square":
        return b + np.sum(alpha / (m + distances_squared), axis=-1)
    if signal_model == "local_bump":
        return _local_bump_signal(
            distances_squared,
            b=b,
            lengthscale=signal_lengthscale,
            amplitude=signal_amplitude,
        )
    raise ValueError("location_signal_model must be one of: inverse_square, local_bump")


def signal_intensities_for_hypotheses(
    hypotheses: np.ndarray,
    query: Location | np.ndarray,
    *,
    config: Any | None = None,
    b: float = 0.1,
    m: float = 1e-4,
    alpha: float = 1.0,
) -> np.ndarray:
    theta = np.asarray(hypotheses, dtype=float)
    query_arr = np.asarray(query, dtype=float)
    distances_squared = np.sum((theta - query_arr) ** 2, axis=-1)
    return signal_intensities_from_distances(
        distances_squared,
        b=b,
        m=m,
        alpha=alpha,
        signal_model=location_signal_model(config),
        signal_lengthscale=float(getattr(config, "location_signal_lengthscale", 0.75)),
        signal_amplitude=float(getattr(config, "location_signal_amplitude", 5.0)),
    )


def signal_intensity_for_hypothesis(
    hypothesis: SourceConfig,
    query: Location,
    b: float = 0.1,
    m: float = 1e-4,
    alpha: float = 1.0,
    config: Any | None = None,
) -> float:
    means = signal_intensities_for_hypotheses(
        np.asarray(hypothesis, dtype=float)[None, :, :],
        query,
        config=config,
        b=b,
        m=m,
        alpha=alpha,
    )
    return float(means[0])


def sample_observation(mean: float, noise_sd: float, rng: np.random.Generator) -> float:
    """Sample multiplicative log-normal observation noise around ``mean``."""
    if mean <= 0.0:
        raise ValueError(f"mean must be positive for log-normal observations (got {mean})")
    return float(mean * math.exp(float(rng.normal(0.0, noise_sd))))


def round_positive_observation(value: float, decimals: int = 2) -> float:
    """Round display-scale observations while preserving positive likelihood support."""
    rounded = round(float(value), decimals)
    if rounded > 0.0:
        return float(rounded)
    return 10.0 ** (-decimals)


def observation_log_likelihood(value: float, mean: float, noise_sd: float) -> float:
    """Log likelihood under log(value) ~ Normal(log(mean), noise_sd).

    The 1 / value Jacobian term is omitted because it is constant across
    hypotheses for a fixed observation and cancels in posterior comparisons.
    """
    if value <= 0.0:
        return float("-inf")
    if mean <= 0.0:
        raise ValueError(f"mean must be positive for log-normal observations (got {mean})")
    z = (math.log(value) - math.log(mean)) / noise_sd
    return -0.5 * z * z - math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp(log_values: list[float] | np.ndarray) -> float:
    values = np.asarray(log_values, dtype=float)
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _hypothesis_log_prior(hypothesis: SourceConfig) -> float:
    theta = np.asarray(hypothesis, dtype=float)
    dimension_count = theta.size
    return float(-0.5 * np.sum(theta ** 2) - 0.5 * dimension_count * math.log(2.0 * math.pi))


hypothesis_log_prior = _hypothesis_log_prior


def hypothesis_log_prior_for_config(hypothesis: SourceConfig, config: Any | None = None) -> float:
    source_prior = str(getattr(config, "location_source_prior", "normal"))
    if source_prior == "normal":
        return _hypothesis_log_prior(hypothesis)
    if source_prior == "branch_decoy":
        theta = np.asarray(hypothesis, dtype=float)
        if theta.ndim != 2 or theta.shape[1] != 2:
            return float("-inf")
        endpoints = branch_decoy_endpoints(float(getattr(config, "location_source_radius", 1.0)))
        jitter_scales = np.asarray([0.05, 0.12, 0.12], dtype=float)
        per_source = []
        for source in theta:
            diff = source[None, :] - endpoints
            component_logs = (
                -0.5 * np.sum((diff / jitter_scales[:, None]) ** 2, axis=1)
                - 2.0 * np.log(jitter_scales)
                - math.log(2.0 * math.pi)
                - math.log(len(endpoints))
            )
            per_source.append(_logsumexp(component_logs))
        return float(sum(per_source))
    raise ValueError("location_source_prior must be one of: normal, branch_decoy")


def source_rmse(predicted: SourceConfig, true_sources: np.ndarray) -> float:
    predicted_arr = np.asarray(predicted, dtype=float)
    if predicted_arr.shape != true_sources.shape:
        raise ValueError("predicted sources and true sources must have matching shape")
    best_mse = min(
        float(np.mean((np.asarray(permutation, dtype=float) - true_sources) ** 2))
        for permutation in permutations(predicted_arr)
    )
    return math.sqrt(best_mse)


def _top_source_rmse(belief_state: BeliefState, true_sources: np.ndarray) -> float:
    if not belief_state.hypotheses:
        return float("inf")
    return source_rmse(belief_state.hypotheses[0], true_sources)


def _signal_grid(
    env: LocationFindingEnv,
    extent: tuple[float, float, float, float] = (-3.0, 3.0, -3.0, 3.0),
    resolution: int = 180,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_min, x_max, y_min, y_max = extent
    x_values = np.linspace(x_min, x_max, resolution)
    y_values = np.linspace(y_min, y_max, resolution)
    grid = np.empty((resolution, resolution), dtype=float)
    for row_idx, y_value in enumerate(y_values):
        for col_idx, x_value in enumerate(x_values):
            grid[row_idx, col_idx] = env.signal_intensity((x_value, y_value))
    return x_values, y_values, grid
