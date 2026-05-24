from __future__ import annotations

import math
from itertools import permutations

import numpy as np

from .types import Location, LocationBeliefState, SourceConfig


def signal_intensity_for_hypothesis(
    hypothesis: SourceConfig,
    query: Location,
    b: float = 0.1,
    m: float = 1e-4,
    alpha: float = 1.0,
) -> float:
    theta = np.asarray(hypothesis, dtype=float)
    query_arr = np.asarray(query, dtype=float)
    distances_squared = np.sum((theta - query_arr) ** 2, axis=1)
    return float(b + np.sum(alpha / (m + distances_squared)))


def _log_normal_pdf(value: float, mean: float, sd: float) -> float:
    z = (value - mean) / sd
    return -0.5 * z * z - math.log(sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp(log_values: list[float] | np.ndarray) -> float:
    values = np.asarray(log_values, dtype=float)
    max_value = float(np.max(values))
    return max_value + float(np.log(np.sum(np.exp(values - max_value))))


def _hypothesis_log_prior(hypothesis: SourceConfig) -> float:
    theta = np.asarray(hypothesis, dtype=float)
    dimension_count = theta.size
    return float(-0.5 * np.sum(theta ** 2) - 0.5 * dimension_count * math.log(2.0 * math.pi))


hypothesis_log_prior = _hypothesis_log_prior


def source_rmse(predicted: SourceConfig, true_sources: np.ndarray) -> float:
    predicted_arr = np.asarray(predicted, dtype=float)
    if predicted_arr.shape != true_sources.shape:
        raise ValueError("predicted sources and true sources must have matching shape")
    best_mse = min(
        float(np.mean((np.asarray(permutation, dtype=float) - true_sources) ** 2))
        for permutation in permutations(predicted_arr)
    )
    return math.sqrt(best_mse)


def _top_source_rmse(belief_state: LocationBeliefState, true_sources: np.ndarray) -> float:
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
