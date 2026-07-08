from __future__ import annotations

import math

import numpy as np

from core import BeliefState
from helpers import Config
from methods.continuous_eig import expected_information_gain_from_means, quadrature_nodes
from .formatting import _log_location
from .beliefs import _posterior_after_observation
from .physics import signal_intensities_for_hypotheses
from .types import Location, LocationObservation


def _quadrature_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes.astype(float), (weights.astype(float) / math.sqrt(math.pi))


def expected_information_gain(
    belief_state: BeliefState,
    query: Location,
    noise_sd: float,
    quadrature_order: int,
    config: Config | None = None,
) -> float:
    if len(belief_state.hypotheses) <= 1:
        return 0.0

    probabilities = np.asarray(belief_state.probabilities, dtype=float)
    means = signal_intensities_for_hypotheses(
        np.asarray(list(belief_state.hypotheses), dtype=float),
        query,
        config=config,
    )
    nodes, weights = quadrature_nodes(quadrature_order)
    return expected_information_gain_from_means(probabilities, means, noise_sd, nodes, weights)


def _log_observation_logpdf_array(log_values: np.ndarray, log_means: np.ndarray, noise_sd: float) -> np.ndarray:
    z = (log_values - log_means) / noise_sd
    return -0.5 * z * z - math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)


def _logsumexp_array(values: np.ndarray, axis: int) -> np.ndarray:
    max_values = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(max_values + np.log(np.sum(np.exp(values - max_values), axis=axis, keepdims=True)), axis=axis)


def _expected_information_gain_from_means(
    probabilities: np.ndarray,
    means: np.ndarray,
    noise_sd: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> float:
    if len(means) <= 1:
        return 0.0

    probabilities = np.asarray(probabilities, dtype=float)
    means = np.asarray(means, dtype=float)
    log_means = np.log(means)
    log_y_values = log_means[:, None] + math.sqrt(2.0) * noise_sd * nodes[None, :]
    component_log_likelihoods = _log_observation_logpdf_array(log_y_values, log_means[:, None], noise_sd)
    all_log_likelihoods = _log_observation_logpdf_array(log_y_values[:, :, None], log_means[None, None, :], noise_sd)
    mixture_log_likelihoods = _logsumexp_array(
        all_log_likelihoods + np.log(np.maximum(probabilities, 1e-300))[None, None, :],
        axis=2,
    )
    value = np.sum(probabilities[:, None] * weights[None, :] * (component_log_likelihoods - mixture_log_likelihoods))
    return max(0.0, float(value))


def _expected_information_gain_batch_from_means(
    probability_rows: np.ndarray,
    means: np.ndarray,
    noise_sd: float,
    nodes: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    if len(means) <= 1:
        return np.zeros(probability_rows.shape[0], dtype=float)

    probability_rows = np.asarray(probability_rows, dtype=float)
    means = np.asarray(means, dtype=float)
    log_means = np.log(means)
    log_y_values = log_means[None, :, None] + math.sqrt(2.0) * noise_sd * nodes[None, None, :]
    component_log_likelihoods = _log_observation_logpdf_array(log_y_values, log_means[None, :, None], noise_sd)
    all_log_likelihoods = _log_observation_logpdf_array(log_y_values[:, :, :, None], log_means[None, None, None, :], noise_sd)
    mixture_log_likelihoods = _logsumexp_array(
        all_log_likelihoods + np.log(np.maximum(probability_rows, 1e-300))[:, None, None, :],
        axis=3,
    )
    values = np.sum(
        probability_rows[:, :, None]
        * weights[None, None, :]
        * (component_log_likelihoods - mixture_log_likelihoods),
        axis=(1, 2),
    )
    return np.maximum(values, 0.0)


def score_candidate_locations(
    belief_state: BeliefState,
    candidates: list[Location],
    config: Config,
    questioner: "Model | None" = None,
    observations: list[LocationObservation] | None = None,
) -> list[float]:
    """Score candidates via :mod:`methods.continuous_eig` (depth-1/2 forward search)."""
    from environments.location_finding.env import LocationBEDEnvironment
    from methods.continuous_eig import score_continuous_forward_search

    if not candidates:
        _log_location("EIG scoring: no candidate locations to score", config)
        return []
    if config.location_search_depth == 2 and (questioner is None or observations is None):
        raise ValueError(
            "location_search_depth=2 requires questioner and observations for branch updates"
        )

    env = LocationBEDEnvironment(config=config)
    history = [(observation.query, observation) for observation in (observations or [])]
    return score_continuous_forward_search(
        belief_state,
        candidates,
        env,
        questioner,
        history,
        config,
        noise_sd=config.location_noise_sd,
        quadrature_order=config.location_eig_quadrature_order,
        search_depth=config.location_search_depth,
    )


def _posterior_probabilities_after_values(
    probabilities: np.ndarray,
    means: np.ndarray,
    values: np.ndarray,
    noise_sd: float,
) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if np.any(values <= 0.0):
        return np.zeros((len(values), len(probabilities)), dtype=float)
    log_values = np.log(values)
    log_means = np.log(means)
    log_scores = (
        np.log(np.maximum(probabilities, 1e-300))[None, :]
        + _log_observation_logpdf_array(log_values[:, None], log_means[None, :], noise_sd)
    )
    normalizers = _logsumexp_array(log_scores, axis=1)
    return np.exp(log_scores - normalizers[:, None])
