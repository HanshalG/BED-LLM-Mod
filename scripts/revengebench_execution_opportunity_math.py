#!/usr/bin/env python3
"""Exact finite-support planner for the frozen RevengeBench opportunity audit."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any


ProbabilityVector = list[float]
LikelihoodMatrix = list[list[float]]


def _require_finite_nonnegative(values: Sequence[float], *, name: str) -> None:
    if not values or any(not math.isfinite(value) or value < 0.0 for value in values):
        raise ValueError(f"{name} must contain finite nonnegative values")


def normalize(values: Sequence[float]) -> ProbabilityVector:
    _require_finite_nonnegative(values, name="probabilities")
    total = math.fsum(values)
    if total <= 0.0:
        raise ValueError("probabilities must have positive mass")
    return [value / total for value in values]


def entropy(probabilities: Sequence[float]) -> float:
    probabilities = normalize(probabilities)
    return -math.fsum(value * math.log(value) for value in probabilities if value > 0.0)


def likelihood_from_distances(distances: Sequence[Sequence[float]], beta: float) -> LikelihoodMatrix:
    """Return L[y][theta] = softmax_y(-beta * D[y][theta])."""
    if not math.isfinite(beta) or beta <= 0.0:
        raise ValueError("beta must be finite and positive")
    size = len(distances)
    if size < 2 or any(len(row) != size for row in distances):
        raise ValueError("distance matrix must be square with at least two hypotheses")
    for row in distances:
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in row):
            raise ValueError("distances must be finite values in [0, 1]")

    result = [[0.0] * size for _ in range(size)]
    for theta in range(size):
        logits = [-beta * distances[y][theta] for y in range(size)]
        offset = max(logits)
        weights = [math.exp(value - offset) for value in logits]
        column = normalize(weights)
        for y, value in enumerate(column):
            result[y][theta] = value
    return result


def validate_likelihood(likelihood: Sequence[Sequence[float]], hypothesis_count: int) -> LikelihoodMatrix:
    if len(likelihood) < 2 or any(len(row) != hypothesis_count for row in likelihood):
        raise ValueError("likelihood must have at least two observations and one column per hypothesis")
    matrix = [list(row) for row in likelihood]
    for row in matrix:
        _require_finite_nonnegative(row, name="likelihood row")
    for theta in range(hypothesis_count):
        column_sum = math.fsum(row[theta] for row in matrix)
        if not math.isclose(column_sum, 1.0, rel_tol=0.0, abs_tol=1e-10):
            raise ValueError("each likelihood column must sum to one")
    return matrix


def observation_update(
    prior: Sequence[float], likelihood: Sequence[Sequence[float]], observation: int
) -> tuple[float, ProbabilityVector]:
    prior = normalize(prior)
    matrix = validate_likelihood(likelihood, len(prior))
    if not 0 <= observation < len(matrix):
        raise IndexError("observation index out of range")
    joint = [prior[theta] * matrix[observation][theta] for theta in range(len(prior))]
    predictive = math.fsum(joint)
    if predictive <= 0.0:
        raise ValueError("observation has zero predictive probability")
    return predictive, [value / predictive for value in joint]


def expected_posterior_entropy(prior: Sequence[float], likelihood: Sequence[Sequence[float]]) -> float:
    prior = normalize(prior)
    matrix = validate_likelihood(likelihood, len(prior))
    terms = []
    for observation in range(len(matrix)):
        predictive, posterior = observation_update(prior, matrix, observation)
        terms.append(predictive * entropy(posterior))
    return math.fsum(terms)


def one_step_eig(prior: Sequence[float], likelihood: Sequence[Sequence[float]]) -> float:
    return entropy(prior) - expected_posterior_entropy(prior, likelihood)


def _argmax_first(values: Sequence[float]) -> int:
    if not values or any(not math.isfinite(value) for value in values):
        raise ValueError("argmax values must be finite and nonempty")
    best = 0
    for index in range(1, len(values)):
        if values[index] > values[best]:
            best = index
    return best


def adaptive_two_step_value(
    prior: Sequence[float], likelihoods: Sequence[Sequence[Sequence[float]]], first_probe: int
) -> dict[str, Any]:
    prior = normalize(prior)
    matrices = [validate_likelihood(item, len(prior)) for item in likelihoods]
    if len(matrices) < 2 or not 0 <= first_probe < len(matrices):
        raise ValueError("invalid first probe")

    first = matrices[first_probe]
    expected_final_entropy = 0.0
    continuation_probes: list[int] = []
    continuation_eigs: list[float] = []
    for observation in range(len(first)):
        predictive, posterior = observation_update(prior, first, observation)
        remaining = [index for index in range(len(matrices)) if index != first_probe]
        values = [one_step_eig(posterior, matrices[index]) for index in remaining]
        selected_position = _argmax_first(values)
        selected = remaining[selected_position]
        continuation_probes.append(selected)
        continuation_eigs.append(values[selected_position])
        expected_final_entropy += predictive * expected_posterior_entropy(posterior, matrices[selected])

    return {
        "first_probe": first_probe,
        "continuation_probes": continuation_probes,
        "continuation_eigs": continuation_eigs,
        "expected_final_entropy": expected_final_entropy,
        "utility": entropy(prior) - expected_final_entropy,
    }


def fixed_two_step_value(
    prior: Sequence[float],
    first_likelihood: Sequence[Sequence[float]],
    second_likelihood: Sequence[Sequence[float]],
) -> float:
    prior = normalize(prior)
    first = validate_likelihood(first_likelihood, len(prior))
    second = validate_likelihood(second_likelihood, len(prior))
    expected_final_entropy = 0.0
    for y1 in range(len(first)):
        p1, posterior1 = observation_update(prior, first, y1)
        expected_final_entropy += p1 * expected_posterior_entropy(posterior1, second)
    return entropy(prior) - expected_final_entropy


def evaluate_policies(
    likelihoods: Sequence[Sequence[Sequence[float]]], prior: Sequence[float] | None = None
) -> dict[str, Any]:
    if not likelihoods:
        raise ValueError("at least one likelihood is required")
    hypothesis_count = len(likelihoods[0][0])
    if prior is None:
        prior = [1.0 / hypothesis_count] * hypothesis_count
    prior = normalize(prior)
    matrices = [validate_likelihood(item, len(prior)) for item in likelihoods]
    if len(matrices) < 2:
        raise ValueError("at least two probes are required")

    root_eigs = [one_step_eig(prior, matrix) for matrix in matrices]
    adaptive = [adaptive_two_step_value(prior, matrices, index) for index in range(len(matrices))]
    depth_two_first = _argmax_first([item["utility"] for item in adaptive])
    receding_first = _argmax_first(root_eigs)

    fixed_candidates = []
    for first in range(len(matrices)):
        for second in range(len(matrices)):
            if second == first:
                continue
            fixed_candidates.append(
                {
                    "probes": [first, second],
                    "utility": fixed_two_step_value(prior, matrices[first], matrices[second]),
                }
            )
    fixed_best = fixed_candidates[_argmax_first([item["utility"] for item in fixed_candidates])]
    random_utility = math.fsum(item["utility"] for item in fixed_candidates) / len(fixed_candidates)

    return {
        "prior_entropy": entropy(prior),
        "root_eigs": root_eigs,
        "depth_two": adaptive[depth_two_first],
        "receding_myopic": {
            **adaptive[receding_first],
            "selection_root_eig": root_eigs[receding_first],
        },
        "fixed": fixed_best,
        "random": {"utility": random_utility},
        "changed_first_action": depth_two_first != receding_first,
        "depth_two_margin": adaptive[depth_two_first]["utility"] - adaptive[receding_first]["utility"],
    }
