#!/usr/bin/env python3
"""Pure finite-belief mechanics for HiddenBench dynamic-belief V3."""

from __future__ import annotations

import math
from typing import Any, Sequence


QUERY_IDS = ("Q1", "Q2", "Q3", "Q4")
CHANNEL_IDS = ("C1", "C2", "C3")


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def tv(left: Sequence[float], right: Sequence[float]) -> float:
    return 0.5 * sum(abs(a - b) for a, b in zip(left, right, strict=True))


def predictive(
    belief: Sequence[float], likelihood: Sequence[Sequence[float]]
) -> list[float]:
    return [
        sum(belief[option] * likelihood[option][channel] for option in range(len(belief)))
        for channel in range(len(CHANNEL_IDS))
    ]


def update(
    belief: Sequence[float],
    likelihood: Sequence[Sequence[float]],
    channel: int,
) -> list[float]:
    joint = [belief[index] * likelihood[index][channel] for index in range(len(belief))]
    evidence = sum(joint)
    if evidence <= 0:
        raise ValueError("observation has zero predictive probability")
    return [value / evidence for value in joint]


def eig(belief: Sequence[float], likelihood: Sequence[Sequence[float]]) -> float:
    channel_probabilities = predictive(belief, likelihood)
    return entropy(belief) - sum(
        probability * entropy(update(belief, likelihood, channel))
        for channel, probability in enumerate(channel_probabilities)
        if probability > 0
    )


def best_query(
    belief: Sequence[float],
    matrices: dict[str, Sequence[Sequence[float]]],
    *,
    exclude: str | None = None,
) -> tuple[str, float, float]:
    scores = {
        query_id: eig(belief, matrix)
        for query_id, matrix in matrices.items()
        if query_id != exclude
    }
    ordered = sorted(scores, key=lambda query_id: (-scores[query_id], query_id))
    margin = scores[ordered[0]] - scores[ordered[1]] if len(ordered) > 1 else math.inf
    return ordered[0], scores[ordered[0]], margin


def dynamic_depth_two(
    prior: Sequence[float],
    matrices: dict[str, Sequence[Sequence[float]]],
    refreshed: dict[str, dict[str, Sequence[float]]],
) -> dict[str, Any]:
    root_entropy = entropy(prior)
    scores: dict[str, float] = {}
    second_actions: dict[str, dict[str, str]] = {}
    for first_id in QUERY_IDS:
        first_predictive = predictive(prior, matrices[first_id])
        expected_terminal = 0.0
        second_actions[first_id] = {}
        for channel_index, channel_id in enumerate(CHANNEL_IDS):
            branch_belief = list(refreshed[first_id][channel_id])
            second_id, _, _ = best_query(branch_belief, matrices, exclude=first_id)
            second_actions[first_id][channel_id] = second_id
            second_predictive = predictive(branch_belief, matrices[second_id])
            terminal_entropy = sum(
                probability
                * entropy(update(branch_belief, matrices[second_id], second_channel))
                for second_channel, probability in enumerate(second_predictive)
                if probability > 0
            )
            expected_terminal += first_predictive[channel_index] * terminal_entropy
        scores[first_id] = root_entropy - expected_terminal
    ordered = sorted(QUERY_IDS, key=lambda query_id: (-scores[query_id], query_id))
    return {
        "scores": scores,
        "first_query_id": ordered[0],
        "margin": scores[ordered[0]] - scores[ordered[1]],
        "second_actions": second_actions,
        "response_contingent_first_queries": sum(
            len(set(actions.values())) >= 2 for actions in second_actions.values()
        ),
    }


def fixed_depth_two(
    prior: Sequence[float],
    matrices: dict[str, Sequence[Sequence[float]]],
) -> dict[str, Any]:
    exact = {
        query_id: {
            channel_id: update(prior, matrices[query_id], channel_index)
            for channel_index, channel_id in enumerate(CHANNEL_IDS)
        }
        for query_id in QUERY_IDS
    }
    return dynamic_depth_two(prior, matrices, exact)


def branch_diagnostics(
    prior: Sequence[float],
    matrices: dict[str, Sequence[Sequence[float]]],
    refreshed: dict[str, dict[str, Sequence[float]]],
) -> dict[str, float | int]:
    compatibility_increases: list[float] = []
    exact_distances: list[float] = []
    within_query_distances: list[float] = []
    for query_id in QUERY_IDS:
        matrix = matrices[query_id]
        for channel_index, channel_id in enumerate(CHANNEL_IDS):
            branch = refreshed[query_id][channel_id]
            prior_compatibility = sum(
                prior[option] * matrix[option][channel_index]
                for option in range(len(prior))
            )
            branch_compatibility = sum(
                branch[option] * matrix[option][channel_index]
                for option in range(len(prior))
            )
            compatibility_increases.append(branch_compatibility - prior_compatibility)
            exact_distances.append(tv(branch, update(prior, matrix, channel_index)))
        branches = [refreshed[query_id][channel_id] for channel_id in CHANNEL_IDS]
        within_query_distances.extend(
            tv(branches[left], branches[right])
            for left in range(3)
            for right in range(left + 1, 3)
        )
    return {
        "obedient_branch_count": sum(value > 0 for value in compatibility_increases),
        "mean_compatibility_increase": sum(compatibility_increases) / len(compatibility_increases),
        "mean_exact_bayes_tv": sum(exact_distances) / len(exact_distances),
        "branches_exact_bayes_tv_at_least_001": sum(value >= 0.01 for value in exact_distances),
        "mean_within_query_pairwise_tv": sum(within_query_distances) / len(within_query_distances),
    }


def endpoint_metrics(belief: Sequence[float], correct_index: int) -> dict[str, float]:
    correct_probability = float(belief[correct_index])
    if not 0 < correct_probability < 1:
        raise ValueError("endpoint correct probability is saturated")
    return {
        "brier": sum(
            (probability - (1.0 if index == correct_index else 0.0)) ** 2
            for index, probability in enumerate(belief)
        ),
        "log_loss": -math.log(correct_probability),
        "correct_probability": correct_probability,
    }
