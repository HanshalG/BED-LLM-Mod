"""Pre-registered naive-primary arbitration for Paprika."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import numpy as np

from core import ActionScore, BeliefState, Environment, Method
from methods.categorical_eig import categorical_eig


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


def categorical_eig_standard_error(
    prior: Sequence[float], likelihoods: np.ndarray
) -> float:
    """Weighted SE of per-hypothesis expected information contributions."""
    probabilities = np.asarray(prior, dtype=float)
    matrix = np.asarray(likelihoods, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != probabilities.size:
        raise ValueError("likelihoods must have shape [num_hypotheses, num_outcomes]")
    if probabilities.size == 0 or np.sum(probabilities) <= 0.0:
        raise ValueError("prior must contain positive mass")
    probabilities = probabilities / np.sum(probabilities)
    row_sums = matrix.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0.0):
        raise ValueError("each likelihood row must have positive mass")
    matrix = matrix / row_sums
    predictive = probabilities @ matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.zeros_like(matrix)
        np.divide(matrix, predictive[None, :], out=ratio, where=predictive[None, :] > 0.0)
        information = np.log(np.maximum(ratio, 1e-300))
    contributions = np.sum(matrix * information, axis=1)
    mean = float(probabilities @ contributions)
    variance = float(probabilities @ np.square(contributions - mean))
    effective_size = 1.0 / float(np.sum(np.square(probabilities)))
    return float(np.sqrt(max(variance, 0.0) / effective_size))


def select_naive_primary_index(
    scores: Sequence[float], standard_errors: Sequence[float]
) -> tuple[int, float, float]:
    """Return selected index, score gap, and one-SE threshold."""
    if not scores or len(scores) != len(standard_errors):
        raise ValueError("scores and standard_errors must be non-empty and aligned")
    best = int(np.argmax(np.asarray(scores, dtype=float)))
    gap = float(scores[best] - scores[0])
    threshold = float(np.hypot(standard_errors[best], standard_errors[0]))
    return (best if best != 0 and gap > threshold else 0), gap, threshold


@dataclass
class PaprikaNaivePrimaryArbitration(Method[H, A, O, S]):
    """Keep native action 0 unless an alternative clears a one-SE EIG margin."""

    skip_candidate_generation: bool = True

    @property
    def name(self) -> str:
        return "NaivePrimaryArbitration"

    def _score(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        matrices: Sequence[np.ndarray],
    ) -> ActionScore[A]:
        if len(candidates) != 3 or len(matrices) != 3:
            raise ValueError("NaivePrimaryArbitration requires exactly three candidates")
        scores = [
            categorical_eig(belief_state.probabilities, matrix) for matrix in matrices
        ]
        standard_errors = [
            categorical_eig_standard_error(belief_state.probabilities, matrix)
            for matrix in matrices
        ]
        selected, gap, threshold = select_naive_primary_index(scores, standard_errors)
        return ActionScore(
            action=candidates[selected],
            score=scores[selected],
            extras={
                "metric_name": "selected_eig",
                "candidate_queries": [str(getattr(candidate, "query", candidate)) for candidate in candidates],
                "candidate_scores": scores,
                "candidate_standard_errors": standard_errors,
                "native_default_index": 0,
                "selected_index": selected,
                "native_overridden": selected != 0,
                "score_gap_vs_native": gap,
                "one_se_threshold": threshold,
            },
        )

    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        del candidates
        proposals = environment.generate_arbitration_actions(
            belief_state, history, model, config
        )
        scorer = getattr(environment, "outcome_likelihoods", None)
        if not callable(scorer):
            raise TypeError("NaivePrimaryArbitration environment lacks outcome_likelihoods")
        matrices = [scorer(belief_state.hypotheses, proposal) for proposal in proposals]
        return self._score(proposals, belief_state, matrices)

    def select_actions(
        self,
        candidates_many: Sequence[Sequence[A]],
        belief_states: Sequence[BeliefState[H]],
        environment: Environment[S, H, A, O],
        model: Any,
        histories: Sequence[Sequence[tuple[A, O]]],
        config: Any,
    ) -> list[ActionScore[A]]:
        del candidates_many
        proposals_many = environment.generate_arbitration_actions_many(
            belief_states, histories, model, config
        )
        if any(len(proposals) != 3 for proposals in proposals_many):
            raise ValueError("NaivePrimaryArbitration requires exactly three candidates")
        requests = [
            (belief_state.hypotheses, proposal)
            for belief_state, proposals in zip(belief_states, proposals_many)
            for proposal in proposals
        ]
        many_scorer = getattr(environment, "outcome_likelihoods_many", None)
        if callable(many_scorer):
            matrices = many_scorer(requests)
        else:
            scorer = getattr(environment, "outcome_likelihoods", None)
            if not callable(scorer):
                raise TypeError("NaivePrimaryArbitration environment lacks outcome likelihoods")
            matrices = [scorer(hypotheses, proposal) for hypotheses, proposal in requests]
        results: list[ActionScore[A]] = []
        cursor = 0
        for proposals, belief_state in zip(proposals_many, belief_states):
            results.append(self._score(proposals, belief_state, matrices[cursor : cursor + 3]))
            cursor += 3
        return results
