"""One-step EIG for actions with action-specific categorical outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import numpy as np

from core import ActionScore, BeliefState, Environment, Method


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


def categorical_eig(prior: Sequence[float], likelihoods: np.ndarray) -> float:
    """Return I(H; O) for likelihood matrix ``[hypothesis, outcome]``."""
    p_h = np.asarray(prior, dtype=float)
    matrix = np.asarray(likelihoods, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] != p_h.size:
        raise ValueError("likelihoods must have shape [num_hypotheses, num_outcomes]")
    rows = matrix.sum(axis=1, keepdims=True)
    if np.any(rows <= 0.0):
        raise ValueError("each likelihood row must have positive mass")
    matrix = matrix / rows
    p_o = p_h @ matrix
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.zeros_like(matrix)
        np.divide(matrix, p_o[None, :], out=ratio, where=p_o[None, :] > 0.0)
        terms = p_h[:, None] * matrix * np.log(np.maximum(ratio, 1e-300))
    return float(np.sum(np.where(np.isfinite(terms), terms, 0.0)))


@dataclass
class CategoricalEIG(Method[H, A, O, S]):
    @property
    def name(self) -> str:
        return "EIG"

    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        del model, history, config
        if not candidates:
            raise ValueError("CategoricalEIG requires at least one candidate")
        scorer = getattr(environment, "outcome_likelihoods", None)
        if not callable(scorer):
            raise TypeError("CategoricalEIG environment must implement outcome_likelihoods")
        scores = [
            categorical_eig(
                belief_state.probabilities,
                scorer(belief_state.hypotheses, candidate),
            )
            for candidate in candidates
        ]
        best = int(np.argmax(scores))
        return ActionScore(
            action=candidates[best],
            score=scores[best],
            extras={"metric_name": "selected_eig", "candidate_scores": scores},
        )


@dataclass
class FullTwoStepCategoricalEIG(Method[H, A, O, S]):
    """Exact categorical branching with model-proposed second-step candidates."""

    @property
    def name(self) -> str:
        return "Full2StepEIG"

    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        if not candidates:
            raise ValueError("FullTwoStepCategoricalEIG requires candidates")
        scorer = getattr(environment, "outcome_likelihoods", None)
        branch_observation = getattr(environment, "branch_observation", None)
        if not callable(scorer) or not callable(branch_observation):
            raise TypeError("Two-step categorical environment lacks branch hooks")

        scores: list[float] = []
        branch_counts: list[int] = []
        prior = np.asarray(belief_state.probabilities, dtype=float)
        for candidate in candidates:
            first_likelihoods = np.asarray(scorer(belief_state.hypotheses, candidate))
            first_eig = categorical_eig(prior, first_likelihoods)
            expected_second_eig = 0.0
            expanded = 0
            for outcome_index in range(first_likelihoods.shape[1]):
                weights = prior * first_likelihoods[:, outcome_index]
                outcome_mass = float(np.sum(weights))
                if outcome_mass <= 0.0:
                    continue
                branch_belief = BeliefState(
                    hypotheses=belief_state.hypotheses,
                    probabilities=tuple(weights / outcome_mass),
                )
                synthetic = branch_observation(candidate, outcome_index)
                branch_history = list(history) + [(candidate, synthetic)]
                followups = environment.generate_candidate_actions(
                    branch_belief, branch_history, model, config
                )
                if followups:
                    best_second = max(
                        categorical_eig(
                            branch_belief.probabilities,
                            scorer(branch_belief.hypotheses, followup),
                        )
                        for followup in followups
                    )
                    expected_second_eig += outcome_mass * best_second
                expanded += 1
            scores.append(first_eig + expected_second_eig)
            branch_counts.append(expanded)
        best = int(np.argmax(scores))
        return ActionScore(
            action=candidates[best],
            score=scores[best],
            extras={
                "metric_name": "selected_eig",
                "candidate_scores": scores,
                "expanded_branch_counts": branch_counts,
                "planning_depth": 2,
            },
        )
