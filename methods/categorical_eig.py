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
