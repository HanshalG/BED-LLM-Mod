"""Generic helpers for continuous-observation EIG methods."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, TypeVar

import numpy as np

from core import ActionScore, BeliefState, Environment, Method


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


class ContinuousEIGEnvironment(Protocol[H, A]):
    """Optional protocol for environments that expose Gaussian predictive means."""

    def predictive_means(self, hypotheses: Sequence[H], action: A) -> np.ndarray:
        """Return one predictive mean per hypothesis for ``action``."""


def quadrature_nodes(order: int) -> tuple[np.ndarray, np.ndarray]:
    nodes, weights = np.polynomial.hermite.hermgauss(order)
    return nodes.astype(float), weights.astype(float) / math.sqrt(math.pi)


def normal_logpdf_array(values: np.ndarray, means: np.ndarray, noise_sd: float) -> np.ndarray:
    z = (values - means) / noise_sd
    return -0.5 * z * z - math.log(noise_sd) - 0.5 * math.log(2.0 * math.pi)


def expected_information_gain_from_means(
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
    y_values = means[:, None] + math.sqrt(2.0) * noise_sd * nodes[None, :]
    component_log_likelihoods = normal_logpdf_array(y_values, means[:, None], noise_sd)
    all_log_likelihoods = normal_logpdf_array(y_values[:, :, None], means[None, None, :], noise_sd)
    max_values = np.max(
        all_log_likelihoods + np.log(np.maximum(probabilities, 1e-300))[None, None, :],
        axis=2,
        keepdims=True,
    )
    mixture = np.squeeze(
        max_values
        + np.log(
            np.sum(
                np.exp(
                    all_log_likelihoods
                    + np.log(np.maximum(probabilities, 1e-300))[None, None, :]
                    - max_values
                ),
                axis=2,
                keepdims=True,
            )
        ),
        axis=2,
    )
    value = np.sum(probabilities[:, None] * weights[None, :] * (component_log_likelihoods - mixture))
    return max(0.0, float(value))


def score_continuous_forward_search(
    belief_state: BeliefState[H],
    candidates: Sequence[A],
    environment: Environment[S, H, A, O],
    model: Any,
    history: Sequence[tuple[A, O]],
    config: Any,
    *,
    noise_sd: float,
    quadrature_order: int,
    search_depth: int,
) -> list[float]:
    """Score continuous-observation candidates with depth-1 or depth-2 forward search."""
    if not candidates:
        return []
    if not hasattr(environment, "predictive_means"):
        raise TypeError(
            f"{type(environment).__name__} must implement predictive_means for ContinuousEIG"
        )

    nodes, weights = quadrature_nodes(quadrature_order)
    probabilities = belief_state.to_numpy()
    hypotheses = belief_state.hypotheses
    candidate_means = [
        np.asarray(environment.predictive_means(hypotheses, candidate), dtype=float)  # type: ignore[attr-defined]
        for candidate in candidates
    ]
    immediate = [
        expected_information_gain_from_means(probabilities, means, noise_sd, nodes, weights)
        for means in candidate_means
    ]
    if search_depth <= 1 or len(hypotheses) <= 1:
        return immediate
    if search_depth != 2:
        raise ValueError("continuous forward search supports search_depth 1 or 2 only")

    batched_depth2 = getattr(environment, "score_continuous_forward_search_depth2_batched", None)
    if (
        callable(batched_depth2)
        and model is not None
        and getattr(config, "location_posterior_mode", None) == "llm_distribution"
    ):
        return batched_depth2(
            belief_state,
            candidates,
            environment,
            model,
            history,
            config,
            noise_sd=noise_sd,
            quadrature_order=quadrature_order,
            immediate_scores=immediate,
        )

    totals = list(immediate)
    for candidate_idx, (candidate, means) in enumerate(zip(candidates, candidate_means)):
        for mean, hypothesis_probability in zip(means, probabilities):
            if hypothesis_probability == 0.0:
                continue
            branch_observation = environment.representative_observation(candidate, float(mean))
            branch_history = list(history) + [(candidate, branch_observation)]
            branch_update = getattr(environment, "belief_after_branch_observation", None)
            if callable(branch_update):
                future_belief = branch_update(belief_state, branch_history, model, config)
            else:
                future_belief = environment.update_belief_state(
                    belief_state,
                    branch_history,
                    model,
                    config,
                )
            future_candidates = list(
                environment.generate_candidate_actions(
                    future_belief,
                    branch_history,
                    model,
                    config,
                )
            )
            if not future_candidates or future_belief.support_size <= 1:
                continue
            future_scores = score_continuous_forward_search(
                future_belief,
                future_candidates,
                environment,
                model,
                branch_history,
                config,
                noise_sd=noise_sd,
                quadrature_order=quadrature_order,
                search_depth=1,
            )
            if future_scores:
                totals[candidate_idx] += float(hypothesis_probability) * max(future_scores)
    return totals


@dataclass
class ContinuousEIG(Method[H, A, O, S]):
    """Gaussian-observation EIG with optional depth-2 forward search."""

    noise_sd: float
    quadrature_order: int
    search_depth: int = 1

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
        if not candidates:
            raise ValueError("ContinuousEIG requires at least one candidate action")
        scoring_belief = belief_state
        subsample = getattr(environment, "belief_state_for_eig_scoring", None)
        if callable(subsample):
            scoring_belief = subsample(belief_state, config)
        scores = score_continuous_forward_search(
            scoring_belief,
            candidates,
            environment,
            model,
            history,
            config,
            noise_sd=self.noise_sd,
            quadrature_order=self.quadrature_order,
            search_depth=self.search_depth,
        )
        best_idx = int(np.argmax(scores)) if scores else 0
        return ActionScore(
            action=candidates[best_idx],
            score=float(scores[best_idx]) if scores else 0.0,
            extras={"all_scores": [float(score) for score in scores], "metric_name": "selected_eig"},
        )

