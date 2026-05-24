"""Generic Expected Information Gain method for binary-observation environments.

For each candidate action ``a`` and belief state ``p(h)``, EIG is

.. math::

   \\text{EIG}(a) = H[p(o \\mid a)] - \\mathbb{E}_{h \\sim p}\\bigl[H[p(o \\mid a, h)]\\bigr]

where ``H`` is the Shannon entropy of a Bernoulli with parameter
``p(o = "label_a" \\mid a)``.

This module works for any environment whose observation space is a
finite two-label set.  It uses :meth:`core.Environment.log_likelihood_many`
to score every hypothesis against both possible observations at once, which
keeps the LLM-batching efficient for the animals environment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import numpy as np

from core import ActionScore, BeliefState, Environment, Method


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


def _binary_entropy(p: float) -> float:
    """Shannon entropy in nats of a Bernoulli with parameter ``p``."""
    if p <= 0.0 or p >= 1.0:
        return 0.0
    q = 1.0 - p
    return -p * math.log(p) - q * math.log(q)


@dataclass
class EIGBinary(Method[H, A, O, S]):
    """EIG scoring over a binary observation space.

    Parameters
    ----------
    observation_labels:
        Two-element sequence of the two possible observation values.  The first
        element is the "positive" label whose probability is plugged into the
        binary entropy formula.  For animals: ``("Yes", "No")``.
    """

    observation_labels: tuple[Any, Any]

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
            raise ValueError("EIGBinary requires at least one candidate action")
        if not belief_state.hypotheses:
            # Can't score without a belief support; return the first candidate.
            return ActionScore(action=candidates[0], score=0.0)

        scores = [
            self._score_action(
                action,
                belief_state,
                environment,
            )
            for action in candidates
        ]
        best_index = int(np.argmax(scores))
        return ActionScore(
            action=candidates[best_index],
            score=float(scores[best_index]),
            extras={
                "all_scores": [float(score) for score in scores],
            },
        )

    # ------------------------------------------------------------------

    def _score_action(
        self,
        action: A,
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
    ) -> float:
        """Compute EIG for a single candidate action."""
        positive_label, _negative_label = self.observation_labels

        # Build a synthetic positive observation; the environment then exposes
        # log p(o_positive | a, h) for every hypothesis in one batched call.
        synthetic_positive = self._synthesise_observation(action, positive_label)
        log_p_yes_given_h = environment.log_likelihood_many(
            belief_state.hypotheses,
            action,
            synthetic_positive,
        )
        p_yes_given_h = np.exp(np.clip(log_p_yes_given_h, a_min=-700.0, a_max=0.0))

        probabilities = belief_state.to_numpy()
        marginal_p_yes = float(np.dot(probabilities, p_yes_given_h))

        # Expected conditional entropy under the belief.
        conditional_entropy = float(
            np.dot(
                probabilities,
                np.array([_binary_entropy(float(p)) for p in p_yes_given_h]),
            )
        )
        marginal_entropy = _binary_entropy(marginal_p_yes)
        return marginal_entropy - conditional_entropy

    @staticmethod
    def _synthesise_observation(action: A, label: Any) -> Any:
        """Construct a placeholder observation carrying a specific label.

        For the animals environment, ``O`` is just the label string itself, so
        we can pass the label through directly.  Environments where ``O`` has
        more structure (e.g. ``LocationObservation``) should not use this
        method — they need a different EIG module entirely.
        """
        return label
