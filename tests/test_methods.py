"""Tests for the generic Method implementations."""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pytest

from core import ActionScore, BeliefState, Environment, Method
from methods import EIGBinary, Naive


# ---------------------------------------------------------------------------
# Minimal mock environment that lets us drive the methods directly.
# ---------------------------------------------------------------------------


class _MockBinaryEnvironment(Environment[str, str, str, str]):
    """Toy environment with deterministic likelihoods for testing EIG/Naive.

    Each hypothesis ``h`` answers each question ``q`` with a fixed probability
    that ``h`` would say "Yes", provided by a lookup table the test supplies.
    """

    def __init__(self, yes_probabilities: dict[tuple[str, str], float]):
        # (hypothesis, action) → P("Yes" | hypothesis, action)
        self._yes_probabilities = yes_probabilities

    @property
    def name(self) -> str:
        return "mock_binary"

    def sample_hidden_state(self, rng: np.random.Generator) -> str:
        return "h0"

    def observe(self, action: str, hidden_state: str, rng: np.random.Generator) -> str:
        p_yes = self._yes_probabilities.get((hidden_state, action), 0.5)
        return "Yes" if rng.random() < p_yes else "No"

    def log_prior(self, hypothesis: str) -> float:
        return 0.0

    def log_likelihood(self, hypothesis: str, action: str, observation: str) -> float:
        p_yes = self._yes_probabilities.get((hypothesis, action), 0.5)
        p = p_yes if observation == "Yes" else 1.0 - p_yes
        return math.log(max(p, 1e-300))

    def log_likelihood_many(
        self,
        hypotheses: Sequence[str],
        action: str,
        observation: str,
    ) -> np.ndarray:
        return np.array(
            [self.log_likelihood(h, action, observation) for h in hypotheses],
            dtype=float,
        )

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[str]:
        raise NotImplementedError("not used in these tests")

    def update_belief_state(
        self, belief_state, history, model, config
    ) -> BeliefState[str]:
        raise NotImplementedError("not used in these tests")

    def generate_candidate_actions(
        self, belief_state, history, model, config
    ) -> list[str]:
        raise NotImplementedError("not used in these tests")

    def round_metrics(self, belief_state, history, hidden_state) -> dict[str, float]:
        return {}

    def generate_naive_action(self, belief_state, history, model, config, **kwargs) -> str:
        return "naive-action"


# ---------------------------------------------------------------------------
# EIGBinary
# ---------------------------------------------------------------------------


def test_eig_binary_prefers_action_that_splits_belief_into_balanced_halves():
    # Two hypotheses; q1 separates them perfectly (Yes <-> h_a, No <-> h_b);
    # q2 doesn't (both answer Yes with probability 0.5).  EIG should pick q1.
    env = _MockBinaryEnvironment(
        yes_probabilities={
            ("h_a", "q1"): 1.0,
            ("h_b", "q1"): 0.0,
            ("h_a", "q2"): 0.5,
            ("h_b", "q2"): 0.5,
        }
    )
    belief = BeliefState.uniform(("h_a", "h_b"))
    method = EIGBinary(observation_labels=("Yes", "No"))

    result = method.select_action(
        candidates=["q1", "q2"],
        belief_state=belief,
        environment=env,
        model=None,
        history=[],
        config=None,
    )

    assert result.action == "q1"
    # EIG for q1 = ln(2) (1 bit of information for a 50/50 binary observation).
    assert result.score == pytest.approx(math.log(2.0))


def test_eig_binary_prefers_discriminative_question_under_skewed_belief():
    env = _MockBinaryEnvironment(
        yes_probabilities={
            ("h_a", "q1"): 1.0,
            ("h_b", "q1"): 0.0,
            ("h_c", "q1"): 0.0,
            ("h_a", "q2"): 0.0,
            ("h_b", "q2"): 1.0,
            ("h_c", "q2"): 1.0,
        }
    )
    # Belief: 0.6/0.2/0.2.  q2 splits 0.6 | 0.4, q1 splits 0.6 | 0.4 too —
    # but both are perfectly informative, so EIG should be equal.
    belief = BeliefState(
        hypotheses=("h_a", "h_b", "h_c"),
        probabilities=(0.6, 0.2, 0.2),
    )
    method = EIGBinary(observation_labels=("Yes", "No"))

    result = method.select_action(
        candidates=["q1", "q2"],
        belief_state=belief,
        environment=env,
        model=None,
        history=[],
        config=None,
    )

    # Tied EIG → either action is acceptable; just verify a valid pick.
    assert result.action in ("q1", "q2")
    all_scores = result.extras["all_scores"]
    assert all_scores[0] == pytest.approx(all_scores[1])


def test_eig_binary_handles_uninformative_question_with_zero_score():
    env = _MockBinaryEnvironment(
        yes_probabilities={
            ("h_a", "q1"): 0.5,
            ("h_b", "q1"): 0.5,
        }
    )
    belief = BeliefState.uniform(("h_a", "h_b"))
    method = EIGBinary(observation_labels=("Yes", "No"))

    result = method.select_action(
        candidates=["q1"],
        belief_state=belief,
        environment=env,
        model=None,
        history=[],
        config=None,
    )

    assert result.score == pytest.approx(0.0, abs=1e-9)


def test_eig_binary_falls_back_when_belief_state_is_empty():
    env = _MockBinaryEnvironment(yes_probabilities={})
    belief: BeliefState[str] = BeliefState()
    method = EIGBinary(observation_labels=("Yes", "No"))

    result = method.select_action(
        candidates=["q1"],
        belief_state=belief,
        environment=env,
        model=None,
        history=[],
        config=None,
    )

    assert result.action == "q1"
    assert result.score == 0.0


def test_eig_binary_raises_with_no_candidates():
    env = _MockBinaryEnvironment(yes_probabilities={})
    belief = BeliefState.uniform(("h_a", "h_b"))
    method = EIGBinary(observation_labels=("Yes", "No"))

    with pytest.raises(ValueError, match="candidate"):
        method.select_action(
            candidates=[],
            belief_state=belief,
            environment=env,
            model=None,
            history=[],
            config=None,
        )


# ---------------------------------------------------------------------------
# Naive
# ---------------------------------------------------------------------------


def test_naive_uses_environment_naive_action_hook():
    env = _MockBinaryEnvironment(yes_probabilities={})
    method = Naive()

    result = method.select_action(
        candidates=["first", "second", "third"],
        belief_state=BeliefState.uniform(("h",)),
        environment=env,
        model=None,
        history=[],
        config=None,
    )

    assert isinstance(result, ActionScore)
    assert result.action == "naive-action"
    assert result.score == 0.0


def test_naive_does_not_require_candidates():
    env = _MockBinaryEnvironment(yes_probabilities={})
    method = Naive()
    result = method.select_action(
        candidates=[],
        belief_state=BeliefState.uniform(("h",)),
        environment=env,
        model=None,
        history=[],
        config=None,
    )
    assert result.action == "naive-action"


def test_naive_maintains_belief_only_for_naive_belief_variant():
    assert Naive(method_name="naive").maintains_belief is False
    assert Naive(method_name="Naive").maintains_belief is False
    assert Naive(method_name="naive+belief").maintains_belief is True
