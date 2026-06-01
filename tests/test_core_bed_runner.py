"""End-to-end tests for ``core.BEDRunner`` with a synthetic environment.

These tests exist to verify that the abstract :class:`core.Environment` /
:class:`core.Method` / :class:`core.BEDRunner` triple wires together
correctly.  They use a tiny "guess a number in [0, N)" environment that has
nothing to do with animals or location finding — proving the runner is truly
environment-agnostic.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np
import pytest

from core import (
    ActionScore,
    BEDRunner,
    BeliefState,
    Environment,
    Method,
    RoundResult,
    TrialResult,
)


# ---------------------------------------------------------------------------
# Toy environment: questioner is trying to identify an integer in [0, N).
# Hypotheses, actions, and observations are all integers; the action ``q`` is
# "is the answer < q?" and the observation is the truthful 0/1 answer.
# ---------------------------------------------------------------------------


class _NumberGuessEnvironment(Environment[int, int, int, int]):
    def __init__(self, n: int, fixed_state: int | None = None):
        self.n = int(n)
        self.fixed_state = fixed_state

    @property
    def name(self) -> str:
        return "number_guess"

    def sample_hidden_state(self, rng: np.random.Generator) -> int:
        if self.fixed_state is not None:
            return self.fixed_state
        return int(rng.integers(low=0, high=self.n))

    def observe(self, action: int, hidden_state: int, rng: np.random.Generator) -> int:
        return int(hidden_state < action)

    def log_prior(self, hypothesis: int) -> float:
        # Uniform over [0, N).
        if 0 <= hypothesis < self.n:
            return -math.log(self.n)
        return float("-inf")

    def log_likelihood(self, hypothesis: int, action: int, observation: int) -> float:
        # Deterministic indicator likelihood.
        return 0.0 if int(hypothesis < action) == observation else float("-inf")

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[int]:
        return BeliefState.uniform(range(self.n))

    def update_belief_state(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        model: Any,
        config: Any,
    ) -> BeliefState[int]:
        # Re-derive the posterior from scratch from the full history.
        hypotheses = list(belief_state.hypotheses)
        if not hypotheses:
            return belief_state
        log_scores = [self.log_prior(h) for h in hypotheses]
        for action, observation in history:
            for idx, h in enumerate(hypotheses):
                log_scores[idx] += self.log_likelihood(h, action, observation)
        # Drop hypotheses with -inf scores (impossible).
        survivors = [
            (h, score)
            for h, score in zip(hypotheses, log_scores)
            if score > float("-inf")
        ]
        if not survivors:
            return BeliefState.uniform(())
        kept_h, kept_scores = zip(*survivors)
        return BeliefState.from_log_scores(kept_h, kept_scores)

    def generate_candidate_actions(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        model: Any,
        config: Any,
    ) -> list[int]:
        # Every integer 1..N-1 is a valid splitting question.
        return list(range(1, self.n))

    def round_metrics(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        hidden_state: int,
    ) -> dict[str, float]:
        top = belief_state.top()
        if top is None:
            return {"support_size": 0.0, "correct": 0.0}
        hypothesis, _probability = top
        return {
            "support_size": float(belief_state.support_size),
            "correct": float(hypothesis == hidden_state),
            "correct_mass": belief_state.probability_of(lambda h: h == hidden_state),
        }

    def early_stop(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        hidden_state: int,
        latest_observation: int,
    ) -> bool:
        # Stop as soon as the support collapses to one hypothesis.
        return belief_state.support_size <= 1


class _BisectionMethod(Method[int, int, int, int]):
    """Pick the action whose ``<`` split is closest to a 50/50 belief split."""

    @property
    def name(self) -> str:
        return "bisection"

    def select_action(
        self,
        candidates: Sequence[int],
        belief_state: BeliefState[int],
        environment: Environment,
        model: Any,
        history: Sequence,
        config: Any,
    ) -> ActionScore[int]:
        if not candidates:
            raise ValueError("bisection method requires candidates")
        best_action = candidates[0]
        best_distance = math.inf
        best_yes_mass = 0.0
        for action in candidates:
            yes_mass = belief_state.probability_of(lambda h, a=action: h < a)
            distance = abs(yes_mass - 0.5)
            if distance < best_distance:
                best_distance = distance
                best_action = action
                best_yes_mass = yes_mass
        # Score = negative distance (so larger is better).
        return ActionScore(action=best_action, score=-best_distance, extras={"yes_mass": best_yes_mass})


class _DirectNoBeliefMethod(Method[int, int, int, int]):
    skip_candidate_generation = True

    @property
    def name(self) -> str:
        return "direct_no_belief"

    def requires_belief_state(self, environment, config) -> bool:
        return False

    def select_action(
        self,
        candidates: Sequence[int],
        belief_state: BeliefState[int],
        environment: Environment,
        model: Any,
        history: Sequence,
        config: Any,
    ) -> ActionScore[int]:
        assert not candidates
        assert belief_state.support_size == 0
        return ActionScore(action=1, score=0.0, extras={"metric_name": "selected_eig"})

    def metrics_after_observation(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        environment: Environment,
        model: Any,
        hidden_state: int,
        config: Any,
    ) -> dict[str, float]:
        assert belief_state.support_size == 0
        return {"history_len": float(len(history))}


class _CountingNumberGuessEnvironment(_NumberGuessEnvironment):
    def __init__(self, n: int, fixed_state: int | None = None):
        super().__init__(n=n, fixed_state=fixed_state)
        self.initial_calls = 0
        self.update_calls = 0
        self.round_metric_calls = 0

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[int]:
        self.initial_calls += 1
        return super().initial_belief_state(model, config)

    def update_belief_state(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        model: Any,
        config: Any,
    ) -> BeliefState[int]:
        self.update_calls += 1
        return super().update_belief_state(belief_state, history, model, config)

    def round_metrics(
        self,
        belief_state: BeliefState[int],
        history: Sequence[tuple[int, int]],
        hidden_state: int,
    ) -> dict[str, float]:
        self.round_metric_calls += 1
        return super().round_metrics(belief_state, history, hidden_state)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_runner_rejects_non_positive_trial_or_round_counts():
    env = _NumberGuessEnvironment(n=4)
    with pytest.raises(ValueError):
        BEDRunner(env, _BisectionMethod(), model=None, config=None, num_trials=0, num_rounds=2)
    with pytest.raises(ValueError):
        BEDRunner(env, _BisectionMethod(), model=None, config=None, num_trials=1, num_rounds=0)


def test_runner_drives_a_single_trial_to_correct_answer():
    env = _NumberGuessEnvironment(n=8, fixed_state=5)
    runner = BEDRunner(
        env,
        _BisectionMethod(),
        model=None,
        config=None,
        num_trials=1,
        num_rounds=10,
        rng=np.random.default_rng(0),
    )

    trial = runner.run_single_trial(0)

    assert isinstance(trial, TrialResult)
    assert trial.trial_index == 0
    assert trial.hidden_state == 5

    # After enough binary splits the belief should collapse onto 5.
    final = trial.final_metrics
    assert final["correct"] == 1.0
    # log2(8) = 3 splits should be enough; allow some slack since we may have
    # short-circuited via early_stop after the first collapse.
    assert len(trial.rounds) <= 8


def test_runner_aggregates_metrics_across_multiple_trials():
    env = _NumberGuessEnvironment(n=4)
    runner = BEDRunner(
        env,
        _BisectionMethod(),
        model=None,
        config=None,
        num_trials=8,
        num_rounds=4,
        rng=np.random.default_rng(123),
    )

    result = runner.run()

    assert len(result.trials) == 8
    assert "correct" in result.aggregate_metrics
    # Bisection on n=4 should always identify the answer within 2 rounds.
    final_correct = [trial.final_metrics["correct"] for trial in result.trials]
    assert all(value == 1.0 for value in final_correct)


def test_runner_records_chosen_action_and_observation_per_round():
    env = _NumberGuessEnvironment(n=4, fixed_state=2)
    runner = BEDRunner(
        env,
        _BisectionMethod(),
        model=None,
        config=None,
        num_trials=1,
        num_rounds=4,
        rng=np.random.default_rng(0),
    )

    trial = runner.run_single_trial(0)

    for round_result in trial.rounds:
        assert isinstance(round_result, RoundResult)
        assert round_result.chosen.action in round_result.candidates
        assert round_result.observation in (0, 1)
        # The chosen ActionScore should carry the method's extras dict.
        assert round_result.chosen.extras is not None
        assert "yes_mass" in round_result.chosen.extras
    assert trial.final_belief_state is not None
    assert trial.final_belief_state.support_size >= 1


def test_runner_raises_on_empty_candidates_by_default():
    class EmptyCandidateEnvironment(_NumberGuessEnvironment):
        def generate_candidate_actions(self, belief_state, history, model, config):
            return []

    runner = BEDRunner(
        EmptyCandidateEnvironment(n=4, fixed_state=2),
        _BisectionMethod(),
        model=None,
        config=None,
        num_trials=1,
        num_rounds=1,
    )

    with pytest.raises(ValueError, match="requires candidates"):
        runner.run_single_trial(0)


def test_runner_skips_belief_lifecycle_when_method_does_not_require_it():
    env = _CountingNumberGuessEnvironment(n=4, fixed_state=2)
    runner = BEDRunner(
        env,
        _DirectNoBeliefMethod(),
        model=None,
        config=None,
        num_trials=1,
        num_rounds=2,
    )

    trial = runner.run_single_trial(0)

    assert env.initial_calls == 0
    assert env.update_calls == 0
    assert env.round_metric_calls == 0
    assert trial.final_belief_state is not None
    assert trial.final_belief_state.support_size == 0
    assert trial.final_metrics["history_len"] == 2.0
    assert trial.final_metrics["selected_eig"] == 0.0


def test_batched_runner_skips_belief_lifecycle_when_method_does_not_require_it():
    env = _CountingNumberGuessEnvironment(n=4, fixed_state=2)
    runner = BEDRunner(
        env,
        _DirectNoBeliefMethod(),
        model=None,
        config=None,
        num_trials=3,
        num_rounds=1,
        trial_batch_size=3,
    )

    result = runner.run()

    assert env.initial_calls == 0
    assert env.update_calls == 0
    assert env.round_metric_calls == 0
    assert len(result.trials) == 3
    assert all(trial.final_belief_state is not None for trial in result.trials)
    assert all(trial.final_belief_state.support_size == 0 for trial in result.trials)


def test_batched_runner_skips_empty_candidate_trials_without_selecting():
    class SkipOneEnvironment(_NumberGuessEnvironment):
        def initial_belief_states(self, trial_indices, model, config):
            return [
                BeliefState.uniform(("skip",)) if trial_index == 0 else BeliefState.uniform(("run",))
                for trial_index in trial_indices
            ]

        def generate_candidate_actions_many(self, belief_states, histories, model, config):
            return [
                [] if belief_state.hypotheses == ("skip",) else [1]
                for belief_state in belief_states
            ]

        def on_empty_candidates(self, belief_state, history, round_index, config):
            return belief_state.hypotheses == ("skip",)

        def update_belief_state(self, belief_state, history, model, config):
            return belief_state

        def round_metrics(self, belief_state, history, hidden_state):
            return {"ran": 1.0}

    class StrictMethod(Method[str, int, int, int]):
        @property
        def name(self):
            return "strict"

        def select_action(self, candidates, belief_state, environment, model, history, config):
            if not candidates:
                raise AssertionError("empty candidate trial should have been skipped")
            return ActionScore(action=candidates[0], score=0.0)

    runner = BEDRunner(
        SkipOneEnvironment(n=2),
        StrictMethod(),
        model=None,
        config=None,
        num_trials=2,
        num_rounds=1,
        trial_batch_size=2,
    )

    result = runner.run()

    assert [len(trial.rounds) for trial in result.trials] == [0, 1]


def test_runner_respects_early_stop():
    # With fixed_state=0, the very first action q=1 collapses the belief to
    # {0}, so the trial should early-stop after exactly one round.
    env = _NumberGuessEnvironment(n=4, fixed_state=0)
    runner = BEDRunner(
        env,
        _BisectionMethod(),
        model=None,
        config=None,
        num_trials=1,
        num_rounds=10,
        rng=np.random.default_rng(0),
    )

    trial = runner.run_single_trial(0)

    # Could take 1 or 2 rounds depending on which split is closest to 50/50,
    # but it must definitely be fewer than the full 10.
    assert 1 <= len(trial.rounds) < 10
    assert trial.final_metrics["correct"] == 1.0
