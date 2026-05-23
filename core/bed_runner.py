"""Shared outer loop for BED experiments.

The :class:`BEDRunner` drives the canonical Bayesian Experimental Design
template:

.. code-block:: text

    for each trial:
        sample hidden state
        bootstrap belief state
        for each round:
            generate candidate actions
            score & pick best action via Method
            observe outcome
            update belief state
            record metrics
        aggregate trial metrics
    return aggregated results

It is intentionally tiny: every environment- and method-specific concern is
delegated to :class:`Environment` and :class:`Method` respectively, which
keeps the loop readable and trivially testable with a mock environment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Generic, Sequence, TypeVar

import numpy as np

from .belief import BeliefState
from .environment import Environment
from .method import ActionScore, Method


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


@dataclass(frozen=True)
class RoundResult(Generic[A, O]):
    """Result of running a single round within a trial."""

    round_index: int
    candidates: tuple[A, ...]
    chosen: ActionScore[A]
    observation: O
    metrics: dict[str, float]


@dataclass(frozen=True)
class TrialResult(Generic[A, O, S]):
    """Result of a single complete trial."""

    trial_index: int
    hidden_state: S
    rounds: tuple[RoundResult[A, O], ...]
    final_metrics: dict[str, float]


@dataclass(frozen=True)
class RunResult(Generic[A, O, S]):
    """Result of running ``num_trials`` trials."""

    trials: tuple[TrialResult[A, O, S], ...]
    aggregate_metrics: dict[str, list[float]] = field(default_factory=dict)


class BEDRunner(Generic[H, A, O, S]):
    """Run a Bayesian Experimental Design experiment end-to-end."""

    def __init__(
        self,
        environment: Environment[S, H, A, O],
        method: Method[H, A, O, S],
        model: Any,  # LLM adapter
        config: Any,
        *,
        num_trials: int,
        num_rounds: int,
        rng: np.random.Generator | None = None,
    ) -> None:
        self.environment = environment
        self.method = method
        self.model = model
        self.config = config
        self.num_trials = int(num_trials)
        self.num_rounds = int(num_rounds)
        if self.num_trials <= 0:
            raise ValueError(f"num_trials must be positive (got {num_trials})")
        if self.num_rounds <= 0:
            raise ValueError(f"num_rounds must be positive (got {num_rounds})")
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------

    def run_single_trial(self, trial_index: int) -> TrialResult[A, O, S]:
        """Run one trial.  Mostly exposed for testing."""
        env = self.environment
        method = self.method
        model = self.model
        config = self.config

        if hasattr(env, "set_questioner"):
            env.set_questioner(model)
        if hasattr(env, "set_active_method"):
            env.set_active_method(getattr(method, "name", type(method).__name__))

        hidden_state = env.sample_hidden_state(self.rng)
        belief_state: BeliefState[H] = env.initial_belief_state(model, config)
        history: list[tuple[A, O]] = []
        rounds: list[RoundResult[A, O]] = []

        for round_index in range(self.num_rounds):
            if getattr(method, "skip_candidate_generation", False):
                candidates: list[A] = []
            else:
                candidates = list(
                    env.generate_candidate_actions(belief_state, history, model, config)
                )

            if not candidates and not getattr(method, "skip_candidate_generation", False):
                if env.on_empty_candidates(belief_state, history, round_index, config):
                    continue
                raise ValueError(
                    f"{type(method).__name__} requires candidates but none were generated"
                )

            chosen = method.select_action(
                candidates,
                belief_state,
                env,
                model,
                history,
                config,
            )
            observation = env.observe(chosen.action, hidden_state, self.rng)
            history.append((chosen.action, observation))
            belief_state = env.update_belief_state(belief_state, history, model, config)

            metrics = dict(env.round_metrics(belief_state, history, hidden_state))
            if hasattr(method, "metrics_after_observation"):
                metrics.update(
                    method.metrics_after_observation(
                        belief_state,
                        history,
                        env,
                        model,
                        hidden_state,
                        config,
                    )
                )
            if chosen.extras and "metric_name" in chosen.extras:
                metrics[str(chosen.extras["metric_name"])] = float(chosen.score)
            rounds.append(
                RoundResult(
                    round_index=round_index,
                    candidates=tuple(candidates),
                    chosen=chosen,
                    observation=observation,
                    metrics=metrics,
                )
            )

            if env.early_stop(belief_state, history, hidden_state, observation):
                break

        final_metrics = rounds[-1].metrics if rounds else {}
        return TrialResult(
            trial_index=trial_index,
            hidden_state=hidden_state,
            rounds=tuple(rounds),
            final_metrics=final_metrics,
        )

    def run(self) -> RunResult[A, O, S]:
        """Run all ``num_trials`` trials and aggregate metrics."""
        trials = tuple(
            self.run_single_trial(trial_index)
            for trial_index in range(self.num_trials)
        )
        aggregate = self._aggregate_metrics(trials)
        return RunResult(trials=trials, aggregate_metrics=aggregate)

    # ------------------------------------------------------------------

    @staticmethod
    def _aggregate_metrics(
        trials: Sequence[TrialResult[A, O, S]],
    ) -> dict[str, list[float]]:
        """Stack per-round metrics across trials.

        The returned dictionary maps each metric name to a list whose length
        equals the longest trial's round count.  Per-round entries are the
        mean over trials that produced a value for that round; trials that
        ended early contribute nothing to later rounds.
        """
        if not trials:
            return {}

        max_rounds = max(len(trial.rounds) for trial in trials)
        if max_rounds == 0:
            return {}

        accumulator: dict[str, list[list[float]]] = {}
        for trial in trials:
            for round_idx, round_result in enumerate(trial.rounds):
                for name, value in round_result.metrics.items():
                    series = accumulator.setdefault(name, [[] for _ in range(max_rounds)])
                    series[round_idx].append(float(value))

        aggregated: dict[str, list[float]] = {}
        for name, series in accumulator.items():
            aggregated[name] = [
                (sum(values) / len(values)) if values else float("nan")
                for values in series
            ]
        return aggregated
