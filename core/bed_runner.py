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
    final_belief_state: BeliefState[Any] | None = None


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
        trial_batch_size: int = 1,
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
        self.trial_batch_size = max(1, int(trial_batch_size))
        self.rng = rng if rng is not None else np.random.default_rng()

    # ------------------------------------------------------------------

    def run_single_trial(self, trial_index: int) -> TrialResult[A, O, S]:
        """Run one trial.  Mostly exposed for testing."""
        self._prepare_components()
        return self._run_single_trial_prepared(trial_index)

    def _prepare_components(self) -> None:
        """Share runner state with optional environment/method hooks."""
        env = self.environment
        method = self.method
        model = self.model

        if hasattr(env, "set_questioner"):
            env.set_questioner(model)
        if hasattr(env, "set_active_method"):
            env.set_active_method(getattr(method, "name", type(method).__name__))
        if hasattr(env, "rng"):
            setattr(env, "rng", self.rng)
        if hasattr(method, "rng") and getattr(method, "rng", None) is None:
            setattr(method, "rng", self.rng)

    def _run_single_trial_prepared(self, trial_index: int) -> TrialResult[A, O, S]:
        env = self.environment
        method = self.method
        model = self.model
        config = self.config

        hidden_state = env.sample_hidden_state_for_trial(trial_index, self.rng)
        requires_belief_state = method.requires_belief_state(env, config)
        belief_state: BeliefState[H] = (
            env.initial_belief_state(model, config)
            if requires_belief_state
            else BeliefState()
        )
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
            if requires_belief_state:
                belief_state = env.update_belief_state(belief_state, history, model, config)

            metrics = (
                dict(env.round_metrics(belief_state, history, hidden_state))
                if requires_belief_state
                else {}
            )
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

            if (
                requires_belief_state or env.early_stop_without_belief_state()
            ) and env.early_stop(belief_state, history, hidden_state, observation):
                break

        final_metrics = rounds[-1].metrics if rounds else {}
        return TrialResult(
            trial_index=trial_index,
            hidden_state=hidden_state,
            rounds=tuple(rounds),
            final_metrics=final_metrics,
            final_belief_state=belief_state,
        )

    def run_trial_batch(self, trial_indices: Sequence[int]) -> list[TrialResult[A, O, S]]:
        """Run a batch of trials round-by-round using environment batch hooks."""
        self._prepare_components()
        env = self.environment
        method = self.method
        model = self.model
        config = self.config

        trial_indices = list(trial_indices)
        hidden_states = env.sample_hidden_states_for_trials(trial_indices, self.rng)
        requires_belief_state = method.requires_belief_state(env, config)
        belief_states: list[BeliefState[H]] = (
            env.initial_belief_states(trial_indices, model, config)
            if requires_belief_state
            else [BeliefState() for _trial_index in trial_indices]
        )
        histories: list[list[tuple[A, O]]] = [[] for _trial_index in trial_indices]
        rounds_by_trial: list[list[RoundResult[A, O]]] = [[] for _trial_index in trial_indices]
        active = [True for _trial_index in trial_indices]

        for round_index in range(self.num_rounds):
            active_positions = [idx for idx, is_active in enumerate(active) if is_active]
            if not active_positions:
                break

            active_beliefs = [belief_states[idx] for idx in active_positions]
            active_histories = [histories[idx] for idx in active_positions]
            active_hidden_states = [hidden_states[idx] for idx in active_positions]

            if getattr(method, "skip_candidate_generation", False):
                candidates_many: list[list[A]] = [[] for _idx in active_positions]
            else:
                candidates_many = env.generate_candidate_actions_many(
                    active_beliefs,
                    active_histories,
                    model,
                    config,
                )

            if len(candidates_many) != len(active_positions):
                raise ValueError("generate_candidate_actions_many returned the wrong number of candidate lists")

            runnable_positions: list[int] = []
            runnable_candidates: list[list[A]] = []
            runnable_beliefs: list[BeliefState[H]] = []
            runnable_histories: list[list[tuple[A, O]]] = []
            runnable_hidden_states: list[S] = []
            for position, candidates, belief_state, history, hidden_state in zip(
                active_positions,
                candidates_many,
                active_beliefs,
                active_histories,
                active_hidden_states,
            ):
                if candidates or getattr(method, "skip_candidate_generation", False):
                    runnable_positions.append(position)
                    runnable_candidates.append(candidates)
                    runnable_beliefs.append(belief_state)
                    runnable_histories.append(history)
                    runnable_hidden_states.append(hidden_state)
                    continue
                if env.on_empty_candidates(belief_state, history, round_index, config):
                    continue
                raise ValueError(
                    f"{type(method).__name__} requires candidates but none were generated"
                )
            if not runnable_positions:
                continue

            prepare_action_batch = getattr(env, "prepare_action_batch", None)
            if callable(prepare_action_batch):
                prepare_action_batch(runnable_hidden_states)
            chosen_many = method.select_actions(
                runnable_candidates,
                runnable_beliefs,
                env,
                model,
                runnable_histories,
                config,
            )
            if len(chosen_many) != len(runnable_positions):
                raise ValueError("select_actions returned the wrong number of actions")

            observations = env.observe_many(
                [chosen.action for chosen in chosen_many],
                runnable_hidden_states,
                self.rng,
            )
            if len(observations) != len(runnable_positions):
                raise ValueError("observe_many returned the wrong number of observations")

            for position, chosen, observation in zip(runnable_positions, chosen_many, observations):
                histories[position].append((chosen.action, observation))

            if requires_belief_state:
                updated_beliefs = env.update_belief_states(
                    runnable_beliefs,
                    [histories[idx] for idx in runnable_positions],
                    model,
                    config,
                )
            else:
                updated_beliefs = list(runnable_beliefs)
            if len(updated_beliefs) != len(runnable_positions):
                raise ValueError("update_belief_states returned the wrong number of belief states")
            for position, belief_state in zip(runnable_positions, updated_beliefs):
                belief_states[position] = belief_state

            method_metrics_many = method.metrics_after_observations(
                updated_beliefs,
                [histories[idx] for idx in runnable_positions],
                env,
                model,
                runnable_hidden_states,
                config,
            )
            if len(method_metrics_many) != len(runnable_positions):
                raise ValueError("metrics_after_observations returned the wrong number of metric dictionaries")

            for position, candidates, chosen, observation, method_metrics in zip(
                runnable_positions,
                runnable_candidates,
                chosen_many,
                observations,
                method_metrics_many,
            ):
                metrics = (
                    dict(env.round_metrics(belief_states[position], histories[position], hidden_states[position]))
                    if requires_belief_state
                    else {}
                )
                metrics.update(method_metrics)
                if chosen.extras and "metric_name" in chosen.extras:
                    metrics[str(chosen.extras["metric_name"])] = float(chosen.score)
                rounds_by_trial[position].append(
                    RoundResult(
                        round_index=round_index,
                        candidates=tuple(candidates),
                        chosen=chosen,
                        observation=observation,
                        metrics=metrics,
                    )
                )
                if (
                    requires_belief_state or env.early_stop_without_belief_state()
                ) and env.early_stop(
                    belief_states[position], histories[position], hidden_states[position], observation
                ):
                    active[position] = False

        return [
            TrialResult(
                trial_index=trial_index,
                hidden_state=hidden_state,
                rounds=tuple(rounds),
                final_metrics=rounds[-1].metrics if rounds else {},
                final_belief_state=belief_state,
            )
            for trial_index, hidden_state, rounds, belief_state in zip(
                trial_indices,
                hidden_states,
                rounds_by_trial,
                belief_states,
            )
        ]

    def run(self) -> RunResult[A, O, S]:
        """Run all ``num_trials`` trials and aggregate metrics."""
        trial_results: list[TrialResult[A, O, S]] = []
        for batch_start in range(0, self.num_trials, self.trial_batch_size):
            batch_stop = min(self.num_trials, batch_start + self.trial_batch_size)
            batch_indices = list(range(batch_start, batch_stop))
            if self.trial_batch_size > 1 and len(batch_indices) > 1:
                trial_results.extend(self.run_trial_batch(batch_indices))
            else:
                trial_results.extend(
                    self.run_single_trial(trial_index)
                    for trial_index in batch_indices
                )
        trials = tuple(trial_results)
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
