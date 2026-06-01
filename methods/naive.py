"""Generic two-phase naive baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from core import ActionScore, BeliefState, Environment, Method


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


@dataclass
class Naive(Method[H, A, O, S]):
    """Ask the environment for a direct naive action, then a naive estimate."""

    method_name: str = "naive"
    skip_candidate_generation: bool = True

    @property
    def name(self) -> str:
        return self.method_name

    def requires_belief_state(
        self,
        environment: Environment[S, H, A, O],
        config: Any,
    ) -> bool:
        del config
        return environment.naive_requires_belief_state(self.method_name)

    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        action = environment.generate_naive_action(
            belief_state,
            history,
            model,
            config,
            method_name=self.method_name,
        )
        return ActionScore(action=action, score=0.0, extras={"metric_name": "selected_eig"})

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
        actions = environment.generate_naive_actions_many(
            belief_states,
            histories,
            model,
            config,
            method_name=self.method_name,
        )
        return [
            ActionScore(action=action, score=0.0, extras={"metric_name": "selected_eig"})
            for action in actions
        ]

    def metrics_after_observation(
        self,
        belief_state: BeliefState[H],
        history: Sequence[tuple[A, O]],
        environment: Environment[S, H, A, O],
        model: Any,
        hidden_state: S,
        config: Any,
    ) -> dict[str, float]:
        return environment.naive_metrics_after_observation(
            belief_state,
            history,
            hidden_state,
            model,
            config,
            method_name=self.method_name,
        )

    def metrics_after_observations(
        self,
        belief_states: Sequence[BeliefState[H]],
        histories: Sequence[Sequence[tuple[A, O]]],
        environment: Environment[S, H, A, O],
        model: Any,
        hidden_states: Sequence[S],
        config: Any,
    ) -> list[dict[str, float]]:
        return environment.naive_metrics_after_observations(
            belief_states,
            histories,
            hidden_states,
            model,
            config,
            method_name=self.method_name,
        )
