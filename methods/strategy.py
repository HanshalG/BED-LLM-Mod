"""Generic StrategyEIG protocol method."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

import numpy as np

from core import ActionScore, BeliefState, Environment, Method


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


@dataclass
class StrategyEIG(Method[H, A, O, S]):
    """Delegate StrategyEIG action selection to environment strategy hooks."""

    fixed_root: bool = False
    rng: np.random.Generator | None = None
    skip_candidate_generation: bool = True

    @property
    def name(self) -> str:
        return "StrategyEIG+root" if self.fixed_root else "StrategyEIG"

    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        if self.rng is None:
            seed = getattr(config, "location_seed", getattr(config, "seed", None))
            self.rng = np.random.default_rng(seed)
        action, score, evaluation = environment.choose_strategy_action(
            belief_state,
            history,
            model,
            config,
            self.rng,
            round_index=len(history),
            fixed_root=self.fixed_root,
        )
        return ActionScore(
            action=action,
            score=float(score),
            extras={"evaluation": evaluation, "metric_name": "selected_eig"},
        )

