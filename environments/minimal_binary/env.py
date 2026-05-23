"""Minimal binary-observation environment for integration tests."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

from core import BeliefState, Environment
from core.llm_likelihood import log_likelihood_many_llm_binary


@dataclass
class MinimalBinaryEnvironment(Environment[str, str, str, str]):
    """Toy Yes/No environment with LLM likelihood from a lookup table."""

    hypotheses: tuple[str, ...] = ("h0", "h1")
    hidden_state: str = "h0"
    observation_labels: tuple[str, str] = ("Yes", "No")
    yes_probabilities: dict[tuple[str, str], float] = field(default_factory=dict)
    config: Any = None
    _questioner: Any = field(default=None, repr=False)

    @property
    def name(self) -> str:
        return "minimal_binary"

    def set_questioner(self, questioner: Any) -> None:
        self._questioner = questioner

    def get_questioner(self) -> Any:
        if self._questioner is None:
            raise RuntimeError("questioner not set")
        return self._questioner

    def build_likelihood_messages(self, hypothesis: str, action: str) -> list[dict[str, str]]:
        return [{"role": "user", "content": f"{hypothesis}|{action}"}]

    def sample_hidden_state(self, rng: np.random.Generator) -> str:
        return self.hidden_state

    def observe(self, action: str, hidden_state: str, rng: np.random.Generator) -> str:
        p_yes = self.yes_probabilities.get((hidden_state, action), 0.5)
        return "Yes" if rng.random() < p_yes else "No"

    def log_prior(self, hypothesis: str) -> float:
        return 0.0

    def log_likelihood(self, hypothesis: str, action: str, observation: str) -> float:
        import math

        rows = log_likelihood_many_llm_binary(self, [hypothesis], action, observation)
        return float(rows[0])

    def log_likelihood_many(
        self,
        hypotheses: Sequence[str],
        action: str,
        observation: str,
    ) -> np.ndarray:
        return log_likelihood_many_llm_binary(self, hypotheses, action, observation)

    def initial_belief_state(self, model: Any, config: Any) -> BeliefState[str]:
        self.set_questioner(model)
        return BeliefState.uniform(self.hypotheses)

    def update_belief_state(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
    ) -> BeliefState[str]:
        return belief_state

    def generate_candidate_actions(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
    ) -> list[str]:
        return ["q1", "q2"]

    def round_metrics(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        hidden_state: str,
    ) -> dict[str, float]:
        return {"correct": float(hidden_state == (belief_state.top() or ("", 0))[0])}

    def generate_naive_action(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        *,
        method_name: str | None = None,
    ) -> str:
        return "naive-q"

    def choose_strategy_action(
        self,
        belief_state: BeliefState[str],
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Any,
        rng: np.random.Generator,
        round_index: int,
        *,
        fixed_root: bool = False,
    ) -> tuple[str, float, Any]:
        return "strategy-q", 1.0, None
