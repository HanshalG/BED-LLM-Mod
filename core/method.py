"""Abstract action-selection strategies (EIG, Naive, StrategyEIG, ...).

A :class:`Method` is the environment-agnostic policy that consumes a list of
candidate actions plus the current belief state and chooses which action to
take.  Splitting this out from :class:`Environment` lets us mix any method
with any environment.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Generic, Sequence, TypeVar

from .belief import BeliefState
from .environment import Environment


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


@dataclass(frozen=True)
class ActionScore(Generic[A]):
    """Result of scoring a single candidate action.

    ``score`` is the value the method maximises (higher = better).  Methods
    are free to populate ``extras`` with diagnostic info that the runner can
    log without the runner needing to know the method's internals.
    """

    action: A
    score: float
    extras: dict[str, Any] | None = None


class Method(ABC, Generic[H, A, O, S]):
    """Abstract action-selection policy.

    Implementations are typically environment-agnostic (an EIG method works
    for any environment that defines a likelihood) but they may be
    specialised when needed.
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier used for logging and run-naming."""

    @abstractmethod
    def select_action(
        self,
        candidates: Sequence[A],
        belief_state: BeliefState[H],
        environment: Environment[S, H, A, O],
        model: Any,  # LLM adapter
        history: Sequence[tuple[A, O]],
        config: Any,
    ) -> ActionScore[A]:
        """Choose the best action from the candidate list.

        Methods that don't actually need candidates (e.g. a pure naive
        question-generator) may ignore the list and produce their own action.
        """

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{type(self).__name__}(name={self.name!r})"
