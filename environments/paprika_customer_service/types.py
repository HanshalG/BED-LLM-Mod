"""Value types used by the Paprika customer-service adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PaprikaTask:
    task_id: str
    scenario: str
    solution: str
    split: str


@dataclass(frozen=True)
class PaprikaAction:
    query: str
    outcomes: tuple[str, ...]
    scenario: str
    transcript: tuple[tuple[str, str], ...] = ()

    def __post_init__(self) -> None:
        query = self.query.strip()
        outcomes = tuple(outcome.strip() for outcome in self.outcomes if outcome.strip())
        if not query:
            raise ValueError("Paprika action query must be non-empty")
        if not 3 <= len(outcomes) <= 5:
            raise ValueError("Paprika actions require 3-5 answer outcomes")
        if len({outcome.casefold() for outcome in outcomes}) != len(outcomes):
            raise ValueError("Paprika action outcomes must be unique")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "outcomes", outcomes)


@dataclass(frozen=True)
class PaprikaObservation:
    reply: str
    mapped_outcome: str | None
    mapped_cleanly: bool
    goal_reached: bool = False

