"""Value types for the MediQ adapter."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MediQTask:
    task_id: str
    source_id: str
    dataset: str
    question: str
    options: tuple[tuple[str, str], ...]
    answer_idx: str
    answer: str
    initial_info: str
    context: tuple[str, ...]
    facts: tuple[str, ...]
    answer_option_text: str = ""
    answer_text_matches_option: bool = True

    @property
    def option_labels(self) -> tuple[str, ...]:
        return tuple(label for label, _text in self.options)

    def option_text(self, label: str) -> str:
        for option_label, text in self.options:
            if option_label == label:
                return text
        raise KeyError(label)


@dataclass(frozen=True)
class MediQProfile:
    """A concrete counterfactual patient state beneath one diagnosis label."""

    profile_id: str
    diagnosis_label: str
    narrative: str


@dataclass(frozen=True)
class MediQAction:
    query: str
    outcomes: tuple[str, ...]
    task: MediQTask
    transcript: tuple[tuple[str, str], ...] = ()
    prior_probabilities: tuple[float, ...] = ()
    support_hypotheses: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        query = self.query.strip()
        outcomes = tuple(outcome.strip() for outcome in self.outcomes if outcome.strip())
        if not query:
            raise ValueError("MediQ query must be non-empty")
        if not 3 <= len(outcomes) <= 5:
            raise ValueError("MediQ actions require 3-5 answer outcomes")
        if len({outcome.casefold() for outcome in outcomes}) != len(outcomes):
            raise ValueError("MediQ action outcomes must be unique")
        support_size = len(self.support_hypotheses) or len(self.task.options)
        if self.prior_probabilities and len(self.prior_probabilities) != support_size:
            raise ValueError("MediQ action prior must match its hypothesis support")
        if self.support_hypotheses and len(set(self.support_hypotheses)) != len(self.support_hypotheses):
            raise ValueError("MediQ action hypothesis support must be unique")
        object.__setattr__(self, "query", query)
        object.__setattr__(self, "outcomes", outcomes)


@dataclass(frozen=True)
class MediQObservation:
    reply: str
    mapped_outcome: str | None
    mapped_cleanly: bool
    selected_fact_indices: tuple[int, ...]
    grounded: bool
    relevant: bool
    cannot_answer: bool
