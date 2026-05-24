"""Animals-only methods (20 Questions) that are not environment-agnostic."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence, TypeVar

from core import ActionScore, BeliefState, Environment, Method
from environments.animals.questions import evaluate_questions_forward_search, generate_candidate_questions
from environments.animals.env import AnimalsBEDEnvironment, _belief_state_to_flat, _history_to_messages


H = TypeVar("H")
A = TypeVar("A")
O = TypeVar("O")
S = TypeVar("S")


def _require_animals_env(environment: Environment[S, H, A, O]) -> AnimalsBEDEnvironment:
    if not isinstance(environment, AnimalsBEDEnvironment):
        raise TypeError(f"{type(environment).__name__} is not an AnimalsBEDEnvironment")
    return environment


@dataclass
class AnimalsForwardSearchEIG(Method[str, str, str, str]):
    """20 Questions EIG via ``evaluate_questions_forward_search``."""

    @property
    def name(self) -> str:
        return "EIG"

    def select_action(
        self,
        candidates: Sequence[str],
        belief_state: BeliefState[str],
        environment: Environment[str, str, str, str],
        model: Any,
        history: Sequence[tuple[str, str]],
        config: Any,
    ) -> ActionScore[str]:
        _require_animals_env(environment)
        flat = _belief_state_to_flat(belief_state)
        history_messages = _history_to_messages(history)
        if not candidates:
            candidates = generate_candidate_questions(
                flat,
                history_messages,
                model,
                config.generation_temperature_diverse,
                config.target_num_questions,
                verbose=False,
            )
        if not candidates:
            raise ValueError("AnimalsForwardSearchEIG could not produce candidate questions")
        if len(candidates) == 1:
            return ActionScore(action=candidates[0], score=0.0)
        scores = evaluate_questions_forward_search(
            flat,
            history_messages,
            list(candidates),
            True,
            False,
            model,
            config,
            depth=config.search_depth,
        )
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return ActionScore(
            action=candidates[best_idx],
            score=float(scores[best_idx]),
            extras={"all_scores": [float(s) for s in scores], "metric_name": "selected_eig"},
        )


@dataclass
class AnimalsEntropy(Method[str, str, str, str]):
    """Entropy minimization (not EIG) for 20 Questions."""

    @property
    def name(self) -> str:
        return "Entropy"

    def select_action(
        self,
        candidates: Sequence[str],
        belief_state: BeliefState[str],
        environment: Environment[str, str, str, str],
        model: Any,
        history: Sequence[tuple[str, str]],
        config: Any,
    ) -> ActionScore[str]:
        _require_animals_env(environment)
        flat = _belief_state_to_flat(belief_state)
        history_messages = _history_to_messages(history)
        if not candidates:
            candidates = generate_candidate_questions(
                flat,
                history_messages,
                model,
                config.generation_temperature_diverse,
                config.target_num_questions,
                verbose=False,
            )
        if not candidates:
            raise ValueError("AnimalsEntropy could not produce candidate questions")
        scores = evaluate_questions_forward_search(
            flat,
            history_messages,
            list(candidates),
            eig=False,
            deterministic=False,
            questioner=model,
            config=config,
            depth=config.search_depth,
        )
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return ActionScore(action=candidates[best_idx], score=float(scores[best_idx]))


@dataclass
class AnimalsSplit(Method[str, str, str, str]):
    """Deterministic belief-split question selection for 20 Questions."""

    @property
    def name(self) -> str:
        return "split"

    def select_action(
        self,
        candidates: Sequence[str],
        belief_state: BeliefState[str],
        environment: Environment[str, str, str, str],
        model: Any,
        history: Sequence[tuple[str, str]],
        config: Any,
    ) -> ActionScore[str]:
        _require_animals_env(environment)
        flat = _belief_state_to_flat(belief_state)
        history_messages = _history_to_messages(history)
        if not candidates:
            candidates = generate_candidate_questions(
                flat,
                history_messages,
                model,
                config.generation_temperature_diverse,
                config.target_num_questions,
                verbose=False,
            )
        if not candidates:
            raise ValueError("AnimalsSplit could not produce candidate questions")
        scores = evaluate_questions_forward_search(
            flat,
            history_messages,
            list(candidates),
            eig=False,
            deterministic=True,
            questioner=model,
            config=config,
            depth=config.search_depth,
        )
        best_idx = int(max(range(len(scores)), key=lambda i: scores[i]))
        return ActionScore(action=candidates[best_idx], score=float(scores[best_idx]))
