"""Exact-posterior Mastermind harness for validating one/two-step BED mechanics."""

from __future__ import annotations

import itertools
import math
from collections import Counter
from dataclasses import dataclass
from typing import Iterable, Sequence


Code = tuple[int, ...]
Feedback = tuple[int, int]


def code_space(num_symbols: int, code_length: int) -> tuple[Code, ...]:
    if num_symbols < 2 or code_length < 1:
        raise ValueError("Mastermind requires at least two symbols and positive code length")
    return tuple(itertools.product(range(num_symbols), repeat=code_length))


def feedback(secret: Code, guess: Code) -> Feedback:
    if len(secret) != len(guess):
        raise ValueError("secret and guess must have equal length")
    exact = sum(left == right for left, right in zip(secret, guess))
    overlap = sum((Counter(secret) & Counter(guess)).values())
    return exact, overlap - exact


def filter_support(support: Sequence[Code], guess: Code, observed: Feedback) -> tuple[Code, ...]:
    return tuple(secret for secret in support if feedback(secret, guess) == observed)


def entropy_uniform(support: Sequence[Code]) -> float:
    return math.log(len(support)) if support else 0.0


def feedback_partitions(support: Sequence[Code], guess: Code) -> dict[Feedback, tuple[Code, ...]]:
    buckets: dict[Feedback, list[Code]] = {}
    for secret in support:
        buckets.setdefault(feedback(secret, guess), []).append(secret)
    return {key: tuple(values) for key, values in buckets.items()}


def expected_entropy_after(support: Sequence[Code], guess: Code) -> float:
    if not support:
        return 0.0
    total = len(support)
    return sum(len(branch) / total * entropy_uniform(branch) for branch in feedback_partitions(support, guess).values())


def one_step_eig(support: Sequence[Code], guess: Code) -> float:
    return entropy_uniform(support) - expected_entropy_after(support, guess)


def two_step_eig(support: Sequence[Code], first_guess: Code, candidates: Sequence[Code]) -> float:
    """Exact depth-two value with an optimal second query in each feedback branch."""
    if not support:
        return 0.0
    total = len(support)
    expected_final_entropy = 0.0
    for branch in feedback_partitions(support, first_guess).values():
        best_remaining_entropy = min(expected_entropy_after(branch, second) for second in candidates)
        expected_final_entropy += len(branch) / total * best_remaining_entropy
    return entropy_uniform(support) - expected_final_entropy


@dataclass(frozen=True)
class MastermindDecision:
    guess: Code
    score: float
    depth: int


def select_query(support: Sequence[Code], candidates: Sequence[Code], *, depth: int) -> MastermindDecision:
    if not support or not candidates:
        raise ValueError("Mastermind selection requires non-empty support and candidates")
    if depth == 1:
        scores = [one_step_eig(support, candidate) for candidate in candidates]
    elif depth == 2:
        scores = [two_step_eig(support, candidate, candidates) for candidate in candidates]
    else:
        raise ValueError("Mastermind harness supports only depth 1 or 2")
    best = max(range(len(candidates)), key=lambda index: (scores[index], tuple(-x for x in candidates[index])))
    return MastermindDecision(candidates[best], scores[best], depth)


@dataclass(frozen=True)
class MastermindState:
    support: tuple[Code, ...]
    history: tuple[tuple[Code, Feedback], ...] = ()

    def observe(self, guess: Code, result: Feedback) -> "MastermindState":
        updated = filter_support(self.support, guess, result)
        if not updated:
            raise ValueError("Feedback is inconsistent with the current Mastermind support")
        return MastermindState(updated, self.history + ((guess, result),))

    def choose(self, candidates: Sequence[Code], *, depth: int) -> MastermindDecision:
        return select_query(self.support, candidates, depth=depth)

