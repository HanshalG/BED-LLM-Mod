"""Generic weighted belief state over a finite hypothesis support.

This is the environment-agnostic replacement for both ``helpers.BeliefState``
(animals: hypotheses are ``str``) and ``location_finding.LocationBeliefState``
(hypotheses are ``SourceConfig`` tuples).  The container is parametric in the
hypothesis type ``H`` so a single module can serve both environments
and any future ones.

The class is intentionally frozen / immutable so it can be safely passed
between trial state, EIG scoring, and logging without aliasing bugs.  Helper
methods (:meth:`renormalized`, :meth:`pruned`, :meth:`sorted_descending`) all
return new :class:`BeliefState` instances rather than mutating self.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Generic, Iterable, Sequence, TypeVar

import numpy as np


H = TypeVar("H")


@dataclass(frozen=True)
class BeliefState(Generic[H]):
    """A weighted set of hypotheses.

    Invariants enforced at construction time:

    - ``len(hypotheses) == len(probabilities)``
    - ``all(p >= 0)``
    - probabilities sum to 1 (renormalised if they don't, raising only when the
      total is zero and we have hypotheses)

    The probabilities list is stored as plain ``list[float]`` rather than a
    numpy array so the container is cheap to hash / compare and works
    identically whether constructed by Python or numpy code.
    """

    hypotheses: tuple[H, ...] = field(default_factory=tuple)
    probabilities: tuple[float, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        # Coerce to tuples so dataclass(frozen=True) stays truly immutable.
        if not isinstance(self.hypotheses, tuple):
            object.__setattr__(self, "hypotheses", tuple(self.hypotheses))
        if not isinstance(self.probabilities, tuple):
            object.__setattr__(self, "probabilities", tuple(float(p) for p in self.probabilities))

        if len(self.hypotheses) != len(self.probabilities):
            raise ValueError(
                f"hypotheses ({len(self.hypotheses)}) and probabilities "
                f"({len(self.probabilities)}) must have the same length"
            )
        for probability in self.probabilities:
            if probability < 0.0 or math.isnan(probability):
                raise ValueError(f"probability must be non-negative and finite (got {probability})")

        if self.probabilities:
            total = sum(self.probabilities)
            if total <= 0.0:
                raise ValueError(
                    "Cannot construct BeliefState with non-empty support and zero total probability"
                )
            if abs(total - 1.0) > 1e-12:
                object.__setattr__(
                    self,
                    "probabilities",
                    tuple(float(p / total) for p in self.probabilities),
                )

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def uniform(cls, hypotheses: Sequence[H]) -> "BeliefState[H]":
        """Build a uniform belief state over the given hypotheses."""
        hypotheses = tuple(hypotheses)
        if not hypotheses:
            return cls(hypotheses=(), probabilities=())
        probability = 1.0 / len(hypotheses)
        return cls(
            hypotheses=hypotheses,
            probabilities=tuple(probability for _ in hypotheses),
        )

    @classmethod
    def from_unnormalized(
        cls,
        hypotheses: Sequence[H],
        weights: Sequence[float],
        *,
        fallback_to_uniform: bool = False,
    ) -> "BeliefState[H]":
        """Build a belief state from unnormalised non-negative weights.

        If the weights sum to zero and ``fallback_to_uniform`` is ``True``, the
        result is a uniform distribution over the hypotheses.  Otherwise a
        :class:`ValueError` is raised.
        """
        hypotheses = tuple(hypotheses)
        weights_list = [float(w) for w in weights]
        if len(hypotheses) != len(weights_list):
            raise ValueError(
                f"hypotheses ({len(hypotheses)}) and weights ({len(weights_list)}) must match"
            )
        total = sum(weights_list)
        if total <= 0.0:
            if fallback_to_uniform:
                return cls.uniform(hypotheses)
            raise ValueError("Cannot normalise weights whose sum is zero")
        return cls(
            hypotheses=hypotheses,
            probabilities=tuple(w / total for w in weights_list),
        )

    @classmethod
    def from_log_scores(
        cls,
        hypotheses: Sequence[H],
        log_scores: Sequence[float],
    ) -> "BeliefState[H]":
        """Build a belief state from (possibly unnormalised) log scores."""
        hypotheses = tuple(hypotheses)
        log_array = np.asarray(list(log_scores), dtype=float)
        if log_array.size != len(hypotheses):
            raise ValueError(
                f"hypotheses ({len(hypotheses)}) and log_scores ({log_array.size}) must match"
            )
        if log_array.size == 0:
            return cls(hypotheses=(), probabilities=())
        log_array = log_array - float(np.max(log_array))
        weights = np.exp(log_array)
        total = float(np.sum(weights))
        if total <= 0.0:
            return cls.uniform(hypotheses)
        return cls(
            hypotheses=hypotheses,
            probabilities=tuple(float(w / total) for w in weights),
        )

    # ------------------------------------------------------------------
    # Inspection
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.hypotheses)

    def __bool__(self) -> bool:
        return bool(self.hypotheses)

    def __iter__(self) -> Iterable[tuple[H, float]]:
        return iter(zip(self.hypotheses, self.probabilities))

    @property
    def support_size(self) -> int:
        return len(self.hypotheses)

    @property
    def is_uniform(self) -> bool:
        """True iff every hypothesis carries the same probability (within 1e-9)."""
        if not self.probabilities:
            return False
        expected = 1.0 / len(self.probabilities)
        return all(abs(p - expected) <= 1e-9 for p in self.probabilities)

    def top(self) -> tuple[H, float] | None:
        """Return the (hypothesis, probability) with the largest probability, or None."""
        if not self.hypotheses:
            return None
        idx = int(np.argmax(self.probabilities))
        return self.hypotheses[idx], float(self.probabilities[idx])

    def top_k(self, k: int) -> list[tuple[H, float]]:
        """Return the k highest-probability (hypothesis, probability) pairs."""
        if k <= 0 or not self.hypotheses:
            return []
        ordered = sorted(
            zip(self.hypotheses, self.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return [(h, float(p)) for h, p in ordered[:k]]

    def effective_sample_size(self) -> float:
        """ESS = 1 / sum(p_i^2). Returns 0.0 for empty supports."""
        if not self.probabilities:
            return 0.0
        squared = sum(p * p for p in self.probabilities)
        return 1.0 / squared if squared > 0.0 else 0.0

    def entropy(self) -> float:
        """Shannon entropy in nats. Returns 0.0 for empty supports."""
        total = 0.0
        for p in self.probabilities:
            if p > 0.0:
                total -= p * math.log(p)
        return total

    def probability_of(self, predicate: Callable[[H], bool]) -> float:
        """Sum the probability mass over hypotheses satisfying the predicate."""
        return sum(
            probability
            for hypothesis, probability in zip(self.hypotheses, self.probabilities)
            if predicate(hypothesis)
        )

    # ------------------------------------------------------------------
    # Transformations (return new instances)
    # ------------------------------------------------------------------

    def sorted_descending(self) -> "BeliefState[H]":
        if not self.hypotheses:
            return self
        ordered = sorted(
            zip(self.hypotheses, self.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )
        return BeliefState(
            hypotheses=tuple(h for h, _ in ordered),
            probabilities=tuple(float(p) for _, p in ordered),
        )

    def pruned(self, max_size: int) -> "BeliefState[H]":
        """Keep the top ``max_size`` hypotheses by probability and renormalise."""
        if max_size <= 0:
            return BeliefState(hypotheses=(), probabilities=())
        if len(self.hypotheses) <= max_size:
            return self
        ordered = sorted(
            zip(self.hypotheses, self.probabilities),
            key=lambda entry: entry[1],
            reverse=True,
        )[:max_size]
        kept_hypotheses = tuple(h for h, _ in ordered)
        kept_weights = [float(p) for _, p in ordered]
        return BeliefState.from_unnormalized(
            kept_hypotheses,
            kept_weights,
            fallback_to_uniform=True,
        )

    def renormalized(self) -> "BeliefState[H]":
        """Force probabilities to sum to 1 (does nothing if already normalised)."""
        total = sum(self.probabilities)
        if total <= 0.0:
            return BeliefState.uniform(self.hypotheses)
        if abs(total - 1.0) <= 1e-12:
            return self
        return BeliefState(
            hypotheses=self.hypotheses,
            probabilities=tuple(p / total for p in self.probabilities),
        )

    def to_numpy(self) -> np.ndarray:
        """Return probabilities as a numpy array (useful for vectorised math)."""
        return np.asarray(self.probabilities, dtype=float)
