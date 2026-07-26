from __future__ import annotations

from fractions import Fraction

import pytest

from scripts.analyze_tau_knowledge_future_first_robustness import (
    exact_one_sided_sign_flip_p,
    future_first_root,
    mean_pairwise_agreement,
)


def test_future_first_root_uses_frozen_lexicographic_ties() -> None:
    assert future_first_root([90, 70, 80], [95, 80, 90]) == 2
    assert future_first_root([80, 90, 70], [90, 100, 80]) == 1
    assert future_first_root([80, 80, 70], [90, 90, 80]) == 0


def test_mean_pairwise_agreement() -> None:
    assert mean_pairwise_agreement([[0, 1], [0, 0], [1, 0]]) == pytest.approx(
        1 / 3
    )


def test_exact_one_sided_sign_flip_p() -> None:
    assert exact_one_sided_sign_flip_p([]) == 1.0
    assert exact_one_sided_sign_flip_p(
        [Fraction(1), Fraction(1)]
    ) == pytest.approx(0.25)
    assert exact_one_sided_sign_flip_p(
        [Fraction(1, 6), Fraction(-1, 6)]
    ) == pytest.approx(0.75)
