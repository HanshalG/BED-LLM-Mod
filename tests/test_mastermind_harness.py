from __future__ import annotations

import math

import pytest

from environments.mastermind_harness import (
    MastermindState,
    code_space,
    feedback,
    one_step_eig,
    select_query,
    two_step_eig,
)


def test_feedback_handles_repeated_symbols() -> None:
    assert feedback((1, 1, 2, 2), (1, 2, 1, 3)) == (1, 2)


def test_exact_posterior_filter_keeps_truth() -> None:
    support = code_space(3, 3)
    truth = (2, 1, 0)
    guess = (0, 0, 0)
    state = MastermindState(support).observe(guess, feedback(truth, guess))
    assert truth in state.support
    assert all(feedback(candidate, guess) == feedback(truth, guess) for candidate in state.support)
    assert len(state.support) < len(support)


def test_one_step_eig_matches_partition_entropy() -> None:
    support = code_space(2, 2)
    # Guess 00 partitions the four secrets into sizes 1, 2, 1.
    expected_final_entropy = 0.5 * math.log(2.0)
    assert one_step_eig(support, (0, 0)) == pytest.approx(math.log(4.0) - expected_final_entropy)


def test_depth_two_value_dominates_one_step_value() -> None:
    support = code_space(3, 3)
    candidates = support
    for first in candidates:
        assert two_step_eig(support, first, candidates) + 1e-12 >= one_step_eig(support, first)


def test_depth_one_and_two_pipeline_returns_valid_queries() -> None:
    support = code_space(3, 3)
    one = select_query(support, support, depth=1)
    two = select_query(support, support, depth=2)
    assert one.guess in support and two.guess in support
    assert one.depth == 1 and two.depth == 2
    assert two.score >= one.score


def test_harness_rejects_depth_greater_than_two() -> None:
    support = code_space(2, 2)
    with pytest.raises(ValueError, match="only depth 1 or 2"):
        select_query(support, support, depth=3)
