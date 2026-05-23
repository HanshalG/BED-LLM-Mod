"""Tests for the generic ``core.BeliefState`` container."""

from __future__ import annotations

import math

import pytest

from core import BeliefState


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


def test_construct_empty_belief_state():
    state: BeliefState[str] = BeliefState()
    assert len(state) == 0
    assert not state
    assert state.support_size == 0
    assert state.effective_sample_size() == 0.0
    assert state.entropy() == 0.0
    assert state.top() is None
    assert state.top_k(3) == []


def test_uniform_construction_distributes_mass_equally():
    state = BeliefState.uniform(["a", "b", "c", "d"])
    assert state.support_size == 4
    assert state.probabilities == (0.25, 0.25, 0.25, 0.25)
    assert state.is_uniform


def test_uniform_construction_with_empty_support():
    state = BeliefState.uniform([])
    assert state.support_size == 0
    assert not state.is_uniform  # an empty support is not "uniform"


def test_from_unnormalized_renormalises_to_sum_one():
    state = BeliefState.from_unnormalized(["a", "b", "c"], [4.0, 1.0, 5.0])
    assert pytest.approx(sum(state.probabilities)) == 1.0
    assert state.probabilities == (0.4, 0.1, 0.5)


def test_from_unnormalized_raises_on_zero_total_without_fallback():
    with pytest.raises(ValueError):
        BeliefState.from_unnormalized(["a", "b"], [0.0, 0.0])


def test_from_unnormalized_falls_back_to_uniform_when_requested():
    state = BeliefState.from_unnormalized(
        ["a", "b"],
        [0.0, 0.0],
        fallback_to_uniform=True,
    )
    assert state.probabilities == (0.5, 0.5)


def test_from_log_scores_matches_softmax():
    # log_scores [0, 0, log(2)] should yield probabilities [1/4, 1/4, 1/2].
    state = BeliefState.from_log_scores(["a", "b", "c"], [0.0, 0.0, math.log(2.0)])
    assert pytest.approx(state.probabilities) == (0.25, 0.25, 0.5)


def test_from_log_scores_handles_large_negative_values_without_underflow():
    # All log-scores are very negative but equal → uniform after subtracting max.
    state = BeliefState.from_log_scores(["a", "b", "c"], [-1e6, -1e6, -1e6])
    assert pytest.approx(state.probabilities) == (1 / 3, 1 / 3, 1 / 3)


def test_construction_rejects_negative_probabilities():
    with pytest.raises(ValueError):
        BeliefState(hypotheses=("a",), probabilities=(-0.1,))


def test_construction_rejects_length_mismatch():
    with pytest.raises(ValueError):
        BeliefState(hypotheses=("a", "b"), probabilities=(1.0,))


def test_construction_renormalizes_unnormalized_weights():
    state = BeliefState(hypotheses=("a", "b"), probabilities=(0.9, 0.9))
    assert pytest.approx(sum(state.probabilities)) == 1.0
    assert state.probabilities == (0.5, 0.5)


def test_construction_raises_on_zero_sum_with_nonempty_support():
    with pytest.raises(ValueError, match="zero total probability"):
        BeliefState(hypotheses=("a", "b"), probabilities=(0.0, 0.0))


# ---------------------------------------------------------------------------
# Inspection
# ---------------------------------------------------------------------------


def test_top_returns_highest_probability_pair():
    state = BeliefState.from_unnormalized(["a", "b", "c"], [1.0, 4.0, 2.0])
    top = state.top()
    assert top is not None
    hypothesis, probability = top
    assert hypothesis == "b"
    assert probability == pytest.approx(4 / 7)


def test_top_k_returns_descending_pairs():
    state = BeliefState.from_unnormalized(["a", "b", "c", "d"], [1.0, 2.0, 3.0, 4.0])
    top2 = state.top_k(2)
    assert [h for h, _ in top2] == ["d", "c"]
    assert top2[0][1] > top2[1][1]


def test_effective_sample_size_matches_definition():
    state = BeliefState(hypotheses=("a", "b"), probabilities=(0.5, 0.5))
    # ESS = 1 / sum(p^2) = 1 / (0.25 + 0.25) = 2.0
    assert state.effective_sample_size() == pytest.approx(2.0)

    skewed = BeliefState(hypotheses=("a", "b"), probabilities=(0.9, 0.1))
    # ESS = 1 / (0.81 + 0.01) = 1 / 0.82 ≈ 1.2195
    assert skewed.effective_sample_size() == pytest.approx(1.0 / 0.82)


def test_entropy_of_uniform_equals_log_n():
    state = BeliefState.uniform(["a", "b", "c", "d"])
    assert state.entropy() == pytest.approx(math.log(4.0))


def test_probability_of_predicate_sums_matching_mass():
    state = BeliefState(
        hypotheses=("dog", "cat", "lion"),
        probabilities=(0.3, 0.5, 0.2),
    )
    feline_mass = state.probability_of(lambda h: h in {"cat", "lion"})
    assert feline_mass == pytest.approx(0.7)


# ---------------------------------------------------------------------------
# Transformations
# ---------------------------------------------------------------------------


def test_sorted_descending_returns_new_state_with_top_first():
    state = BeliefState(
        hypotheses=("a", "b", "c"),
        probabilities=(0.2, 0.5, 0.3),
    )
    sorted_state = state.sorted_descending()
    assert sorted_state.hypotheses == ("b", "c", "a")
    assert sorted_state.probabilities == (0.5, 0.3, 0.2)
    # Original unchanged (frozen dataclass + tuples).
    assert state.hypotheses == ("a", "b", "c")


def test_pruned_keeps_top_k_and_renormalises():
    state = BeliefState.from_unnormalized(
        ["a", "b", "c", "d"],
        [1.0, 2.0, 3.0, 4.0],
    )
    pruned = state.pruned(2)
    assert pruned.support_size == 2
    assert set(pruned.hypotheses) == {"c", "d"}
    assert pytest.approx(sum(pruned.probabilities)) == 1.0


def test_pruned_is_a_noop_when_smaller_than_limit():
    state = BeliefState.uniform(["a", "b"])
    assert state.pruned(5) is state


def test_renormalized_is_identity_for_already_normalized_state():
    state = BeliefState.uniform(["a", "b", "c"])
    assert state.renormalized() is state
