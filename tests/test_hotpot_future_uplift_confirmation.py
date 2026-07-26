from __future__ import annotations

from scripts.hotpot_future_uplift_confirmation import (
    _pairwise_accuracy,
    _sign_flip_p,
    future_first_root,
    policy_selections,
    title_bm25_order,
)


def _row(aligned_max: int, aligned_index: int = 0) -> dict[str, list[int]]:
    aligned = [0] * 9
    initial = [0] * 9
    shuffled = [0] * 9
    aligned[aligned_index] = aligned_max
    initial[(aligned_index + 1) % 9] = max(0, aligned_max - 10)
    shuffled[(aligned_index + 2) % 9] = max(0, aligned_max - 5)
    return {
        "aligned_scores": aligned,
        "initial_scores": initial,
        "shuffled_scores": shuffled,
    }


def test_future_first_reproduces_the_open_hotpot_smoke_choice() -> None:
    rows = [_row(value) for value in (56, 100, 100, 99)]
    assert future_first_root([98, 42, 0, 1], rows) == 1


def test_future_first_breaks_future_ties_with_immediate_then_order() -> None:
    rows = [_row(100) for _ in range(4)]
    assert future_first_root([20, 30, 30, 10], rows) == 1


def test_policy_controls_use_shared_rows_but_distinct_objectives() -> None:
    rows = [_row(value, index) for index, value in enumerate((56, 100, 90, 80))]
    policies = policy_selections(
        immediate_scores=[98, 42, 0, 1],
        continuation_rows=rows,
        random_seed=7,
    )
    assert policies["future_uplift"]["root_index"] == 1
    assert policies["myopic"]["root_index"] == 0
    assert policies["old_total"]["root_index"] == 0
    assert policies["future_uplift"]["followup_candidate_index"] == 1


def test_pairwise_accuracy_ignores_value_and_score_ties() -> None:
    assert _pairwise_accuracy([3, 2, 1], [3, 1, 2]) == (2, 3)
    assert _pairwise_accuracy([3, 3, 1], [3, 2, 1]) == (2, 2)
    assert _pairwise_accuracy([3, 2, 1], [2, 2, 1]) == (2, 2)


def test_exact_one_sided_sign_flip_probability() -> None:
    assert _sign_flip_p([1, 1, 1, 1]) == 0.0625
    assert _sign_flip_p([1, 1, 1, 1, 1]) == 0.03125
    assert _sign_flip_p([1, -1]) == 0.75
    assert _sign_flip_p([0, 0]) == 1.0


def test_title_bm25_order_is_deterministic() -> None:
    titles = ["red planet", "blue ocean", "planetary science", "plain"]
    first = title_bm25_order("red planet science", titles)
    second = title_bm25_order("red planet science", titles)
    assert first == second
    assert first[0] == 0
