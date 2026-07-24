import pytest

from scripts.movielens_confidence_gated_explicit_v10 import (
    CONFIDENCE_MARGIN,
    ENROLLMENT_COUNT,
    FORMAL_EXPECTED_REQUESTS,
    FORMAL_SCREEN_USER_IDS,
    INITIAL_HISTORY_MOVIE_IDS,
    SMOKE_EXPECTED_REQUESTS,
    confidence_summary,
    selected_user_ids,
)
from scripts.movielens_profile_dynamics_gate import load_movielens


def _record(user_id, scores, nlls):
    return {
        "user_id": user_id,
        "semantic_lookahead_scores": scores,
        "branches": [{"heldout_nll": value} for value in nlls],
    }


def test_protocol_sizes_are_frozen():
    assert len(INITIAL_HISTORY_MOVIE_IDS) == 4
    assert len(FORMAL_SCREEN_USER_IDS) == 44
    assert ENROLLMENT_COUNT == 16
    assert CONFIDENCE_MARGIN == 0.02
    assert FORMAL_EXPECTED_REQUESTS == 856
    assert SMOKE_EXPECTED_REQUESTS == 12


def test_frozen_fresh_user_selection_reproduces():
    ratings, _items = load_movielens("external/ml-100k")
    selected = selected_user_ids(ratings)
    assert len(selected) == 46
    assert selected[:2] == (750, 782)


def test_confidence_gate_defers_below_margin_and_activates_above():
    records = []
    for index in range(ENROLLMENT_COUNT):
        if index == 0:
            records.append(
                _record(index, [0.0, 0.03, -0.1, -0.2], [1.0, 0.8, 1.2, 1.3])
            )
        elif index == 1:
            records.append(
                _record(index, [0.0, 0.019, -0.1, -0.2], [1.0, 0.7, 1.2, 1.3])
            )
        else:
            records.append(
                _record(index, [0.0, -0.1, -0.2, -0.3], [1.0, 1.1, 1.2, 1.3])
            )
    result = confidence_summary(
        {
            "records": records,
            "usage": {"physical_requests": FORMAL_EXPECTED_REQUESTS, "reasoning_tokens": 0},
        }
    )
    assert result["active_user_count"] == 1
    assert result["per_user"][0]["selected_index"] == 1
    assert result["per_user"][0]["nll_improvement_vs_immediate"] == pytest.approx(0.2)
    assert result["per_user"][1]["selected_index"] == 0
    assert result["per_user"][1]["nll_improvement_vs_immediate"] == 0.0
