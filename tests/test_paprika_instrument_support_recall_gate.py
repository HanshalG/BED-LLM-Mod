import json

import pytest

from scripts.paprika_instrument_support_recall_gate import (
    ACTIVITY_INDICES,
    EXPECTED_REQUESTS,
    FORMAL_INDICES,
    SMOKE_INDICES,
    checked_yes_labels,
    exact_contains,
    recover_yes_labels_from_reversed_no,
)


def test_recover_yes_labels_restores_both_reversed_axes():
    yes = [
        [True, False, True],
        [False, False, True],
        [True, True, False],
    ]
    reversed_no = [
        [not value for value in reversed(row)]
        for row in reversed(yes)
    ]
    assert recover_yes_labels_from_reversed_no(reversed_no) == yes


def test_exact_contains_normalizes_punctuation_but_not_aliases():
    assert exact_contains(["Oboe d Amore"], "Oboe d'amore")
    assert not exact_contains(["Violin"], "Fiddle")


def test_complementary_label_disagreement_fails_closed():
    direct = json.dumps(
        {"labels": [[True], [False], [True]]}
    )
    inconsistent_reversed_no = json.dumps(
        {"labels": [[False], [False], [False]]}
    )
    with pytest.raises(
        ValueError, match="complementary semantic label passes disagree"
    ):
        checked_yes_labels(
            direct,
            inconsistent_reversed_no,
            item_count=1,
        )


def test_request_counts_include_complementary_label_passes():
    ranked_per_case = 2 + 3 + 1 + 2 + 18 + 2 + 1
    activity_per_case = ranked_per_case - 1
    assert EXPECTED_REQUESTS["serving_smoke"] == (
        len(SMOKE_INDICES) * ranked_per_case
    )
    assert EXPECTED_REQUESTS["activity"] == (
        len(ACTIVITY_INDICES) * activity_per_case
    )
    assert EXPECTED_REQUESTS["confirmation"] == (
        len(FORMAL_INDICES) * ranked_per_case
    )
