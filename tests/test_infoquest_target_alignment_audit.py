from __future__ import annotations

import pytest

from scripts import infoquest_target_alignment_audit as audit


def test_average_ranks_and_spearman_handle_ties():
    assert audit.average_ranks([1, 1, 3, 2]) == [1.5, 1.5, 4.0, 3.0]
    assert audit.spearman([1, 2, 3, 4], [1, 2, 3, 4]) == pytest.approx(1.0)
    assert audit.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert audit.spearman([1, 1, 1, 1], [1, 2, 3, 4]) is None


def test_additive_checklist_gain_and_union():
    current = (1, 0, 0, 1, 0)
    candidate = (0, 1, 0, 1, 1)
    assert audit.additive_gain(current, candidate) == 2
    assert audit.union_bits(current, candidate) == (1, 1, 0, 1, 1)

    with pytest.raises(ValueError, match="five bits"):
        audit.additive_gain((1,), candidate)
