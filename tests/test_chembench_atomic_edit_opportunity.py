from __future__ import annotations

import numpy as np

from environments.chembench_mopen.factored import FactoredModelBank
from scripts.chembench_atomic_edit_opportunity import (
    compare,
    is_standard_atomic_successor,
    reduced_atomic_bank,
)


def _bank() -> FactoredModelBank:
    names = (
        "c0_michaelis_menten",
        "c8_hill_cooperativity",
        "c10_mm_competitive_arrhenius",
        "c33_hill_competitive",
        "c19_ordered_bi_bi",
    )
    likelihoods = np.full((len(names), 2, 3), 1.0 / 3.0)
    features = np.arange(len(names) * 2, dtype=float).reshape(len(names), 2)
    return FactoredModelBank(
        likelihoods,
        features,
        names,
        ("a", "b"),
        ("x", "y"),
        (0, 1),
        evidence_slots=2,
        diversity_slots=1,
    )


def test_atomic_successor_filter_and_reduction() -> None:
    bank = _bank()
    assert not is_standard_atomic_successor(bank, 0)
    assert not is_standard_atomic_successor(bank, 2)
    assert is_standard_atomic_successor(bank, 3)
    assert not is_standard_atomic_successor(bank, 4)

    reduced, truths, names = reduced_atomic_bank(bank)
    assert reduced.model_names == (
        "c0_michaelis_menten",
        "c8_hill_cooperativity",
        "c33_hill_competitive",
    )
    assert truths == (2,)
    assert names == ("c33_hill_competitive",)


def test_compare_uses_paired_truth_cells() -> None:
    result = compare([4.0, 2.0, 1.0], [2.0, 2.0, 3.0])
    assert result["left_mean"] == 7.0 / 3.0
    assert result["right_mean"] == 7.0 / 3.0
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
