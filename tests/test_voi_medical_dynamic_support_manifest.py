import pytest

from scripts.voi_medical_dynamic_support_manifest import (
    SEED,
    SPLIT_SIZES,
    ordered_hash,
    split_indices,
)


def test_split_is_deterministic_disjoint_and_exhaustive() -> None:
    first = split_indices(12, seed=7, split_sizes={"a": 3, "b": 4, "c": 5})
    second = split_indices(12, seed=7, split_sizes={"a": 3, "b": 4, "c": 5})
    assert first == second
    flattened = [index for values in first.values() for index in values]
    assert len(flattened) == len(set(flattened)) == 12
    assert set(flattened) == set(range(12))
    assert ordered_hash(first["a"]) == ordered_hash(second["a"])


def test_frozen_split_sizes_exhaust_source() -> None:
    assert SEED == 24423
    assert SPLIT_SIZES == {
        "mechanics": 5,
        "opportunity": 40,
        "development": 20,
        "holdout": 434,
    }
    with pytest.raises(ValueError, match="exhaust"):
        split_indices(10, split_sizes={"a": 9})
