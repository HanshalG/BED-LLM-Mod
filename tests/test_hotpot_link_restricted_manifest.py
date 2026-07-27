from scripts.hotpot_link_restricted_manifest import (
    SPLIT_SIZES,
    link_restricted_splits,
    ordered_list_hash,
)


def test_link_restricted_splits_are_ordered_disjoint_and_exhaustive() -> None:
    ids = [f"id-{index}" for index in range(sum(SPLIT_SIZES.values()))]
    splits = link_restricted_splits(ids)
    assert {name: len(values) for name, values in splits.items()} == SPLIT_SIZES
    flattened = [value for values in splits.values() for value in values]
    assert flattened == ids
    assert len(set(flattened)) == len(flattened)


def test_ordered_list_hash_is_order_sensitive() -> None:
    assert ordered_list_hash(["a", "b"]) != ordered_list_hash(["b", "a"])
