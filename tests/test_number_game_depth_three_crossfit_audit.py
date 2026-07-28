import pytest

from scripts.number_game_depth_three_crossfit_audit import (
    circular_validation_indices,
    select_minimum_risk_root,
)


def test_circular_validation_indices_excludes_own_tree() -> None:
    assert circular_validation_indices(
        3,
        tree_count=5,
        validation_count=4,
    ) == [4, 0, 1, 2]


def test_circular_validation_indices_rejects_invalid_counts() -> None:
    with pytest.raises(ValueError):
        circular_validation_indices(
            0,
            tree_count=1,
            validation_count=1,
        )
    with pytest.raises(ValueError):
        circular_validation_indices(
            0,
            tree_count=4,
            validation_count=4,
        )


def test_minimum_risk_selection_uses_source_root_order_for_ties() -> None:
    roots = [40, 10, 70]
    assert select_minimum_risk_root(
        roots,
        {40: 0.2 + 5e-13, 10: 0.2, 70: 0.3},
    ) == 40
