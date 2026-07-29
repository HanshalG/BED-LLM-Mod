from __future__ import annotations

from scripts import number_game_crossplanner_first_link_mechanism128 as mechanism
from scripts.number_game_crossplanner_canonical_pooled128 import load_blocks


def test_crossplanner_advantage_uses_same_tree_roots_and_endpoint() -> None:
    blocks = load_blocks()
    rows = mechanism.rows_for_baseline(blocks, root_key="myopic_root")
    row = rows[0]
    tree = blocks[0]["trees"][0]
    selection = tree["selection"]
    candidate = str(selection["crossfit_depth_three_root"])
    baseline = str(selection["myopic_root"])

    assert len(rows) == 128
    assert row["predicted_advantage"] == (
        selection["crossfit_depth_three_brier"][baseline]
        - selection["crossfit_depth_three_brier"][candidate]
    )
    assert row["realized_advantage"] == (
        tree["per_root_endpoint_brier"][baseline]
        - tree["per_root_endpoint_brier"][candidate]
    )


def test_source_blocks_cover_two_families_and_are_disjoint() -> None:
    blocks = load_blocks()
    seeds = [
        {int(tree["tree_seed"]) for tree in block["trees"]}
        for block in blocks
    ]

    assert len(blocks) == 4
    assert all(len(block["trees"]) == 32 for block in blocks)
    assert len({block["family"] for block in blocks}) == 2
    assert all(
        not seeds[left] & seeds[right]
        for left in range(4)
        for right in range(left + 1, 4)
    )


def test_protocol_constants_preserve_retrospective_boundary() -> None:
    assert mechanism.BOOTSTRAP_SAMPLES == 20_000
    assert mechanism.BOOTSTRAP_SEED == 58_000
    assert mechanism.BASELINES == {
        "myopic_eig": "myopic_root",
        "fixed_support_depth_three": "fixed_support_depth_three_root",
        "crossfit_depth_two": "crossfit_depth_two_root",
    }
