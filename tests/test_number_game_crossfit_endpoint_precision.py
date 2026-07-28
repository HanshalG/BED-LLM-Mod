import copy
import json

import pytest

from scripts import number_game_crossfit_endpoint_precision as precision


def test_endpoint_seeds_are_unique_and_disjoint_from_source() -> None:
    seeds = [
        seed
        for tree_index in range(precision.TREE_COUNT)
        for seed in precision.endpoint_seeds_for_tree(tree_index)
    ]
    assert len(seeds) == precision.EXPECTED_REQUESTS == 512
    assert len(set(seeds)) == len(seeds)
    assert not set(seeds) & set(range(28300, 28756))


def test_source_hashes_are_frozen() -> None:
    assert (
        precision.sha256_file(precision.SOURCE_RESULT)
        == precision.SOURCE_RESULT_SHA256
    )
    assert (
        precision.sha256_file(precision.SOURCE_TREES)
        == precision.SOURCE_TREES_SHA256
    )


def test_starting_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        precision.require_starting_balance(1.139)
    precision.require_starting_balance(1.14)


def test_fixed_tree_scoring_does_not_change_selected_roots() -> None:
    source_trees = json.loads(precision.SOURCE_TREES.read_text())["trees"]
    source_result = json.loads(precision.SOURCE_RESULT.read_text())["trees"]
    endpoint_supports = [
        copy.deepcopy(source_trees[0]["validation_supports"][index % 8])
        for index in range(16)
    ]

    scored = precision.score_fixed_tree(
        source_trees[0],
        source_result[0],
        endpoint_supports,
    )

    assert (
        scored["selection"]["crossfit_depth_three_root"]
        == source_result[0]["selection"]["crossfit_depth_three_root"]
    )
    assert (
        scored["selection"]["crossfit_depth_two_root"]
        == source_result[0]["selection"]["crossfit_depth_two_root"]
    )
    assert scored["mechanics"]["endpoint_draw_count"] == 16
    assert "crossfit_depth_two" in scored["comparisons"]
