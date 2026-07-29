from __future__ import annotations

import json

from scripts import number_game_external_canonical_replay as replay
from scripts.number_game_crossfit_endpoint_precision import score_fixed_tree


def test_canonical_target_bank_is_exact_and_unique() -> None:
    targets = replay.canonical_targets()
    by_name = {target.name: target for target in targets}

    assert len(targets) == 33
    assert len({target.extension for target in targets}) == 33
    assert by_name["numbers_less_than_100"].extension[0]
    assert not by_name["numbers_less_than_100"].extension[100]
    assert by_name["perfect_squares"].extension[0]
    assert by_name["powers_of_2"].extension[1]
    assert by_name["powers_of_2"].extension[64]
    assert not by_name["powers_of_2"].extension[0]
    assert by_name["both_digits_equal"].extension[11]
    assert by_name["both_digits_equal"].extension[99]


def test_one_tree_scores_all_external_targets_without_model_calls() -> None:
    source_result = json.loads(replay.SOURCE_RESULT.read_text())
    source_trees = json.loads(replay.SOURCE_TREES.read_text())
    public_targets = [
        target.public_dict() for target in replay.canonical_targets()
    ]

    scored = score_fixed_tree(
        source_trees["trees"][0],
        source_result["trees"][0],
        [public_targets],
    )

    assert scored["tree_index"] == 0
    assert scored["mechanics"]["endpoint_draw_count"] == 1
    assert scored["mechanics"]["minimum_endpoint_support_valid"] == 33
    assert set(scored["endpoint"]) >= {
        "crossfit_depth_three",
        "crossfit_depth_two",
        "myopic_eig",
        "fixed_support_depth_three",
        "positive_test_strategy",
    }
    assert all(
        0.0 <= endpoint["mean_posterior_predictive_brier"] <= 1.0
        for endpoint in scored["endpoint"].values()
    )


def test_source_hashes_are_frozen() -> None:
    assert replay.sha256_file(replay.SOURCE_RESULT) == (
        replay.SOURCE_RESULT_SHA256
    )
    assert replay.sha256_file(replay.SOURCE_TREES) == (
        replay.SOURCE_TREES_SHA256
    )
