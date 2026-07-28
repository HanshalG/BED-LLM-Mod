from scripts import number_game_pooled_support_confirmation as confirmation


def test_confirmation_seeds_are_fresh():
    prior_tree_seeds = {
        *range(26400, 26432),
        *range(27000, 27032),
    }
    prior_target_seeds = {
        *range(26500, 26532),
        *range(27100, 27132),
    }

    assert confirmation.TREE_SEEDS == tuple(range(27200, 27232))
    assert confirmation.TARGET_SEEDS == tuple(range(27300, 27332))
    assert not set(confirmation.TREE_SEEDS) & prior_tree_seeds
    assert not set(confirmation.TARGET_SEEDS) & prior_target_seeds
