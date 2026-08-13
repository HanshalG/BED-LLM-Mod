from scripts import number_game_atomic_particle_v2_codec as codec


def test_v2_cohort_is_fresh_and_exact():
    assert codec.INTERFACE_VERSION == "number-game-atomic-particle-depth3-2"
    assert codec.MODEL_SEEDS == tuple(range(202608160000, 202608166400))
    assert codec.TREE_SEEDS == tuple(range(202608167000, 202608167004))
    assert len(codec.MODEL_SEEDS) == 6400
