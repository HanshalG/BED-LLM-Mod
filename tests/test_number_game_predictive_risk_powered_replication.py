from scripts.number_game_predictive_risk_powered_replication import (
    EXPECTED_REQUESTS,
    EXPECTED_REQUESTS_PER_TREE,
    TARGET_SEEDS,
    TREE_SEEDS,
)


def test_powered_replication_seed_and_request_budget_is_frozen():
    assert TREE_SEEDS == tuple(range(26400, 26432))
    assert TARGET_SEEDS == tuple(range(26500, 26532))
    assert len(TREE_SEEDS) == len(TARGET_SEEDS) == 32
    assert EXPECTED_REQUESTS_PER_TREE == 18
    assert EXPECTED_REQUESTS == 576
    assert not set(TREE_SEEDS) & set(range(26080, 26088))
    assert not set(TARGET_SEEDS) & set(range(26180, 26188))
