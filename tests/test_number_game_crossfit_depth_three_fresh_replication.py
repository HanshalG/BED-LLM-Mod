import json
from pathlib import Path

import pytest

from scripts import (
    number_game_crossfit_depth_three_fresh_replication as replication,
)


ROOT = Path(__file__).resolve().parents[1]
PRECISION_RESULT = (
    ROOT
    / "results/nonmyopic/number_game_crossfit_endpoint_precision"
    / "number-game-crossfit-endpoint-precision-20260728/RESULT.json"
)


def test_fresh_replication_seed_spaces_are_unique() -> None:
    validation = [
        seed
        for tree_index in range(len(replication.TREE_SEEDS))
        for seed in replication.validation_seeds_for_tree(tree_index)
    ]
    endpoints = [
        seed
        for tree_index in range(len(replication.TREE_SEEDS))
        for seed in replication.extra_endpoint_seeds_for_tree(tree_index)
    ]
    all_seeds = [
        *replication.TREE_SEEDS,
        *replication.TARGET_SEEDS,
        *validation,
        *endpoints,
    ]
    assert len(all_seeds) == len(set(all_seeds))
    assert len(validation) == 32 * 8
    assert len(endpoints) == 32 * 15


def test_fresh_replication_request_count_is_frozen() -> None:
    assert replication.EXPECTED_REQUESTS_PER_TREE == 73
    assert replication.EXPECTED_REQUESTS == 2336


def test_fresh_replication_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        replication.require_starting_balance(6.89)
    replication.require_starting_balance(6.90)


def test_replication_gate_contract_accepts_prior_positive_shape() -> None:
    source = json.loads(PRECISION_RESULT.read_text())
    usage = {
        "adapter_requests": replication.EXPECTED_REQUESTS,
        "http_attempts": replication.EXPECTED_REQUESTS,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 6.9,
    }

    gates = replication.replication_gates(
        scored_trees=source["trees"],
        usage=usage,
        aggregate=source["aggregate"],
    )

    assert all(gates.values())
