import json
from pathlib import Path

import pytest

from scripts.number_game_qwen_planner_depth_three import (
    ENDPOINT_DRAWS_PER_TREE,
    FORMAL_BUDGET_USD,
    FORMAL_EXPECTED_REQUESTS,
    FORMAL_EXTRA_ENDPOINT_SEED_START,
    FORMAL_TREE_SEEDS,
    FORMAL_VALIDATION_SEED_START,
    REQUESTS_PER_TREE,
    extra_endpoint_seeds_for_tree,
    formal_gates,
    mechanics_gates,
    require_starting_balance,
    validation_seeds_for_tree,
)


ROOT = Path(__file__).resolve().parents[1]
POSITIVE_RESULT = (
    ROOT
    / "results/nonmyopic/number_game_crossfit_endpoint_precision"
    / "number-game-crossfit-endpoint-precision-20260728"
    / "RESULT.json"
)


def _clean_tree() -> dict:
    return {
        "mechanics": {
            "initial_valid": 20,
            "validation_support_count": 8,
            "minimum_validation_support_valid": 18,
            "endpoint_draw_count": 16,
            "minimum_endpoint_support_valid": 18,
            "total_novel_endpoint_hypotheses": 140,
            "minimum_first_branch_valid": 9,
            "minimum_retained_second_branch_valid": 5,
        }
    }


def _clean_usage(*, requests: int, cost: float) -> dict:
    return {
        "adapter_requests": requests,
        "http_attempts": requests,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": cost,
    }


def test_seed_blocks_are_unique_and_disjoint() -> None:
    validation = [
        seed
        for index in range(len(FORMAL_TREE_SEEDS))
        for seed in validation_seeds_for_tree(
            index,
            start=FORMAL_VALIDATION_SEED_START,
        )
    ]
    endpoints = [
        seed
        for index in range(len(FORMAL_TREE_SEEDS))
        for seed in extra_endpoint_seeds_for_tree(
            index,
            start=FORMAL_EXTRA_ENDPOINT_SEED_START,
        )
    ]
    all_seeds = [
        *FORMAL_TREE_SEEDS,
        *validation,
        *endpoints,
    ]
    assert len(set(all_seeds)) == len(all_seeds)
    assert len(validation) == 32 * 8
    assert len(endpoints) == 32 * 15
    assert FORMAL_EXPECTED_REQUESTS == 32 * REQUESTS_PER_TREE
    assert ENDPOINT_DRAWS_PER_TREE == 16


def test_mechanics_gates_accept_clean_tree() -> None:
    gates = mechanics_gates(
        scored_trees=[_clean_tree()],
        usage=_clean_usage(requests=REQUESTS_PER_TREE, cost=0.1),
        expected_requests=REQUESTS_PER_TREE,
        run_budget_usd=0.2,
    )
    assert all(gates.values())


def test_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        require_starting_balance(3.79)
    require_starting_balance(3.80)


def test_formal_gates_accept_clean_positive_result() -> None:
    comparison = {
        "relative_brier_reduction": 0.08,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.03, -0.01],
        "brier_tree_wins": 20,
        "mean_candidate_minus_baseline_hamming": -0.01,
        "mean_coverage_difference": 0.01,
    }
    aggregate = {
        "comparisons": {
            key: dict(comparison)
            for key in (
                "crossfit_depth_two",
                "myopic_eig",
                "fixed_support_depth_three",
                "positive_test_strategy",
            )
        },
        "root_differences": {"crossfit_depth_two": 20},
        "novel_target_mean_differences": {
            "candidate_minus_baseline_brier": -0.01,
            "candidate_minus_baseline_hamming": -0.01,
            "coverage_difference": 0.01,
        },
        "ranking": {
            "crossfit_depth_three_spearman_brier": {"mean": 0.9},
            "crossfit_depth_two_spearman_brier": {"mean": 0.5},
        },
    }
    gates = formal_gates(
        scored_trees=[_clean_tree() for _ in range(32)],
        usage=_clean_usage(
            requests=FORMAL_EXPECTED_REQUESTS,
            cost=FORMAL_BUDGET_USD - 0.1,
        ),
        aggregate=aggregate,
    )
    assert all(gates.values())


def test_formal_contract_accepts_existing_positive_shape() -> None:
    source = json.loads(POSITIVE_RESULT.read_text())
    usage = _clean_usage(
        requests=FORMAL_EXPECTED_REQUESTS,
        cost=3.5,
    )
    gates = formal_gates(
        scored_trees=source["trees"],
        usage=usage,
        aggregate=source["aggregate"],
    )
    assert all(gates.values())
