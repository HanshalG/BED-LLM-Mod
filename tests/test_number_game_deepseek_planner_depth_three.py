import json

import pytest

from scripts import number_game_deepseek_planner_depth_three as deepseek
from scripts import number_game_qwen_planner_depth_three as engine
from scripts.number_game_deepseek_planner_serving_smoke import (
    INTERFACE_VERSION as SMOKE_INTERFACE_VERSION,
    MODEL_ID,
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


def _comparison() -> dict:
    return {
        "relative_brier_reduction": 0.05,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, -0.005],
        "brier_tree_wins": 16,
        "mean_candidate_minus_baseline_brier": -0.01,
    }


def test_formal_seeds_are_disjoint_and_request_count_is_exact() -> None:
    validation = [
        seed
        for index in range(32)
        for seed in engine.validation_seeds_for_tree(
            index,
            start=deepseek.FORMAL_VALIDATION_SEED_START,
        )
    ]
    endpoints = [
        seed
        for index in range(32)
        for seed in engine.extra_endpoint_seeds_for_tree(
            index,
            start=deepseek.FORMAL_EXTRA_ENDPOINT_SEED_START,
        )
    ]
    seeds = [
        *deepseek.FORMAL_TREE_SEEDS,
        *deepseek.FORMAL_TARGET_SEEDS,
        *validation,
        *endpoints,
    ]

    assert len(seeds) == len(set(seeds))
    assert deepseek.FORMAL_EXPECTED_REQUESTS == 32 * engine.REQUESTS_PER_TREE


def test_validate_smoke_requires_exact_frozen_contract(tmp_path) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol": {
                    "interface_version": SMOKE_INTERFACE_VERSION,
                    "model": MODEL_ID,
                    "expected_requests": 10,
                    "efficacy_used_for_authorization": False,
                },
                "gates": {"all_pass": True},
            }
        )
    )

    assert deepseek.validate_smoke_result(path)["status"] == "passed"

    value = json.loads(path.read_text())
    value["protocol"]["expected_requests"] = 9
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        deepseek.validate_smoke_result(path)


def test_formal_gates_accept_primary_proper_score_positive() -> None:
    comparisons = {
        name: _comparison()
        for name in (
            "crossfit_depth_two",
            "myopic_eig",
            "fixed_support_depth_three",
            "positive_test_strategy",
        )
    }
    aggregate = {
        "comparisons": comparisons,
        "root_differences": {"crossfit_depth_two": 20},
        "novel_target_mean_differences": {
            "candidate_minus_baseline_brier": -0.005,
            "candidate_minus_baseline_hamming": 0.01,
            "coverage_difference": -0.01,
        },
        "ranking": {
            "crossfit_depth_three_spearman_brier": {"mean": 0.9},
            "crossfit_depth_two_spearman_brier": {"mean": 0.5},
        },
    }
    usage = {
        "adapter_requests": deepseek.FORMAL_EXPECTED_REQUESTS,
        "http_attempts": deepseek.FORMAL_EXPECTED_REQUESTS,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 4.0,
    }

    gates = deepseek.formal_gates(
        scored_trees=[_clean_tree() for _ in range(32)],
        usage=usage,
        aggregate=aggregate,
    )

    assert all(gates.values())


def test_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        deepseek.require_starting_balance(4.99)
    deepseek.require_starting_balance(5.00)
