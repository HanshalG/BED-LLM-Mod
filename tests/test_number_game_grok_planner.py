import json

import pytest

from scripts import number_game_grok_planner_depth_three as grok
from scripts import number_game_grok_planner_serving_smoke as smoke
from scripts import number_game_qwen_planner_depth_three as engine


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


def test_grok_smoke_contract_is_distinct_and_exact_ten() -> None:
    assert smoke.MODEL_ID == "x-ai/grok-4.3"
    assert smoke.MODEL_SEED == 44_000
    assert len(smoke.serving_cases()) == 10


def test_formal_seed_blocks_are_disjoint() -> None:
    validation = [
        seed
        for index in range(32)
        for seed in engine.validation_seeds_for_tree(
            index,
            start=grok.FORMAL_VALIDATION_SEED_START,
        )
    ]
    endpoints = [
        seed
        for index in range(32)
        for seed in engine.extra_endpoint_seeds_for_tree(
            index,
            start=grok.FORMAL_EXTRA_ENDPOINT_SEED_START,
        )
    ]
    seeds = [
        *grok.FORMAL_TREE_SEEDS,
        *grok.FORMAL_TARGET_SEEDS,
        *validation,
        *endpoints,
    ]

    assert len(seeds) == len(set(seeds))
    assert grok.FORMAL_EXPECTED_REQUESTS == 32 * engine.REQUESTS_PER_TREE


def test_validate_smoke_requires_frozen_grok_contract(tmp_path) -> None:
    path = tmp_path / "RESULT.json"
    path.write_text(
        json.dumps(
            {
                "status": "passed",
                "protocol": {
                    "interface_version": smoke.INTERFACE_VERSION,
                    "model": smoke.MODEL_ID,
                    "expected_requests": 10,
                    "efficacy_used_for_authorization": False,
                },
                "gates": {"all_pass": True},
            }
        )
    )

    assert grok.validate_smoke_result(path)["status"] == "passed"

    value = json.loads(path.read_text())
    value["protocol"]["model"] = "x-ai/other"
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        grok.validate_smoke_result(path)


def test_formal_gates_accept_primary_proper_score_positive() -> None:
    aggregate = {
        "comparisons": {
            name: _comparison()
            for name in (
                "crossfit_depth_two",
                "myopic_eig",
                "fixed_support_depth_three",
                "positive_test_strategy",
            )
        },
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
        "adapter_requests": grok.FORMAL_EXPECTED_REQUESTS,
        "http_attempts": grok.FORMAL_EXPECTED_REQUESTS,
        "retry_count": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 6.0,
    }

    gates = grok.formal_gates(
        scored_trees=[_clean_tree() for _ in range(32)],
        usage=usage,
        aggregate=aggregate,
    )

    assert all(gates.values())


def test_balance_check_is_fail_closed() -> None:
    with pytest.raises(RuntimeError):
        grok.require_starting_balance(7.49)
    grok.require_starting_balance(7.50)
