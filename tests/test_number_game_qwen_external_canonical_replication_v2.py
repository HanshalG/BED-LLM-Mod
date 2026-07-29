from __future__ import annotations

import copy

from scripts import (
    number_game_qwen_external_canonical_confirmation as engine,
)
from scripts import (
    number_game_qwen_external_canonical_replication_v2 as replication,
)


def _comparison(
    *,
    relative: float,
    upper: float,
    wins: int,
) -> dict[str, float | int | list[float]]:
    return {
        "relative_brier_reduction": relative,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.02, upper],
        "brier_tree_wins": wins,
    }


def _aggregate() -> dict:
    return {
        "comparisons": {
            "crossfit_depth_two": _comparison(
                relative=0.02,
                upper=-0.001,
                wins=13,
            ),
            "myopic_eig": _comparison(
                relative=0.09,
                upper=-0.003,
                wins=21,
            ),
            "fixed_support_depth_three": _comparison(
                relative=0.04,
                upper=-0.001,
                wins=18,
            ),
            "positive_test_strategy": _comparison(
                relative=0.04,
                upper=-0.001,
                wins=18,
            ),
            "uniform_random_candidate_root": _comparison(
                relative=0.06,
                upper=-0.001,
                wins=20,
            ),
        },
        "ranking": {
            "crossfit_depth_three_spearman_brier": {"mean": 0.4},
            "crossfit_depth_two_spearman_brier": {"mean": 0.2},
        },
    }


def _tree() -> dict:
    return {
        "mechanics": {
            "initial_valid": 20,
            "validation_support_count": 8,
            "minimum_validation_support_valid": 18,
            "minimum_first_branch_valid": 10,
            "minimum_retained_second_branch_valid": 6,
        }
    }


def test_primary_is_only_depth_three_versus_myopic() -> None:
    aggregate = _aggregate()
    assert all(replication.primary_gates(aggregate).values())

    failed = copy.deepcopy(aggregate)
    failed["comparisons"]["crossfit_depth_two"][
        "tree_cluster_brier_difference_95pct_bootstrap"
    ][1] = 0.1
    assert all(replication.primary_gates(failed).values())
    assert not replication.diagnostic_gates(failed)[
        "depth_three_vs_depth_two_ci_below_zero"
    ]


def test_mechanics_allow_bounded_provider_retries() -> None:
    usage = {
        "adapter_requests": replication.EXPECTED_REQUESTS,
        "http_attempts": replication.EXPECTED_REQUESTS + 2,
        "retry_count": 2,
        "provider_error_retries": 2,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 2.5,
    }
    targets = [
        type("Target", (), {"extension": (index,)})()
        for index in range(replication.TARGET_COUNT)
    ]
    gates = replication.mechanics_gates(
        scored_trees=[_tree() for _ in range(replication.TREE_COUNT)],
        usage=usage,
        targets=targets,
    )

    assert all(gates.values())
    usage["provider_error_retries"] = replication.MAX_RETRIES + 1
    assert not replication.mechanics_gates(
        scored_trees=[_tree() for _ in range(replication.TREE_COUNT)],
        usage=usage,
        targets=targets,
    )["provider_error_retries_within_cap"]


def test_configured_engine_restores_historical_configuration() -> None:
    original_seeds = engine.TREE_SEEDS
    original_primary = engine.primary_gates

    with replication.configured_engine():
        assert engine.TREE_SEEDS == replication.TREE_SEEDS
        assert engine.primary_gates is replication.primary_gates

    assert engine.TREE_SEEDS == original_seeds
    assert engine.primary_gates is original_primary


def test_seed_schedule_is_fresh_and_exact() -> None:
    validation = {
        seed
        for index in range(replication.TREE_COUNT)
        for seed in range(
            replication.VALIDATION_SEED_START + index * 8,
            replication.VALIDATION_SEED_START + index * 8 + 8,
        )
    }
    assert len(validation) == 256
    assert not validation.intersection(replication.TREE_SEEDS)
    assert not validation.intersection(replication.TARGET_SEEDS)
    assert replication.EXPECTED_REQUESTS == 1856
