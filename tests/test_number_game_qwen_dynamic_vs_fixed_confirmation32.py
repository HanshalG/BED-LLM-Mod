from __future__ import annotations

import pytest

from scripts import (
    number_game_qwen_dynamic_vs_fixed_confirmation32 as run,
)


def scored_tree() -> dict:
    return {
        "mechanics": {
            "initial_valid": 24,
            "minimum_first_branch_valid": 12,
            "minimum_retained_second_branch_valid": 8,
            "validation_support_count": 16,
            "minimum_validation_support_valid": 16,
        }
    }


def test_frozen_request_and_parse_counts() -> None:
    assert run.REQUESTS_PER_TREE == 115
    assert run.EXPECTED_REQUESTS == 3680
    assert run.EXPECTED_POOLED_PARSE_EVENTS == 1568
    assert run.EXPECTED_PARSE_EVENTS == 2112
    assert len(
        {
            seed
            for index in range(run.TREE_COUNT)
            for seed in range(
                run.VALIDATION_SEED_START
                + index * run.VALIDATION_DRAWS_PER_TREE,
                run.VALIDATION_SEED_START
                + (index + 1) * run.VALIDATION_DRAWS_PER_TREE,
            )
        }
    ) == 512


def test_committed_smoke_authorizes_confirmation() -> None:
    result = run.validate_smoke_result(run.SMOKE_RESULT)

    assert result["status"] == "passed"
    assert result["protocol"]["interface_version"] == (
        "number-game-qwen-dynamic-vs-fixed-serving-smoke-1"
    )
    assert result["usage"]["adapter_requests"] == 10


def test_starting_balance_gate() -> None:
    run.require_starting_balance(run.MIN_STARTING_BALANCE_USD)
    with pytest.raises(RuntimeError):
        run.require_starting_balance(run.MIN_STARTING_BALANCE_USD - 0.01)


def test_mechanics_accept_exact_frozen_counts() -> None:
    usage = {
        "adapter_requests": run.EXPECTED_REQUESTS,
        "http_attempts": run.EXPECTED_REQUESTS,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 4.2,
    }
    targets = [
        type("Target", (), {"extension": (index,)})()
        for index in range(run.TARGET_COUNT)
    ]
    parse_summary = {
        "parse_events": run.EXPECTED_PARSE_EVENTS,
        "pooled_parse_events": run.EXPECTED_POOLED_PARSE_EVENTS,
        "item_salvaged_draws": 0,
    }

    gates = run.mechanics_gates(
        scored_trees=[scored_tree() for _ in range(run.TREE_COUNT)],
        usage=usage,
        targets=targets,
        parse_summary=parse_summary,
    )

    assert all(gates.values())
    assert "all_sixteen_validation_supports_valid" in gates


def test_dynamic_support_gates_match_frozen_thresholds() -> None:
    comparison = {
        "relative_brier_reduction": 0.03,
        "tree_cluster_brier_difference_95pct_bootstrap": [-0.01, -0.001],
        "brier_tree_wins": 16,
    }

    assert all(
        run.dynamic_support_gates(
            comparison=comparison,
            root_differences=24,
        ).values()
    )


def test_configured_engine_uses_sixteen_validation_draws() -> None:
    original_qwen_draws = run.qwen.VALIDATION_DRAWS_PER_TREE

    with run.configured_engine(run.SMOKE_RESULT, []):
        assert run.qwen.VALIDATION_DRAWS_PER_TREE == 16
        assert run.engine.VALIDATION_DRAWS_PER_TREE == 16
        assert run.engine.validation_seeds_for_tree(0) == tuple(
            range(66_200, 66_216)
        )
        assert run.engine.validation_seeds_for_tree(31) == tuple(
            range(66_696, 66_712)
        )

    assert run.qwen.VALIDATION_DRAWS_PER_TREE == original_qwen_draws


def test_frozen_bootstrap_is_deterministic(monkeypatch) -> None:
    trees = [
        {
            "comparisons": {
                "fixed_support_depth_three": {
                    "candidate_minus_baseline_brier": value,
                    "candidate_minus_baseline_hamming": value / 2.0,
                    "coverage_difference": -value,
                }
            }
        }
        for value in (-0.2, -0.1, 0.1, 0.2)
    ]
    monkeypatch.setattr(
        run.engine,
        "aggregate_scored_trees",
        lambda _: {
            "comparisons": {
                "fixed_support_depth_three": {
                    "relative_brier_reduction": 0.1,
                    "brier_tree_wins": 2,
                }
            }
        },
    )

    first = run.comparison_with_frozen_bootstrap(
        trees,
        baseline="fixed_support_depth_three",
        seed=7,
        samples=100,
    )
    second = run.comparison_with_frozen_bootstrap(
        trees,
        baseline="fixed_support_depth_three",
        seed=7,
        samples=100,
    )

    assert first == second
    assert "tree_cluster_coverage_difference_95pct_bootstrap" in first
