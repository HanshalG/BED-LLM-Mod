from __future__ import annotations

from scripts import (
    number_game_qwen_pooled_second_refresh_confirmation32 as run,
)


def scored_tree():
    return {
        "mechanics": {
            "initial_valid": 24,
            "minimum_first_branch_valid": 12,
            "minimum_retained_second_branch_valid": 8,
            "validation_support_count": 8,
            "minimum_validation_support_valid": 16,
        }
    }


def test_mechanics_gate_only_deployed_support_minima() -> None:
    usage = {
        "adapter_requests": run.EXPECTED_REQUESTS,
        "http_attempts": run.EXPECTED_REQUESTS,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 4.0,
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
    assert not any("generated_branches" in key for key in gates)


def test_second_refresh_gates_match_registered_thresholds() -> None:
    comparison = {
        "root_differences": 12,
        "relative_brier_reduction": 0.02,
        "tree_bootstrap_brier_difference_95pct": [-0.01, -0.001],
        "wins_minus_losses": 8,
    }

    assert all(run.second_refresh_gates(comparison).values())


def test_fresh_seeds_do_not_overlap_source_cohort() -> None:
    assert set(run.TREE_SEEDS).isdisjoint(range(63_100, 63_132))
    assert set(run.TARGET_SEEDS).isdisjoint(range(63_200, 63_232))
    assert run.EXPECTED_REQUESTS == 3424
    assert run.SMOKE_RESULT_SHA256 == (
        "f4c5371e9cbe80e4a344fd7cb649e7e02d883c00768456c472ee2474f60b880a"
    )
