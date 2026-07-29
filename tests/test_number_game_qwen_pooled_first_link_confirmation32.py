from __future__ import annotations

from scripts import number_game_qwen_pooled_first_link_confirmation32 as run


def scored_tree():
    return {
        "mechanics": {
            "initial_valid": 24,
            "minimum_generated_first_branch_valid": 8,
            "minimum_generated_second_branch_valid": 8,
            "minimum_first_branch_valid": 12,
            "minimum_retained_second_branch_valid": 8,
            "validation_support_count": 8,
            "minimum_validation_support_valid": 16,
        }
    }


def test_parser_accounting_counts_nested_salvage() -> None:
    events = [
        {
            "pool_size": 2,
            "draw_novel_contributions": [20, 5],
            "draw_diagnostics": [
                {"codec_mode": "strict_json"},
                {"codec_mode": "complete_item_salvage"},
            ],
        },
        {
            "pool_size": 1,
            "codec_mode": "strict_json",
        },
    ]

    summary = run.parser_accounting(events)

    assert summary["pooled_parse_events"] == 1
    assert summary["single_parse_events"] == 1
    assert summary["provider_draws_parsed"] == 3
    assert summary["item_salvaged_draws"] == 1
    assert summary["minimum_second_draw_novel_contribution"] == 5


def test_mechanics_gates_accept_exact_frozen_counts() -> None:
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


def test_first_link_gates_use_32_tree_thresholds() -> None:
    summary = {
        "root_differences": 28,
        "mean_realized_advantage": 0.008,
        "mean_realized_advantage_95pct_bootstrap": [0.001, 0.02],
        "score_to_realized_advantage_spearman": 0.25,
        "score_to_realized_spearman_95pct_bootstrap": [0.01, 0.6],
        "wins": 18,
        "losses": 10,
    }

    assert all(run.first_link_gates(summary).values())


def test_committed_smoke_authorizes_confirmation() -> None:
    result = run.validate_smoke_result(run.SMOKE_RESULT)

    assert result["status"] == "passed"
    assert result["protocol"]["pool_size"] == 2
