from __future__ import annotations

from copy import deepcopy

from scripts import number_game_qwen_history_blind_matched32 as base
from scripts import number_game_qwen_history_blind_matched32_v2 as v2


def _pool_row(second_novelty: int) -> dict:
    return {
        "valid_unique_count": 24,
        "diagnostic": {
            "draw_novel_contributions": [23, second_novelty],
            "draw_diagnostics": [
                {
                    "codec_mode": "strict_json",
                    "valid_unique_count": 23,
                },
                {
                    "codec_mode": "strict_json",
                    "valid_unique_count": 21,
                },
            ],
        },
    }


def _usage() -> dict:
    return {
        "adapter_requests": base.EXPECTED_REQUESTS,
        "http_attempts": base.EXPECTED_REQUESTS,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 3.2,
    }


def test_v1_failure_binding_is_exact() -> None:
    failure = v2.validate_v1_failure()

    assert failure["status"] == "mechanics_failed"
    assert failure["protocol"]["endpoint_accessed"] is False


def test_v2_removes_only_second_draw_novelty_gate() -> None:
    controls = {"trees": [{} for _ in range(base.TREE_COUNT)]}
    pool_rows = [
        _pool_row(1)
        for _ in range(base.TREE_COUNT * base.SLOTS_PER_TREE)
    ]
    v1_gates = base.mechanics_gates(
        usage=_usage(),
        controls=controls,
        pool_rows=pool_rows,
    )
    v2_gates = v2.mechanics_gates(
        usage=_usage(),
        controls=controls,
        pool_rows=pool_rows,
    )

    assert v1_gates["every_second_draw_adds_at_least_two_extensions"] is False
    assert "every_second_draw_adds_at_least_two_extensions" not in v2_gates
    assert all(v2_gates.values())


def test_v2_seed_override_is_scoped() -> None:
    original = base.CONTROL_SEED_START
    with v2.configured_base():
        assert base.CONTROL_SEED_START == 9_200_000
        assert base.control_seed(0, 0, 0) == 9_200_000
        assert base.INTERFACE_VERSION == v2.INTERFACE_VERSION
    assert base.CONTROL_SEED_START == original


def test_second_draw_novelty_is_descriptive() -> None:
    branch = {
        "diagnostic": {
            "draw_novel_contributions": [20, 1],
        }
    }
    controls = {
        "trees": [
            {
                "branches": {
                    str(index): deepcopy(branch)
                    for index in range(base.SLOTS_PER_TREE)
                }
            }
            for _ in range(base.TREE_COUNT)
        ]
    }

    summary = v2.second_draw_novelty(controls)

    assert summary["minimum"] == 1
    assert summary["fraction_at_least_two"] == 0.0
    assert summary["used_as_gate"] is False
