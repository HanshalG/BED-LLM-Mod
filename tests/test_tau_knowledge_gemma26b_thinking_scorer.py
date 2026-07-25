from __future__ import annotations

from scripts.tau_knowledge_gemma26b_thinking_scorer import (
    apply_thinking_gates,
)


def _payload() -> dict:
    return {
        "status": "gate_failed",
        "protocol": {"expected_physical_requests": 14},
        "summary": {
            "gates": {
                "all_pass": False,
                "exact_physical_request_count": False,
                "zero_reasoning_tokens": False,
                "scores_vary_on_at_least_8_of_10_roots": True,
                "focused_pairwise_accuracy_at_least_0_55": True,
                "focused_optimal_followup_at_least_7_of_10": True,
            }
        },
        "usage": {
            "physical_requests": 20,
            "reasoning_tokens": 1200,
            "adapter_cost_usd": 0.10,
            "generator": {
                "forced_exits": 6,
                "forced_final_requests": 6,
                "forced_final_successes": 6,
            },
        },
    }


def test_thinking_gates_replace_nonreasoning_request_gates() -> None:
    payload = apply_thinking_gates(_payload(), stage="serving_smoke")
    gates = payload["summary"]["gates"]
    assert payload["status"] == "passed"
    assert "zero_reasoning_tokens" not in gates
    assert "exact_physical_request_count" not in gates
    assert payload["protocol"]["maximum_physical_requests"] == 28


def test_thinking_gate_rejects_unfinalized_forced_exit() -> None:
    payload = _payload()
    payload["usage"]["generator"]["forced_final_successes"] = 5
    result = apply_thinking_gates(payload, stage="serving_smoke")
    assert result["status"] == "gate_failed"
    assert not result["summary"]["gates"]["all_forced_exits_finalized"]


def test_thinking_gate_rejects_stage_cost_overrun() -> None:
    payload = _payload()
    payload["usage"]["adapter_cost_usd"] = 0.251
    result = apply_thinking_gates(payload, stage="serving_smoke")
    assert result["status"] == "gate_failed"
    assert not result["summary"]["gates"]["cost_within_stage_cap"]
