from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from scripts import number_game_qwen_history_blind_matched32 as base
from scripts import number_game_qwen_history_blind_matched32_v3 as v3
from scripts import number_game_qwen_history_blind_serving_smoke as smoke


class _FakeFormalAdapter:
    def __init__(self, first_response: str, second_response: str) -> None:
        self.first_response = first_response
        self.second_response = second_response
        self.request_count = 0

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages,
        seeds,
        **kwargs,
    ):
        del kwargs
        assert len(batch_messages) == base.EXPECTED_REQUESTS
        assert len(seeds) == base.EXPECTED_REQUESTS
        self.request_count = len(seeds)
        return [
            self.first_response if index % 2 == 0 else self.second_response
            for index in range(len(seeds))
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.request_count,
            "http_attempts": self.request_count,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


def _pool_row(second_novelty: int = 1) -> dict:
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


def test_prior_failure_bindings_are_exact() -> None:
    v1 = v3.validate_v1_failure()
    v2 = v3.validate_v2_failure()

    assert v1["status"] == "mechanics_failed"
    assert v1["protocol"]["endpoint_accessed"] is False
    assert v2 == {
        "schema_version": 1,
        "status": "failed_closed",
        "error_type": "RecursionError",
        "error": "maximum recursion depth exceeded",
    }
    assert v3.V2_ENDPOINT_ACCESSED is False


def test_v3_dispatch_inside_context_does_not_recurse() -> None:
    controls = {"trees": [{} for _ in range(base.TREE_COUNT)]}
    pool_rows = [
        _pool_row()
        for _ in range(base.TREE_COUNT * base.SLOTS_PER_TREE)
    ]

    with v3.configured_base():
        gates = base.mechanics_gates(
            usage=_usage(),
            controls=controls,
            pool_rows=pool_rows,
        )

    assert "every_second_draw_adds_at_least_two_extensions" not in gates
    assert all(gates.values())


def test_v3_context_restores_every_base_global() -> None:
    originals = {
        "INTERFACE_VERSION": base.INTERFACE_VERSION,
        "CONTROL_SEED_START": base.CONTROL_SEED_START,
        "mechanics_gates": base.mechanics_gates,
    }

    with v3.configured_base():
        assert base.INTERFACE_VERSION == v3.INTERFACE_VERSION
        assert base.CONTROL_SEED_START == 9_400_000
        assert base.control_seed(0, 0, 0) == 9_400_000
        assert base.mechanics_gates is v3.mechanics_gates

    assert {
        "INTERFACE_VERSION": base.INTERFACE_VERSION,
        "CONTROL_SEED_START": base.CONTROL_SEED_START,
        "mechanics_gates": base.mechanics_gates,
    } == originals


def test_second_draw_novelty_remains_descriptive() -> None:
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

    summary = v3.second_draw_novelty(controls)

    assert summary["minimum"] == 1
    assert summary["fraction_at_least_two"] == 0.0
    assert summary["used_as_gate"] is False


def test_zero_call_full_v3_formal_path(
    tmp_path,
    monkeypatch,
) -> None:
    targets = json.loads(base.SOURCE_TARGETS.read_text())["targets"]

    def response(items) -> str:
        return json.dumps(
            {
                "hypotheses": [
                    {
                        "name": item["name"],
                        "expression": item["expression"],
                    }
                    for item in items
                ]
            }
        )

    adapter = _FakeFormalAdapter(
        response(targets[:24]),
        response(targets[9:33]),
    )
    smoke_path = tmp_path / "SMOKE.json"
    smoke_payload = {
        "status": "passed",
        "protocol": {
            "interface_version": smoke.INTERFACE_VERSION,
            "model": smoke.MODEL_ID,
            "expected_requests": 10,
            "efficacy_used_for_authorization": False,
        },
        "usage": {"run_cost_usd": 0.0},
        "gates": {"all_pass": True},
    }
    smoke_path.write_text(json.dumps(smoke_payload))
    monkeypatch.setattr(
        base,
        "SMOKE_RESULT_SHA256",
        hashlib.sha256(smoke_path.read_bytes()).hexdigest(),
    )

    result = v3.run_formal(
        output_dir=tmp_path / "formal",
        run_id="zero-call-v3-integration",
        smoke_result_path=smoke_path,
        adapter=adapter,
        remaining_credit=10.0,
        bootstrap_samples=20,
    )

    assert adapter.request_count == base.EXPECTED_REQUESTS
    assert all(result["mechanics_gates"].values())
    assert len(result["trees"]) == base.TREE_COUNT
    assert result["protocol"]["interface_version"] == v3.INTERFACE_VERSION
    assert result["protocol"]["control_seed_start"] == 9_400_000
    assert result["protocol"]["v1_responses_reused"] is False
    assert result["protocol"]["v2_responses_reused"] is False
    assert result["second_draw_novelty"]["used_as_gate"] is False
    assert result["usage"]["run_cost_usd"] == 0.0
    assert (tmp_path / "formal" / "CONTROLS.json").exists()
    assert (tmp_path / "formal" / "RESULT.json").exists()
