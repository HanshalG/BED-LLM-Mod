from __future__ import annotations

import json
from threading import Lock

from scripts import number_game_validator_fallback_serving_smoke_v2 as run


def diagnostic(*, valid: int = 24) -> dict:
    return {
        "raw_count": 24,
        "valid_unique_count": valid,
        "rejected": {
            "wrong_fields": 0,
            "invalid_name": 0,
            "invalid_expression": 0,
            "inconsistent": 0,
            "duplicate_extension": 24 - valid,
        },
    }


def test_corrected_gate_matches_base_parser_contract() -> None:
    usage = {
        "adapter_requests": 10,
        "http_attempts": 10,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.02,
    }
    supports = [[object() for _ in range(24)] for _ in range(10)]
    gates = run.smoke_gates(
        supports=supports,
        diagnostics=[diagnostic() for _ in range(10)],
        usage=usage,
        fallback_events=[],
    )
    assert all(gates.values())
    assert "all_ten_draws_strict_json" not in gates


class SyntheticAdapter:
    def __init__(self) -> None:
        self.model_name = "google/gemini-2.5-flash"
        self.requests = 0
        self.http_attempts = 0
        self.retry_count = 0
        self.provider_error_retries = 0
        self._usage_lock = Lock()

    def _post(self, payload):
        raise AssertionError("synthetic structured path does not call _post")

    def chat_complete_messages_batched_structured(
        self,
        batch_messages,
        **kwargs,
    ):
        del kwargs
        self.requests += len(batch_messages)
        self.http_attempts += len(batch_messages)
        hypotheses = [
            {"name": f"singleton_{value}", "expression": f"n == {value}"}
            for value in range(24)
        ]
        response = json.dumps({"hypotheses": hypotheses})
        return [response for _ in batch_messages]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.http_attempts,
            "retry_count": self.retry_count,
            "provider_error_retries": self.provider_error_retries,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def test_synthetic_full_smoke_serializes_pass(tmp_path, monkeypatch) -> None:
    adapters = []

    def factory(**kwargs):
        del kwargs
        adapter = SyntheticAdapter()
        adapters.append(adapter)
        return adapter

    monkeypatch.setattr(run.base.depth, "_adapter", factory)
    result = run.run_smoke(output_dir=tmp_path, run_id="synthetic")

    assert result["status"] == "passed"
    assert result["protocol"]["interface_version"] == run.INTERFACE_VERSION
    assert result["protocol"]["seeds"] == list(run.SEEDS)
    assert result["usage"]["adapter_requests"] == 10
    assert all(result["gates"].values())
    assert len(adapters) == 10
    assert (tmp_path / "RESULT.json").is_file()
    assert (tmp_path / "private" / "RAW_RESPONSES.json").is_file()
