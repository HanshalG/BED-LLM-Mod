from __future__ import annotations

from threading import Lock

import pytest

from scripts.number_game_provider_seed_fallback import (
    PROVIDER_ERROR_MESSAGE,
    install_provider_seed_fallback,
)
from scripts import number_game_validator_fallback_serving_smoke as smoke


class FakeAdapter:
    def __init__(self, outcomes):
        self.model_name = "fake/model"
        self.outcomes = list(outcomes)
        self.payloads = []
        self.retry_count = 0
        self.provider_error_retries = 0
        self.http_attempts = 0
        self._usage_lock = Lock()

    def _post(self, payload):
        self.payloads.append(dict(payload))
        self.http_attempts += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_fallback_changes_seed_only_after_exhausted_provider_error() -> None:
    events = []
    adapter = FakeAdapter(
        [
            RuntimeError(PROVIDER_ERROR_MESSAGE),
            {"choices": [{"finish_reason": "stop"}]},
        ]
    )
    install_provider_seed_fallback(
        adapter,
        events=events,
        event_lock=Lock(),
        fallback_offsets=(100, 200),
    )

    result = adapter._post({"seed": 7, "messages": []})

    assert result["choices"][0]["finish_reason"] == "stop"
    assert [payload["seed"] for payload in adapter.payloads] == [7, 107]
    assert adapter.retry_count == 1
    assert adapter.provider_error_retries == 1
    assert events == [
        {
            "model": "fake/model",
            "original_seed": 7,
            "exhausted_seed": 7,
            "fallback_seed": 107,
            "fallback_group": 1,
        }
    ]


def test_fallback_never_catches_other_runtime_errors() -> None:
    adapter = FakeAdapter([RuntimeError("paid malformed response")])
    install_provider_seed_fallback(
        adapter,
        events=[],
        event_lock=Lock(),
    )
    with pytest.raises(RuntimeError, match="paid malformed"):
        adapter._post({"seed": 7})
    assert adapter.retry_count == 0


def test_fallback_exhausts_frozen_schedule() -> None:
    adapter = FakeAdapter(
        [RuntimeError(PROVIDER_ERROR_MESSAGE) for _ in range(3)]
    )
    events = []
    install_provider_seed_fallback(
        adapter,
        events=events,
        event_lock=Lock(),
        fallback_offsets=(100, 200),
    )
    with pytest.raises(RuntimeError, match=PROVIDER_ERROR_MESSAGE):
        adapter._post({"seed": 7})
    assert [payload["seed"] for payload in adapter.payloads] == [7, 107, 207]
    assert adapter.retry_count == 2
    assert len(events) == 2


def test_fallback_offsets_are_validated() -> None:
    adapter = FakeAdapter([{}])
    with pytest.raises(ValueError):
        install_provider_seed_fallback(
            adapter,
            events=[],
            event_lock=Lock(),
            fallback_offsets=(),
        )
    with pytest.raises(ValueError):
        install_provider_seed_fallback(
            adapter,
            events=[],
            event_lock=Lock(),
            fallback_offsets=(1, 1),
        )


def test_validator_smoke_gates_exact_clean_fixture() -> None:
    usage = {
        "adapter_requests": 10,
        "http_attempts": 11,
        "retry_count": 1,
        "provider_error_retries": 1,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.02,
    }
    diagnostics = [{"codec_mode": "strict_json"} for _ in range(10)]
    supports = [[object() for _ in range(16)] for _ in range(10)]
    gates = smoke.smoke_gates(
        supports=supports,
        diagnostics=diagnostics,
        usage=usage,
        fallback_events=[{"fallback_seed": 1}],
    )
    assert all(gates.values())
