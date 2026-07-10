from __future__ import annotations

import json
import urllib.error
from pathlib import Path

import pytest

from helpers import Config, ModelSpec, load_config
from model_factory import build_model_adapter
from openrouter_model import OpenRouterAdapter, OpenRouterBudgetError


class _Response:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def read(self):
        return json.dumps(self.payload).encode()


def _completion(*, content="ok", cost=0.01, reasoning_tokens=0, finish_reason="stop"):
    return {
        "choices": [{"message": {"content": content}, "finish_reason": finish_reason}],
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
            "cost": cost,
            "completion_tokens_details": {"reasoning_tokens": reasoning_tokens},
        },
    }


def _config(tmp_path: Path, **overrides) -> Config:
    values = {
        "run_id": "smoke-run",
        "log_path": tmp_path / "run.log",
        "openrouter_spend_path": str(tmp_path / "spend.json"),
        "openrouter_budget_usd": 20.0,
        "openrouter_projected_cost_usd": 1.0,
        "openrouter_backoff_seconds": 0.001,
    }
    values.update(overrides)
    return Config(**values)


def test_openrouter_smoke_config_is_nonthinking_and_uses_verified_slug() -> None:
    config = load_config("configs/config_paprika_step0a_smoke_openrouter.yaml")
    spec = config.model_pairs[0].questioner
    assert spec.backend == "openrouter"
    assert spec.model == "google/gemma-4-26b-a4b-it"
    assert spec.thinking is False
    assert config.openrouter_projected_cost_usd == 1.0
    assert config.openrouter_concurrency == 24


def test_openrouter_adapter_tracks_native_cost_without_reasoning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.headers["Authorization"]
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="google/gemma-4-26b-a4b-it",
            backend="openrouter",
            thinking=False,
            max_model_len=32768,
        ),
        _config(tmp_path),
    )
    assert adapter.chat_complete([{"role": "user", "content": "hello"}], 0.0) == ["ok"]
    assert "reasoning" not in captured["payload"]
    assert captured["payload"]["max_tokens"] == 2048
    assert captured["authorization"] == "Bearer secret-test-key"
    snapshot = adapter.usage_snapshot()
    assert snapshot["adapter_cost_usd"] == pytest.approx(0.01)
    assert snapshot["adapter_requests"] == 1
    assert snapshot["adapter_reasoning_tokens"] == 0
    assert json.loads((tmp_path / "spend.json").read_text())["total_spent_usd"] == pytest.approx(0.01)
    assert "secret-test-key" not in (tmp_path / "run.log").read_text()


def test_openrouter_thinking_payload_and_forced_exit_are_measured(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured.update(json.loads(request.data))
        return _Response(_completion(reasoning_tokens=50, finish_reason="length"))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="google/gemma-4-26b-a4b-it",
            backend="openrouter",
            thinking=True,
            thinking_max_new_tokens=1024,
            thinking_final_max_new_tokens=256,
        ),
        _config(tmp_path),
    )
    adapter.chat_complete([{"role": "user", "content": "think"}], 0.2)
    assert captured["reasoning"] == {"enabled": True, "exclude": False}
    assert captured["max_tokens"] == 1280
    assert adapter.usage_snapshot()["forced_exits"] == 1
    assert "Forced thinking exit" in (tmp_path / "run.log").read_text()


def test_openrouter_retries_429(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    calls = 0

    def fake_urlopen(request, timeout):
        nonlocal calls
        del request, timeout
        calls += 1
        if calls == 1:
            raise urllib.error.HTTPError("url", 429, "rate", {}, None)
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _delay: None)
    adapter = OpenRouterAdapter(
        ModelSpec(model="google/gemma-4-26b-a4b-it", backend="openrouter"),
        _config(tmp_path),
    )
    assert adapter.chat_complete([{"role": "user", "content": "hello"}], 0.0) == ["ok"]
    assert calls == 2


def test_openrouter_refuses_projected_overspend(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    spend = tmp_path / "spend.json"
    spend.write_text(json.dumps({"budget_usd": 20.0, "total_spent_usd": 19.5, "runs": {}}))
    with pytest.raises(OpenRouterBudgetError, match="exceeds"):
        OpenRouterAdapter(
            ModelSpec(model="google/gemma-4-26b-a4b-it", backend="openrouter"),
            _config(tmp_path, openrouter_projected_cost_usd=1.0),
        )


def test_lazy_factory_builds_openrouter_without_gpu_import(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    adapter = build_model_adapter(
        ModelSpec(model="google/gemma-4-26b-a4b-it", backend="openrouter"),
        _config(tmp_path),
    )
    assert isinstance(adapter, OpenRouterAdapter)
