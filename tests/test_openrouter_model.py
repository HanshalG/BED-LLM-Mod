from __future__ import annotations

import json
import http.client
import multiprocessing
import time
import urllib.error
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from helpers import Config, ModelSpec, load_config
from model_factory import build_model_adapter
from openrouter_model import OpenRouterAdapter, OpenRouterBudgetError, OpenRouterBudgetTracker


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


def _concurrent_tracker_writer(path: str, run_id: str, count: int) -> None:
    config = Config(
        run_id=run_id,
        openrouter_spend_path=path,
        openrouter_budget_usd=20.0,
        openrouter_projected_cost_usd=0.0,
    )
    tracker = OpenRouterBudgetTracker(config, "test-model")
    usage = {
        "prompt_tokens": 2,
        "completion_tokens": 1,
        "completion_tokens_details": {"reasoning_tokens": 0},
    }
    for _ in range(count):
        tracker.add(0.001, usage)


def test_openrouter_max_request_cost_parses_and_validates(tmp_path: Path) -> None:
    config_path = tmp_path / "config.yaml"
    config_path.write_text("openrouter_max_request_cost_usd: 0.012\n")
    assert load_config(str(config_path)).openrouter_max_request_cost_usd == pytest.approx(
        0.012
    )

    for value in (0.0, -0.01, float("nan"), float("inf")):
        with pytest.raises(
            ValueError,
            match="openrouter_max_request_cost_usd must be positive or null",
        ):
            Config(openrouter_max_request_cost_usd=value)


def test_openrouter_smoke_config_is_nonthinking_and_uses_verified_slug() -> None:
    config = load_config("configs/config_paprika_step0a_smoke_openrouter.yaml")
    spec = config.model_pairs[0].questioner
    assert spec.backend == "openrouter"
    assert spec.model == "google/gemma-4-26b-a4b-it"
    assert spec.thinking is False
    assert config.openrouter_projected_cost_usd == 1.0
    assert config.openrouter_concurrency == 128


def test_openrouter_adapter_tracks_native_cost_without_reasoning(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["payload"] = json.loads(request.data)
        captured["authorization"] = request.headers["Authorization"]
        captured["timeout"] = timeout
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="google/gemma-4-26b-a4b-it",
            backend="openrouter",
            thinking=False,
            max_model_len=32768,
        ),
        _config(
            tmp_path,
            task="paprika_customer_service",
            paprika_seed=1304,
            openrouter_request_timeout_seconds=42.0,
        ),
    )
    assert adapter.chat_complete([{"role": "user", "content": "hello"}], 0.0) == ["ok"]
    assert "reasoning" not in captured["payload"]
    assert captured["payload"]["max_tokens"] == 2048
    assert captured["payload"]["seed"] == 1304
    assert captured["authorization"] == "Bearer secret-test-key"
    assert captured["timeout"] == 42.0
    snapshot = adapter.usage_snapshot()
    assert snapshot["adapter_cost_usd"] == pytest.approx(0.01)
    assert snapshot["adapter_requests"] == 1
    assert snapshot["adapter_reasoning_tokens"] == 0
    assert json.loads((tmp_path / "spend.json").read_text())["total_spent_usd"] == pytest.approx(0.01)
    assert "secret-test-key" not in (tmp_path / "run.log").read_text()


def test_openrouter_structured_batch_forwards_strict_schema(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    captured = []

    def fake_urlopen(request, timeout):
        del timeout
        captured.append(json.loads(request.data))
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="openai/gpt-5.4",
            backend="openrouter",
            thinking=False,
            max_model_len=32768,
        ),
        _config(tmp_path),
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    assert adapter.chat_complete_messages_batched_structured(
        [[{"role": "user", "content": "hello"}]],
        temperature=0.0,
        block_size=1,
        response_format=response_format,
    ) == ["ok"]
    assert captured[0]["response_format"] == response_format
    assert captured[0]["provider"] == {"require_parameters": True}


def test_openrouter_responses_structured_batch_tracks_cost(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["url"] = request.full_url
        captured["payload"] = json.loads(request.data)
        return _Response(
            {
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {
                                "type": "output_text",
                                "text": '{"answer":"ok"}',
                            }
                        ],
                    }
                ],
                "usage": {
                    "input_tokens": 11,
                    "output_tokens": 7,
                    "output_tokens_details": {"reasoning_tokens": 0},
                    "total_tokens": 18,
                    "cost": 0.012,
                },
            }
        )

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="openai/gpt-5.4",
            backend="openrouter",
            thinking=False,
            reasoning_effort="none",
            max_model_len=32768,
        ),
        _config(tmp_path),
    )
    response_format = {
        "type": "json_schema",
        "json_schema": {
            "name": "answer",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }
    assert adapter.responses_complete_messages_batched_structured(
        [[{"role": "user", "content": "hello"}]],
        temperature=0.0,
        block_size=1,
        response_format=response_format,
    ) == ['{"answer":"ok"}']
    assert captured["url"].endswith("/api/v1/responses")
    assert captured["payload"]["text"]["format"] == {
        "name": "answer",
        "strict": True,
        "schema": response_format["json_schema"]["schema"],
        "type": "json_schema",
    }
    assert captured["payload"]["provider"] == {"require_parameters": True}
    assert captured["payload"]["reasoning"] == {"effort": "none"}
    snapshot = adapter.usage_snapshot()
    assert snapshot["adapter_requests"] == 1
    assert snapshot["adapter_prompt_tokens"] == 11
    assert snapshot["adapter_completion_tokens"] == 7
    assert snapshot["adapter_cost_usd"] == pytest.approx(0.012)


def test_openrouter_budget_warning_uses_configured_threshold_and_emits_once(
    monkeypatch, tmp_path: Path, capsys
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    spend = tmp_path / "spend.json"
    spend.write_text(
        json.dumps({"budget_usd": 1.0, "total_spent_usd": 0.89, "runs": {}})
    )
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(_completion(cost=0.02)),
    )
    adapter = OpenRouterAdapter(
        ModelSpec(model="openai/gpt-5.4-mini", backend="openrouter"),
        _config(
            tmp_path,
            openrouter_budget_usd=1.0,
            openrouter_projected_cost_usd=0.0,
        ),
    )

    adapter.chat_complete([{"role": "user", "content": "one"}], 0.0)
    adapter.chat_complete([{"role": "user", "content": "two"}], 0.0)

    output = capsys.readouterr().out
    assert output.count("cumulative OpenRouter spend") == 1
    assert "$1.00" in output


def test_openrouter_adapter_uses_mediq_seed(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured["payload"] = json.loads(request.data)
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="google/gemma-4-26b-a4b-it",
            backend="openrouter",
            thinking=False,
            max_model_len=32768,
        ),
        _config(tmp_path, task="mediq", mediq_seed=1304),
    )
    assert adapter.chat_complete([{"role": "user", "content": "hello"}], 0.0) == ["ok"]
    assert captured["payload"]["seed"] == 1304


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


def test_openrouter_retry_attempts_are_measured(monkeypatch, tmp_path: Path) -> None:
    responses = [
        urllib.error.URLError("temporary"),
        _Response(
            {
                "choices": [
                    {
                        "message": {"content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "cost": 0.001,
                },
            }
        ),
    ]

    def fake_urlopen(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr("openrouter_model.time.sleep", lambda _delay: None)
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    adapter = OpenRouterAdapter(
        ModelSpec(model="openai/gpt-5.4", backend="openrouter"),
        _config(tmp_path),
    )

    assert adapter.chat_complete([{"role": "user", "content": "hi"}], 0.0) == [
        "ok"
    ]
    snapshot = adapter.usage_snapshot()
    assert snapshot["http_attempts"] == 2
    assert snapshot["retry_count"] == 1


def test_openrouter_reasoning_only_length_response_gets_bounded_finalization(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    payloads = []

    def fake_urlopen(request, timeout):
        del timeout
        payloads.append(json.loads(request.data))
        if len(payloads) == 1:
            response = _completion(
                content=None,
                reasoning_tokens=1024,
                finish_reason="length",
            )
            response["choices"][0]["message"]["reasoning_details"] = [
                {
                    "type": "reasoning.text",
                    "text": "preserved analysis",
                    "format": "unknown",
                    "id": None,
                }
            ]
            return _Response(response)
        return _Response(_completion(content='{"r0":["move-SOUTH","check-5"]}'))

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

    completion = adapter.chat_complete(
        [{"role": "user", "content": "return json"}], 0.0
    )

    assert completion == ['{"r0":["move-SOUTH","check-5"]}']
    assert len(payloads) == 2
    assert payloads[1]["max_tokens"] == 256
    assert payloads[1]["reasoning"] == {"enabled": False, "exclude": True}
    assert payloads[1]["messages"][-2]["reasoning_details"][0]["text"] == (
        "preserved analysis"
    )
    snapshot = adapter.usage_snapshot()
    assert snapshot["forced_exits"] == 1
    assert snapshot["forced_final_requests"] == 1
    assert snapshot["forced_final_successes"] == 1
    assert snapshot["adapter_requests"] == 2


def test_openrouter_wafer_truncation_notice_gets_bounded_finalization(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    payloads = []
    notice = (
        "[Wafer: response was truncated before the model finished its internal "
        "reasoning. Increase max_tokens, or disable thinking on this model "
        "(e.g. chat_template_kwargs.enable_thinking=false), then retry.]"
    )

    def fake_urlopen(request, timeout):
        del timeout
        payloads.append(json.loads(request.data))
        if len(payloads) == 1:
            return _Response(
                _completion(
                    content=notice,
                    reasoning_tokens=1024,
                    finish_reason="length",
                )
            )
        return _Response(_completion(content='{"r0":4,"r1":7,"r2":6,"r3":1}'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="google/gemma-4-26b-a4b-it",
            backend="openrouter",
            thinking=True,
            thinking_max_new_tokens=1024,
            thinking_final_max_new_tokens=128,
        ),
        _config(tmp_path),
    )

    completion = adapter.chat_complete(
        [{"role": "user", "content": "return target json"}], 0.0
    )

    assert completion == ['{"r0":4,"r1":7,"r2":6,"r3":1}']
    assert len(payloads) == 2
    assert payloads[1]["max_tokens"] == 128
    assert payloads[1]["reasoning"] == {"enabled": False, "exclude": True}
    assert payloads[1]["messages"][-2]["content"] == ""
    snapshot = adapter.usage_snapshot()
    assert snapshot["forced_final_requests"] == 1
    assert snapshot["forced_final_successes"] == 1
    assert "provider truncation notice normalized" in (
        tmp_path / "run.log"
    ).read_text()


def test_openrouter_native_reasoning_length_stop_gets_bounded_finalization(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    payloads = []

    def fake_urlopen(request, timeout):
        del timeout
        payloads.append(json.loads(request.data))
        if len(payloads) == 1:
            return _Response(
                _completion(
                    content=None,
                    reasoning_tokens=2048,
                    finish_reason="length",
                )
            )
        return _Response(_completion(content='{"answer":"done"}'))

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="openai/gpt-5.4",
            backend="openrouter",
            reasoning_effort="high",
        ),
        _config(tmp_path),
    )

    completion = adapter.chat_complete(
        [{"role": "user", "content": "return json"}], 0.0
    )

    assert completion == ['{"answer":"done"}']
    assert len(payloads) == 2
    assert payloads[0]["reasoning"] == {"effort": "high", "exclude": False}
    assert payloads[1]["max_tokens"] == 512
    assert payloads[1]["reasoning"] == {"enabled": False, "exclude": True}
    snapshot = adapter.usage_snapshot()
    assert snapshot["forced_final_requests"] == 1
    assert snapshot["forced_final_successes"] == 1


def test_openrouter_explicit_reasoning_effort_overrides_model_default(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured.update(json.loads(request.data))
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="openai/gpt-5.4",
            backend="openrouter",
            reasoning_effort="none",  # type: ignore[arg-type]
        ),
        _config(tmp_path),
    )
    adapter.chat_complete([{"role": "user", "content": "return json"}], 0.0)
    assert captured["reasoning"] == {"effort": "none", "exclude": False}
    assert adapter.usage_snapshot()["reasoning_enabled"] is False


def test_openrouter_explicit_reasoning_token_budget_reserves_the_final_response(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    captured = {}

    def fake_urlopen(request, timeout):
        del timeout
        captured.update(json.loads(request.data))
        return _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    adapter = OpenRouterAdapter(
        ModelSpec(
            model="qwen/qwen3.5-397b-a17b",
            backend="openrouter",
            reasoning_max_tokens=512,
        ),
        _config(tmp_path, openrouter_max_output_tokens=768),
    )
    adapter.chat_complete([{"role": "user", "content": "return json"}], 0.0)
    assert captured["reasoning"] == {"max_tokens": 512, "exclude": False}
    assert captured["max_tokens"] == 768


def test_openrouter_ledger_attributes_cost_by_model(tmp_path: Path) -> None:
    config = _config(tmp_path, openrouter_projected_cost_usd=0.0)
    usage = _completion()["usage"]
    OpenRouterBudgetTracker(config, "model-a").add(0.01, usage)
    OpenRouterBudgetTracker(config, "model-b").add(0.02, usage)
    run = json.loads((tmp_path / "spend.json").read_text())["runs"]["smoke-run"]
    assert run["cost_usd"] == pytest.approx(0.03)
    assert run["model_usage"]["model-a"]["cost_usd"] == pytest.approx(0.01)
    assert run["model_usage"]["model-b"]["cost_usd"] == pytest.approx(0.02)


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


def test_openrouter_retries_incomplete_chunked_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    calls = 0

    class IncompleteResponse(_Response):
        def read(self):
            raise http.client.IncompleteRead(b"partial")

    def fake_urlopen(request, timeout):
        nonlocal calls
        del request, timeout
        calls += 1
        return IncompleteResponse({}) if calls == 1 else _Response(_completion())

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("time.sleep", lambda _delay: None)
    adapter = OpenRouterAdapter(
        ModelSpec(model="google/gemma-4-26b-a4b-it", backend="openrouter"),
        _config(tmp_path),
    )
    assert adapter.chat_complete([{"role": "user", "content": "hello"}], 0.0) == ["ok"]
    assert calls == 2


def test_openrouter_retries_truncated_json_response(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    calls = 0

    class TruncatedResponse(_Response):
        def read(self):
            return b'{"choices": ['

    def fake_urlopen(request, timeout):
        nonlocal calls
        del request, timeout
        calls += 1
        return TruncatedResponse({}) if calls == 1 else _Response(_completion())

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


def test_openrouter_run_budget_is_enforced_under_the_ledger_lock(tmp_path: Path) -> None:
    tracker = OpenRouterBudgetTracker(
        _config(tmp_path, openrouter_projected_cost_usd=0.0, openrouter_run_budget_usd=0.02),
        "test-model",
    )
    tracker.add(0.015, _completion()["usage"])
    with pytest.raises(OpenRouterBudgetError, match="run budget"):
        tracker.add(0.006, _completion()["usage"])
    snapshot = tracker.snapshot()
    assert snapshot["run_budget_usd"] == pytest.approx(0.02)
    assert snapshot["run_remaining_usd"] == pytest.approx(0.005)


def test_request_reservations_prevent_concurrent_run_budget_oversubscription(
    tmp_path: Path,
) -> None:
    tracker = OpenRouterBudgetTracker(
        _config(
            tmp_path,
            openrouter_projected_cost_usd=0.0,
            openrouter_run_budget_usd=0.02,
        ),
        "test-model",
    )
    first = tracker.reserve_request(0.015, timeout_seconds=0.01)
    with pytest.raises(OpenRouterBudgetError, match="run budget"):
        tracker.reserve_request(0.01, timeout_seconds=0.01)

    reserved = tracker.snapshot()
    assert reserved["run_reserved_usd"] == pytest.approx(0.015)
    tracker.add(0.005, _completion()["usage"], reservation_id=first)

    second = tracker.reserve_request(0.01, timeout_seconds=0.01)
    tracker.add(0.009, _completion()["usage"], reservation_id=second)
    settled = tracker.snapshot()
    assert settled["run_reserved_usd"] == pytest.approx(0.0)
    assert settled["run_cost_usd"] == pytest.approx(0.014)


def test_waiting_request_reserves_only_after_prior_request_settles(
    tmp_path: Path,
) -> None:
    tracker = OpenRouterBudgetTracker(
        _config(
            tmp_path,
            openrouter_projected_cost_usd=0.0,
            openrouter_run_budget_usd=0.02,
        ),
        "test-model",
    )
    first = tracker.reserve_request(0.015, timeout_seconds=0.5)
    with ThreadPoolExecutor(max_workers=1) as executor:
        waiting = executor.submit(
            tracker.reserve_request, 0.01, timeout_seconds=0.5
        )
        time.sleep(0.03)
        assert not waiting.done()
        tracker.add(0.005, _completion()["usage"], reservation_id=first)
        second = waiting.result(timeout=0.5)

    assert tracker.snapshot()["run_reserved_usd"] == pytest.approx(0.01)
    tracker.release_request(second)


def test_request_reservation_rejects_charge_above_bound_and_stays_fail_closed(
    tmp_path: Path,
) -> None:
    tracker = OpenRouterBudgetTracker(
        _config(
            tmp_path,
            openrouter_projected_cost_usd=0.0,
            openrouter_run_budget_usd=0.02,
        ),
        "test-model",
    )
    reservation = tracker.reserve_request(0.005, timeout_seconds=0.01)

    with pytest.raises(OpenRouterBudgetError, match="exceeds reserved"):
        tracker.add(0.006, _completion()["usage"], reservation_id=reservation)

    snapshot = tracker.snapshot()
    assert snapshot["run_cost_usd"] == pytest.approx(0.0)
    assert snapshot["run_reserved_usd"] == pytest.approx(0.005)


def test_adapter_settles_request_reservation_after_usage(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    monkeypatch.setattr(
        "urllib.request.urlopen",
        lambda request, timeout: _Response(_completion(cost=0.01)),
    )
    adapter = OpenRouterAdapter(
        ModelSpec(model="test/model", backend="openrouter"),
        _config(
            tmp_path,
            openrouter_projected_cost_usd=0.0,
            openrouter_run_budget_usd=0.02,
            openrouter_max_request_cost_usd=0.015,
        ),
    )

    assert adapter.chat_complete([{"role": "user", "content": "x"}], 0.0) == [
        "ok"
    ]
    snapshot = adapter.usage_snapshot()
    assert snapshot["run_cost_usd"] == pytest.approx(0.01)
    assert snapshot["run_reserved_usd"] == pytest.approx(0.0)


def test_ambiguous_retry_keeps_first_attempt_reserved(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    responses = [
        urllib.error.URLError("response lost"),
        _Response(_completion(cost=0.005)),
    ]

    def fake_urlopen(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("openrouter_model.time.sleep", lambda _delay: None)
    adapter = OpenRouterAdapter(
        ModelSpec(model="test/model", backend="openrouter"),
        _config(
            tmp_path,
            openrouter_projected_cost_usd=0.0,
            openrouter_run_budget_usd=0.03,
            openrouter_max_request_cost_usd=0.01,
        ),
    )

    assert adapter.chat_complete([{"role": "user", "content": "x"}], 0.0) == [
        "ok"
    ]
    snapshot = adapter.usage_snapshot()
    assert snapshot["run_cost_usd"] == pytest.approx(0.005)
    assert snapshot["run_reserved_usd"] == pytest.approx(0.01)


def test_explicit_http_error_releases_attempt_before_retry(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-test-key")
    responses = [
        urllib.error.HTTPError("url", 429, "rate", {}, None),
        _Response(_completion(cost=0.005)),
    ]

    def fake_urlopen(*args, **kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)
    monkeypatch.setattr("openrouter_model.time.sleep", lambda _delay: None)
    adapter = OpenRouterAdapter(
        ModelSpec(model="test/model", backend="openrouter"),
        _config(
            tmp_path,
            openrouter_projected_cost_usd=0.0,
            openrouter_run_budget_usd=0.015,
            openrouter_max_request_cost_usd=0.01,
        ),
    )

    assert adapter.chat_complete([{"role": "user", "content": "x"}], 0.0) == [
        "ok"
    ]
    snapshot = adapter.usage_snapshot()
    assert snapshot["run_cost_usd"] == pytest.approx(0.005)
    assert snapshot["run_reserved_usd"] == pytest.approx(0.0)


def test_spend_tracker_serializes_concurrent_processes(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    context = multiprocessing.get_context("fork")
    processes = [
        context.Process(target=_concurrent_tracker_writer, args=(str(path), f"run-{index}", 50))
        for index in range(4)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=20)
        assert process.exitcode == 0

    payload = json.loads(path.read_text())
    assert payload["total_spent_usd"] == pytest.approx(0.2)
    assert sum(run["requests"] for run in payload["runs"].values()) == 200
    assert not list(tmp_path.glob(".spend.json.*.tmp"))


def test_spend_tracker_does_not_downgrade_top_up_from_older_worker(tmp_path: Path) -> None:
    path = tmp_path / "spend.json"
    old = OpenRouterBudgetTracker(
        _config(tmp_path, openrouter_spend_path=str(path), openrouter_budget_usd=30.0),
        "test-model",
    )
    path.write_text(json.dumps({"budget_usd": 40.0, "total_spent_usd": 31.0, "runs": {}}))

    old.add(0.01, _completion()["usage"])

    payload = json.loads(path.read_text())
    assert payload["budget_usd"] == pytest.approx(40.0)
    assert old.snapshot()["remaining_usd"] == pytest.approx(8.99)


def test_lazy_factory_builds_openrouter_without_gpu_import(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret")
    adapter = build_model_adapter(
        ModelSpec(model="google/gemma-4-26b-a4b-it", backend="openrouter"),
        _config(tmp_path),
    )
    assert isinstance(adapter, OpenRouterAdapter)
