from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_support_recovery_daily as daily
from scripts import regretbench_deepseek_dynamic_depth2_policy as policy


def _live(*, usage: float = 100.15) -> dict[str, float]:
    return {
        "total_credits_usd": 120.0,
        "total_usage_usd": usage,
        "balance_usd": 120.0 - usage,
    }


def _catalog(
    *, prompt: float = 0.00000009, completion: float = 0.00000018
) -> dict:
    return {
        "data": [
            {
                "id": daily.recovery.MODEL_ID,
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["seed", "structured_outputs"],
                "top_provider": {
                    "context_length": 1_048_576,
                    "max_completion_tokens": 65_536,
                },
                "pricing": {
                    "prompt": str(prompt),
                    "completion": str(completion),
                },
            }
        ]
    }


def _install_baseline_files(tmp_path, monkeypatch, *, recorded: float = 0.15):
    result = tmp_path / "baseline" / "RESULT.json"
    ledger = tmp_path / "baseline-ledger.json"
    execution = result.parent / "EXECUTION.json"
    result.parent.mkdir()
    result.write_text("{}")
    ledger.write_text(
        json.dumps(
            {
                "date": daily.DATE,
                "timezone": daily.TIMEZONE,
                "daily_cap_usd": 5.0,
                "opening_total_credits_usd": 120.0,
                "opening_total_usage_usd": 100.0,
                "opening_balance_usd": 20.0,
                "recorded_actual_spend_usd": recorded,
                "account_wide_usage_counts_against_cap": True,
                "naive_first_link": {"status": "passed"},
            }
        )
    )
    execution.write_text(
        json.dumps(
            {
                "status": "complete_reconciled",
                "result_sha256": daily.recovery.sha256_file(result),
                "ledger_sha256": daily.recovery.sha256_file(ledger),
            }
        )
    )
    monkeypatch.setattr(daily.baseline_daily, "SMOKE_RESULT", result)
    monkeypatch.setattr(daily.baseline_daily, "SMOKE_LEDGER", ledger)
    monkeypatch.setattr(daily.baseline_daily, "SMOKE_DIR", result.parent)
    monkeypatch.setattr(daily.baseline, "verify_smoke_result", lambda path: {"verified": True})
    return result, ledger


def test_preflight_inherits_account_wide_baseline_opening(tmp_path, monkeypatch) -> None:
    _install_baseline_files(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "regret-ledger.json")

    result = daily.preflight(
        now=datetime(2026, 8, 8, 12, tzinfo=ZoneInfo("Europe/London")),
        live_reader=_live,
        catalog_reader=_catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["spent_before_regretbench_usd"] == pytest.approx(0.15)
    assert result["budget"]["remaining_after_full_caps_usd"] == pytest.approx(4.15)
    assert result["model_calls_made"] == 0
    assert result["model"]["id"] == daily.recovery.MODEL_ID
    assert result["model"]["covered_prompt_tokens_at_live_price"] > 12_000


def test_model_catalog_rejects_price_drift_beyond_request_reservation() -> None:
    with pytest.raises(RuntimeError, match="reservation no longer covers"):
        daily.validate_deepseek_model_catalog(
            _catalog(prompt=0.0000003, completion=0.0000003)
        )


def test_model_catalog_requires_seed_and_structured_output() -> None:
    catalog = _catalog()
    catalog["data"][0]["supported_parameters"] = []

    with pytest.raises(RuntimeError, match="seeded requests"):
        daily.validate_deepseek_model_catalog(catalog)


def test_model_catalog_rejects_non_numeric_pricing() -> None:
    catalog = _catalog()
    catalog["data"][0]["pricing"]["prompt"] = "not-a-price"

    with pytest.raises(RuntimeError, match="live pricing is invalid"):
        daily.validate_deepseek_model_catalog(catalog)


def test_reserved_prompt_floor_covers_frozen_request_envelope() -> None:
    cigs = daily.recovery.load_stage_cigs("development")
    maximum_answer_bytes = max(
        len(str(value).encode("utf-8"))
        for cig in cigs
        for intent in cig.intents
        for value in (intent.slots or {}).values()
    )
    answer_width = max(200, maximum_answer_bytes)
    dialogues = [
        [],
        [
            {"role": "assistant", "content": "Q" * 240},
            {"role": "user", "content": "A" * answer_width},
        ],
        [
            {"role": "assistant", "content": "Q" * 240},
            {"role": "user", "content": "A" * answer_width},
            {"role": "assistant", "content": "Q" * 240},
            {"role": "user", "content": "A" * answer_width},
        ],
    ]
    interfaces = [
        (
            daily.recovery.messages_for,
            daily.recovery.support_response_format(),
            daily.recovery.MODEL_ID,
            daily.recovery.MAX_TOKENS,
        ),
        (
            policy.messages_for,
            policy.enriched_response_format(),
            policy.MODEL_ID,
            policy.MAX_TOKENS,
        ),
        (
            policy.naive_messages_for,
            policy.naive_response_format(),
            policy.NAIVE_MODEL_ID,
            policy.NAIVE_MAX_TOKENS,
        ),
    ]
    request_bytes = []
    for messages_for, response_format, model, max_tokens in interfaces:
        for cig in cigs:
            for dialogue in dialogues:
                messages, _ = messages_for(cig, dialogue)
                payload = {
                    "model": model,
                    "messages": messages,
                    "temperature": 0.7,
                    "top_p": 0.95,
                    "top_k": 50,
                    "max_tokens": max_tokens,
                    "n": 1,
                    "seed": 20260808,
                    "response_format": response_format,
                    "provider": {"require_parameters": True},
                }
                request_bytes.append(
                    len(
                        json.dumps(
                            payload,
                            ensure_ascii=True,
                            separators=(",", ":"),
                        ).encode("utf-8")
                    )
                )

    # Any tokenizer uses at most one token per input byte.
    assert max(request_bytes) == 3_354
    assert max(request_bytes) <= daily.MIN_RESERVED_PROMPT_TOKENS


def test_preflight_refuses_support_core_hash_change(monkeypatch) -> None:
    monkeypatch.setattr(daily, "RECOVERY_CORE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="core binding changed"):
        daily.preflight(
            now=datetime(2026, 8, 8, 12, tzinfo=ZoneInfo("Europe/London")),
            live_reader=_live,
            catalog_reader=_catalog,
        )


def test_preflight_rejects_when_combined_caps_do_not_fit(tmp_path, monkeypatch) -> None:
    _install_baseline_files(tmp_path, monkeypatch, recorded=4.5)
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "regret-ledger.json")

    with pytest.raises(RuntimeError, match="remaining account-wide day"):
        daily.preflight(
            now=datetime(2026, 8, 8, 12, tzinfo=ZoneInfo("Europe/London")),
            live_reader=lambda: _live(usage=104.5),
            catalog_reader=_catalog,
        )


class _Adapter:
    def usage_snapshot(self):
        return {
            "adapter_requests": 0,
            "http_attempts": 0,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


def test_execute_runs_development_only_after_passing_smoke(tmp_path, monkeypatch) -> None:
    _, baseline_ledger = _install_baseline_files(tmp_path, monkeypatch)
    smoke_dir = tmp_path / "smoke"
    development_dir = tmp_path / "development"
    ledger_path = tmp_path / "regret-ledger.json"
    root = tmp_path / "root"
    monkeypatch.setattr(daily, "SMOKE_DIR", smoke_dir)
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", development_dir)
    monkeypatch.setattr(daily, "LEDGER", ledger_path)
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(
        daily,
        "preflight",
        lambda **kwargs: {
            "budget": {"spent_before_regretbench_usd": 0.15},
            "predecessor": {},
        },
    )
    monkeypatch.setattr(
        daily.baseline_daily,
        "SMOKE_LEDGER",
        baseline_ledger,
    )
    monkeypatch.setattr(daily.recovery, "build_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(daily, "_budget_status", lambda *args, **kwargs: {"authorized": True})
    calls = []

    def fake_run_stage(*, stage, output_dir, **kwargs):
        calls.append(stage)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {
            "status": "passed",
            "usage": {"run_cost_usd": 0.01 if stage == "smoke" else 0.03},
        }
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(daily.recovery, "run_stage", fake_run_stage)
    monkeypatch.setattr(
        daily.result_verify,
        "verify_support",
        lambda *args, **kwargs: {"status": "verified", "model_calls": 0},
    )

    result = daily.execute(live_reader=_live)

    assert calls == ["smoke", "development"]
    assert result["status"] == "complete_reconciled"
    assert result["development_opened"] is True
    assert result["independent_replay_passed"] is True
    ledger = json.loads(ledger_path.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.19)
    assert ledger["stages"]["smoke"]["status"] == "passed"
    assert ledger["stages"]["development"]["status"] == "passed"
