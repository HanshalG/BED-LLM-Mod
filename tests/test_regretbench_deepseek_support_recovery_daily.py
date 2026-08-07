from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_support_recovery_daily as daily


def _live(*, usage: float = 100.15) -> dict[str, float]:
    return {
        "total_credits_usd": 120.0,
        "total_usage_usd": usage,
        "balance_usd": 120.0 - usage,
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
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["spent_before_regretbench_usd"] == pytest.approx(0.15)
    assert result["budget"]["remaining_after_full_caps_usd"] == pytest.approx(4.15)
    assert result["model_calls_made"] == 0


def test_preflight_refuses_support_core_hash_change(monkeypatch) -> None:
    monkeypatch.setattr(daily, "RECOVERY_CORE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="core binding changed"):
        daily.preflight(
            now=datetime(2026, 8, 8, 12, tzinfo=ZoneInfo("Europe/London")),
            live_reader=_live,
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

    result = daily.execute(live_reader=_live)

    assert calls == ["smoke", "development"]
    assert result["status"] == "complete_reconciled"
    assert result["development_opened"] is True
    ledger = json.loads(ledger_path.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.19)
    assert ledger["stages"]["smoke"]["status"] == "passed"
    assert ledger["stages"]["development"]["status"] == "passed"
