from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_smc_confirmation_daily as daily


NOW = datetime(2026, 8, 10, 12, tzinfo=ZoneInfo("Europe/London"))


def _live() -> dict[str, float]:
    return {
        "total_credits_usd": 200.0,
        "total_usage_usd": 150.0,
        "balance_usd": 50.0,
    }


def _catalog() -> dict:
    return {
        "data": [
            {
                "id": daily.core.MODEL_ID,
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
                    "prompt": "0.00000009",
                    "completion": "0.00000018",
                },
            }
        ]
    }


def _authorization() -> dict:
    return {
        "status": "authorized",
        "development_status": "passed",
        "independent_replay_status": "verified",
        "report_tier": "smc_provisional_development_signal_confirmation_required",
    }


def _install_paths(tmp_path, monkeypatch) -> None:
    root = tmp_path / "confirmation"
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(daily, "PARENT_DIR", root / "parent")
    monkeypatch.setattr(daily, "RUN_DIR", root / "run")
    monkeypatch.setattr(daily, "DAILY_RESULT", root / "DAILY_RESULT.json")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(daily, "_assert_bongard_unopened", lambda: None)


def test_preflight_reserves_370_without_calls_or_writes(tmp_path, monkeypatch) -> None:
    _install_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily.confirmation, "validate_protocol_binding", lambda: None)
    monkeypatch.setattr(daily, "validate_development_predecessor", _authorization)

    result = daily.preflight(now=NOW, live_reader=_live, catalog_reader=_catalog)

    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["aggregate_cap_usd"] == pytest.approx(3.70)
    assert result["budget"]["remaining_after_full_cap_usd"] == pytest.approx(1.30)
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert not daily.LEDGER.exists()


def test_date_gate_precedes_external_or_predecessor_access(monkeypatch) -> None:
    touched: list[str] = []

    def forbidden():
        touched.append("external")
        raise AssertionError("external reader reached")

    monkeypatch.setattr(
        daily,
        "validate_development_predecessor",
        lambda: touched.append("predecessor"),
    )
    with pytest.raises(RuntimeError, match="only on 2026-08-10"):
        daily.preflight(
            now=datetime(2026, 8, 9, 23, 59, tzinfo=ZoneInfo("Europe/London")),
            live_reader=forbidden,
            catalog_reader=forbidden,
        )
    assert touched == []


def test_bongard_opened_fails_before_catalog_or_credit(tmp_path, monkeypatch) -> None:
    _install_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily.confirmation, "validate_protocol_binding", lambda: None)
    monkeypatch.setattr(daily, "validate_development_predecessor", _authorization)
    monkeypatch.setattr(
        daily,
        "_assert_bongard_unopened",
        lambda: (_ for _ in ()).throw(RuntimeError("Aug 10 Bongard branch is already open")),
    )
    touched: list[str] = []

    def forbidden():
        touched.append("external")
        raise AssertionError("external reader reached")

    with pytest.raises(RuntimeError, match="Bongard branch"):
        daily.preflight(now=NOW, live_reader=forbidden, catalog_reader=forbidden)
    assert touched == []
