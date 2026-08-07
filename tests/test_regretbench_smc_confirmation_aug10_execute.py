from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from scripts import regretbench_smc_confirmation_aug10_execute as execute


def test_frozen_bindings_verify_without_model_calls() -> None:
    result = execute.validate_bindings()
    assert result["status"] == "verified_frozen_execution"
    assert result["bound_files"] == 19
    assert result["model_calls_made"] == 0


def test_wrapper_preflight_routes_to_daily_without_writing(monkeypatch) -> None:
    monkeypatch.setattr(
        execute,
        "validate_bindings",
        lambda: {"status": "verified_frozen_execution"},
    )
    monkeypatch.setattr(
        execute.daily,
        "preflight",
        lambda **kwargs: {
            "status": "ready_without_paid_calls",
            "model_calls_made": 0,
            "files_written": 0,
        },
    )
    result = execute.preflight(
        now=datetime(2026, 8, 10, 12, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: {},
        catalog_reader=lambda: {},
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["next_stage"] == "smc_dynamic_depth2_confirmation"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_wrapper_writes_derived_outputs_only_after_complete_daily(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        execute,
        "validate_bindings",
        lambda: {"status": "verified_frozen_execution"},
    )
    monkeypatch.setattr(
        execute.daily,
        "execute",
        lambda **kwargs: {
            "status": "complete_reconciled",
            "confirmation_status": "gated_null",
        },
    )
    monkeypatch.setattr(
        execute.frozen_report,
        "write_report",
        lambda *args, **kwargs: calls.append("report")
        or {"claim_tier": "smc_confirmation_null_no_headline_result"},
    )
    monkeypatch.setattr(
        execute.paper_fragment,
        "write_fragment",
        lambda *args, **kwargs: calls.append("fragment")
        or {"claim_tier": "smc_confirmation_null_no_headline_result"},
    )

    result = execute.execute(live_reader=lambda: {})

    assert calls == ["report", "fragment"]
    assert result["reporting"]["frozen_report"]["claim_tier"] == (
        "smc_confirmation_null_no_headline_result"
    )


def test_wrapper_emits_no_derived_result_when_parent_bank_stops(monkeypatch) -> None:
    monkeypatch.setattr(execute, "validate_bindings", lambda: {})
    monkeypatch.setattr(
        execute.daily,
        "execute",
        lambda **kwargs: {"status": "parent_bank_stopped"},
    )
    monkeypatch.setattr(
        execute.frozen_report,
        "write_report",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("report should remain closed")
        ),
    )

    result = execute.execute(live_reader=lambda: {})

    assert result["reporting"] is None
