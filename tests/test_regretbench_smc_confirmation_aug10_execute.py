from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo

from scripts import regretbench_smc_confirmation_aug10_execute as execute


def test_frozen_bindings_verify_without_model_calls() -> None:
    result = execute.validate_bindings()
    assert result["status"] == "verified_frozen_execution"
    assert result["bound_files"] == 15
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
