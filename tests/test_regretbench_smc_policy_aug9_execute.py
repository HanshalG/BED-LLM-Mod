from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_smc_policy_aug9_execute as execute


def test_frozen_execution_bindings_are_exact() -> None:
    result = execute.validate_bindings()

    assert result["status"] == "verified_frozen_execution"
    assert result["bound_files"] == 14
    assert result["derived_bound_files"] == 9
    assert result["reporting_bindings_verified"] is True
    assert result["model_calls_made"] == 0
    assert result["cost_usd"] == 0.0


def test_preflight_validates_bindings_before_delegating(monkeypatch) -> None:
    order = []
    monkeypatch.setattr(
        execute,
        "validate_bindings",
        lambda: order.append("bindings") or {"status": "verified"},
    )
    monkeypatch.setattr(
        execute.daily,
        "preflight",
        lambda **kwargs: order.append("daily")
        or {"status": "ready_without_paid_calls"},
    )

    result = execute.preflight(
        now=datetime(2026, 8, 9, 12, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: {},
        catalog_reader=lambda: {},
    )

    assert order == ["bindings", "daily"]
    assert result["status"] == "ready_without_paid_calls"
    assert result["next_stage"] == "smc_dynamic_depth2_policy"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_binding_tamper_refuses_before_daily_execution(tmp_path, monkeypatch) -> None:
    tampered = tmp_path / "EXECUTION_BINDINGS.json"
    payload = json.loads(execute.BINDINGS.read_text())
    payload["maximum_combined_aug9_spend_usd"] = 4.5
    tampered.write_text(json.dumps(payload))
    monkeypatch.setattr(execute, "BINDINGS", tampered)
    called = []
    monkeypatch.setattr(
        execute.daily,
        "execute",
        lambda **kwargs: called.append(True),
    )

    with pytest.raises(RuntimeError, match="binding artifact changed"):
        execute.execute(live_reader=lambda: {})
    assert called == []


@pytest.mark.parametrize(
    ("stage_status", "expected_reporting"),
    [("complete_reconciled", True), ("smoke_stopped", False)],
)
def test_reporting_runs_only_after_complete_development(
    monkeypatch, stage_status, expected_reporting
) -> None:
    calls = []
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"status": "verified"})
    monkeypatch.setattr(
        execute.daily,
        "execute",
        lambda **kwargs: {"status": stage_status},
    )
    monkeypatch.setattr(
        execute.frozen_report,
        "write_report",
        lambda *args, **kwargs: calls.append("report") or {"status": "written"},
    )
    monkeypatch.setattr(
        execute.paper_fragment,
        "write_fragment",
        lambda *args, **kwargs: calls.append("fragment") or {"status": "written"},
    )

    result = execute.execute(live_reader=lambda: {})

    assert (result["reporting"] is not None) is expected_reporting
    assert calls == (["report", "fragment"] if expected_reporting else [])
