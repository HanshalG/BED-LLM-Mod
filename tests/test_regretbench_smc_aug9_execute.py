from __future__ import annotations

import pytest

from scripts import regretbench_smc_aug9_execute as execute


def test_execution_bindings_are_exact() -> None:
    result = execute.validate_bindings()

    assert result["status"] == "verified_frozen_execution"
    assert result["bound_files"] == 4
    assert result["model_calls_made"] == 0


def test_preflight_routes_only_to_dated_daily_gate(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        execute.daily,
        "preflight",
        lambda **kwargs: calls.append(kwargs) or {"status": "ready_without_paid_calls"},
    )

    result = execute.preflight(live_reader=lambda: {})

    assert result["status"] == "ready_without_paid_calls"
    assert result["next_stage"] == "smc_support_recovery"
    assert result["model_calls_made"] == 0
    assert len(calls) == 1


def test_execute_does_not_bypass_daily_gate(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        execute.daily,
        "execute",
        lambda **kwargs: calls.append(kwargs)
        or {"status": "complete_reconciled", "development_status": "gated_null"},
    )

    result = execute.execute(live_reader=lambda: {})

    assert result["status"] == "complete"
    assert result["stage_result"]["development_status"] == "gated_null"
    assert len(calls) == 1


def test_binding_hash_tamper_fails_closed(monkeypatch) -> None:
    monkeypatch.setattr(execute, "BINDINGS_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="binding artifact changed"):
        execute.validate_bindings()
