from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_factorized_static_likelihood_aug9_execute as daily
from scripts import regretbench_factorized_static_likelihood_smoke as smoke
from tests.test_regretbench_deepseek_smc_dynamic_depth2_experiment import (
    _install_primary_stage,
)
from tests.test_regretbench_factorized_static_likelihood_smoke import (
    _FactorizedAdapter,
)


NOW = datetime(2026, 8, 9, 9, 0, tzinfo=ZoneInfo("Europe/London"))


def _live(usage: float = 100.0) -> dict[str, float]:
    return {
        "total_credits_usd": 130.0,
        "total_usage_usd": usage,
        "balance_usd": 130.0 - usage,
    }


def _install(tmp_path, monkeypatch):
    root = tmp_path / "factorized"
    primary_dir = tmp_path / "primary-smoke"
    info = _install_primary_stage(primary_dir, "smoke")
    predecessor = {
        "support": {
            "support_smoke_sha256": "smoke",
            "support_development_sha256": "development",
        },
        "support_smoke_result": tmp_path / "support-smoke.json",
        "support_development_result": tmp_path / "support-development.json",
        "primary_smoke_dir": primary_dir,
        "policy_result_sha256": "policy-result",
        "policy_verification_sha256": "policy-verification",
        "policy_daily_result_sha256": "policy-daily",
        "policy_ledger_sha256": "policy-ledger",
        "frozen_report_sha256": "report",
        "paper_fragment_sha256": "paper",
        "fidelity_sha256": "fidelity",
        "decision_sha256": "decision",
        "aug8_close_usage_boundary_usd": 100.0,
    }
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(daily, "RUN_DIR", root / "smoke-20260809")
    monkeypatch.setattr(daily, "DAILY_RESULT", root / "DAILY_RESULT_20260809.json")
    monkeypatch.setattr(daily, "DAILY_FAILURE", root / "DAILY_FAILURE_20260809.json")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "daily-ledger.json")
    monkeypatch.setattr(daily.primary_daily, "SMOKE_DIR", primary_dir)
    monkeypatch.setattr(
        daily,
        "validate_bindings",
        lambda: {"status": "verified_frozen_execution"},
    )
    monkeypatch.setattr(
        daily, "validate_policy_null_predecessor", lambda: predecessor
    )
    monkeypatch.setattr(
        daily.primary_daily,
        "validate_deepseek_model_catalog",
        lambda value: {"id": smoke.core.MODEL_ID, "available": True},
    )
    monkeypatch.setattr(
        smoke.transport,
        "validate_support_predecessors",
        lambda **kwargs: predecessor["support"],
    )
    return info, predecessor


def test_aug9_executes_exact_ten_replays_and_is_idempotent(
    tmp_path, monkeypatch
) -> None:
    info, _ = _install(tmp_path, monkeypatch)
    adapter = _FactorizedAdapter(info)
    result = daily.execute(
        now=NOW,
        live_reader=lambda: _live(),
        catalog_reader=lambda: {},
        adapter_builder=lambda **kwargs: adapter,
    )
    assert result["status"] == "complete_reconciled"
    assert result["smoke_status"] == "passed"
    assert result["independent_replay_passed"] is True
    assert result["development_opened"] is False
    assert adapter.requests == 10
    verification = json.loads(
        (daily.RUN_DIR / "VERIFICATION.json").read_text()
    )
    assert verification["status"] == "verified"
    ledger = json.loads(daily.LEDGER.read_text())
    assert ledger["date"] == "2026-08-09"
    assert ledger["daily_cap_usd"] == 5.0
    assert ledger["stage"]["maximum_cost_usd"] == 0.2
    assert ledger["opening_boundary_derived_from_reconciled_aug8_close"] is True

    def no_external_read():
        raise AssertionError("completed execution read live credits")

    repeated = daily.execute(now=NOW, live_reader=no_external_read)
    assert repeated == result
    assert adapter.requests == 10


def test_completed_result_tamper_refuses_without_repeat(tmp_path, monkeypatch) -> None:
    info, _ = _install(tmp_path, monkeypatch)
    adapter = _FactorizedAdapter(info)
    daily.execute(
        now=NOW,
        live_reader=lambda: _live(),
        catalog_reader=lambda: {},
        adapter_builder=lambda **kwargs: adapter,
    )
    result_path = daily.RUN_DIR / "RESULT.json"
    value = json.loads(result_path.read_text())
    value["status"] = "mechanics_failed"
    result_path.write_text(json.dumps(value))
    with pytest.raises(RuntimeError, match="does not replay exactly"):
        daily.execute(now=NOW, live_reader=lambda: _live())
    assert adapter.requests == 10


def test_preflight_counts_usage_since_aug8_close(tmp_path, monkeypatch) -> None:
    _install(tmp_path, monkeypatch)
    ready = daily.preflight(
        now=NOW,
        live_reader=lambda: _live(104.8),
        catalog_reader=lambda: {},
    )
    assert ready["budget"]["spent_before_factorized_smoke_usd"] == pytest.approx(
        4.8
    )
    assert ready["budget"]["remaining_after_full_cap_usd"] == pytest.approx(0.0)
    with pytest.raises(RuntimeError, match="remaining account-wide day"):
        daily.preflight(
            now=NOW,
            live_reader=lambda: _live(104.800001),
            catalog_reader=lambda: {},
        )


def test_preflight_refuses_wrong_date_and_partial_paths(tmp_path, monkeypatch) -> None:
    _install(tmp_path, monkeypatch)
    wrong = datetime(2026, 8, 8, 9, 0, tzinfo=ZoneInfo("Europe/London"))
    with pytest.raises(RuntimeError, match="only on 2026-08-09"):
        daily.preflight(
            now=wrong,
            live_reader=lambda: _live(),
            catalog_reader=lambda: {},
        )
    daily.LEDGER.write_text("{}")
    with pytest.raises(RuntimeError, match="not pristine"):
        daily.preflight(
            now=NOW,
            live_reader=lambda: _live(),
            catalog_reader=lambda: {},
        )


def test_preflight_refuses_non_null_predecessor(tmp_path, monkeypatch) -> None:
    _install(tmp_path, monkeypatch)

    def reject():
        raise RuntimeError("dynamic policy predecessor is not a verified daily null")

    monkeypatch.setattr(daily, "validate_policy_null_predecessor", reject)
    with pytest.raises(RuntimeError, match="not a verified daily null"):
        daily.preflight(
            now=NOW,
            live_reader=lambda: _live(),
            catalog_reader=lambda: {},
        )
