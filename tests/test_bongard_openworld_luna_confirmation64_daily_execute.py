from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import bongard_openworld_luna_confirmation64_daily_execute as daily


def _now(block_id: str = "a") -> datetime:
    return datetime.fromisoformat(
        f"{daily.confirmation.BLOCK_DATES[block_id]}T12:00:00"
    ).replace(tzinfo=ZoneInfo(daily.TIMEZONE))


def _redirect(tmp_path: Path, monkeypatch) -> None:
    block_dirs = {
        block_id: tmp_path / f"block-{block_id}"
        for block_id in daily.confirmation.BLOCK_ORDER
    }
    ledgers = {
        block_id: tmp_path / f"ledger-{block_id}.json"
        for block_id in daily.confirmation.BLOCK_ORDER
    }
    monkeypatch.setattr(daily, "ROOT", tmp_path)
    monkeypatch.setattr(daily, "BLOCK_DIRS", block_dirs)
    monkeypatch.setattr(daily, "LEDGERS", ledgers)
    monkeypatch.setattr(daily, "COMBINED_RESULT", tmp_path / "combined.json")


def _live() -> dict[str, float]:
    return {
        "total_credits_usd": 250.0,
        "total_usage_usd": 220.0,
        "balance_usd": 30.0,
    }


def test_preflight_waits_for_full_development_without_live_call(
    tmp_path, monkeypatch
) -> None:
    _redirect(tmp_path, monkeypatch)
    monkeypatch.setattr(
        daily.confirmation,
        "verify_protocol_manifest",
        lambda: {"verified": True, "manifest_sha256": "m" * 64},
    )
    monkeypatch.setattr(
        daily.confirmation,
        "verify_development_authorization",
        lambda: (_ for _ in ()).throw(
            RuntimeError("development claim report is missing")
        ),
    )
    result = daily.preflight_daily_block(
        block_id="a",
        live_reader=lambda: pytest.fail("live credits must remain unopened"),
    )
    assert result["status"] == "waiting_for_development"
    assert result["runtime"] is None
    assert result["model_calls_made"] == result["files_written"] == 0


def test_preflight_propagates_present_development_replay_failure(
    tmp_path, monkeypatch
) -> None:
    _redirect(tmp_path, monkeypatch)
    monkeypatch.setattr(
        daily.confirmation,
        "verify_protocol_manifest",
        lambda: {"verified": True, "manifest_sha256": "m" * 64},
    )
    monkeypatch.setattr(
        daily.confirmation,
        "verify_development_authorization",
        lambda: (_ for _ in ()).throw(
            RuntimeError("development claim report does not replay exactly")
        ),
    )
    with pytest.raises(RuntimeError, match="does not replay exactly"):
        daily.preflight_daily_block(
            block_id="a",
            live_reader=lambda: pytest.fail("live credits must remain unopened"),
        )


def test_preflight_ready_checks_live_model_and_pristine_paths(
    tmp_path, monkeypatch
) -> None:
    _redirect(tmp_path, monkeypatch)
    monkeypatch.setattr(
        daily.confirmation,
        "verify_protocol_manifest",
        lambda: {"verified": True, "manifest_sha256": "m" * 64},
    )
    monkeypatch.setattr(
        daily.confirmation,
        "verify_development_authorization",
        lambda: {
            "verified": True,
            "claim_tier": "full_path_dependent_llm_native_development_signal",
        },
    )
    monkeypatch.setattr(
        daily.aug10,
        "_validate_model_catalog",
        lambda catalog: {"model": daily.confirmation.MODEL_ID},
    )
    result = daily.preflight_daily_block(
        block_id="a",
        live_reader=_live,
        model_catalog_reader=lambda: {"data": []},
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["maximum_precharged_exposure_usd"] == 2.752


def test_wrong_date_refuses_before_development_or_live(tmp_path, monkeypatch) -> None:
    _redirect(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="can run only"):
        daily.execute_daily_block(
            block_id="a",
            now=datetime(2026, 8, 14, 12, tzinfo=ZoneInfo(daily.TIMEZONE)),
            live_reader=lambda: pytest.fail("live call is forbidden"),
        )


def test_execute_block_a_reconciles_and_keeps_endpoint_sealed(
    tmp_path, monkeypatch
) -> None:
    _redirect(tmp_path, monkeypatch)
    manifest = {"verified": True, "manifest_sha256": "m" * 64}
    authorization = {
        "verified": True,
        "claim_tier": "full_path_dependent_llm_native_development_signal",
    }
    monkeypatch.setattr(
        daily.confirmation, "verify_protocol_manifest", lambda: manifest
    )
    monkeypatch.setattr(
        daily.confirmation,
        "verify_development_authorization",
        lambda: authorization,
    )
    monkeypatch.setattr(
        daily.development,
        "verify_mechanics_result",
        lambda path: {"cost_usd": 0.10, "request_count": 82},
    )
    monkeypatch.setattr(
        daily.aug10,
        "_validate_model_catalog",
        lambda catalog: {"model": daily.confirmation.MODEL_ID},
    )

    def fake_runner(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True)
        result = {
            "status": "block_mechanics_pass",
            "usage": {"run_cost_usd": 1.25},
        }
        (output_dir / "RESULT.json").write_text(json.dumps(result), encoding="utf-8")
        return result

    verification = {
        "verified": True,
        "block_id": "a",
        "result_sha256": "r" * 64,
        "raw_responses_sha256": "q" * 64,
        "execution_sha256": "e" * 64,
        "ledger_sha256": "l" * 64,
        "recorded_daily_spend_usd": 1.25,
    }
    monkeypatch.setattr(
        daily, "validate_block_result", lambda **kwargs: verification
    )
    result = daily.execute_daily_block(
        block_id="a",
        now=_now("a"),
        live_reader=_live,
        model_catalog_reader=lambda: {"data": []},
        block_runner=fake_runner,
    )
    assert result["status"] == "block_complete_verified"
    assert result["combined_endpoint_accessed"] is False
    assert not daily.COMBINED_RESULT.exists()
    ledger = json.loads(daily.LEDGERS["a"].read_text(encoding="utf-8"))
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(1.25)
    assert ledger["first_authorized_block"]["status"] == "block_mechanics_pass"


def test_validate_ledger_rejects_changed_authorized_cap(
    tmp_path, monkeypatch
) -> None:
    _redirect(tmp_path, monkeypatch)
    ledger = daily._initialize_ledger(
        path=daily.LEDGERS["a"], live=_live(), block_id="a", now=_now("a")
    )
    ledger = daily._reconcile(
        ledger=ledger,
        block_id="a",
        measured_cost_usd=1.25,
        live_after={**_live(), "total_usage_usd": 221.25},
        status="block_mechanics_pass",
    )
    ledger["first_authorized_block"]["maximum_cost_usd"] = 4.74
    daily.checkpoint(daily.LEDGERS["a"], ledger)
    with pytest.raises(RuntimeError, match="ledger is invalid"):
        daily.validate_ledger(path=daily.LEDGERS["a"], block_id="a")
