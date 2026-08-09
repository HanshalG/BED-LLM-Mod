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


def _prior_close(_: str = "a") -> dict:
    return {
        "closing_total_usage_boundary_usd": 220.0,
        "source": {"kind": "fixture"},
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
        prior_close_validator=_prior_close,
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["maximum_accepted_responses"] == 1_056
    assert result["budget"]["maximum_http_attempts"] == 1_078
    assert result["budget"]["maximum_precharged_exposure_usd"] == pytest.approx(
        4.312
    )


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
        prior_close_validator=_prior_close,
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
        path=daily.LEDGERS["a"],
        live=_live(),
        block_id="a",
        now=_now("a"),
        prior_close=_prior_close(),
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


def test_preflight_counts_prior_account_spend_and_opens_nothing(
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
        lambda: {"verified": True},
    )
    monkeypatch.setattr(
        daily.aug10,
        "_validate_model_catalog",
        lambda catalog: {"model": daily.confirmation.MODEL_ID},
    )
    with pytest.raises(RuntimeError, match="remaining account-wide day"):
        daily.preflight_daily_block(
            block_id="a",
            live_reader=lambda: {
                "total_credits_usd": 250.0,
                "total_usage_usd": 220.251,
                "balance_usd": 29.749,
            },
            model_catalog_reader=lambda: {"data": []},
            prior_close_validator=_prior_close,
        )
    assert not daily.LEDGERS["a"].exists()
    assert not daily.BLOCK_DIRS["a"].exists()


def test_aug15_boundary_uses_verified_aug14_paired_close(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(daily, "REPO_ROOT", tmp_path)
    ledger_path = tmp_path / (
        "results/nonmyopic/openrouter_daily_budget/"
        "2026-08-14-naive-first-link.json"
    )
    handoff_path = tmp_path / (
        "results/nonmyopic/bongard_openworld_development_daily_handoff/"
        "block-d-20260814/RESULT.json"
    )
    ledger = {
        "interface_version": daily.development_daily.EXPECTED_NAIVE_DAILY_INTERFACE,
        "date": "2026-08-14",
        "timezone": daily.TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 220.0,
        "recorded_actual_spend_usd": 2.75,
        "account_wide_usage_counts_against_cap": True,
        "opening_boundary_derived_from_main_reconciled_ledger": True,
        "opening_boundary_amendment_sha256": (
            daily.development_daily.BUDGET_CHAIN_AMENDMENT_SHA256
        ),
        "unspent_allowance_does_not_roll_over": True,
        "naive_first_link": {"status": "passed"},
        "reconciliation": {
            "recorded_spend_is_max_of_posted_and_local": True,
            "remaining_daily_allowance_usd": 2.25,
        },
    }
    daily.checkpoint(ledger_path, ledger)
    daily.checkpoint(
        handoff_path,
        {
            "interface_version": (
                daily.development_daily.EXPECTED_PAIRED_HANDOFF_INTERFACE
            ),
            "status": "paired_daily_complete",
            "block_id": "d",
            "recorded_daily_spend_usd": 2.75,
            "components": {
                "naive_ledger": {
                    "path": str(ledger_path.resolve()),
                    "sha256": daily._sha256(ledger_path),
                    "status": "reconciled",
                }
            },
        },
    )

    close = daily._development_close()
    assert close["closing_total_usage_boundary_usd"] == pytest.approx(222.75)

    ledger["opening_boundary_derived_from_main_reconciled_ledger"] = False
    daily.checkpoint(ledger_path, ledger)
    with pytest.raises(RuntimeError, match="paired development close is invalid"):
        daily._development_close()
