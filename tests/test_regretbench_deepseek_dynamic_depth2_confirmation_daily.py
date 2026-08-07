from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_dynamic_depth2_confirmation_daily as daily


def _live(usage: float = 103.0) -> dict[str, float]:
    return {
        "total_credits_usd": 140.0,
        "total_usage_usd": usage,
        "balance_usd": 140.0 - usage,
    }


def _catalog() -> dict:
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
        "development_result_sha256": "1" * 64,
        "development_verification_sha256": "2" * 64,
        "development_daily_result_sha256": "3" * 64,
        "development_ledger_sha256": "4" * 64,
        "development_ledger": {
            "opening_total_usage_usd": 100.0,
            "recorded_actual_spend_usd": 2.8,
        },
    }


def _set_paths(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(daily, "ROOT", tmp_path / "root")
    monkeypatch.setattr(daily, "RUN_DIR", tmp_path / "root" / "run")
    monkeypatch.setattr(daily, "DAILY_RESULT", tmp_path / "root" / "daily.json")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")


def test_preflight_is_zero_call_and_counts_usage_since_aug8_close(
    tmp_path, monkeypatch
) -> None:
    _set_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "validate_execution_bindings", lambda: {"ok": True})
    monkeypatch.setattr(daily, "validate_development_predecessor", _authorization)

    result = daily.preflight(
        now=datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: _live(103.0),
        catalog_reader=_catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["opening_total_usage_boundary_usd"] == pytest.approx(
        102.8
    )
    assert result["budget"]["spent_before_confirmation_usd"] == pytest.approx(
        0.2
    )
    assert result["budget"]["remaining_after_full_cap_usd"] == pytest.approx(
        1.3
    )
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_preflight_refuses_when_account_usage_leaves_too_little_room(
    tmp_path, monkeypatch
) -> None:
    _set_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "validate_execution_bindings", lambda: {"ok": True})
    monkeypatch.setattr(daily, "validate_development_predecessor", _authorization)

    with pytest.raises(RuntimeError, match="remaining account-wide day"):
        daily.preflight(
            now=datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Europe/London")),
            live_reader=lambda: _live(104.4),
            catalog_reader=_catalog,
        )


def test_preflight_refuses_wrong_date_before_writes(tmp_path, monkeypatch) -> None:
    _set_paths(tmp_path, monkeypatch)

    with pytest.raises(RuntimeError, match="only on 2026-08-09"):
        daily.preflight(
            now=datetime(2026, 8, 8, 10, tzinfo=ZoneInfo("Europe/London")),
            live_reader=_live,
            catalog_reader=_catalog,
        )

    assert not daily.ROOT.exists()


def test_literal_development_null_cannot_authorize_confirmation(
    tmp_path, monkeypatch
) -> None:
    policy_root = tmp_path / "policy"
    development = policy_root / "development"
    development.mkdir(parents=True)
    ledger_path = tmp_path / "aug8-ledger.json"
    monkeypatch.setattr(daily.policy_daily, "ROOT", policy_root)
    monkeypatch.setattr(daily.policy_daily, "DEVELOPMENT_DIR", development)
    monkeypatch.setattr(daily.policy_daily, "LEDGER", ledger_path)
    result = {
        "status": "gated_null",
        "protocol": {"confirmation_opened": False},
        "mechanics_gates": {"all_pass": True},
        "science": {"gates": {"all_pass": False}},
    }
    (development / "RESULT.json").write_text(json.dumps(result))
    verification = {
        "status": "verified",
        "result_status": "gated_null",
        "model_calls": 0,
        "mismatches": [],
        "artifact_sha256": {
            "RESULT.json": daily.recovery.sha256_file(
                development / "RESULT.json"
            )
        },
    }
    (development / "VERIFICATION.json").write_text(json.dumps(verification))
    ledger = {
        "date": "2026-08-08",
        "timezone": daily.TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 1.0,
    }
    ledger_path.write_text(json.dumps(ledger))
    daily_result = {
        "status": "complete_reconciled",
        "development_status": "gated_null",
        "development_opened": True,
        "confirmation_opened": False,
        "independent_replay_passed": True,
        "development_result_sha256": daily.recovery.sha256_file(
            development / "RESULT.json"
        ),
        "development_verification_sha256": daily.recovery.sha256_file(
            development / "VERIFICATION.json"
        ),
        "ledger_sha256": daily.recovery.sha256_file(ledger_path),
    }
    (policy_root / "DAILY_RESULT.json").write_text(json.dumps(daily_result))

    with pytest.raises(RuntimeError, match="literal independently verified"):
        daily.validate_development_predecessor()


class _Adapter:
    def __init__(self) -> None:
        self.requests = 0

    def usage_snapshot(self):
        return {"adapter_cost_usd": 0.0}


def test_execute_banks_verified_gated_null_without_retry(tmp_path, monkeypatch) -> None:
    _set_paths(tmp_path, monkeypatch)
    authorization = _authorization()
    ready = {
        "live_credits": _live(103.0),
        "budget": {
            "opening_total_usage_boundary_usd": 102.8,
            "spent_before_confirmation_usd": 0.2,
        },
        "development_authorization": {
            key: value
            for key, value in authorization.items()
            if key != "development_ledger"
        },
    }
    monkeypatch.setattr(daily, "preflight", lambda **kwargs: ready)
    adapter = _Adapter()
    monkeypatch.setattr(daily.policy, "build_adapter", lambda **kwargs: adapter)
    calls = []

    def fake_run(*, output_dir, **kwargs):
        calls.append("producer")
        output_dir.mkdir(parents=True)
        result = {
            "status": "gated_null",
            "usage": {
                "combined_cost_usd": 0.3,
                "deepseek_primary": {"adapter_requests": 8300},
            },
        }
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    def fake_verify(run_dir):
        calls.append("verifier")
        return {"status": "verified", "model_calls": 0}

    monkeypatch.setattr(daily.confirmation, "run_confirmation", fake_run)
    monkeypatch.setattr(daily.result_verify, "verify_confirmation", fake_verify)
    result = daily.execute(
        now=datetime(2026, 8, 9, 10, tzinfo=ZoneInfo("Europe/London")),
        live_reader=lambda: _live(103.0),
    )

    assert calls == ["producer", "verifier"]
    assert result["status"] == "complete_reconciled"
    assert result["confirmation_status"] == "gated_null"
    assert result["authorizes"] == "nothing"
    assert result["independent_replay_passed"] is True
    assert daily.DAILY_RESULT.exists()
    ledger = json.loads(daily.LEDGER.read_text())
    assert ledger["stage"]["status"] == "gated_null"
