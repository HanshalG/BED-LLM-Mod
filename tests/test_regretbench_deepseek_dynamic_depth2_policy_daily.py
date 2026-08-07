from __future__ import annotations

from datetime import datetime
import json
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_dynamic_depth2_policy_daily as daily


def _live(usage: float = 100.9) -> dict[str, float]:
    return {
        "total_credits_usd": 130.0,
        "total_usage_usd": usage,
        "balance_usd": 130.0 - usage,
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
                "pricing": {"prompt": "0.00000009", "completion": "0.00000018"},
            },
            {
                "id": daily.policy.NAIVE_MODEL_ID,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["reasoning", "structured_outputs"],
                "top_provider": {
                    "context_length": 1_050_000,
                    "max_completion_tokens": 128_000,
                },
                "pricing": {"prompt": "0.0000001", "completion": "0.0000006"},
            },
        ]
    }


def _prior_ledger(recorded: float = 0.9) -> dict:
    return {
        "date": daily.DATE,
        "timezone": daily.TIMEZONE,
        "daily_cap_usd": 5.0,
        "opening_total_credits_usd": 130.0,
        "opening_total_usage_usd": 100.0,
        "opening_balance_usd": 30.0,
        "recorded_actual_spend_usd": recorded,
        "account_wide_usage_counts_against_cap": True,
    }


def test_preflight_inherits_recovery_day_and_fits_480_total(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        daily,
        "validate_recovery_predecessor",
        lambda: {"support": {}, "ledger": _prior_ledger()},
    )
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", tmp_path / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")

    result = daily.preflight(
        now=datetime(2026, 8, 8, 14, tzinfo=ZoneInfo("Europe/London")),
        live_reader=_live,
        catalog_reader=_catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["spent_before_policy_usd"] == pytest.approx(0.9)
    assert result["budget"]["remaining_after_full_caps_usd"] == pytest.approx(0.2)
    assert result["model_calls_made"] == 0
    assert result["models"]["deepseek"]["id"] == daily.recovery.MODEL_ID
    assert result["models"]["luna"]["id"] == daily.policy.NAIVE_MODEL_ID


def test_preflight_marks_missing_luna_unavailable_without_vetoing_primary(
    tmp_path, monkeypatch
) -> None:
    monkeypatch.setattr(
        daily,
        "validate_recovery_predecessor",
        lambda: {"support": {}, "ledger": _prior_ledger()},
    )
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", tmp_path / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")
    catalog = _catalog()
    catalog["data"] = [catalog["data"][0]]

    result = daily.preflight(
        now=datetime(2026, 8, 8, 14, tzinfo=ZoneInfo("Europe/London")),
        live_reader=_live,
        catalog_reader=lambda: catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["models"]["deepseek"]["status"] == "available"
    assert result["models"]["luna"]["status"] == "unavailable"
    assert result["models"]["luna"]["can_affect_primary_status"] is False


def test_preflight_marks_invalid_luna_pricing_unavailable(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        daily,
        "validate_recovery_predecessor",
        lambda: {"support": {}, "ledger": _prior_ledger()},
    )
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", tmp_path / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")
    catalog = _catalog()
    catalog["data"][1]["pricing"]["completion"] = "not-a-price"

    result = daily.preflight(
        now=datetime(2026, 8, 8, 14, tzinfo=ZoneInfo("Europe/London")),
        live_reader=_live,
        catalog_reader=lambda: catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["models"]["luna"]["status"] == "unavailable"
    assert result["models"]["luna"]["error_type"] == "ValueError"


def test_preflight_refuses_when_full_policy_caps_do_not_fit(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        daily,
        "validate_recovery_predecessor",
        lambda: {"support": {}, "ledger": _prior_ledger(1.2)},
    )
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", tmp_path / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")

    with pytest.raises(RuntimeError, match="remaining account-wide day"):
        daily.preflight(
            now=datetime(2026, 8, 8, 14, tzinfo=ZoneInfo("Europe/London")),
            live_reader=lambda: _live(101.2),
            catalog_reader=_catalog,
        )


def test_preflight_refuses_policy_core_hash_change(monkeypatch) -> None:
    monkeypatch.setattr(daily, "POLICY_CORE_SHA256", "0" * 64)

    with pytest.raises(RuntimeError, match="dynamic policy core binding changed"):
        daily.preflight(
            now=datetime(2026, 8, 8, 14, tzinfo=ZoneInfo("Europe/London")),
            live_reader=_live,
            catalog_reader=_catalog,
        )


class _Adapter:
    def usage_snapshot(self):
        return {"adapter_cost_usd": 0.0}


def _install_verifier(monkeypatch) -> None:
    monkeypatch.setattr(
        daily.result_verify,
        "verify_policy_smoke",
        lambda *args, **kwargs: {"status": "verified", "model_calls": 0},
    )
    monkeypatch.setattr(
        daily.result_verify,
        "verify_policy",
        lambda *args, **kwargs: {"status": "verified", "model_calls": 0},
    )


def test_execute_orders_enriched_smoke_before_policy_development(
    tmp_path, monkeypatch
) -> None:
    prior_path = tmp_path / "prior-ledger.json"
    prior_path.write_text(json.dumps(_prior_ledger()))
    smoke_dir = tmp_path / "smoke"
    naive_smoke_dir = tmp_path / "naive-smoke"
    development_dir = tmp_path / "development"
    ledger_path = tmp_path / "ledger.json"
    root = tmp_path / "root"
    support_smoke_dir = tmp_path / "support-smoke"
    support_dev_dir = tmp_path / "support-development"
    support_smoke_dir.mkdir()
    support_dev_dir.mkdir()
    (support_smoke_dir / "RESULT.json").write_text("{}")
    (support_dev_dir / "RESULT.json").write_text("{}")
    monkeypatch.setattr(daily, "SMOKE_DIR", smoke_dir)
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", naive_smoke_dir)
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", development_dir)
    monkeypatch.setattr(daily, "LEDGER", ledger_path)
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(daily.recovery_daily, "LEDGER", prior_path)
    monkeypatch.setattr(daily.recovery_daily, "SMOKE_DIR", support_smoke_dir)
    monkeypatch.setattr(daily.recovery_daily, "DEVELOPMENT_DIR", support_dev_dir)
    monkeypatch.setattr(
        daily,
        "preflight",
        lambda **kwargs: {
            "budget": {"spent_before_policy_usd": 0.9},
            "predecessor": {},
            "models": {"luna": {"status": "available"}},
        },
    )
    adapter_builds = []

    def fake_build_adapter(**kwargs):
        adapter_builds.append(("deepseek", kwargs))
        return _Adapter()

    def fake_build_naive_adapter(**kwargs):
        adapter_builds.append(("luna", kwargs))
        return _Adapter()

    monkeypatch.setattr(daily.policy, "build_adapter", fake_build_adapter)
    monkeypatch.setattr(daily.policy, "build_naive_adapter", fake_build_naive_adapter)
    monkeypatch.setattr(daily, "_budget_status", lambda *args, **kwargs: {"authorized": True})
    calls = []

    def fake_smoke(*, output_dir, **kwargs):
        calls.append("smoke")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "passed", "usage": {"run_cost_usd": 0.01}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    def fake_development(*, output_dir, **kwargs):
        calls.append("development")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "gated_null", "usage": {"combined_cost_usd": 0.03}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    def fake_naive_smoke(*, output_dir, **kwargs):
        calls.append("naive_smoke")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "passed", "usage": {"run_cost_usd": 0.01}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(daily.policy, "run_smoke", fake_smoke)
    monkeypatch.setattr(daily.policy, "run_naive_smoke", fake_naive_smoke)
    monkeypatch.setattr(daily.policy, "run_development", fake_development)
    _install_verifier(monkeypatch)

    result = daily.execute(live_reader=_live)

    assert calls == ["smoke", "naive_smoke", "development"]
    assert result["status"] == "complete_reconciled"
    assert result["development_status"] == "gated_null"
    assert result["independent_replay_passed"] is True
    ledger = json.loads(ledger_path.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.95)
    assert ledger["stages"]["enriched_smoke"]["status"] == "passed"
    assert ledger["stages"]["naive_smoke"]["status"] == "passed"
    assert ledger["stages"]["policy_development"]["status"] == "gated_null"
    development_builds = [
        kwargs for _, kwargs in adapter_builds if kwargs["stage"] == "development"
    ]
    assert len(development_builds) == 3
    assert len({kwargs["run_id"] for kwargs in development_builds}) == 1


def test_luna_catalog_unavailable_skips_baseline_and_runs_primary(
    tmp_path, monkeypatch
) -> None:
    prior_path = tmp_path / "prior-ledger.json"
    prior_path.write_text(json.dumps(_prior_ledger()))
    support_smoke_dir = tmp_path / "support-smoke"
    support_dev_dir = tmp_path / "support-development"
    support_smoke_dir.mkdir()
    support_dev_dir.mkdir()
    (support_smoke_dir / "RESULT.json").write_text("{}")
    (support_dev_dir / "RESULT.json").write_text("{}")
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", tmp_path / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(daily, "ROOT", tmp_path / "root")
    monkeypatch.setattr(daily.recovery_daily, "LEDGER", prior_path)
    monkeypatch.setattr(daily.recovery_daily, "SMOKE_DIR", support_smoke_dir)
    monkeypatch.setattr(daily.recovery_daily, "DEVELOPMENT_DIR", support_dev_dir)
    monkeypatch.setattr(
        daily,
        "preflight",
        lambda **kwargs: {
            "budget": {"spent_before_policy_usd": 0.9},
            "predecessor": {},
            "models": {
                "luna": {
                    "status": "unavailable",
                    "can_affect_primary_status": False,
                }
            },
        },
    )
    monkeypatch.setattr(daily.policy, "build_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(
        daily.policy,
        "build_naive_adapter",
        lambda **kwargs: pytest.fail("Luna adapter must not be constructed"),
    )
    monkeypatch.setattr(
        daily, "_budget_status", lambda *args, **kwargs: {"authorized": True}
    )
    calls = []

    def fake_smoke(*, output_dir, **kwargs):
        calls.append("smoke")
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "passed", "usage": {"run_cost_usd": 0.01}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    def fake_development(*, output_dir, **kwargs):
        calls.append("development")
        assert kwargs["naive_baseline_enabled"] is False
        assert kwargs["naive_adapter"] is None
        assert kwargs["naive_endpoint_adapter"] is None
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "gated_null", "usage": {"combined_cost_usd": 0.02}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(daily.policy, "run_smoke", fake_smoke)
    monkeypatch.setattr(
        daily.policy,
        "run_naive_smoke",
        lambda **kwargs: pytest.fail("naive smoke must not run"),
    )
    monkeypatch.setattr(daily.policy, "run_development", fake_development)
    _install_verifier(monkeypatch)

    result = daily.execute(live_reader=_live)

    assert calls == ["smoke", "development"]
    assert result["status"] == "complete_reconciled"
    assert result["naive_smoke_status"] == "unavailable_preflight"
    assert result["naive_baseline_enabled"] is False
    assert result["development_status"] == "gated_null"
    ledger = json.loads((tmp_path / "ledger.json").read_text())
    assert ledger["stages"]["naive_smoke"]["actual_cost_usd"] == 0.0


def test_naive_smoke_failure_disables_baseline_but_still_runs_primary(
    tmp_path, monkeypatch
) -> None:
    prior_path = tmp_path / "prior-ledger.json"
    prior_path.write_text(json.dumps(_prior_ledger()))
    support_smoke_dir = tmp_path / "support-smoke"
    support_dev_dir = tmp_path / "support-development"
    support_smoke_dir.mkdir()
    support_dev_dir.mkdir()
    (support_smoke_dir / "RESULT.json").write_text("{}")
    (support_dev_dir / "RESULT.json").write_text("{}")
    monkeypatch.setattr(daily, "SMOKE_DIR", tmp_path / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", tmp_path / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", tmp_path / "development")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(daily, "ROOT", tmp_path / "root")
    monkeypatch.setattr(daily.recovery_daily, "LEDGER", prior_path)
    monkeypatch.setattr(daily.recovery_daily, "SMOKE_DIR", support_smoke_dir)
    monkeypatch.setattr(daily.recovery_daily, "DEVELOPMENT_DIR", support_dev_dir)
    monkeypatch.setattr(
        daily,
        "preflight",
        lambda **kwargs: {
            "budget": {"spent_before_policy_usd": 0.9},
            "predecessor": {},
            "models": {"luna": {"status": "available"}},
        },
    )
    monkeypatch.setattr(daily.policy, "build_adapter", lambda **kwargs: _Adapter())
    monkeypatch.setattr(
        daily.policy, "build_naive_adapter", lambda **kwargs: _Adapter()
    )
    monkeypatch.setattr(
        daily, "_budget_status", lambda *args, **kwargs: {"authorized": True}
    )

    def fake_smoke(*, output_dir, **kwargs):
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "passed", "usage": {"run_cost_usd": 0.01}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    def fail_naive_smoke(*args, **kwargs):
        raise ValueError("descriptive baseline smoke failed")

    seen = {}

    def fake_development(*, output_dir, **kwargs):
        seen.update(kwargs)
        output_dir.mkdir(parents=True, exist_ok=True)
        result = {"status": "gated_null", "usage": {"combined_cost_usd": 0.02}}
        (output_dir / "RESULT.json").write_text(json.dumps(result))
        return result

    monkeypatch.setattr(daily.policy, "run_smoke", fake_smoke)
    monkeypatch.setattr(daily.policy, "run_naive_smoke", fail_naive_smoke)
    monkeypatch.setattr(daily.policy, "run_development", fake_development)
    _install_verifier(monkeypatch)

    result = daily.execute(live_reader=_live)

    assert result["status"] == "complete_reconciled"
    assert result["naive_smoke_status"] == "failed_closed"
    assert result["naive_baseline_enabled"] is False
    assert seen["naive_baseline_enabled"] is False
    assert seen["naive_adapter"] is None
    assert seen["naive_endpoint_adapter"] is None
    assert result["development_status"] == "gated_null"
