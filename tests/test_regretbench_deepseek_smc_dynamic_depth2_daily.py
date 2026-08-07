from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import regretbench_deepseek_smc_dynamic_depth2_daily as daily


NOW = datetime(2026, 8, 9, 12, tzinfo=ZoneInfo("Europe/London"))


def _live(usage: float = 100.5) -> dict[str, float]:
    return {
        "total_credits_usd": 130.0,
        "total_usage_usd": usage,
        "balance_usd": 130.0 - usage,
    }


def _support_ledger(recorded: float = 0.5) -> dict:
    return {
        "date": daily.DATE,
        "timezone": daily.TIMEZONE,
        "daily_cap_usd": daily.DAILY_CAP_USD,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": recorded,
        "account_wide_usage_counts_against_cap": True,
        "stage": {"status": "passed"},
    }


def _predecessor(recorded: float = 0.5) -> dict:
    return {
        "status": "authorized_smc_policy_development_only",
        "result_sha256": "result",
        "verification_sha256": "verification",
        "daily_result_sha256": "daily",
        "ledger_sha256": "ledger",
        "support_raw_sha256": "raw",
        "support_controls_sha256": "controls",
        "support_ledger": _support_ledger(recorded),
    }


def _catalog(*, include_luna: bool = True) -> dict:
    rows = [
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
            "pricing": {"prompt": "0.00000009", "completion": "0.00000018"},
        }
    ]
    if include_luna:
        rows.append(
            {
                "id": daily.transport.NAIVE_MODEL_ID,
                "architecture": {
                    "input_modalities": ["text", "image"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["reasoning", "structured_outputs"],
                "top_provider": {
                    "context_length": 1_050_000,
                    "max_completion_tokens": 128_000,
                },
                "pricing": {
                    "prompt": "0.0000001",
                    "completion": "0.0000006",
                },
            }
        )
    return {"data": rows}


def _install_paths(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "policy"
    monkeypatch.setattr(daily, "ROOT", root)
    monkeypatch.setattr(daily, "SMOKE_DIR", root / "smoke")
    monkeypatch.setattr(daily, "NAIVE_SMOKE_DIR", root / "naive-smoke")
    monkeypatch.setattr(daily, "DEVELOPMENT_DIR", root / "development")
    monkeypatch.setattr(daily, "DAILY_RESULT", root / "DAILY_RESULT.json")
    monkeypatch.setattr(daily, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(daily, "_forbidden_descendants", lambda: [])


def test_preflight_carries_aug9_account_spend_and_reserves_all_caps(
    tmp_path, monkeypatch
) -> None:
    _install_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "_validate_hashes", lambda: None)
    monkeypatch.setattr(daily, "validate_smc_predecessor", _predecessor)

    result = daily.preflight(
        now=NOW,
        live_reader=_live,
        catalog_reader=_catalog,
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["budget"]["spent_before_policy_usd"] == pytest.approx(0.5)
    assert result["budget"]["policy_worst_case_usd"] == pytest.approx(3.9)
    assert result["budget"]["remaining_after_full_caps_usd"] == pytest.approx(0.6)
    assert result["models"]["deepseek"]["policy_max_tokens"] == 2_400
    assert result["models"]["luna"]["status"] == "available"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0


def test_preflight_missing_luna_disables_only_descriptive_baseline(
    tmp_path, monkeypatch
) -> None:
    _install_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "_validate_hashes", lambda: None)
    monkeypatch.setattr(daily, "validate_smc_predecessor", _predecessor)

    result = daily.preflight(
        now=NOW,
        live_reader=_live,
        catalog_reader=lambda: _catalog(include_luna=False),
    )

    assert result["status"] == "ready_without_paid_calls"
    assert result["models"]["deepseek"]["status"] == "available"
    assert result["models"]["luna"]["status"] == "unavailable"
    assert result["models"]["luna"]["can_affect_primary_status"] is False


def test_preflight_refuses_when_cumulative_support_plus_policy_caps_do_not_fit(
    tmp_path, monkeypatch
) -> None:
    _install_paths(tmp_path, monkeypatch)
    monkeypatch.setattr(daily, "_validate_hashes", lambda: None)
    monkeypatch.setattr(
        daily, "validate_smc_predecessor", lambda: _predecessor(1.2)
    )

    with pytest.raises(RuntimeError, match="remaining account-wide day"):
        daily.preflight(
            now=NOW,
            live_reader=lambda: _live(101.2),
            catalog_reader=_catalog,
        )


def test_date_gate_precedes_catalog_or_credit_access(monkeypatch) -> None:
    touched = []

    def forbidden():
        touched.append(True)
        raise AssertionError("external preflight reader was reached")

    with pytest.raises(RuntimeError, match="only on 2026-08-09"):
        daily.preflight(
            now=datetime(
                2026, 8, 8, 23, 59, tzinfo=ZoneInfo("Europe/London")
            ),
            live_reader=forbidden,
            catalog_reader=forbidden,
        )
    assert touched == []


def test_completed_run_replays_without_catalog_credit_or_model_access(
    monkeypatch,
) -> None:
    touched = []
    monkeypatch.setattr(daily, "_validate_hashes", lambda: None)
    monkeypatch.setattr(
        daily,
        "_validated_existing_complete",
        lambda: {
            "status": "complete_reconciled",
            "resume_status": "already_complete_verified",
        },
    )
    monkeypatch.setattr(
        daily,
        "validate_smc_predecessor",
        lambda: touched.append("predecessor"),
    )

    def forbidden():
        touched.append("external")
        raise AssertionError("external reader was reached")

    result = daily.preflight(
        now=NOW,
        live_reader=forbidden,
        catalog_reader=forbidden,
    )

    assert result["status"] == "already_complete_verified"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert touched == []


class _Adapter:
    def __init__(self, label: str) -> None:
        self.label = label

    def usage_snapshot(self):
        return {
            "adapter_requests": 0,
            "http_attempts": 0,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
        }


def test_execute_verifies_smoke_before_development_and_naive_failure_cannot_veto(
    tmp_path, monkeypatch
) -> None:
    _install_paths(tmp_path, monkeypatch)
    order = []
    ready = {
        "status": "ready_without_paid_calls",
        "live_credits": _live(),
        "budget": {
            "opening_total_usage_boundary_usd": 100.0,
            "spent_before_policy_usd": 0.5,
        },
        "predecessor": {},
        "models": {"luna": {"status": "available"}},
    }
    monkeypatch.setattr(daily, "preflight", lambda **kwargs: ready)
    monkeypatch.setattr(
        daily,
        "build_deepseek_adapter",
        lambda **kwargs: _Adapter(f"deepseek-{kwargs['stage']}"),
    )
    monkeypatch.setattr(
        daily.transport,
        "build_naive_adapter",
        lambda **kwargs: _Adapter(f"luna-{kwargs['stage']}"),
    )

    def smoke(**kwargs):
        order.append("smoke")
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        (output / "RESULT.json").write_text(json.dumps({"status": "passed"}))
        return {
            "status": "passed",
            "usage": {"run_cost_usd": 0.0},
        }

    def verify_smoke(*args, **kwargs):
        order.append("verify_smoke")
        return {"status": "verified", "model_calls": 0, "mismatches": []}

    def naive_smoke(**kwargs):
        order.append("naive_smoke")
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        (output / "RESULT.json").write_text(json.dumps({"status": "passed"}))
        return {"status": "passed", "usage": {"run_cost_usd": 0.0}}

    def planning(**kwargs):
        order.append("planning")
        return {
            "contexts": [object()],
            "initial_supports": [{}],
            "privacy": [],
        }

    def realized(**kwargs):
        order.append("realized")
        return {"actual_privacy": [], "tasks": [], "usage": {}}

    def naive_baseline(**kwargs):
        order.append("naive_baseline")
        raise RuntimeError("descriptive baseline failed")

    def finalize(**kwargs):
        order.append("finalize")
        assert kwargs["naive_result"] is None
        assert kwargs["naive_error"]["error"] == "descriptive baseline failed"
        result = {
            "status": "gated_null",
            "authorizes": "nothing",
            "usage": {"combined_cost_usd": 0.0},
        }
        kwargs["output_dir"].mkdir(parents=True, exist_ok=True)
        (kwargs["output_dir"] / "RESULT.json").write_text(json.dumps(result))
        return result

    def verify_development(*args, **kwargs):
        order.append("verify_development")
        return {"status": "verified", "model_calls": 0, "mismatches": []}

    monkeypatch.setattr(daily.experiment, "run_smoke", smoke)
    monkeypatch.setattr(daily.verifier, "verify_smoke", verify_smoke)
    monkeypatch.setattr(daily.experiment, "run_naive_smoke", naive_smoke)
    monkeypatch.setattr(
        daily.experiment,
        "validate_policy_smoke",
        lambda path: {"path": str(path), "sha256": "smoke"},
    )
    monkeypatch.setattr(
        daily.experiment,
        "validate_naive_smoke",
        lambda path: {"path": str(path), "sha256": "naive"},
    )
    monkeypatch.setattr(
        daily.experiment, "build_development_planning_tree", planning
    )
    monkeypatch.setattr(daily.experiment, "run_realized_primary", realized)
    monkeypatch.setattr(daily.experiment, "run_naive_baseline", naive_baseline)
    monkeypatch.setattr(
        daily.experiment, "finalize_development_result", finalize
    )
    monkeypatch.setattr(daily.verifier, "verify", verify_development)

    result = daily.execute(now=NOW, live_reader=_live)

    assert result["status"] == "complete_reconciled"
    assert result["development_status"] == "gated_null"
    assert result["naive_baseline_enabled"] is True
    assert result["authorizes"] == "nothing"
    assert order.index("verify_smoke") < order.index("planning")
    assert order[-1] == "verify_development"
    ledger = json.loads(daily.LEDGER.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(0.5)
    assert ledger["stages"]["development"]["status"] == "gated_null"


def test_smoke_replay_failure_stops_before_any_development_call(
    tmp_path, monkeypatch
) -> None:
    _install_paths(tmp_path, monkeypatch)
    ready = {
        "status": "ready_without_paid_calls",
        "live_credits": _live(),
        "budget": {
            "opening_total_usage_boundary_usd": 100.0,
            "spent_before_policy_usd": 0.5,
        },
        "predecessor": {},
        "models": {"luna": {"status": "unavailable"}},
    }
    adapter = _Adapter("smoke")
    monkeypatch.setattr(daily, "preflight", lambda **kwargs: ready)
    monkeypatch.setattr(
        daily, "build_deepseek_adapter", lambda **kwargs: adapter
    )

    def smoke(**kwargs):
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        (output / "RESULT.json").write_text(json.dumps({"status": "passed"}))
        return {"status": "passed", "usage": {"run_cost_usd": 0.0}}

    development_opened = []
    monkeypatch.setattr(daily.experiment, "run_smoke", smoke)
    monkeypatch.setattr(
        daily.verifier,
        "verify_smoke",
        lambda *args, **kwargs: {
            "status": "verification_failed",
            "model_calls": 0,
            "mismatches": ["$.gates"],
        },
    )
    monkeypatch.setattr(
        daily.experiment,
        "build_development_planning_tree",
        lambda **kwargs: development_opened.append(True),
    )

    with pytest.raises(RuntimeError, match="smoke independent replay failed"):
        daily.execute(now=NOW, live_reader=_live)

    assert development_opened == []
    ledger = json.loads(daily.LEDGER.read_text())
    assert ledger["stages"]["enriched_smoke"]["status"] == "failed_closed"
    assert not daily.DEVELOPMENT_DIR.exists()
