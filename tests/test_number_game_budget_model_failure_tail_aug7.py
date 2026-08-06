from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_budget_model_failure_tail_aug7 as tail


def _now() -> datetime:
    return datetime(2026, 8, 7, 12, tzinfo=ZoneInfo(tail.TIMEZONE))


def _copy_ledger(tmp_path: Path, monkeypatch) -> Path:
    path = tmp_path / "ledger.json"
    ledger = json.loads(tail.aug7.DAILY_LEDGER.read_text(encoding="utf-8"))
    ledger["additional_paid_blocks_authorized"] = False
    ledger.pop("authorized_tail_blocks", None)
    ledger.pop("tail_authorization_reason", None)
    ledger.pop("failure_tail_authorization_amendment_sha256", None)
    ledger.pop("control_status_is_immutable", None)
    for key in list(ledger):
        if key.startswith("budget_model_reliability128_"):
            ledger.pop(key)
    ledger["recorded_actual_spend_usd"] = 2.6901734400000024
    ledger["reconciliation"]["remaining_daily_allowance_usd"] = 2.30982656
    path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(tail, "INITIAL_LEDGER_SHA256", tail._sha256(path))
    return path


def test_frozen_control_failure_boundary_is_transport_clean_and_endpoint_blind() -> None:
    result = tail.validate_control_failure_boundary()
    assert result["verified"] is True
    assert result["accepted_requests"] == 3_072
    assert result["endpoint_accessed"] is False
    assert result["failure_values_used_for_authorization"] is False


def test_authorization_uses_only_fixed_tail_and_remaining_allowance(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    ledger = tail.authorize_ledger(
        ledger_path=ledger_path,
        reliability_root=tmp_path / "reliability",
        stress_output_dir=tmp_path / "stress",
    )
    assert ledger["additional_paid_blocks_authorized"] is True
    assert len(ledger["authorized_tail_blocks"]) == 3
    assert sum(
        item["maximum_cost_usd"] for item in ledger["authorized_tail_blocks"]
    ) == pytest.approx(1.75)
    assert all(
        item["efficacy_used_for_authorization"] is False
        for item in ledger["authorized_tail_blocks"]
    )


def test_authorization_refuses_insufficient_remaining_allowance(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["reconciliation"]["remaining_daily_allowance_usd"] = 1.74
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(tail, "INITIAL_LEDGER_SHA256", tail._sha256(ledger_path))
    with pytest.raises(ValueError, match="cannot fund"):
        tail.authorize_ledger(
            ledger_path=ledger_path,
            reliability_root=tmp_path / "reliability",
            stress_output_dir=tmp_path / "stress",
        )


def test_authorization_refuses_nonpristine_paid_path(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    reliability_root = tmp_path / "reliability"
    first_run = next(iter(tail.aug7.RELIABILITY_RUNS.values()))
    dirty = reliability_root / first_run
    dirty.mkdir(parents=True)
    (dirty / "partial.txt").write_text("partial", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not pristine"):
        tail.authorize_ledger(
            ledger_path=ledger_path,
            reliability_root=reliability_root,
            stress_output_dir=tmp_path / "stress",
        )


def test_wrong_date_refuses_before_authorization(tmp_path) -> None:
    with pytest.raises(RuntimeError, match="restricted to Aug 7"):
        tail.execute_failure_tail(
            output_dir=tmp_path / "wrapper",
            ledger_path=tmp_path / "ledger.json",
            reliability_root=tmp_path / "reliability",
            stress_output_dir=tmp_path / "stress",
            now=datetime(2026, 8, 8, 0, tzinfo=ZoneInfo(tail.TIMEZONE)),
        )


def test_existing_wrapper_result_is_resumable_without_calls(tmp_path) -> None:
    output = tmp_path / "wrapper"
    output.mkdir()
    expected = {"status": "complete", "sentinel": 7}
    (output / "RESULT.json").write_text(json.dumps(expected), encoding="utf-8")
    assert tail.execute_failure_tail(
        output_dir=output,
        ledger_path=tmp_path / "missing.json",
        reliability_root=tmp_path / "reliability",
        stress_output_dir=tmp_path / "stress",
        now=_now(),
    ) == expected


def test_preflight_is_read_only_and_reports_exact_slack(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    before = ledger_path.read_bytes()
    result = tail.preflight_failure_tail(
        ledger_path=ledger_path,
        reliability_root=tmp_path / "reliability",
        stress_output_dir=tmp_path / "stress",
        live_reader=lambda: {
            "total_credits_usd": 245.0,
            "total_usage_usd": 219.823780823,
            "balance_usd": 25.176219177,
        },
        model_catalog_reader=lambda: _model_catalog(),
        now=_now(),
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["maximum_failure_tail_cost_usd"] == pytest.approx(1.75)
    assert result["post_tail_minimum_slack_usd"] == pytest.approx(
        0.55982656
    )
    assert result["model_calls_made"] == result["files_written"] == 0
    assert ledger_path.read_bytes() == before


def _model_catalog() -> dict:
    return {
        "data": [
            {
                "id": model,
                "context_length": 1_048_576,
                "top_provider": {"max_completion_tokens": 131_072},
                "architecture": {"input_modalities": ["text", "image"]},
                "supported_parameters": ["structured_outputs"],
                "pricing": {
                    "prompt": str(prompt / 1_000_000),
                    "completion": str(completion / 1_000_000),
                },
            }
            for model, prompt, completion in (
                ("qwen/qwen3.7-plus", 0.32, 1.28),
                ("openai/gpt-5.6-luna", 0.10, 0.60),
                ("deepseek/deepseek-v4-flash-0731", 0.09, 0.18),
            )
        ]
    }


def test_end_to_end_runs_both_models_then_stress(tmp_path, monkeypatch) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    reliability_root = tmp_path / "reliability"
    stress_dir = tmp_path / "stress"
    calls = []

    def fake_one(*, model, output_dir, ledger_path, **kwargs):
        calls.append(model)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        for item in ledger["authorized_tail_blocks"]:
            if item.get("model") == model:
                item["status"] = "passed"
                item["actual_cost_usd"] = 0.01
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        output_dir.mkdir(parents=True)
        (output_dir / "RESULT.json").write_text("{}", encoding="utf-8")
        return {"verified": True, "model": model, "status": "passed"}

    def fake_stress(*, output_dir, ledger_path, **kwargs):
        calls.append("stress")
        output_dir.mkdir(parents=True)
        (output_dir / "RESULT.json").write_text("{}", encoding="utf-8")
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["recorded_actual_spend_usd"] = 4.2
        ledger["reconciliation"]["remaining_daily_allowance_usd"] = 0.8
        ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
        return {"status": "passed"}

    monkeypatch.setattr(tail, "execute_one_reliability", fake_one)

    def fake_replay_stress(*, result_path, reliability_paths):
        assert result_path == stress_dir / "RESULT.json"
        assert reliability_paths == tail._reliability_paths(reliability_root)
        return {"verified": True, "status": "passed"}

    monkeypatch.setattr(
        tail.stress, "replay_stress_result", fake_replay_stress
    )
    result = tail.execute_failure_tail(
        output_dir=tmp_path / "wrapper",
        ledger_path=ledger_path,
        reliability_root=reliability_root,
        stress_output_dir=stress_dir,
        stress_executor=fake_stress,
        now=_now(),
    )
    assert calls == [*tail.aug7.RELIABILITY_RUNS, "stress"]
    assert result["status"] == "complete"
    assert result["qwen_control_status_changed"] is False
    assert result["authorizes_aug8_diversity"] is False


def test_first_model_exception_still_runs_second_and_closes_stress(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    calls = []
    first = next(iter(tail.aug7.RELIABILITY_RUNS))

    def fake_one(*, model, **kwargs):
        calls.append(model)
        if model == first:
            raise RuntimeError("first model failed")
        return {"verified": True, "model": model, "status": "passed"}

    def forbidden_stress(**kwargs):
        raise AssertionError("stress must stay closed without both results")

    monkeypatch.setattr(tail, "execute_one_reliability", fake_one)
    result = tail.execute_failure_tail(
        output_dir=tmp_path / "wrapper",
        ledger_path=ledger_path,
        reliability_root=tmp_path / "reliability",
        stress_output_dir=tmp_path / "stress",
        stress_executor=forbidden_stress,
        now=_now(),
    )
    assert calls == list(tail.aug7.RELIABILITY_RUNS)
    assert result["status"] == "failed_closed"
    assert first in result["reliability_failures"]
    assert result["stress_verification"] is None


def test_post_call_resume_replays_existing_stress_without_execution(
    tmp_path, monkeypatch
) -> None:
    ledger_path = _copy_ledger(tmp_path, monkeypatch)
    reliability_root = tmp_path / "reliability"
    stress_dir = tmp_path / "stress"
    tail.authorize_ledger(
        ledger_path=ledger_path,
        reliability_root=reliability_root,
        stress_output_dir=stress_dir,
    )
    stress_dir.mkdir()
    stress_result = {"status": "gated_null", "decision": "no_eligible_model"}
    (stress_dir / "RESULT.json").write_text(
        json.dumps(stress_result), encoding="utf-8"
    )

    monkeypatch.setattr(
        tail,
        "execute_one_reliability",
        lambda **kwargs: {
            "verified": True,
            "model": kwargs["model"],
            "status": "gated_null",
        },
    )

    def forbidden_stress(**kwargs):
        raise AssertionError("existing stress bytes must only be replayed")

    monkeypatch.setattr(
        tail.stress,
        "replay_stress_result",
        lambda **kwargs: {"verified": True, "status": "gated_null"},
    )
    result = tail.execute_failure_tail(
        output_dir=tmp_path / "wrapper",
        ledger_path=ledger_path,
        reliability_root=reliability_root,
        stress_output_dir=stress_dir,
        stress_executor=forbidden_stress,
        now=_now(),
    )
    assert result["status"] == "complete"
    assert result["stress_status"] == "gated_null"
