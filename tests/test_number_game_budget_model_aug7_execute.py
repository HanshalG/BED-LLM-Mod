from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_budget_model_aug7_execute as execute
from scripts import number_game_budget_model_reliability128 as reliability
from scripts import number_game_budget_model_stress3584 as stress


NOW = datetime(2026, 8, 7, 12, tzinfo=ZoneInfo("Europe/London"))


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _ledger(*, stress_authorized: bool = True) -> dict:
    tails = [
        {
            "interface": reliability.INTERFACE_VERSION,
            "model": model,
            "maximum_cost_usd": 0.1,
            "status": "authorized_pending",
        }
        for model in execute.RELIABILITY_RUNS
    ]
    if stress_authorized:
        tails.append(
            {
                "interface": stress.INTERFACE_VERSION,
                "model": None,
                "maximum_cost_usd": stress.RUN_BUDGET_USD,
                "status": "waiting_for_reliability_results",
            }
        )
    return {
        "date": "2026-08-07",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": 3.2,
        "additional_paid_blocks_authorized": True,
        "authorized_tail_blocks": tails,
    }


def _paths(tmp_path: Path) -> dict:
    ledger = tmp_path / "ledger.json"
    _write(ledger, _ledger())
    return {
        "output_dir": tmp_path / "sequence",
        "control_run_dir": tmp_path / "control",
        "daily_ledger": ledger,
        "reliability_root": tmp_path / "reliability",
        "stress_output_dir": tmp_path / "stress",
    }


def _control_runner(calls: list[str]):
    def run(*, run_dir: Path, ledger_path: Path, **_):
        calls.append("control")
        result = {
            "decision": "complete_composite_endpoint",
            "control": {
                "usage": {"adapter_requests": 3072},
                "mechanics_gates": {"all": True},
            },
        }
        _write(run_dir / "RESULT.json", result)
        execution = {"status": "complete_verified", "decision": "complete"}
        _write(run_dir / "CONTROL_DAILY_EXECUTION.json", execution)
        return execution

    return run


def _reliability_runner(calls: list[str], *, fail_model: str | None = None):
    def run(*, output_dir: Path, model_id: str, ledger_path: Path, **_):
        calls.append(model_id)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        if model_id == fail_model:
            artifact = {
                "interface_version": reliability.INTERFACE_VERSION,
                "status": "failed_closed",
                "model": model_id,
            }
            _write(output_dir / "FAILURE.json", artifact)
            status = "failed_closed_posted_spend_reconciled"
            for item in ledger["authorized_tail_blocks"]:
                if item.get("model") == model_id:
                    item["status"] = status
            _write(ledger_path, ledger)
            raise RuntimeError("model failed")
        artifact = {
            "status": "passed",
            "decision": "eligible",
            "protocol": {
                "interface_version": reliability.INTERFACE_VERSION,
                "model": model_id,
            },
            "gates": {"all_pass": True},
            "conditioned_valid_summary": {
                "minimum": 8,
                "mean": 12.0,
                "maximum": 24,
            },
            "initial_parse_failures": 0,
            "format_retry_requests": 0,
            "usage": {
                "forced_exits": 0,
                "retry_count": 0,
                "run_cost_usd": 0.04,
            },
        }
        _write(output_dir / "RESULT.json", artifact)
        for item in ledger["authorized_tail_blocks"]:
            if item.get("model") == model_id:
                item["status"] = "passed"
        _write(ledger_path, ledger)
        return artifact

    return run


def _stress_runner(calls: list[str]):
    def run(*, output_dir: Path, reliability_paths: dict, **_):
        calls.append("stress")
        assert set(reliability_paths) == set(execute.RELIABILITY_RUNS)
        result = {
            "status": "passed",
            "decision": "scale_reliable",
            "selection": {
                "selected_model": "deepseek/deepseek-v4-flash-0731"
            },
        }
        _write(output_dir / "RESULT.json", result)
        return result

    return run


def test_sequence_runs_all_components_in_exact_order(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls = []
    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=_stress_runner(calls),
    )

    assert calls == [
        "control",
        "openai/gpt-5.6-luna",
        "deepseek/deepseek-v4-flash-0731",
        "stress",
    ]
    assert result["status"] == "complete"
    assert result["selected_model"] == "deepseek/deepseek-v4-flash-0731"
    assert (paths["output_dir"] / "RESULT.json").exists()


def test_banked_failed_model_does_not_block_other_gate(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls = []
    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(
            calls, fail_model="openai/gpt-5.6-luna"
        ),
        stress_runner=_stress_runner(calls),
    )

    assert calls[-1] == "stress"
    assert result["components"]["openai/gpt-5.6-luna"]["status"] == (
        "failed_closed"
    )
    assert result["status"] == "complete"


def test_stress_is_skipped_when_control_did_not_authorize_headroom(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _ledger(stress_authorized=False))
    calls = []
    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=lambda **_: pytest.fail("stress must not run"),
    )

    assert result["decision"] == "stress_not_authorized_by_control_headroom"
    assert "stress" not in calls


def test_resume_never_repeats_banked_components(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls = []
    execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=_stress_runner(calls),
    )
    first_calls = list(calls)

    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=lambda **_: pytest.fail("control repeated"),
        reliability_runner=lambda **_: pytest.fail("gate repeated"),
        stress_runner=lambda **_: pytest.fail("stress repeated"),
    )

    assert calls == first_calls
    assert result["status"] == "complete"


def test_wrong_calendar_day_refuses_before_components(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls = []
    with pytest.raises(RuntimeError, match="only on August 7"):
        execute.execute_aug7_sequence(
            **paths,
            now=datetime(
                2026, 8, 8, 0, 1, tzinfo=ZoneInfo("Europe/London")
            ),
            control_runner=_control_runner(calls),
        )
    assert calls == []


def test_stress_failure_is_banked_without_reexecution(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls = []

    def failed_stress(*, output_dir: Path, **_):
        calls.append("stress")
        output_dir.mkdir(parents=True, exist_ok=True)
        raise RuntimeError("stress transport failed")

    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=failed_stress,
    )

    assert result["status"] == "failed_closed"
    assert (paths["stress_output_dir"] / "FAILURE.json").exists()
    assert calls.count("stress") == 1


def test_control_failure_is_banked_at_wrapper_boundary(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    calls = []

    def failed_control(**_):
        calls.append("control")
        raise RuntimeError("control failed")

    with pytest.raises(RuntimeError, match="control failed"):
        execute.execute_aug7_sequence(
            **paths,
            now=NOW,
            control_runner=failed_control,
        )

    failure = paths["control_run_dir"] / (
        "CONTROL_DAILY_EXECUTION_FAILURE.json"
    )
    assert json.loads(failure.read_text(encoding="utf-8"))["status"] == (
        "failed_closed"
    )
    assert calls == ["control"]


def test_stress_result_survives_post_result_reconciliation_error(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = []

    def result_then_error(*, output_dir: Path, **_):
        calls.append("stress")
        result = {
            "status": "passed",
            "decision": "scale_reliable",
            "selection": {"selected_model": "openai/gpt-5.6-luna"},
        }
        _write(output_dir / "RESULT.json", result)
        raise RuntimeError("posted credits read failed")

    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=result_then_error,
    )

    assert result["status"] == "complete"
    assert result["selected_model"] == "openai/gpt-5.6-luna"
    assert (paths["stress_output_dir"] / "RESULT.json").exists()
    assert not (paths["stress_output_dir"] / "FAILURE.json").exists()
