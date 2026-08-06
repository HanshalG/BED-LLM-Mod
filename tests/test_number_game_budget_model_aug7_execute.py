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
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "additional_paid_blocks_authorized": True,
        "authorized_tail_blocks": tails,
    }


def _pristine_ledger() -> dict:
    return {
        "date": "2026-08-07",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_credits_usd": 245.0,
        "opening_total_usage_usd": 217.25,
        "opening_balance_usd": 27.75,
        "opening_baseline_frozen_on_previous_closed_day": True,
        "recorded_actual_spend_usd": 0.0,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "additional_paid_blocks_authorized": False,
        "first_authorized_block": {
            "maximum_cost_usd": 4.25,
            "expected_cost_usd": 3.21,
            "actual_cost_usd": None,
            "status": "authorized_pending_later_day",
        },
    }


def _control_validator(*, run_dir: Path, **_) -> dict:
    path = run_dir / "CONTROL_DAILY_EXECUTION.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verified": True,
        "status": payload["status"],
        "decision": payload.get("decision"),
        "artifact_sha256": reliability.sha256_file(path),
    }


def _reliability_validator(path: Path, model: str) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verified": True,
        "model": model,
        "status": payload["status"],
        "decision": payload.get("decision"),
        "artifact_sha256": reliability.sha256_file(path),
    }


def _stress_validator(path: Path, _) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "verified": True,
        "status": payload["status"],
        "decision": payload.get("decision"),
        "selected_model": (payload.get("selection") or {}).get(
            "selected_model"
        ),
        "artifact_sha256": reliability.sha256_file(path),
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
        "control_validator": _control_validator,
        "reliability_validator": _reliability_validator,
        "stress_validator": _stress_validator,
    }


def _preflight_paths(tmp_path: Path) -> dict:
    paths = _paths(tmp_path)
    _write(paths["daily_ledger"], _pristine_ledger())
    for name in (
        "control_validator",
        "reliability_validator",
        "stress_validator",
    ):
        paths.pop(name)
    return paths


def _source_validator(path: Path) -> dict:
    return {"verified": True, "run_dir": str(path)}


def _case_validator() -> dict:
    return {
        "reliability_case_count_per_model": 128,
        "stress_case_count": 3_584,
    }


def _live_credits(*, usage: float = 217.25) -> dict[str, float]:
    return {
        "total_credits_usd": 275.0,
        "total_usage_usd": usage,
        "balance_usd": 275.0 - usage,
    }


def test_preflight_is_read_only_and_reports_exact_budget_pack(
    tmp_path: Path,
) -> None:
    paths = _preflight_paths(tmp_path)
    before = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    result = execute.preflight_aug7_sequence(
        **paths,
        live_reader=_live_credits,
        source_validator=_source_validator,
        case_validator=_case_validator,
    )

    after = {
        path: path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }
    assert before == after
    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == 0
    assert result["files_written"] == 0
    assert result["budget"]["expected_full_sequence_cost_usd"] == 4.96
    assert result["budget"]["expected_full_sequence_slack_usd"] == (
        pytest.approx(0.04)
    )
    assert result["budget"]["maximum_control_cost_for_full_stress_usd"] == (
        pytest.approx(3.25)
    )


def test_preflight_rejects_usage_after_frozen_opening(tmp_path: Path) -> None:
    paths = _preflight_paths(tmp_path)

    with pytest.raises(RuntimeError, match="usage advanced"):
        execute.preflight_aug7_sequence(
            **paths,
            live_reader=lambda: _live_credits(usage=217.250001),
            source_validator=_source_validator,
            case_validator=_case_validator,
        )


def test_preflight_rejects_partial_tail_without_writing(tmp_path: Path) -> None:
    paths = _preflight_paths(tmp_path)
    partial = paths["reliability_root"] / next(
        iter(execute.RELIABILITY_RUNS.values())
    )
    _write(partial / "RAW_RESPONSES.json", {"partial": True})
    before = (partial / "RAW_RESPONSES.json").read_bytes()

    with pytest.raises(RuntimeError, match="partial or terminal reliability"):
        execute.preflight_aug7_sequence(
            **paths,
            live_reader=_live_credits,
            source_validator=_source_validator,
            case_validator=_case_validator,
        )

    assert (partial / "RAW_RESPONSES.json").read_bytes() == before
    assert not paths["output_dir"].exists()


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
                item["actual_cost_usd"] = 0.04
        ledger["additional_paid_blocks_authorized"] = any(
            item["status"]
            in {"authorized_pending", "waiting_for_reliability_results"}
            for item in ledger["authorized_tail_blocks"]
        )
        _write(ledger_path, ledger)
        return artifact

    return run


def _stress_runner(calls: list[str]):
    def run(
        *,
        output_dir: Path,
        reliability_paths: dict,
        ledger_path: Path,
        **_,
    ):
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
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        entry = ledger["authorized_tail_blocks"][-1]
        entry["status"] = "passed"
        entry["model"] = result["selection"]["selected_model"]
        entry["actual_cost_usd"] = 0.3
        ledger["additional_paid_blocks_authorized"] = False
        ledger["recorded_actual_spend_usd"] = 3.5
        _write(ledger_path, ledger)
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


def test_resume_after_banked_stress_does_not_misclassify_authorization(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = []
    first = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=_stress_runner(calls),
    )
    (paths["output_dir"] / "RESULT.json").unlink()

    resumed = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=lambda **_: pytest.fail("control repeated"),
        reliability_runner=lambda **_: pytest.fail("gate repeated"),
        stress_runner=lambda **_: pytest.fail("stress repeated"),
    )

    assert resumed["decision"] == first["decision"] == "scale_reliable"
    assert resumed["selected_model"] == first["selected_model"]
    assert calls.count("stress") == 1


def test_completed_wrapper_refuses_changed_model_artifact(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = []
    execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=_control_runner(calls),
        reliability_runner=_reliability_runner(calls),
        stress_runner=_stress_runner(calls),
    )
    luna_path = paths["reliability_root"] / (
        "number-game-budget-model-reliability128-luna-20260807/RESULT.json"
    )
    luna = json.loads(luna_path.read_text(encoding="utf-8"))
    luna["decision"] = "tampered"
    _write(luna_path, luna)

    with pytest.raises(RuntimeError, match="component changed"):
        execute.execute_aug7_sequence(
            **paths,
            now=NOW,
            control_runner=lambda **_: pytest.fail("control repeated"),
            reliability_runner=lambda **_: pytest.fail("gate repeated"),
            stress_runner=lambda **_: pytest.fail("stress repeated"),
        )
    assert calls == [
        "control",
        "openai/gpt-5.6-luna",
        "deepseek/deepseek-v4-flash-0731",
        "stress",
    ]


def test_verified_control_without_gate_headroom_stops_before_model_calls(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = []

    def control_without_headroom(*, ledger_path: Path, **kwargs):
        result = _control_runner(calls)(ledger_path=ledger_path, **kwargs)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["additional_paid_blocks_authorized"] = False
        ledger["authorized_tail_blocks"] = []
        _write(ledger_path, ledger)
        return result

    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=control_without_headroom,
        reliability_runner=lambda **_: pytest.fail("gate must remain closed"),
        stress_runner=lambda **_: pytest.fail("stress must remain closed"),
    )

    assert calls == ["control"]
    assert result["status"] == "complete"
    assert result["decision"] == (
        "reliability_not_authorized_by_control_headroom"
    )


def test_incomplete_control_is_hash_bound_and_stops_before_model_calls(
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    calls = []

    def incomplete_control(*, run_dir: Path, ledger_path: Path, **_):
        calls.append("control")
        execution = {
            "interface_version": execute.control.INTERFACE_VERSION,
            "status": "control_incomplete_or_mechanics_failed",
            "decision": "control_null",
        }
        _write(run_dir / "CONTROL_DAILY_EXECUTION.json", execution)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["additional_paid_blocks_authorized"] = False
        ledger["authorized_tail_blocks"] = []
        _write(ledger_path, ledger)
        return execution

    result = execute.execute_aug7_sequence(
        **paths,
        now=NOW,
        control_runner=incomplete_control,
        reliability_runner=lambda **_: pytest.fail("gate must remain closed"),
    )

    assert calls == ["control"]
    assert result["status"] == "stopped_after_control"
    assert result["components"]["control"]["verification"]["verified"] is False


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

    def failed_stress(*, output_dir: Path, ledger_path: Path, **_):
        calls.append("stress")
        output_dir.mkdir(parents=True, exist_ok=True)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["authorized_tail_blocks"][-1]["status"] = (
            "failed_closed_posted_spend_reconciled"
        )
        ledger["additional_paid_blocks_authorized"] = False
        _write(ledger_path, ledger)
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

    def result_then_error(*, output_dir: Path, ledger_path: Path, **_):
        calls.append("stress")
        result = {
            "status": "passed",
            "decision": "scale_reliable",
            "selection": {"selected_model": "openai/gpt-5.6-luna"},
        }
        _write(output_dir / "RESULT.json", result)
        ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        ledger["authorized_tail_blocks"][-1].update(
            {
                "status": "passed",
                "model": "openai/gpt-5.6-luna",
                "actual_cost_usd": 0.3,
            }
        )
        ledger["additional_paid_blocks_authorized"] = False
        _write(ledger_path, ledger)
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
