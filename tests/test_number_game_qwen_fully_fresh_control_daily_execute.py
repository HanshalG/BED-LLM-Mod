from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import number_game_qwen_fully_fresh_control_daily_execute as execute


def _ledger(*, recorded: float = 0.0) -> dict:
    return {
        "schema_version": 1,
        "date": "2026-08-07",
        "timezone": "Europe/London",
        "daily_cap_usd": 5.0,
        "opening_total_usage_usd": 100.0,
        "recorded_actual_spend_usd": recorded,
        "first_authorized_block": {
            "description": "control",
            "maximum_cost_usd": 4.25,
            "actual_cost_usd": None,
            "status": "authorized_pending_later_day",
        },
        "additional_paid_blocks_authorized": False,
    }


def _result(
    *,
    cost: float = 3.2,
    requests: int = 3072,
    mechanics: bool = True,
) -> dict:
    return {
        "status": "gated_null",
        "decision": (
            "complete_composite_endpoint"
            if mechanics
            else "stop_at_control_mechanics"
        ),
        "usage": {
            "control_cost_usd": cost,
            "control_requests": requests,
        },
        "control": {
            "usage": {
                "run_cost_usd": cost,
                "adapter_requests": requests,
            },
            "mechanics_gates": {"mechanics": mechanics},
        },
    }


def _write_ledger(path: Path, *, recorded: float = 0.0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_ledger(recorded=recorded)), encoding="utf-8")


def _live_reader(*values: dict[str, float]):
    remaining = iter(values)
    return lambda: next(remaining)


def _live(usage: float) -> dict[str, float]:
    return {
        "total_credits_usd": 120.0,
        "total_usage_usd": usage,
        "balance_usd": 120.0 - usage,
    }


def _stage(result: dict, calls: list[dict]):
    def run(**kwargs):
        calls.append(kwargs)
        run_dir = kwargs["output_dir"]
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "RESULT.json").write_text(
            json.dumps(result), encoding="utf-8"
        )
        return result

    return run


def _verifier(calls: list[Path]):
    def run(*, run_dir: Path):
        calls.append(run_dir)
        verification = {"status": "verified"}
        (run_dir / "CONTROL_VERIFICATION.json").write_text(
            json.dumps(verification), encoding="utf-8"
        )
        return verification

    return run


def test_provider_lag_uses_local_measured_cost(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)
    stage_calls = []
    verifier_calls = []

    execution = execute.run_control_daily(
        run_dir=run_dir,
        run_id="daily",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.0), _live(100.0)),
        stage_runner=_stage(_result(), stage_calls),
        verifier=_verifier(verifier_calls),
    )

    ledger = json.loads(ledger_path.read_text())
    assert execution["status"] == "complete_verified"
    assert execution["verification_status"] == "verified"
    assert execution["recorded_actual_spend_usd"] == pytest.approx(3.2)
    assert execution["remaining_daily_allowance_usd"] == pytest.approx(1.8)
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.2)
    assert ledger["first_authorized_block"]["actual_cost_usd"] == 3.2
    assert ledger["first_authorized_block"]["status"] == "complete"
    assert ledger["additional_paid_blocks_authorized"]
    assert [
        item["model"]
        for item in ledger["authorized_tail_blocks"]
        if item["interface"]
        == "number-game-budget-model-reliability128-1"
    ] == list(execute.AUTHORIZED_RELIABILITY_TAILS)
    stress = ledger["authorized_tail_blocks"][-1]
    assert stress["interface"] == execute.STRESS_INTERFACE_VERSION
    assert stress["status"] == "waiting_for_reliability_results"
    assert len(stage_calls) == 1
    assert verifier_calls == [run_dir]
    assert (run_dir / "CONTROL_DAILY_EXECUTION.json").exists()


def test_posted_account_usage_dominates_local_cost(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)

    execution = execute.run_control_daily(
        run_dir=run_dir,
        run_id="posted",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.4), _live(104.6)),
        stage_runner=_stage(_result(cost=3.2), []),
        verifier=_verifier([]),
    )

    ledger = json.loads(ledger_path.read_text())
    assert execution["recorded_actual_spend_usd"] == pytest.approx(4.6)
    assert execution["remaining_daily_allowance_usd"] == pytest.approx(0.4)
    assert ledger["reconciliation"][
        "posted_spend_since_frozen_opening_usd"
    ] == pytest.approx(4.6)


def test_prior_recorded_spend_is_added_to_measured_control(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path, recorded=0.5)

    execute.run_control_daily(
        run_dir=run_dir,
        run_id="prior",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.5), _live(100.5)),
        stage_runner=_stage(_result(cost=3.2), []),
        verifier=_verifier([]),
    )

    ledger = json.loads(ledger_path.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.7)
    assert ledger["reconciliation"]["locally_measured_spend_usd"] == 3.7


def test_mechanics_failure_is_recorded_without_verifier(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)
    verifier_calls = []

    execution = execute.run_control_daily(
        run_dir=run_dir,
        run_id="mechanics",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.0), _live(103.0)),
        stage_runner=_stage(_result(cost=3.0, mechanics=False), []),
        verifier=_verifier(verifier_calls),
    )

    assert execution["status"] == "control_incomplete_or_mechanics_failed"
    assert execution["verification_status"] is None
    assert execution["recorded_actual_spend_usd"] == 3.0
    assert not execution["additional_paid_blocks_authorized"]
    assert verifier_calls == []


def test_stress_is_omitted_but_small_gates_remain_when_headroom_is_0_60(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)

    execution = execute.run_control_daily(
        run_dir=run_dir,
        run_id="no-tail",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.0), _live(104.4)),
        stage_runner=_stage(_result(cost=3.2), []),
        verifier=_verifier([]),
    )

    assert execution["remaining_daily_allowance_usd"] == pytest.approx(0.6)
    assert execution["additional_paid_blocks_authorized"]
    assert len(execution["authorized_tail_blocks"]) == 2
    assert all(
        item["interface"] == "number-game-budget-model-reliability128-1"
        for item in execution["authorized_tail_blocks"]
    )


def test_no_tail_is_authorized_when_headroom_is_below_0_20(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)

    execution = execute.run_control_daily(
        run_dir=run_dir,
        run_id="no-tail-at-all",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.0), _live(104.9)),
        stage_runner=_stage(_result(cost=3.2), []),
        verifier=_verifier([]),
    )

    assert execution["remaining_daily_allowance_usd"] == pytest.approx(0.1)
    assert not execution["additional_paid_blocks_authorized"]
    assert execution["authorized_tail_blocks"] == []


def test_wrong_request_count_does_not_invoke_verifier(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)
    verifier_calls = []

    execution = execute.run_control_daily(
        run_dir=run_dir,
        run_id="requests",
        ledger_path=ledger_path,
        live_reader=_live_reader(_live(100.0), _live(103.0)),
        stage_runner=_stage(_result(cost=3.0, requests=3071), []),
        verifier=_verifier(verifier_calls),
    )

    assert execution["status"] == "control_incomplete_or_mechanics_failed"
    assert verifier_calls == []


def test_verifier_failure_cannot_lose_spend_record(tmp_path: Path) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)

    def fail_verifier(**_):
        raise ValueError("replay drift")

    with pytest.raises(ValueError, match="replay drift"):
        execute.run_control_daily(
            run_dir=run_dir,
            run_id="verify-fail",
            ledger_path=ledger_path,
            live_reader=_live_reader(_live(100.0), _live(103.2)),
            stage_runner=_stage(_result(cost=3.2), []),
            verifier=fail_verifier,
        )

    ledger = json.loads(ledger_path.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.2)
    assert ledger["first_authorized_block"]["status"] == "complete"
    assert not (run_dir / "CONTROL_DAILY_EXECUTION.json").exists()


def test_post_run_credits_failure_cannot_lose_spend_record(
    tmp_path: Path,
) -> None:
    ledger_path = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _write_ledger(ledger_path)
    calls = 0

    def live_reader():
        nonlocal calls
        calls += 1
        if calls == 1:
            return _live(100.0)
        raise RuntimeError("credits unavailable")

    with pytest.raises(RuntimeError, match="credits unavailable"):
        execute.run_control_daily(
            run_dir=run_dir,
            run_id="credits-fail",
            ledger_path=ledger_path,
            live_reader=live_reader,
            stage_runner=_stage(_result(cost=3.2), []),
            verifier=_verifier([]),
        )

    ledger = json.loads(ledger_path.read_text())
    assert ledger["recorded_actual_spend_usd"] == pytest.approx(3.2)
    assert ledger["first_authorized_block"]["actual_cost_usd"] == 3.2
    assert ledger["first_authorized_block"]["status"] == "complete"


def test_negative_measured_cost_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        execute.reconcile_ledger(
            ledger=_ledger(),
            measured_control_cost_usd=-0.1,
            live_after=_live(100.0),
        )
