from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_two_draw_diversity_bonus_confirmation64_daily_execute as execute
from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged


LONDON = ZoneInfo("Europe/London")


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _control(path: Path, *, verified: bool = True) -> None:
    _write(
        path,
        {
            "status": "complete_verified" if verified else "failed_closed",
            "verification_status": "verified" if verified else None,
            "measured_control_requests": 3072 if verified else 0,
        },
    )


def _live(usage: float = 100.0) -> dict[str, float]:
    return {
        "total_credits_usd": 130.0,
        "total_usage_usd": usage,
        "balance_usd": 130.0 - usage,
    }


def test_block_a_initializes_exact_ledger_and_executes_once(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    calls = []

    def block_executor(**kwargs):
        calls.append(kwargs)
        execution = {
            "status": "complete",
            "block": "a",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_A_DAILY_EXECUTION.json", execution)
        return execution

    result = execute.execute_formal_block(
        block="a",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
        live_reader=lambda: _live(),
        block_executor=block_executor,
    )

    assert result["status"] == "complete"
    assert len(calls) == 1
    opened = json.loads(ledger.read_text(encoding="utf-8"))
    assert opened["date"] == "2026-08-08"
    assert opened["daily_cap_usd"] == 5.0
    assert opened["opening_total_usage_usd"] == 100.0
    assert opened["first_authorized_block"]["block"] == "a"
    assert opened["first_authorized_block"]["status"] == "complete"
    assert opened["first_authorized_block"]["actual_cost_usd"] == 4.2
    assert not opened["additional_paid_blocks_authorized"]

    execute.execute_formal_block(
        block="a",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 8, 1, 0, tzinfo=LONDON),
        live_reader=lambda: pytest.fail("resume read live credits"),
        block_executor=lambda **_: pytest.fail("resume repeated block"),
    )
    assert len(calls) == 1


def test_wrong_day_refuses_before_live_or_ledger(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    _control(control)
    with pytest.raises(RuntimeError, match="only on 2026-08-08"):
        execute.execute_formal_block(
            block="a",
            run_dir=tmp_path / "run",
            ledger_path=tmp_path / "ledger.json",
            control_execution_path=control,
            now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("wrong day read live credits"),
        )
    assert not (tmp_path / "ledger.json").exists()


def test_unverified_aug7_control_refuses_before_opening_day(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    _control(control, verified=False)
    with pytest.raises(RuntimeError, match="not complete and verified"):
        execute.execute_formal_block(
            block="a",
            run_dir=tmp_path / "run",
            ledger_path=tmp_path / "ledger.json",
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("invalid predecessor read live"),
        )


def test_partial_formal_block_is_never_retried(tmp_path: Path) -> None:
    control = tmp_path / "control.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "block_a" / "source" / "partial.json", {})

    with pytest.raises(RuntimeError, match="partial Block A"):
        execute.execute_formal_block(
            block="a",
            run_dir=run_dir,
            ledger_path=tmp_path / "ledger.json",
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: pytest.fail("partial block read live"),
        )


def test_failed_block_reconciles_posted_spend_and_banks_failure(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    live_values = iter([_live(100.0), _live(103.7)])

    with pytest.raises(RuntimeError, match="provider failed"):
        execute.execute_formal_block(
            block="a",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
            live_reader=lambda: next(live_values),
            block_executor=lambda **_: (_ for _ in ()).throw(
                RuntimeError("provider failed")
            ),
        )

    reconciled = json.loads(ledger.read_text(encoding="utf-8"))
    assert reconciled["recorded_actual_spend_usd"] == pytest.approx(3.7)
    assert (
        reconciled["diversity_bonus_confirmation64_block_a"]["status"]
        == "failed_closed_posted_spend_reconciled"
    )
    failure = json.loads(
        (run_dir / "BLOCK_A_RUNNER_FAILURE.json").read_text(encoding="utf-8")
    )
    assert failure["status"] == "failed_closed"
    assert reconciled["first_authorized_block"]["status"] == "failed_closed"


def test_started_ledger_is_never_reused_after_ambiguous_crash(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger_path = tmp_path / "ledger.json"
    _control(control)
    ledger = execute.initialize_daily_ledger(
        path=ledger_path,
        block="a",
        live=_live(),
        local_now=datetime(2026, 8, 8, 0, 1, tzinfo=LONDON),
    )
    ledger["first_authorized_block"]["status"] = "execution_started"
    _write(ledger_path, ledger)

    with pytest.raises(RuntimeError, match="authorization is not reusable"):
        execute.execute_formal_block(
            block="a",
            run_dir=tmp_path / "run",
            ledger_path=ledger_path,
            control_execution_path=control,
            now=datetime(2026, 8, 8, 0, 2, tzinfo=LONDON),
            live_reader=lambda: _live(),
            block_executor=lambda **_: pytest.fail("ambiguous block rerun"),
        )


def test_block_b_writes_verified_report_before_completion(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "BLOCK_A_STAGE.json", {"status": "banked"})
    calls = []

    def block_executor(**_):
        execution = {
            "status": "complete",
            "block": "b",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_B_DAILY_EXECUTION.json", execution)
        return execution

    def reporter(*, run_dir: Path):
        calls.append(run_dir)
        return {"path": "report.md", "sha256": "report-hash"}

    result = execute.execute_formal_block(
        block="b",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
        live_reader=lambda: _live(),
        block_executor=block_executor,
        reporter=reporter,
    )

    assert calls == [run_dir]
    assert result["verified_report"]["sha256"] == "report-hash"
    stored = json.loads(
        (run_dir / "BLOCK_B_DAILY_EXECUTION.json").read_text(encoding="utf-8")
    )
    assert stored["verified_report"] == result["verified_report"]


def test_report_failure_recovers_without_repeating_paid_block(
    tmp_path: Path,
) -> None:
    control = tmp_path / "control.json"
    ledger = tmp_path / "ledger.json"
    run_dir = tmp_path / "run"
    _control(control)
    _write(run_dir / "BLOCK_A_STAGE.json", {"status": "banked"})
    block_calls = []

    def block_executor(**_):
        block_calls.append("b")
        execution = {
            "status": "complete",
            "block": "b",
            "measured_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "measured_cost_usd": 4.2,
        }
        _write(run_dir / "BLOCK_B_DAILY_EXECUTION.json", execution)
        return execution

    with pytest.raises(RuntimeError, match="report failed"):
        execute.execute_formal_block(
            block="b",
            run_dir=run_dir,
            ledger_path=ledger,
            control_execution_path=control,
            now=datetime(2026, 8, 9, 0, 1, tzinfo=LONDON),
            live_reader=lambda: _live(),
            block_executor=block_executor,
            reporter=lambda **_: (_ for _ in ()).throw(
                RuntimeError("report failed")
            ),
        )

    assert block_calls == ["b"]
    assert not (run_dir / "BLOCK_B_RUNNER_FAILURE.json").exists()
    report_failure = json.loads(
        (run_dir / "BLOCK_B_REPORT_FAILURE.json").read_text(encoding="utf-8")
    )
    assert report_failure["block_execution_remains_complete"]
    current_ledger = json.loads(ledger.read_text(encoding="utf-8"))
    assert current_ledger["first_authorized_block"]["status"] == "complete"

    recovered = execute.execute_formal_block(
        block="b",
        run_dir=run_dir,
        ledger_path=ledger,
        control_execution_path=control,
        now=datetime(2026, 8, 9, 0, 2, tzinfo=LONDON),
        live_reader=lambda: pytest.fail("report recovery read live credits"),
        block_executor=lambda **_: pytest.fail("paid block repeated"),
        reporter=lambda **_: {"path": "report.md", "sha256": "recovered"},
    )
    assert recovered["verified_report"]["sha256"] == "recovered"
    assert block_calls == ["b"]
