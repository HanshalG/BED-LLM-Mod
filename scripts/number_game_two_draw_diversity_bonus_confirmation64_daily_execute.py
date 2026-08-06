#!/usr/bin/env python3
"""Execute one exact daily block of the staged diversity confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_two_draw_diversity_bonus_audit as audit
from scripts import number_game_two_draw_diversity_bonus_confirmation64_staged as staged
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = (
    "number-game-two-draw-diversity-bonus-confirmation64-daily-execute-1"
)
RUN_ID = "number-game-two-draw-diversity-bonus-confirmation64-20260808"
RUN_DIR = REPO_ROOT / (
    "results/nonmyopic/number_game_two_draw_diversity_bonus_confirmation64/"
    f"{RUN_ID}"
)
BLOCK_DATES = staged.FORMAL_BLOCK_DATES
LEDGER_PATHS = {
    block: REPO_ROOT
    / f"results/nonmyopic/openrouter_daily_budget/{date}.json"
    for block, date in BLOCK_DATES.items()
}
CONTROL_EXECUTION = REPO_ROOT / (
    "results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/"
    "number-game-qwen-fully-fresh-daily-stages-20260806T000200Z/"
    "CONTROL_DAILY_EXECUTION.json"
)
EXECUTION_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_CONFIRMATION64_"
    "EXECUTION_AMENDMENT.md"
)
EXECUTION_AMENDMENT_SHA256 = (
    "4c3e947bf9b68f71668c05593b4a224d609de19be52070aefc0ce989c426b39b"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_date(*, block: str, now: datetime | None) -> datetime:
    if block not in BLOCK_DATES:
        raise ValueError(f"unknown block: {block}")
    timezone = ZoneInfo("Europe/London")
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if local_now.date().isoformat() != BLOCK_DATES[block]:
        raise RuntimeError(
            f"Block {block.upper()} can run only on {BLOCK_DATES[block]}"
        )
    return local_now


def _validate_control_predecessor(path: Path) -> dict[str, Any]:
    execution = _load(path)
    if execution.get("status") != "complete_verified":
        raise RuntimeError("August 7 control is not complete and verified")
    if execution.get("verification_status") != "verified":
        raise RuntimeError("August 7 control verification is missing")
    if execution.get("measured_control_requests") != 3072:
        raise RuntimeError("August 7 control request count is not exact")
    return execution


def initialize_daily_ledger(
    *,
    path: Path,
    block: str,
    live: dict[str, float],
    local_now: datetime,
) -> dict[str, Any]:
    expected_date = BLOCK_DATES[block]
    if path.exists():
        ledger = _load(path)
        if ledger.get("date") != expected_date:
            raise RuntimeError("existing ledger date does not match block")
        if ledger.get("timezone") != "Europe/London":
            raise RuntimeError("existing ledger timezone is not Europe/London")
        if float(ledger.get("daily_cap_usd", 0.0)) != staged.DAILY_CAP_USD:
            raise RuntimeError("existing ledger does not preserve the $5 cap")
        authorization = ledger.get("first_authorized_block") or {}
        if (
            authorization.get("interface_version") != staged.INTERFACE_VERSION
            or authorization.get("run_id") != RUN_ID
            or authorization.get("block") != block
            or authorization.get("expected_requests")
            != staged.EXPECTED_REQUESTS_PER_BLOCK
            or float(authorization.get("maximum_cost_usd", 0.0))
            != staged.DAILY_CAP_USD
            or authorization.get("status") != "authorized_pending"
        ):
            raise RuntimeError("existing ledger authorization is not reusable")
        return ledger
    if float(live["balance_usd"]) + 1e-12 < staged.MIN_STARTING_BALANCE_USD:
        raise RuntimeError("OpenRouter balance is below the exact block gate")
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "date": expected_date,
        "timezone": "Europe/London",
        "daily_cap_usd": staged.DAILY_CAP_USD,
        "opening_total_credits_usd": float(live["total_credits_usd"]),
        "opening_total_usage_usd": float(live["total_usage_usd"]),
        "opening_balance_usd": float(live["balance_usd"]),
        "opening_frozen_at_london": local_now.isoformat(),
        "recorded_actual_spend_usd": 0.0,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "first_authorized_block": {
            "interface_version": staged.INTERFACE_VERSION,
            "run_id": RUN_ID,
            "block": block,
            "expected_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "maximum_cost_usd": staged.DAILY_CAP_USD,
            "status": "authorized_pending",
        },
        "additional_paid_blocks_authorized": False,
    }
    checkpoint(path, ledger)
    return ledger


def _partial_block_exists(*, run_dir: Path, block: str) -> bool:
    if block == "a":
        return run_dir.exists() and any(run_dir.iterdir())
    return any(
        path.exists()
        for path in (
            staged.block_directory(run_dir, "b"),
            staged.block_stage_path(run_dir, "b"),
            run_dir / "RESULT.json",
            run_dir / "VERIFICATION.json",
        )
    )


def _set_authorization_status(
    *,
    ledger_path: Path,
    status: str,
    actual_cost_usd: float | None = None,
) -> dict[str, Any]:
    ledger = _load(ledger_path)
    authorization = ledger["first_authorized_block"]
    authorization["status"] = status
    if actual_cost_usd is not None:
        authorization["actual_cost_usd"] = actual_cost_usd
    checkpoint(ledger_path, ledger)
    return ledger


def _write_verified_report(*, run_dir: Path) -> dict[str, str]:
    from scripts import number_game_two_draw_diversity_bonus_confirmation64_report as report

    report.write_report(run_dir=run_dir, output_path=report.REPORT_PATH)
    return {
        "path": str(report.REPORT_PATH),
        "sha256": audit.sha256_file(report.REPORT_PATH),
    }


def execute_formal_block(
    *,
    block: str,
    run_dir: Path = RUN_DIR,
    ledger_path: Path | None = None,
    control_execution_path: Path = CONTROL_EXECUTION,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    block_executor: Callable[..., dict[str, Any]] = staged.execute_daily_block,
    reporter: Callable[..., dict[str, str]] = _write_verified_report,
) -> dict[str, Any]:
    if audit.sha256_file(EXECUTION_AMENDMENT) != EXECUTION_AMENDMENT_SHA256:
        raise ValueError("diversity execution amendment hash changed")
    local_now = _validate_date(block=block, now=now)
    _validate_control_predecessor(control_execution_path)
    ledger_path = ledger_path or LEDGER_PATHS[block]
    execution_path = run_dir / f"BLOCK_{block.upper()}_DAILY_EXECUTION.json"
    failure_path = run_dir / f"BLOCK_{block.upper()}_RUNNER_FAILURE.json"
    if execution_path.exists():
        execution = _load(execution_path)
        if execution.get("status") != "complete":
            raise RuntimeError("banked daily execution is not complete")
        if block == "b" and "verified_report" not in execution:
            execution["verified_report"] = reporter(run_dir=run_dir)
            checkpoint(execution_path, execution)
        return execution
    if failure_path.exists():
        raise RuntimeError(f"Block {block.upper()} already failed closed")
    if _partial_block_exists(run_dir=run_dir, block=block):
        raise RuntimeError(
            f"partial Block {block.upper()} artifacts exist; refusing rerun"
        )
    live_opening = live_reader()
    initialize_daily_ledger(
        path=ledger_path,
        block=block,
        live=live_opening,
        local_now=local_now,
    )
    _set_authorization_status(
        ledger_path=ledger_path,
        status="execution_started",
    )
    try:
        execution = block_executor(
            run_dir=run_dir,
            run_id=RUN_ID,
            block=block,
            ledger_path=ledger_path,
            live_reader=live_reader,
            now=now,
        )
    except Exception as exc:
        reconciliation_error = None
        try:
            ledger = _load(ledger_path)
            live_failure = live_reader()
            reconciled = staged.reconcile_ledger(
                ledger=ledger,
                block=block,
                measured_cost_usd=0.0,
                live_after=live_failure,
                status="failed_closed_posted_spend_reconciled",
            )
            reconciled["first_authorized_block"]["status"] = "failed_closed"
            checkpoint(ledger_path, reconciled)
        except Exception as reconcile_exc:
            reconciliation_error = (
                f"{type(reconcile_exc).__name__}: {reconcile_exc}"
            )
        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            failure_path,
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "block": block,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    _set_authorization_status(
        ledger_path=ledger_path,
        status="complete",
        actual_cost_usd=float(execution["measured_cost_usd"]),
    )
    if block == "b":
        try:
            execution = dict(execution)
            execution["verified_report"] = reporter(run_dir=run_dir)
            checkpoint(execution_path, execution)
        except Exception as exc:
            checkpoint(
                run_dir / "BLOCK_B_REPORT_FAILURE.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "interface_version": INTERFACE_VERSION,
                    "status": "zero_call_report_failed",
                    "block_execution_remains_complete": True,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=tuple(BLOCK_DATES), required=True)
    args = parser.parse_args()
    execution = execute_formal_block(block=args.block)
    print(json.dumps(execution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
