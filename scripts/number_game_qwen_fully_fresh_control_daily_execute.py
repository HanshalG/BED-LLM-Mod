#!/usr/bin/env python3
"""Execute, reconcile, and independently verify the later-day Qwen control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_fully_fresh_daily_stages as staged
from scripts.number_game_qwen_fully_fresh_control_verify import (
    sha256_file,
    verify_completed_control,
)
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-fully-fresh-control-daily-execute-1"
AUTHORIZED_RELIABILITY_TAILS = (
    "openai/gpt-5.6-luna",
    "deepseek/deepseek-v4-flash-0731",
)
RELIABILITY_TAIL_CAP_USD = 0.10
STRESS_INTERFACE_VERSION = "number-game-budget-model-stress3584-1"
STRESS_TAIL_CAP_USD = 1.55
RELIABILITY_GATE_TOTAL_CAP_USD = (
    len(AUTHORIZED_RELIABILITY_TAILS) * RELIABILITY_TAIL_CAP_USD
)
FULL_TAIL_TOTAL_CAP_USD = RELIABILITY_GATE_TOTAL_CAP_USD + STRESS_TAIL_CAP_USD


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _measured_control_cost(result: dict[str, Any]) -> float:
    usage = result.get("usage") or {}
    if "control_cost_usd" in usage:
        return float(usage["control_cost_usd"])
    control_usage = (result.get("control") or {}).get("usage") or {}
    if "run_cost_usd" in control_usage:
        return float(control_usage["run_cost_usd"])
    raise ValueError("completed control result has no measured control cost")


def _measured_control_requests(result: dict[str, Any]) -> int:
    usage = result.get("usage") or {}
    if "control_requests" in usage:
        return int(usage["control_requests"])
    control_usage = (result.get("control") or {}).get("usage") or {}
    if "adapter_requests" in control_usage:
        return int(control_usage["adapter_requests"])
    raise ValueError("completed control result has no control request count")


def reconcile_ledger(
    *,
    ledger: dict[str, Any],
    measured_control_cost_usd: float,
    live_after: dict[str, float],
) -> dict[str, Any]:
    if measured_control_cost_usd < 0:
        raise ValueError("measured control cost must be non-negative")
    updated = json.loads(json.dumps(ledger))
    previous_recorded = float(updated.get("recorded_actual_spend_usd", 0.0))
    opening_usage = float(updated["opening_total_usage_usd"])
    posted_spend = max(
        0.0,
        float(live_after["total_usage_usd"]) - opening_usage,
    )
    locally_measured = previous_recorded + measured_control_cost_usd
    recorded = max(posted_spend, locally_measured)
    updated["recorded_actual_spend_usd"] = recorded
    block = updated["first_authorized_block"]
    block["actual_cost_usd"] = max(
        float(block.get("actual_cost_usd") or 0.0),
        measured_control_cost_usd,
    )
    block["status"] = "complete"
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_frozen_opening_usd": posted_spend,
        "locally_measured_spend_usd": locally_measured,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(
            0.0,
            float(updated["daily_cap_usd"]) - recorded,
        ),
    }
    updated["additional_paid_blocks_authorized"] = False
    return updated


def authorize_reliability_tails(
    *,
    ledger: dict[str, Any],
    control_complete: bool,
    verification: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    remaining = float(
        (updated.get("reconciliation") or {}).get(
            "remaining_daily_allowance_usd",
            0.0,
        )
    )
    verified = (
        control_complete
        and verification is not None
        and verification.get("status") == "verified"
    )
    gates_authorized = (
        verified and remaining + 1e-12 >= RELIABILITY_GATE_TOTAL_CAP_USD
    )
    stress_authorized = (
        verified and remaining + 1e-12 >= FULL_TAIL_TOTAL_CAP_USD
    )
    updated["additional_paid_blocks_authorized"] = gates_authorized
    updated["authorized_tail_blocks"] = (
        [
            {
                "interface": "number-game-budget-model-reliability128-1",
                "model": model,
                "maximum_cost_usd": RELIABILITY_TAIL_CAP_USD,
                "status": "authorized_pending",
            }
            for model in AUTHORIZED_RELIABILITY_TAILS
        ]
        + ([
            {
                "interface": STRESS_INTERFACE_VERSION,
                "model": None,
                "maximum_cost_usd": STRESS_TAIL_CAP_USD,
                "status": "waiting_for_reliability_results",
                "selection_rule": (
                    "pass_then_conditioned_min_mean_then_failure_tail_then_cost"
                ),
            }
        ] if stress_authorized else [])
        if gates_authorized
        else []
    )
    updated["tail_authorization_reason"] = (
        "verified_control_full_stress_allowance"
        if stress_authorized
        else "verified_control_reliability_gates_only"
        if gates_authorized
        else "control_not_verified_or_insufficient_gate_allowance"
    )
    return updated


def run_control_daily(
    *,
    run_dir: Path,
    run_id: str,
    ledger_path: Path,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    stage_runner: Callable[..., dict[str, Any]] = staged.run_control_stage,
    verifier: Callable[..., dict[str, Any]] = verify_completed_control,
) -> dict[str, Any]:
    ledger = _load(ledger_path)
    live_before = live_reader()
    result = stage_runner(
        output_dir=run_dir,
        run_id=run_id,
        ledger=ledger,
        total_usage_usd=float(live_before["total_usage_usd"]),
        balance_usd=float(live_before["balance_usd"]),
    )
    measured_cost = _measured_control_cost(result)
    measured_requests = _measured_control_requests(result)
    locally_reconciled_ledger = reconcile_ledger(
        ledger=ledger,
        measured_control_cost_usd=measured_cost,
        live_after=live_before,
    )
    checkpoint(ledger_path, locally_reconciled_ledger)
    live_after = live_reader()
    updated_ledger = reconcile_ledger(
        ledger=locally_reconciled_ledger,
        measured_control_cost_usd=0.0,
        live_after=live_after,
    )
    checkpoint(ledger_path, updated_ledger)

    control = result.get("control") or {}
    mechanics = control.get("mechanics_gates") or {}
    complete = (
        result.get("decision") == "complete_composite_endpoint"
        and mechanics
        and all(mechanics.values())
        and measured_requests == staged.base.CONTROL_EXPECTED_REQUESTS
    )
    verification = None
    if complete:
        verification = verifier(run_dir=run_dir)
    updated_ledger = authorize_reliability_tails(
        ledger=updated_ledger,
        control_complete=complete,
        verification=verification,
    )
    checkpoint(ledger_path, updated_ledger)

    execution = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "complete_verified"
            if verification is not None
            else "control_incomplete_or_mechanics_failed"
        ),
        "scientific_status": result.get("status"),
        "decision": result.get("decision"),
        "run_id": run_id,
        "daily_ledger": str(ledger_path),
        "daily_ledger_sha256": sha256_file(ledger_path),
        "live_before": live_before,
        "live_after": live_after,
        "measured_control_requests": measured_requests,
        "measured_control_cost_usd": measured_cost,
        "recorded_actual_spend_usd": updated_ledger[
            "recorded_actual_spend_usd"
        ],
        "remaining_daily_allowance_usd": updated_ledger[
            "reconciliation"
        ]["remaining_daily_allowance_usd"],
        "additional_paid_blocks_authorized": updated_ledger[
            "additional_paid_blocks_authorized"
        ],
        "authorized_tail_blocks": updated_ledger[
            "authorized_tail_blocks"
        ],
        "verification_status": (
            verification["status"] if verification is not None else None
        ),
        "verification_sha256": (
            sha256_file(run_dir / "CONTROL_VERIFICATION.json")
            if verification is not None
            else None
        ),
        "composite_result_sha256": sha256_file(run_dir / "RESULT.json"),
    }
    checkpoint(run_dir / "CONTROL_DAILY_EXECUTION.json", execution)
    return execution


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    try:
        execution = run_control_daily(
            run_dir=args.run_dir,
            run_id=args.run_id,
            ledger_path=args.daily_ledger,
        )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "daily_ledger": str(args.daily_ledger),
            "daily_ledger_sha256": (
                sha256_file(args.daily_ledger)
                if args.daily_ledger.exists()
                else None
            ),
        }
        checkpoint(
            args.run_dir / "CONTROL_DAILY_EXECUTION_FAILURE.json",
            failure,
        )
        raise
    print(json.dumps(execution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
