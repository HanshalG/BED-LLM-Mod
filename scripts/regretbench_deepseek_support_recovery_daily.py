#!/usr/bin/env python3
"""Execute the Aug 8 RegretBench support-recovery smoke and development gate."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_naive_first_link as baseline
from scripts import bongard_openworld_luna_naive_first_link_daily_execute as baseline_daily
from scripts import regretbench_deepseek_support_recovery as recovery
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-support-recovery-daily-1"
DATE = "2026-08-08"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
ROOT = REPO_ROOT / "results/nonmyopic/regretbench_deepseek_support_recovery"
SMOKE_DIR = ROOT / "smoke-20260808"
DEVELOPMENT_DIR = ROOT / "development-20260808"
LEDGER = (
    REPO_ROOT
    / "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-08-regretbench-support-recovery.json"
)
RECOVERY_CORE_SHA256 = (
    "7e227e4d3a125b817dd45c31ce6b1fc94c24bae6ee982ce59f2a9082065752c2"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_date(now: datetime | None = None) -> None:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"RegretBench daily gate can run only on {DATE}")


def _day_spent(ledger: Mapping[str, Any], live: Mapping[str, float]) -> float:
    posted = max(
        0.0,
        float(live["total_usage_usd"])
        - float(ledger["opening_total_usage_usd"]),
    )
    return max(posted, float(ledger["recorded_actual_spend_usd"]))


def validate_baseline_predecessor() -> dict[str, Any]:
    verification = baseline.verify_smoke_result(baseline_daily.SMOKE_RESULT)
    ledger = _load(baseline_daily.SMOKE_LEDGER)
    execution = _load(baseline_daily.SMOKE_DIR / "EXECUTION.json")
    section = ledger.get("naive_first_link") or {}
    if (
        ledger.get("date") != DATE
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or section.get("status") != "passed"
        or execution.get("status") != "complete_reconciled"
        or execution.get("result_sha256")
        != recovery.sha256_file(baseline_daily.SMOKE_RESULT)
        or execution.get("ledger_sha256")
        != recovery.sha256_file(baseline_daily.SMOKE_LEDGER)
    ):
        raise RuntimeError("Aug 8 baseline smoke predecessor is not a clean pass")
    return {
        "verification": verification,
        "ledger": ledger,
        "result_sha256": recovery.sha256_file(baseline_daily.SMOKE_RESULT),
        "ledger_sha256": recovery.sha256_file(baseline_daily.SMOKE_LEDGER),
        "execution_sha256": recovery.sha256_file(
            baseline_daily.SMOKE_DIR / "EXECUTION.json"
        ),
    }


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    _validate_date(now)
    if (
        recovery.sha256_file(Path(recovery.__file__).resolve())
        != RECOVERY_CORE_SHA256
    ):
        raise RuntimeError("support-recovery core binding changed")
    recovery.validate_source_bindings()
    predecessor = validate_baseline_predecessor()
    for path in (SMOKE_DIR, DEVELOPMENT_DIR):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RuntimeError(f"RegretBench output path is not pristine: {path}")
    if LEDGER.exists():
        raise RuntimeError("RegretBench supplemental daily ledger already exists")
    live = live_reader()
    spent = _day_spent(predecessor["ledger"], live)
    maximum = (
        recovery.STAGES["smoke"]["run_budget_usd"]
        + recovery.STAGES["development"]["run_budget_usd"]
    )
    if spent + maximum > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("RegretBench stages exceed remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < maximum:
        raise RuntimeError("OpenRouter balance is below RegretBench stage caps")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": DATE,
        "predecessor": {
            key: value for key, value in predecessor.items() if key != "ledger"
        },
        "live_credits": live,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "spent_before_regretbench_usd": spent,
            "smoke_cap_usd": recovery.STAGES["smoke"]["run_budget_usd"],
            "development_cap_usd": recovery.STAGES["development"][
                "run_budget_usd"
            ],
            "remaining_after_full_caps_usd": DAILY_CAP_USD - spent - maximum,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def _initial_ledger(preflight_result: Mapping[str, Any]) -> dict[str, Any]:
    baseline_ledger = _load(baseline_daily.SMOKE_LEDGER)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": baseline_ledger[
            "opening_total_credits_usd"
        ],
        "opening_total_usage_usd": baseline_ledger["opening_total_usage_usd"],
        "opening_balance_usd": baseline_ledger["opening_balance_usd"],
        "recorded_actual_spend_usd": preflight_result["budget"][
            "spent_before_regretbench_usd"
        ],
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "baseline_predecessor": preflight_result["predecessor"],
        "stages": {
            "smoke": {
                "status": "authorized_pending",
                "maximum_cost_usd": recovery.STAGES["smoke"]["run_budget_usd"],
            },
            "development": {
                "status": "smoke_gated",
                "maximum_cost_usd": recovery.STAGES["development"][
                    "run_budget_usd"
                ],
            },
        },
    }


def _reconcile(
    ledger: Mapping[str, Any],
    *,
    stage: str,
    status: str,
    measured_cost: float,
    live: Mapping[str, float],
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    prior = float(updated["recorded_actual_spend_usd"])
    posted = max(0.0, float(live["total_usage_usd"]) - opening)
    recorded = max(posted, prior + measured_cost)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("RegretBench reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stages"][stage].update(
        {"status": status, "actual_cost_usd": measured_cost}
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_day_opening_usd": posted,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": DAILY_CAP_USD - recorded,
    }
    return updated


def _budget_status(
    ledger: Mapping[str, Any],
    *,
    projected_cost: float,
    live: Mapping[str, float],
) -> dict[str, Any]:
    status = require_budget(
        dict(ledger),
        projected_cost_usd=projected_cost,
        total_usage_usd=float(live["total_usage_usd"]),
    )
    status.update(live)
    return status


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader)
    ledger = _initial_ledger(ready)
    checkpoint(LEDGER, ledger)
    smoke_adapter = recovery.build_adapter(
        stage="smoke",
        run_id="regretbench-deepseek-support-recovery-smoke-20260808",
        output_dir=SMOKE_DIR,
    )
    try:
        smoke_live = live_reader()
        smoke = recovery.run_stage(
            stage="smoke",
            output_dir=SMOKE_DIR,
            run_id="regretbench-deepseek-support-recovery-smoke-20260808",
            adapter=smoke_adapter,
            daily_budget_status=_budget_status(
                ledger,
                projected_cost=recovery.STAGES["smoke"]["run_budget_usd"],
                live=smoke_live,
            ),
        )
    except Exception:
        usage = recovery.summarize_usage(smoke_adapter.usage_snapshot())
        ledger = _reconcile(
            ledger,
            stage="smoke",
            status="failed_closed",
            measured_cost=float(usage["run_cost_usd"]),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="smoke",
        status=smoke["status"],
        measured_cost=float(smoke["usage"]["run_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    if smoke["status"] != "passed":
        result = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "smoke_stopped",
            "smoke_result_sha256": recovery.sha256_file(SMOKE_DIR / "RESULT.json"),
            "development_opened": False,
            "ledger_sha256": recovery.sha256_file(LEDGER),
        }
        checkpoint(ROOT / "DAILY_RESULT.json", result)
        return result

    ledger["stages"]["development"]["status"] = "authorized_pending"
    checkpoint(LEDGER, ledger)
    development_live = live_reader()
    development_adapter = recovery.build_adapter(
        stage="development",
        run_id="regretbench-deepseek-support-recovery-development-20260808",
        output_dir=DEVELOPMENT_DIR,
    )
    try:
        development = recovery.run_stage(
            stage="development",
            output_dir=DEVELOPMENT_DIR,
            run_id="regretbench-deepseek-support-recovery-development-20260808",
            adapter=development_adapter,
            daily_budget_status=_budget_status(
                ledger,
                projected_cost=recovery.STAGES["development"]["run_budget_usd"],
                live=development_live,
            ),
            smoke_result_path=SMOKE_DIR / "RESULT.json",
        )
    except Exception:
        usage = recovery.summarize_usage(development_adapter.usage_snapshot())
        ledger = _reconcile(
            ledger,
            stage="development",
            status="failed_closed",
            measured_cost=float(usage["run_cost_usd"]),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="development",
        status=development["status"],
        measured_cost=float(development["usage"]["run_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "smoke_result_sha256": recovery.sha256_file(SMOKE_DIR / "RESULT.json"),
        "development_result_sha256": recovery.sha256_file(
            DEVELOPMENT_DIR / "RESULT.json"
        ),
        "development_status": development["status"],
        "development_opened": True,
        "confirmation_opened": False,
        "policy_endpoint_opened": False,
        "ledger_sha256": recovery.sha256_file(LEDGER),
        "recorded_daily_spend_usd": ledger["recorded_actual_spend_usd"],
    }
    checkpoint(ROOT / "DAILY_RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight() if args.preflight else execute()
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "development_opened": False,
        }
        if not args.preflight:
            ROOT.mkdir(parents=True, exist_ok=True)
            checkpoint(ROOT / "DAILY_FAILURE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {"ready_without_paid_calls", "complete_reconciled"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
