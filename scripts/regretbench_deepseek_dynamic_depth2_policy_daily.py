#!/usr/bin/env python3
"""Execute the Aug 8 enriched, naive, and dynamic depth-two policy gates."""

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

from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_support_recovery as recovery
from scripts import regretbench_deepseek_support_recovery_daily as recovery_daily
from scripts import regretbench_deepseek_result_verify as result_verify
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_naive_first_link_daily_execute as baseline_daily
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-dynamic-depth2-policy-daily-1"
DATE = "2026-08-08"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
ROOT = REPO_ROOT / "results/nonmyopic/regretbench_deepseek_dynamic_depth2_policy"
SMOKE_DIR = ROOT / "smoke-20260808"
NAIVE_SMOKE_DIR = ROOT / "naive-smoke-20260808"
DEVELOPMENT_DIR = ROOT / "development-20260808"
LEDGER = (
    REPO_ROOT
    / "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-08-regretbench-dynamic-policy.json"
)
RESULT_VERIFIER_SHA256 = recovery_daily.RESULT_VERIFIER_SHA256
POLICY_CORE_SHA256 = (
    "d0aaf34e0299ba0512a0da18aaa340b467e07458e429c881246261f021b01776"
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_date(now: datetime | None = None) -> None:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"RegretBench policy can run only on {DATE}")


def _spent(ledger: Mapping[str, Any], live: Mapping[str, float]) -> float:
    posted = max(
        0.0,
        float(live["total_usage_usd"])
        - float(ledger["opening_total_usage_usd"]),
    )
    return max(posted, float(ledger["recorded_actual_spend_usd"]))


def validate_recovery_predecessor() -> dict[str, Any]:
    daily_result = _load(recovery_daily.ROOT / "DAILY_RESULT.json")
    ledger = _load(recovery_daily.LEDGER)
    support = policy.validate_support_predecessors(
        support_smoke_result=recovery_daily.SMOKE_DIR / "RESULT.json",
        support_development_result=recovery_daily.DEVELOPMENT_DIR / "RESULT.json",
    )
    smoke_verification = _load(recovery_daily.SMOKE_DIR / "VERIFICATION.json")
    development_verification = _load(
        recovery_daily.DEVELOPMENT_DIR / "VERIFICATION.json"
    )
    if (
        daily_result.get("status") != "complete_reconciled"
        or daily_result.get("development_status") != "passed"
        or daily_result.get("development_opened") is not True
        or daily_result.get("confirmation_opened") is not False
        or daily_result.get("policy_endpoint_opened") is not False
        or daily_result.get("independent_replay_passed") is not True
        or daily_result.get("smoke_result_sha256")
        != support["support_smoke_sha256"]
        or daily_result.get("development_result_sha256")
        != support["support_development_sha256"]
        or smoke_verification.get("status") != "verified"
        or development_verification.get("status") != "verified"
        or daily_result.get("smoke_verification_sha256")
        != recovery.sha256_file(recovery_daily.SMOKE_DIR / "VERIFICATION.json")
        or daily_result.get("development_verification_sha256")
        != recovery.sha256_file(
            recovery_daily.DEVELOPMENT_DIR / "VERIFICATION.json"
        )
        or daily_result.get("ledger_sha256")
        != recovery.sha256_file(recovery_daily.LEDGER)
        or ledger.get("date") != DATE
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("stages", {}).get("smoke", {}).get("status") != "passed"
        or ledger.get("stages", {}).get("development", {}).get("status")
        != "passed"
    ):
        raise RuntimeError("support-recovery daily predecessor is not a clean pass")
    return {
        "support": support,
        "daily_result_sha256": recovery.sha256_file(
            recovery_daily.ROOT / "DAILY_RESULT.json"
        ),
        "ledger_sha256": recovery.sha256_file(recovery_daily.LEDGER),
        "ledger": ledger,
    }


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = aug10.read_openrouter_model_catalog,
) -> dict[str, Any]:
    _validate_date(now)
    policy.validate_protocol_binding()
    if (
        recovery.sha256_file(Path(policy.__file__).resolve())
        != POLICY_CORE_SHA256
    ):
        raise RuntimeError("dynamic policy core binding changed")
    if (
        recovery.sha256_file(Path(result_verify.__file__).resolve())
        != RESULT_VERIFIER_SHA256
    ):
        raise RuntimeError("RegretBench result-verifier binding changed")
    predecessor = validate_recovery_predecessor()
    for path in (SMOKE_DIR, NAIVE_SMOKE_DIR, DEVELOPMENT_DIR):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RuntimeError(f"policy output path is not pristine: {path}")
    if LEDGER.exists():
        raise RuntimeError("policy supplemental ledger already exists")
    catalog = catalog_reader()
    deepseek = recovery_daily.validate_deepseek_model_catalog(catalog)
    deepseek["status"] = "available"
    try:
        luna = baseline_daily._validate_model_catalog(catalog)
    except (RuntimeError, TypeError, ValueError) as exc:
        luna = {
            "id": policy.NAIVE_MODEL_ID,
            "status": "unavailable",
            "can_affect_primary_status": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    else:
        luna["status"] = "available"
    models = {"deepseek": deepseek, "luna": luna}
    live = live_reader()
    spent = _spent(predecessor["ledger"], live)
    maximum = (
        policy.SMOKE_BUDGET_USD
        + policy.NAIVE_SMOKE_BUDGET_USD
        + policy.RUN_BUDGET_USD
    )
    if spent + maximum > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("policy stages exceed remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < maximum:
        raise RuntimeError("OpenRouter balance is below policy stage caps")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": DATE,
        "models": models,
        "predecessor": {
            key: value for key, value in predecessor.items() if key != "ledger"
        },
        "live_credits": live,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "spent_before_policy_usd": spent,
            "enriched_smoke_cap_usd": policy.SMOKE_BUDGET_USD,
            "naive_smoke_cap_usd": policy.NAIVE_SMOKE_BUDGET_USD,
            "policy_development_cap_usd": policy.RUN_BUDGET_USD,
            "remaining_after_full_caps_usd": DAILY_CAP_USD - spent - maximum,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def _initial_ledger(ready: Mapping[str, Any]) -> dict[str, Any]:
    prior = _load(recovery_daily.LEDGER)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": prior["opening_total_credits_usd"],
        "opening_total_usage_usd": prior["opening_total_usage_usd"],
        "opening_balance_usd": prior["opening_balance_usd"],
        "recorded_actual_spend_usd": ready["budget"]["spent_before_policy_usd"],
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "support_recovery_predecessor": ready["predecessor"],
        "stages": {
            "enriched_smoke": {
                "status": "authorized_pending",
                "maximum_cost_usd": policy.SMOKE_BUDGET_USD,
            },
            "policy_development": {
                "status": "smokes_gated",
                "maximum_cost_usd": policy.RUN_BUDGET_USD,
            },
            "naive_smoke": {
                "status": "enriched_smoke_gated",
                "maximum_cost_usd": policy.NAIVE_SMOKE_BUDGET_USD,
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
        raise RuntimeError("policy reconciliation exceeds daily cap")
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
    projected: float,
    live: Mapping[str, float],
) -> dict[str, Any]:
    status = require_budget(
        dict(ledger),
        projected_cost_usd=projected,
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
    support_smoke = recovery_daily.SMOKE_DIR / "RESULT.json"
    support_development = recovery_daily.DEVELOPMENT_DIR / "RESULT.json"
    smoke_adapter = policy.build_adapter(
        stage="smoke",
        run_id="regretbench-deepseek-dynamic-policy-smoke-20260808",
        output_dir=SMOKE_DIR,
    )
    try:
        live = live_reader()
        smoke = policy.run_smoke(
            output_dir=SMOKE_DIR,
            run_id="regretbench-deepseek-dynamic-policy-smoke-20260808",
            support_smoke_result=support_smoke,
            support_development_result=support_development,
            adapter=smoke_adapter,
            daily_budget_status=_budget_status(
                ledger,
                projected=policy.SMOKE_BUDGET_USD,
                live=live,
            ),
        )
        smoke_verification = result_verify.verify_policy_smoke(SMOKE_DIR)
        checkpoint(SMOKE_DIR / "VERIFICATION.json", smoke_verification)
        if smoke_verification["status"] != "verified":
            raise RuntimeError("policy smoke independent replay failed")
    except Exception:
        usage = summarize_adapter(smoke_adapter)
        ledger = _reconcile(
            ledger,
            stage="enriched_smoke",
            status="failed_closed",
            measured_cost=usage,
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="enriched_smoke",
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
            "enriched_smoke_result_sha256": recovery.sha256_file(
                SMOKE_DIR / "RESULT.json"
            ),
            "enriched_smoke_verification_sha256": recovery.sha256_file(
                SMOKE_DIR / "VERIFICATION.json"
            ),
            "development_opened": False,
            "confirmation_opened": False,
            "ledger_sha256": recovery.sha256_file(LEDGER),
        }
        checkpoint(ROOT / "DAILY_RESULT.json", result)
        return result

    luna_available = (
        ready.get("models", {}).get("luna", {}).get("status") == "available"
    )
    if not luna_available:
        naive_smoke_status = "unavailable_preflight"
        naive_smoke_path = ROOT / "NAIVE_SMOKE_STATUS.json"
        ledger["stages"]["naive_smoke"].update(
            {"status": naive_smoke_status, "actual_cost_usd": 0.0}
        )
        checkpoint(LEDGER, ledger)
    else:
        ledger["stages"]["naive_smoke"]["status"] = "authorized_pending"
        checkpoint(LEDGER, ledger)
        naive_smoke_adapter = policy.build_naive_adapter(
            stage="smoke",
            run_id="regretbench-luna-naive-policy-smoke-20260808",
            output_dir=NAIVE_SMOKE_DIR,
        )
        try:
            live = live_reader()
            naive_smoke = policy.run_naive_smoke(
                output_dir=NAIVE_SMOKE_DIR,
                run_id="regretbench-luna-naive-policy-smoke-20260808",
                support_smoke_result=support_smoke,
                support_development_result=support_development,
                policy_smoke_result=SMOKE_DIR / "RESULT.json",
                adapter=naive_smoke_adapter,
                daily_budget_status=_budget_status(
                    ledger,
                    projected=policy.NAIVE_SMOKE_BUDGET_USD,
                    live=live,
                ),
            )
        except Exception as exc:
            ledger = _reconcile(
                ledger,
                stage="naive_smoke",
                status="failed_closed",
                measured_cost=summarize_adapter(naive_smoke_adapter),
                live=live_reader(),
            )
            checkpoint(LEDGER, ledger)
            naive_smoke_status = "failed_closed"
            naive_smoke_path = NAIVE_SMOKE_DIR / "FAILURE.json"
            checkpoint(
                naive_smoke_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "interface_version": policy.INTERFACE_VERSION,
                    "status": naive_smoke_status,
                    "authorizes": "nothing",
                    "can_affect_primary_status": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        else:
            naive_smoke_status = naive_smoke["status"]
            naive_smoke_path = NAIVE_SMOKE_DIR / "RESULT.json"
            ledger = _reconcile(
                ledger,
                stage="naive_smoke",
                status=naive_smoke_status,
                measured_cost=float(naive_smoke["usage"]["run_cost_usd"]),
                live=live_reader(),
            )
            checkpoint(LEDGER, ledger)
    naive_baseline_enabled = naive_smoke_status == "passed"
    if not naive_baseline_enabled:
        checkpoint(
            ROOT / "NAIVE_SMOKE_STATUS.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": naive_smoke_status,
                "development_opened": True,
                "naive_baseline_enabled": False,
                "can_affect_primary_status": False,
                "naive_smoke_artifact": str(naive_smoke_path),
                "catalog_status": ready.get("models", {}).get("luna"),
            },
        )

    ledger["stages"]["policy_development"]["status"] = "authorized_pending"
    checkpoint(LEDGER, ledger)
    development_adapter = policy.build_adapter(
        stage="development",
        run_id="regretbench-deepseek-dynamic-policy-development-20260808",
        output_dir=DEVELOPMENT_DIR,
    )
    development_naive_adapter = (
        policy.build_naive_adapter(
            stage="development",
            run_id="regretbench-deepseek-dynamic-policy-development-20260808",
            output_dir=DEVELOPMENT_DIR,
        )
        if naive_baseline_enabled
        else None
    )
    development_naive_endpoint_adapter = (
        policy.build_adapter(
            stage="development",
            run_id="regretbench-deepseek-dynamic-policy-development-20260808",
            output_dir=DEVELOPMENT_DIR,
        )
        if naive_baseline_enabled
        else None
    )
    try:
        live = live_reader()
        development = policy.run_development(
            output_dir=DEVELOPMENT_DIR,
            run_id="regretbench-deepseek-dynamic-policy-development-20260808",
            support_smoke_result=support_smoke,
            support_development_result=support_development,
            policy_smoke_result=SMOKE_DIR / "RESULT.json",
            naive_smoke_result=naive_smoke_path,
            adapter=development_adapter,
            naive_adapter=development_naive_adapter,
            naive_endpoint_adapter=development_naive_endpoint_adapter,
            naive_baseline_enabled=naive_baseline_enabled,
            daily_budget_status=_budget_status(
                ledger,
                projected=policy.RUN_BUDGET_USD,
                live=live,
            ),
        )
        development_verification = result_verify.verify_policy(DEVELOPMENT_DIR)
        checkpoint(
            DEVELOPMENT_DIR / "VERIFICATION.json", development_verification
        )
        if development_verification["status"] != "verified":
            raise RuntimeError("policy development independent replay failed")
    except Exception:
        usage = summarize_adapter(development_adapter)
        if development_naive_adapter is not None:
            usage += summarize_adapter(development_naive_adapter)
        if development_naive_endpoint_adapter is not None:
            usage += summarize_adapter(development_naive_endpoint_adapter)
        ledger = _reconcile(
            ledger,
            stage="policy_development",
            status="failed_closed",
            measured_cost=usage,
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="policy_development",
        status=development["status"],
        measured_cost=float(development["usage"]["combined_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "enriched_smoke_result_sha256": recovery.sha256_file(
            SMOKE_DIR / "RESULT.json"
        ),
        "enriched_smoke_verification_sha256": recovery.sha256_file(
            SMOKE_DIR / "VERIFICATION.json"
        ),
        "naive_smoke_status": naive_smoke_status,
        "naive_smoke_result_sha256": recovery.sha256_file(naive_smoke_path),
        "naive_baseline_enabled": naive_baseline_enabled,
        "development_result_sha256": recovery.sha256_file(
            DEVELOPMENT_DIR / "RESULT.json"
        ),
        "development_verification_sha256": recovery.sha256_file(
            DEVELOPMENT_DIR / "VERIFICATION.json"
        ),
        "independent_replay_passed": True,
        "development_status": development["status"],
        "development_opened": True,
        "confirmation_opened": False,
        "ledger_sha256": recovery.sha256_file(LEDGER),
        "recorded_daily_spend_usd": ledger["recorded_actual_spend_usd"],
    }
    checkpoint(ROOT / "DAILY_RESULT.json", result)
    return result


def summarize_adapter(adapter: Any) -> float:
    snapshot = adapter.usage_snapshot()
    return float(snapshot.get("adapter_cost_usd", 0.0))


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
            "confirmation_opened": False,
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
