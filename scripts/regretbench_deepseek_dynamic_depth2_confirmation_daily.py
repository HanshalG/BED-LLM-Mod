#!/usr/bin/env python3
"""Execute the frozen RegretBench confirmation on its Aug 9 budget day."""

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

from scripts import regretbench_deepseek_confirmation_result_verify as result_verify
from scripts import regretbench_deepseek_confirmation_protocol_verify as protocol_verify
from scripts import regretbench_deepseek_dynamic_depth2_confirmation as confirmation
from scripts import regretbench_deepseek_dynamic_depth2_policy as policy
from scripts import regretbench_deepseek_dynamic_depth2_policy_daily as policy_daily
from scripts import regretbench_deepseek_support_recovery as recovery
from scripts import regretbench_deepseek_support_recovery_daily as recovery_daily
from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-dynamic-depth2-confirmation-daily-1"
DATE = "2026-08-09"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
ROOT = REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_dynamic_depth2_confirmation"
)
RUN_DIR = ROOT / "confirmation-20260809"
DAILY_RESULT = ROOT / "DAILY_RESULT.json"
LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-09-regretbench-confirmation.json"
)
EXECUTION_BINDINGS = ROOT / "EXECUTION_BINDINGS.json"
EXECUTION_BINDINGS_SHA256 = (
    "34b03e7466007a81534a513234cf1ca9f83bdc04fa35977779874285fddf7816"
)
PRODUCER_SHA256 = (
    "7cbe10ec1dde5406d21dfb2ee02431ca5771e5e760c8bcb939f2cea94ae129d0"
)
CONFIRMATION_VERIFIER_SHA256 = (
    "f8a88ba92a079f2993838ea50cac5fb78546ba66f1428a173927b57c72eaa6aa"
)
SHARED_VERIFIER_SHA256 = (
    "a5b18ebc48f6df25da4df4827f39ae96f5efde02d2d8aeb347443f97ff6c0eca"
)
SUPPORT_DAILY_SHA256 = (
    "d183e5b2785741d8d410c90d0b3f983da31db1a5b698a75fbd4b52b234c49399"
)
POLICY_DAILY_SHA256 = (
    "59fc575fb716b45ae76d0a1505b9293b5883c4cd99088f67e5e6159d64f0ebf7"
)
POLICY_CORE_SHA256 = (
    "d6aeda8f6ea5de994bf33844fea0a08e35b4213e565bb2a0f955618008378ce0"
)


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"RegretBench confirmation can run only on {DATE}")
    return local


def validate_execution_bindings() -> dict[str, Any]:
    confirmation.validate_protocol_binding()
    protocol_verification = protocol_verify.verify_protocol()
    if protocol_verification.get("status") != "verified_frozen_protocol":
        raise RuntimeError("confirmation supplemental protocol verification failed")
    if recovery.sha256_file(EXECUTION_BINDINGS) != EXECUTION_BINDINGS_SHA256:
        raise RuntimeError("confirmation execution bindings changed")
    if recovery.sha256_file(Path(confirmation.__file__).resolve()) != PRODUCER_SHA256:
        raise RuntimeError("confirmation producer binding changed")
    if (
        recovery.sha256_file(Path(result_verify.__file__).resolve())
        != CONFIRMATION_VERIFIER_SHA256
    ):
        raise RuntimeError("confirmation result-verifier binding changed")
    if recovery.sha256_file(Path(result_verify.base.__file__).resolve()) != (
        SHARED_VERIFIER_SHA256
    ):
        raise RuntimeError("shared result-verifier binding changed")
    if recovery.sha256_file(Path(recovery_daily.__file__).resolve()) != (
        SUPPORT_DAILY_SHA256
    ):
        raise RuntimeError("support daily binding changed")
    if recovery.sha256_file(Path(policy_daily.__file__).resolve()) != (
        POLICY_DAILY_SHA256
    ):
        raise RuntimeError("policy daily binding changed")
    if recovery.sha256_file(Path(policy.__file__).resolve()) != POLICY_CORE_SHA256:
        raise RuntimeError("policy core binding changed")
    bindings = _load(EXECUTION_BINDINGS)
    if (
        bindings.get("status") != "prospectively_amended_before_responses"
        or bindings.get("scientific_contract_changed") is not True
        or bindings.get("confirmation_producer", {}).get("sha256")
        != PRODUCER_SHA256
        or bindings.get("confirmation_result_verifier", {}).get("sha256")
        != CONFIRMATION_VERIFIER_SHA256
        or bindings.get("shared_result_verifier", {}).get("sha256")
        != SHARED_VERIFIER_SHA256
        or bindings.get("policy_core", {}).get("sha256")
        != POLICY_CORE_SHA256
        or bindings.get("first_reply_alignment_amendment", {}).get("sha256")
        != policy.FIRST_REPLY_ALIGNMENT_AMENDMENT_SHA256
        or bindings.get("first_reply_endpoint_amendment", {}).get("sha256")
        != policy.FIRST_REPLY_ENDPOINT_AMENDMENT_SHA256
        or bindings.get("matched_utility_myopic_amendment", {}).get("sha256")
        != policy.MATCHED_UTILITY_MYOPIC_AMENDMENT_SHA256
        or bindings.get("refresh_matched_myopic_amendment", {}).get("sha256")
        != policy.REFRESH_MATCHED_MYOPIC_AMENDMENT_SHA256
        or bindings.get("requirements", {}).get(
            "minimum_truth_consistent_first_reply_matches_per_policy"
        )
        != 40
        or bindings.get("requirements", {}).get(
            "unmodelled_truth_consistent_first_reply_scored_mass"
        )
        != 0.0
        or bindings.get("requirements", {}).get(
            "matched_utility_myopic_policy_required"
        )
        is not True
        or bindings.get("requirements", {}).get(
            "entropy_myopic_width_control_remains_required"
        )
        is not True
        or bindings.get("requirements", {}).get(
            "maximum_distinct_roots_per_task"
        )
        != 4
        or bindings.get("requirements", {}).get(
            "matched_utility_additional_model_calls"
        )
        != 0
        or bindings.get("requirements", {}).get(
            "refresh_matched_myopic_policy_required"
        )
        is not True
        or bindings.get("requirements", {}).get(
            "refresh_matched_is_headline_horizon_control"
        )
        is not True
        or bindings.get("requirements", {}).get(
            "refresh_matched_additional_model_calls"
        )
        != 0
    ):
        raise RuntimeError("confirmation execution-binding manifest is invalid")
    return bindings


def validate_development_predecessor() -> dict[str, Any]:
    daily = _load(policy_daily.ROOT / "DAILY_RESULT.json")
    result_path = policy_daily.DEVELOPMENT_DIR / "RESULT.json"
    verification_path = policy_daily.DEVELOPMENT_DIR / "VERIFICATION.json"
    result = _load(result_path)
    verification = _load(verification_path)
    ledger = _load(policy_daily.LEDGER)
    result_sha = recovery.sha256_file(result_path)
    verification_sha = recovery.sha256_file(verification_path)
    daily_sha = recovery.sha256_file(policy_daily.ROOT / "DAILY_RESULT.json")
    ledger_sha = recovery.sha256_file(policy_daily.LEDGER)
    artifact_result_sha = verification.get("artifact_sha256", {}).get(
        "RESULT.json"
    )
    science = result.get("science") or {}
    if (
        daily.get("status") != "complete_reconciled"
        or daily.get("development_status") != "passed"
        or daily.get("development_opened") is not True
        or daily.get("confirmation_opened") is not False
        or daily.get("independent_replay_passed") is not True
        or daily.get("development_result_sha256") != result_sha
        or daily.get("development_verification_sha256") != verification_sha
        or daily.get("ledger_sha256") != ledger_sha
        or result.get("status") != "passed"
        or result.get("confirmation_opened") is not None
        and result.get("confirmation_opened") is not False
        or result.get("protocol", {}).get("confirmation_opened") is not False
        or result.get("mechanics_gates", {}).get("all_pass") is not True
        or science.get("gates", {}).get("all_pass") is not True
        or verification.get("status") != "verified"
        or verification.get("result_status") != "passed"
        or verification.get("model_calls") != 0
        or verification.get("mismatches") != []
        or artifact_result_sha != result_sha
        or ledger.get("date") != "2026-08-08"
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or float(ledger.get("recorded_actual_spend_usd", 99.0))
        > DAILY_CAP_USD
    ):
        raise RuntimeError(
            "confirmation requires a literal independently verified development pass"
        )
    return {
        "status": "authorized",
        "development_status": "passed",
        "independent_replay_status": "verified",
        "development_result_sha256": result_sha,
        "development_verification_sha256": verification_sha,
        "development_daily_result_sha256": daily_sha,
        "development_ledger_sha256": ledger_sha,
        "development_ledger": ledger,
    }


def _day_boundary(ledger: Mapping[str, Any]) -> float:
    return float(ledger["opening_total_usage_usd"]) + float(
        ledger["recorded_actual_spend_usd"]
    )


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = aug10.read_openrouter_model_catalog,
) -> dict[str, Any]:
    local = _validate_date(now)
    bindings = validate_execution_bindings()
    authorization = validate_development_predecessor()
    for path in (RUN_DIR, DAILY_RESULT, LEDGER):
        if path.exists() and (not path.is_dir() or any(path.iterdir())):
            raise RuntimeError(f"confirmation output path is not pristine: {path}")
        if path.is_file():
            raise RuntimeError(f"confirmation output path is not pristine: {path}")
    catalog = catalog_reader()
    model = recovery_daily.validate_deepseek_model_catalog(catalog)
    live = live_reader()
    boundary = _day_boundary(authorization["development_ledger"])
    prior_usage = max(0.0, float(live["total_usage_usd"]) - boundary)
    if prior_usage + confirmation.RUN_BUDGET_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("confirmation cap exceeds remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < confirmation.RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below confirmation cap")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": local.date().isoformat(),
        "model": model,
        "development_authorization": {
            key: value
            for key, value in authorization.items()
            if key != "development_ledger"
        },
        "execution_bindings_sha256": EXECUTION_BINDINGS_SHA256,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "opening_total_usage_boundary_usd": boundary,
            "spent_before_confirmation_usd": prior_usage,
            "confirmation_cap_usd": confirmation.RUN_BUDGET_USD,
            "remaining_after_full_cap_usd": (
                DAILY_CAP_USD - prior_usage - confirmation.RUN_BUDGET_USD
            ),
        },
        "live_credits": live,
        "bindings": bindings,
        "model_calls_made": 0,
        "files_written": 0,
    }


def _initial_ledger(ready: Mapping[str, Any]) -> dict[str, Any]:
    live = ready["live_credits"]
    boundary = ready["budget"]["opening_total_usage_boundary_usd"]
    spent = ready["budget"]["spent_before_confirmation_usd"]
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": float(live["total_credits_usd"]),
        "opening_total_usage_usd": boundary,
        "opening_balance_usd": float(live["total_credits_usd"]) - boundary,
        "recorded_actual_spend_usd": spent,
        "account_wide_usage_counts_against_cap": True,
        "opening_boundary_derived_from_reconciled_aug8_close": True,
        "unspent_allowance_does_not_roll_over": True,
        "development_authorization": ready["development_authorization"],
        "stage": {
            "status": "authorized_pending",
            "maximum_cost_usd": confirmation.RUN_BUDGET_USD,
        },
    }


def _budget_status(
    ledger: Mapping[str, Any],
    live: Mapping[str, float],
    *,
    now: datetime | None = None,
) -> dict[str, Any]:
    status = require_budget(
        dict(ledger),
        projected_cost_usd=confirmation.RUN_BUDGET_USD,
        total_usage_usd=float(live["total_usage_usd"]),
        now=now,
    )
    status.update(live)
    return status


def _reconcile(
    ledger: Mapping[str, Any],
    *,
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
        raise RuntimeError("confirmation reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update(
        {"status": status, "actual_cost_usd": measured_cost}
    )
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_boundary_usd": posted,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": DAILY_CAP_USD - recorded,
    }
    return updated


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader)
    ledger = _initial_ledger(ready)
    checkpoint(LEDGER, ledger)
    adapter = policy.build_adapter(
        stage="development",
        run_id="regretbench-deepseek-dynamic-confirmation-20260809",
        output_dir=RUN_DIR,
    )
    try:
        live = live_reader()
        result = confirmation.run_confirmation(
            output_dir=RUN_DIR,
            run_id="regretbench-deepseek-dynamic-confirmation-20260809",
            support_smoke_result=recovery_daily.SMOKE_DIR / "RESULT.json",
            support_development_result=recovery_daily.DEVELOPMENT_DIR
            / "RESULT.json",
            policy_smoke_result=policy_daily.SMOKE_DIR / "RESULT.json",
            development_authorization=ready["development_authorization"],
            adapter=adapter,
            daily_budget_status=_budget_status(ledger, live, now=now),
        )
        verification = result_verify.verify_confirmation(RUN_DIR)
        checkpoint(RUN_DIR / "VERIFICATION.json", verification)
        if verification["status"] != "verified":
            raise RuntimeError("confirmation independent replay failed")
    except Exception:
        ledger = _reconcile(
            ledger,
            status="failed_closed",
            measured_cost=float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0)),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        status=result["status"],
        measured_cost=float(result["usage"]["combined_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    daily = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "confirmation_status": result["status"],
        "confirmation_result_sha256": recovery.sha256_file(
            RUN_DIR / "RESULT.json"
        ),
        "confirmation_verification_sha256": recovery.sha256_file(
            RUN_DIR / "VERIFICATION.json"
        ),
        "independent_replay_passed": True,
        "authorizes": (
            "confirmed_llm_native_nonmyopic_result"
            if result["status"] == "passed"
            else "nothing"
        ),
        "ledger_sha256": recovery.sha256_file(LEDGER),
        "recorded_daily_spend_usd": ledger["recorded_actual_spend_usd"],
        "model_calls_made": int(
            result["usage"]["deepseek_primary"]["adapter_requests"]
        ),
    }
    checkpoint(DAILY_RESULT, daily)
    return daily


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
            "confirmation_opened": False,
            "authorizes": "nothing",
        }
        if not args.preflight:
            checkpoint(ROOT / "DAILY_FAILURE.json", result)
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] in {
        "ready_without_paid_calls",
        "complete_reconciled",
    } else 1


if __name__ == "__main__":
    raise SystemExit(main())
