#!/usr/bin/env python3
"""Execute the sealed Aug 10 RegretBench SMC confirmation branch."""

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

from helpers import Config, ModelSpec
from scripts import bongard_openworld_luna_aug10_execute as bongard
from scripts import regretbench_deepseek_smc_confirmation as confirmation
from scripts import regretbench_deepseek_smc_confirmation_verify as verifier
from scripts import regretbench_deepseek_smc_dynamic_depth2_daily as development_daily
from scripts import regretbench_deepseek_smc_dynamic_depth2_experiment as experiment
from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_smc_dynamic_depth2_verify as development_verify
from scripts import regretbench_deepseek_smc_frozen_report as development_report
from scripts import regretbench_deepseek_support_recovery as primary
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-confirmation-daily-1"
DATE = "2026-08-10"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
ROOT = REPO_ROOT / (
    "results/nonmyopic/regretbench_deepseek_smc_dynamic_depth2_confirmation"
)
PARENT_DIR = ROOT / "parent-bank-20260810"
RUN_DIR = ROOT / "confirmation-20260810"
DAILY_RESULT = ROOT / "DAILY_RESULT.json"
LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-10-regretbench-smc-confirmation.json"
)
DEVELOPMENT_DIR = development_daily.DEVELOPMENT_DIR
DEVELOPMENT_RESULT = DEVELOPMENT_DIR / "RESULT.json"
DEVELOPMENT_VERIFICATION = DEVELOPMENT_DIR / "VERIFICATION.json"
DEVELOPMENT_REPORT = DEVELOPMENT_DIR / "FROZEN_REPORT.json"
MAX_REQUEST_COST_USD = development_daily.MAX_REQUEST_COST_USD
MIN_RESERVED_PROMPT_TOKENS = development_daily.MIN_RESERVED_PROMPT_TOKENS


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"SMC confirmation can run only on {DATE}")
    return local


def _pristine(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_file():
        return False
    return not any(path.iterdir())


def _assert_bongard_unopened() -> None:
    paths = [
        bongard.OUTPUT_DIR,
        bongard.SERVING_DIR,
        bongard.MECHANICS_DIR,
        bongard.DAILY_LEDGER,
    ]
    if any(not _pristine(path) for path in paths):
        raise RuntimeError("Aug 10 Bongard branch is already open")


def validate_development_predecessor() -> dict[str, Any]:
    required = [
        DEVELOPMENT_RESULT,
        DEVELOPMENT_VERIFICATION,
        DEVELOPMENT_REPORT,
        development_daily.DAILY_RESULT,
        development_daily.LEDGER,
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("complete SMC development artifacts are required")
    result = _load(DEVELOPMENT_RESULT)
    stored = _load(DEVELOPMENT_VERIFICATION)
    report = _load(DEVELOPMENT_REPORT)
    daily = _load(development_daily.DAILY_RESULT)
    replay = development_verify.verify(
        DEVELOPMENT_DIR,
        primary_dir=development_daily.PRIMARY_DEVELOPMENT_DIR,
    )
    rebuilt_report = development_report.build_report(
        DEVELOPMENT_DIR,
        primary_dir=development_daily.PRIMARY_DEVELOPMENT_DIR,
    )
    if (
        result.get("status") != "passed"
        or result.get("authorizes") != "smc_confirmation_preregistration_only"
        or stored.get("status") != "verified"
        or stored.get("mismatches") != []
        or replay.get("status") != "verified"
        or replay.get("mismatches") != []
        or daily.get("status") != "complete_reconciled"
        or daily.get("development_status") != "passed"
        or daily.get("independent_replay_passed") is not True
        or daily.get("confirmation_opened") is not False
        or report.get("claim_tier")
        != "smc_provisional_development_signal_confirmation_required"
        or rebuilt_report != report
    ):
        raise RuntimeError("SMC development does not authorize confirmation")
    return {
        "status": "authorized",
        "development_status": "passed",
        "independent_replay_status": "verified",
        "report_tier": report["claim_tier"],
        "development_result_sha256": primary.sha256_file(DEVELOPMENT_RESULT),
        "development_verification_sha256": primary.sha256_file(
            DEVELOPMENT_VERIFICATION
        ),
        "development_report_sha256": primary.sha256_file(DEVELOPMENT_REPORT),
        "development_daily_sha256": primary.sha256_file(
            development_daily.DAILY_RESULT
        ),
        "development_ledger_sha256": primary.sha256_file(
            development_daily.LEDGER
        ),
    }


def _validated_existing_complete() -> dict[str, Any] | None:
    if not DAILY_RESULT.exists():
        return None
    required = [
        PARENT_DIR / "RESULT.json",
        PARENT_DIR / "VERIFICATION.json",
        RUN_DIR / "RESULT.json",
        RUN_DIR / "VERIFICATION.json",
        LEDGER,
    ]
    if not all(path.is_file() for path in required):
        raise RuntimeError("partial SMC confirmation artifacts require manual audit")
    daily = _load(DAILY_RESULT)
    replay = verifier.verify(RUN_DIR, parent_dir=PARENT_DIR)
    stored = _load(RUN_DIR / "VERIFICATION.json")
    if (
        daily.get("status") != "complete_reconciled"
        or replay.get("status") != "verified"
        or stored != replay
        or daily.get("result_sha256")
        != primary.sha256_file(RUN_DIR / "RESULT.json")
        or daily.get("verification_sha256")
        != primary.sha256_file(RUN_DIR / "VERIFICATION.json")
        or daily.get("ledger_sha256") != primary.sha256_file(LEDGER)
    ):
        raise RuntimeError("completed SMC confirmation replay failed")
    return daily


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = bongard.read_openrouter_model_catalog,
) -> dict[str, Any]:
    _validate_date(now)
    existing = _validated_existing_complete()
    if existing is not None:
        return {
            "status": "already_complete_verified",
            "daily_result": existing,
            "model_calls_made": 0,
            "files_written": 0,
        }
    confirmation.validate_protocol_binding()
    authorization = validate_development_predecessor()
    _assert_bongard_unopened()
    for path in (PARENT_DIR, RUN_DIR, DAILY_RESULT, LEDGER):
        if not _pristine(path):
            raise RuntimeError(f"SMC confirmation output is not pristine: {path}")
    model = development_daily._validate_deepseek(catalog_reader())
    live = live_reader()
    if float(live["balance_usd"]) + 1e-12 < confirmation.RUN_BUDGET_USD:
        raise RuntimeError("insufficient OpenRouter balance for SMC confirmation")
    return {
        "status": "ready_without_paid_calls",
        "authorization": authorization,
        "model": model,
        "live_credits": live,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "opening_total_usage_boundary_usd": float(live["total_usage_usd"]),
            "parent_cap_usd": confirmation.PARENT_BUDGET_USD,
            "policy_cap_usd": confirmation.POLICY_BUDGET_USD,
            "aggregate_cap_usd": confirmation.RUN_BUDGET_USD,
            "remaining_after_full_cap_usd": DAILY_CAP_USD
            - confirmation.RUN_BUDGET_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def _initial_ledger(ready: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_usage_usd": ready["budget"][
            "opening_total_usage_boundary_usd"
        ],
        "recorded_actual_spend_usd": 0.0,
        "account_wide_usage_counts_against_cap": True,
        "development_authorization": ready["authorization"],
        "stages": {
            "parent_bank": {"status": "pending", "maximum_cost_usd": 0.20},
            "confirmation": {"status": "pending", "maximum_cost_usd": 3.50},
        },
    }


def _budget_status(
    ledger: Mapping[str, Any],
    *,
    projected: float,
    live: Mapping[str, float],
    now: datetime | None,
) -> dict[str, Any]:
    value = require_budget(
        dict(ledger),
        projected_cost_usd=projected,
        total_usage_usd=float(live["total_usage_usd"]),
        now=now,
    )
    value["live_balance_usd"] = float(live["balance_usd"])
    if value["live_balance_usd"] + 1e-12 < projected:
        raise RuntimeError("live balance is below the requested stage cap")
    return value


def _usage(adapter: Any) -> dict[str, Any]:
    return experiment.summarize_usage(adapter.usage_snapshot())


def _reconcile(
    ledger: Mapping[str, Any],
    *,
    stage: str,
    status: str,
    measured_cost: float,
    live: Mapping[str, float],
) -> dict[str, Any]:
    value = json.loads(json.dumps(ledger))
    posted = max(
        0.0,
        float(live["total_usage_usd"])
        - float(value["opening_total_usage_usd"]),
    )
    recorded = max(
        posted,
        float(value.get("recorded_actual_spend_usd", 0.0)),
        float(measured_cost)
        + sum(
            float(row.get("actual_cost_usd", 0.0))
            for name, row in value["stages"].items()
            if name != stage
        ),
    )
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("SMC confirmation exceeded the account-wide daily cap")
    value["stages"][stage].update(
        {"status": status, "actual_cost_usd": float(measured_cost)}
    )
    value["recorded_actual_spend_usd"] = recorded
    value["latest_total_usage_usd"] = float(live["total_usage_usd"])
    value["latest_balance_usd"] = float(live["balance_usd"])
    return value


def build_adapter(
    *, stage: str, run_id: str, output_dir: Path
) -> PerRequestSeedStructuredAdapter:
    if stage not in {"parent", "confirmation"}:
        raise ValueError("invalid SMC confirmation adapter stage")
    parent = stage == "parent"
    max_tokens = primary.MAX_TOKENS if parent else core.MAX_TOKENS
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=275.0,
        openrouter_run_budget_usd=confirmation.RUN_BUDGET_USD,
        openrouter_projected_cost_usd=(0.05 if parent else 3.10),
        openrouter_concurrency=64 if parent else 128,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=max_tokens,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=core.MODEL_ID, backend="openrouter", max_model_len=65_536),
        config,
    )


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = bongard.read_openrouter_model_catalog,
) -> dict[str, Any]:
    ready = preflight(now=now, live_reader=live_reader, catalog_reader=catalog_reader)
    if ready["status"] == "already_complete_verified":
        return dict(ready["daily_result"])
    ledger = _initial_ledger(ready)
    checkpoint(LEDGER, ledger)
    run_id = "regretbench-deepseek-smc-confirmation-20260810"

    parent_live = live_reader()
    parent_budget = _budget_status(
        ledger,
        projected=confirmation.PARENT_BUDGET_USD,
        live=parent_live,
        now=now,
    )
    parent_adapter = build_adapter(stage="parent", run_id=run_id, output_dir=PARENT_DIR)
    try:
        parent = confirmation.build_parent_bank(
            output_dir=PARENT_DIR,
            adapter=parent_adapter,
            daily_budget_status=parent_budget,
        )
        parent_verification = verifier.verify_parent_bank(PARENT_DIR)
        checkpoint(PARENT_DIR / "VERIFICATION.json", parent_verification)
        if parent_verification["status"] != "verified":
            raise RuntimeError("SMC confirmation parent replay failed")
    except Exception:
        ledger = _reconcile(
            ledger,
            stage="parent_bank",
            status="failed_closed",
            measured_cost=float(_usage(parent_adapter)["run_cost_usd"]),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="parent_bank",
        status=parent["status"],
        measured_cost=float(parent["usage"]["run_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    if parent["status"] != "passed":
        daily = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "parent_bank_stopped",
            "authorizes": "nothing",
            "parent_status": parent["status"],
            "confirmation_opened": False,
            "ledger_sha256": primary.sha256_file(LEDGER),
        }
        checkpoint(DAILY_RESULT, daily)
        return daily

    policy_live = live_reader()
    policy_budget = _budget_status(
        ledger,
        projected=confirmation.POLICY_BUDGET_USD,
        live=policy_live,
        now=now,
    )
    policy_adapter = build_adapter(
        stage="confirmation", run_id=run_id, output_dir=RUN_DIR
    )
    try:
        result = confirmation.run_confirmation(
            output_dir=RUN_DIR,
            parent_dir=PARENT_DIR,
            adapter=policy_adapter,
            development_authorization=ready["authorization"],
            daily_budget_status=policy_budget,
        )
        replay = verifier.verify(RUN_DIR, parent_dir=PARENT_DIR)
        checkpoint(RUN_DIR / "VERIFICATION.json", replay)
        if replay["status"] != "verified":
            raise RuntimeError("SMC confirmation independent replay failed")
    except Exception:
        ledger = _reconcile(
            ledger,
            stage="confirmation",
            status="failed_closed",
            measured_cost=float(_usage(policy_adapter)["run_cost_usd"]),
            live=live_reader(),
        )
        checkpoint(LEDGER, ledger)
        raise
    ledger = _reconcile(
        ledger,
        stage="confirmation",
        status=result["status"],
        measured_cost=float(result["usage"]["combined_cost_usd"]),
        live=live_reader(),
    )
    checkpoint(LEDGER, ledger)
    daily = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "parent_status": parent["status"],
        "confirmation_status": result["status"],
        "independent_replay_passed": True,
        "authorizes": result["authorizes"],
        "bongard_aug10_opened": False,
        "result_sha256": primary.sha256_file(RUN_DIR / "RESULT.json"),
        "verification_sha256": primary.sha256_file(RUN_DIR / "VERIFICATION.json"),
        "parent_result_sha256": primary.sha256_file(PARENT_DIR / "RESULT.json"),
        "parent_verification_sha256": primary.sha256_file(
            PARENT_DIR / "VERIFICATION.json"
        ),
        "ledger_sha256": primary.sha256_file(LEDGER),
    }
    checkpoint(DAILY_RESULT, daily)
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
