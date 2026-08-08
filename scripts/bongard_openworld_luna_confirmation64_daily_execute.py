#!/usr/bin/env python3
"""Execute one frozen Bongard confirmation96 daily block."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import math
from pathlib import Path
import sys
import tempfile
from typing import Any, Callable, Mapping, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_aug10_execute as aug10
from scripts import bongard_openworld_luna_confirmation64 as confirmation
from scripts import bongard_openworld_luna_confirmation64_execute_verify as execute_verify
from scripts import bongard_openworld_luna_vlm_development as development
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-confirmation96-daily-execute-5"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
MAX_ACCEPTED_RESPONSES = confirmation.MAX_REQUESTS_PER_BLOCK
MAX_HTTP_ATTEMPTS = confirmation.MAX_HTTP_ATTEMPTS_PER_BLOCK
MAX_PRECHARGED_EXPOSURE_USD = (
    MAX_HTTP_ATTEMPTS * confirmation.serving.MAX_REQUEST_COST_USD
)
ROOT = REPO_ROOT / "results/nonmyopic/bongard_openworld_luna_confirmation64"
BLOCK_DIRS = {
    block_id: ROOT
    / f"block-{block_id}-{confirmation.BLOCK_DATES[block_id].replace('-', '')}"
    for block_id in confirmation.BLOCK_ORDER
}
BLOCK_RUN_IDS = {
    block_id: (
        f"bongard-openworld-luna-confirmation64-{block_id}-"
        f"{confirmation.BLOCK_DATES[block_id].replace('-', '')}"
    )
    for block_id in confirmation.BLOCK_ORDER
}
LEDGERS = {
    block_id: REPO_ROOT
    / "results/nonmyopic/openrouter_daily_budget"
    / f"{confirmation.BLOCK_DATES[block_id]}.json"
    for block_id in confirmation.BLOCK_ORDER
}
COMBINED_RESULT = ROOT / "COMBINED_RESULT.json"
MECHANICS_RESULT = confirmation.freeze_verify.MECHANICS_RESULT


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return confirmation.sha256_file(path)


def _validate_date(block_id: str, now: datetime | None = None) -> datetime:
    if block_id not in confirmation.BLOCK_ORDER:
        raise ValueError(f"unknown confirmation block {block_id!r}")
    timezone = ZoneInfo(TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    expected = confirmation.BLOCK_DATES[block_id]
    if local_now.date().isoformat() != expected:
        raise RuntimeError(f"confirmation block {block_id} can run only on {expected}")
    return local_now


def _result_paths() -> dict[str, Path]:
    return {
        block_id: BLOCK_DIRS[block_id] / "RESULT.json"
        for block_id in confirmation.BLOCK_ORDER
    }


def _initialize_ledger(
    *, path: Path, live: Mapping[str, float], block_id: str, now: datetime
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"confirmation ledger already exists: {path}")
    ledger = {
        "schema_version": SCHEMA_VERSION,
        "date": now.date().isoformat(),
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": float(live["total_credits_usd"]),
        "opening_total_usage_usd": float(live["total_usage_usd"]),
        "opening_balance_usd": float(live["balance_usd"]),
        "opening_frozen_at_london": now.isoformat(),
        "recorded_actual_spend_usd": 0.0,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "first_authorized_block": {
            "interface_version": confirmation.INTERFACE_VERSION,
            "block_id": block_id,
            "model": confirmation.MODEL_ID,
            "maximum_cost_usd": confirmation.RUN_BUDGET_USD,
            "maximum_accepted_responses": MAX_ACCEPTED_RESPONSES,
            "maximum_http_attempts": MAX_HTTP_ATTEMPTS,
            "maximum_precharged_exposure_usd": MAX_PRECHARGED_EXPOSURE_USD,
            "status": "authorized_pending",
        },
        "additional_paid_blocks_authorized": False,
        "protocol_manifest_sha256": confirmation.PROTOCOL_MANIFEST_SHA256,
    }
    checkpoint(path, ledger)
    return ledger


def _reconcile(
    *,
    ledger: Mapping[str, Any],
    block_id: str,
    measured_cost_usd: float,
    live_after: Mapping[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    key = f"bongard_luna_confirmation64_block_{block_id}"
    old_cost = float((updated.get(key) or {}).get("actual_cost_usd", 0.0))
    block_cost = max(old_cost, measured_cost_usd)
    updated["recorded_actual_spend_usd"] = recorded
    updated[key] = {
        "interface_version": confirmation.INTERFACE_VERSION,
        "model": confirmation.MODEL_ID,
        "status": status,
        "actual_cost_usd": block_cost,
        "maximum_cost_usd": confirmation.RUN_BUDGET_USD,
    }
    authorization = updated["first_authorized_block"]
    authorization["status"] = status
    authorization["actual_cost_usd"] = block_cost
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(0.0, DAILY_CAP_USD - recorded),
    }
    return updated


def validate_ledger(*, path: Path, block_id: str) -> dict[str, Any]:
    ledger = _load(path)
    key = f"bongard_luna_confirmation64_block_{block_id}"
    block = ledger.get(key) or {}
    authorization = ledger.get("first_authorized_block") or {}
    reconciliation = ledger.get("reconciliation") or {}
    recorded = float(ledger.get("recorded_actual_spend_usd", math.inf))
    block_cost = float(block.get("actual_cost_usd", math.inf))
    remaining = float(reconciliation.get("remaining_daily_allowance_usd", math.inf))
    if (
        ledger.get("schema_version") != SCHEMA_VERSION
        or ledger.get("date") != confirmation.BLOCK_DATES[block_id]
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("unspent_allowance_does_not_roll_over") is not True
        or ledger.get("additional_paid_blocks_authorized") is not False
        or ledger.get("protocol_manifest_sha256")
        != confirmation.PROTOCOL_MANIFEST_SHA256
        or not 0.0 <= recorded <= DAILY_CAP_USD
        or not 0.0 <= block_cost <= confirmation.RUN_BUDGET_USD
        or authorization.get("interface_version")
        != confirmation.INTERFACE_VERSION
        or authorization.get("block_id") != block_id
        or authorization.get("model") != confirmation.MODEL_ID
        or float(authorization.get("maximum_cost_usd", math.inf))
        != confirmation.RUN_BUDGET_USD
        or authorization.get("maximum_accepted_responses")
        != MAX_ACCEPTED_RESPONSES
        or authorization.get("maximum_http_attempts") != MAX_HTTP_ATTEMPTS
        or float(
            authorization.get("maximum_precharged_exposure_usd", math.inf)
        )
        != MAX_PRECHARGED_EXPOSURE_USD
        or authorization.get("status") != "block_mechanics_pass"
        or float(authorization.get("actual_cost_usd", math.inf)) != block_cost
        or block.get("interface_version") != confirmation.INTERFACE_VERSION
        or block.get("model") != confirmation.MODEL_ID
        or float(block.get("maximum_cost_usd", math.inf))
        != confirmation.RUN_BUDGET_USD
        or block.get("status") != "block_mechanics_pass"
        or reconciliation.get("recorded_spend_is_max_of_posted_and_local") is not True
        or remaining != max(0.0, DAILY_CAP_USD - recorded)
    ):
        raise RuntimeError(f"confirmation block {block_id} ledger is invalid")
    return {
        "verified": True,
        "ledger_sha256": _sha256(path),
        "recorded_daily_spend_usd": recorded,
        "block_cost_usd": block_cost,
    }


def validate_block_result(
    *, path: Path, block_id: str, ledger_path: Path
) -> dict[str, Any]:
    replay = confirmation.replay_block(result_path=path)
    if replay["block_id"] != block_id:
        raise RuntimeError("confirmation result block ID changed")
    ledger = validate_ledger(path=ledger_path, block_id=block_id)
    execution_path = path.parent / "EXECUTION.json"
    execution = _load(execution_path)
    if (
        execution.get("interface_version") != confirmation.INTERFACE_VERSION
        or execution.get("block_id") != block_id
        or execution.get("status") != "complete_reconciled"
        or execution.get("result_sha256") != replay["result_sha256"]
        or execution.get("ledger_sha256") != ledger["ledger_sha256"]
    ):
        raise RuntimeError("confirmation execution record is invalid")
    return {
        "verified": True,
        "block_id": block_id,
        "result_sha256": replay["result_sha256"],
        "raw_responses_sha256": replay["raw_responses_sha256"],
        "execution_sha256": _sha256(execution_path),
        "ledger_sha256": ledger["ledger_sha256"],
        "recorded_daily_spend_usd": ledger["recorded_daily_spend_usd"],
    }


def _validate_daily_record(
    *, path: Path, block_id: str, block_verification: Mapping[str, Any]
) -> dict[str, Any]:
    daily = _load(path)
    combined_accessed = daily.get("combined_endpoint_accessed")
    expected_status = (
        "complete_combined_verified"
        if combined_accessed is True
        else "block_complete_verified"
    )
    if (
        daily.get("interface_version") != INTERFACE_VERSION
        or daily.get("status") != expected_status
        or daily.get("block_id") != block_id
        or daily.get("date") != confirmation.BLOCK_DATES[block_id]
        or daily.get("block_verification") != block_verification
        or type(combined_accessed) is not bool
        or daily.get("sealed_test_accessed") is not False
        or daily.get("reserve_accessed") is not False
        or (block_id != "d" and combined_accessed is not False)
    ):
        raise RuntimeError("banked confirmation daily record changed")
    return daily


def preflight_daily_block(
    *,
    block_id: str,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        aug10.read_openrouter_model_catalog
    ),
) -> dict[str, Any]:
    if block_id not in confirmation.BLOCK_ORDER:
        raise ValueError(f"unknown confirmation block {block_id!r}")
    execution_bindings = execute_verify.verify_execution_bindings()
    manifest = confirmation.verify_protocol_manifest()
    paths = _result_paths()
    target_index = confirmation.BLOCK_ORDER.index(block_id)
    for future in confirmation.BLOCK_ORDER[target_index + 1 :]:
        if LEDGERS[future].exists() or (
            BLOCK_DIRS[future].exists() and any(BLOCK_DIRS[future].iterdir())
        ):
            raise RuntimeError("future confirmation block exists out of order")
    try:
        development_authorization = confirmation.verify_development_authorization()
    except RuntimeError as exc:
        if str(exc) != "development claim report is missing":
            raise
        development_authorization = {
            "verified": False,
            "status": "waiting_for_full_development_tier",
        }
    prior = {}
    waiting_for = None if development_authorization["verified"] else "development"
    for prior_id in confirmation.BLOCK_ORDER[:target_index]:
        result_path = paths[prior_id]
        daily_path = result_path.parent / "DAILY_EXECUTION.json"
        if not (result_path.is_file() and daily_path.is_file() and LEDGERS[prior_id].is_file()):
            if any((result_path.exists(), daily_path.exists(), LEDGERS[prior_id].exists())):
                raise RuntimeError(f"confirmation predecessor {prior_id} is partial")
            waiting_for = waiting_for or f"block_{prior_id}"
            break
        verification = validate_block_result(
            path=result_path, block_id=prior_id, ledger_path=LEDGERS[prior_id]
        )
        _validate_daily_record(
            path=daily_path,
            block_id=prior_id,
            block_verification=verification,
        )
        prior[prior_id] = verification
    runtime = None
    if waiting_for is None:
        block_dir = BLOCK_DIRS[block_id]
        if LEDGERS[block_id].exists() or (
            block_dir.exists() and any(block_dir.iterdir())
        ):
            raise RuntimeError("fresh confirmation block path is not pristine")
        live = live_reader()
        model = aug10._validate_model_catalog(model_catalog_reader())
        if float(live["balance_usd"]) + 1e-12 < DAILY_CAP_USD:
            raise RuntimeError("live balance is below confirmation $5 start gate")
        runtime = {"live_credits": live, "model": model}
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "ready_without_paid_calls" if waiting_for is None else f"waiting_for_{waiting_for}"
        ),
        "block_id": block_id,
        "date": confirmation.BLOCK_DATES[block_id],
        "protocol_manifest": manifest,
        "execution_bindings": execution_bindings,
        "development_authorization": development_authorization,
        "prior_blocks": prior,
        "runtime": runtime,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "run_cap_usd": confirmation.RUN_BUDGET_USD,
            "maximum_accepted_responses": MAX_ACCEPTED_RESPONSES,
            "maximum_http_attempts": MAX_HTTP_ATTEMPTS,
            "maximum_precharged_exposure_usd": MAX_PRECHARGED_EXPOSURE_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def execute_daily_block(
    *,
    block_id: str,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        aug10.read_openrouter_model_catalog
    ),
    block_runner: Callable[..., dict[str, Any]] = confirmation.run_block,
    analyzer: Callable[..., dict[str, Any]] = confirmation.analyze_combined,
) -> dict[str, Any]:
    local_now = _validate_date(block_id, now)
    execution_bindings = execute_verify.verify_execution_bindings()
    manifest = confirmation.verify_protocol_manifest()
    development_authorization = confirmation.verify_development_authorization()
    paths = _result_paths()
    target_index = confirmation.BLOCK_ORDER.index(block_id)
    previous = []
    prior_verifications = {}
    for prior_id in confirmation.BLOCK_ORDER[:target_index]:
        path = paths[prior_id]
        verification = validate_block_result(
            path=path, block_id=prior_id, ledger_path=LEDGERS[prior_id]
        )
        _validate_daily_record(
            path=path.parent / "DAILY_EXECUTION.json",
            block_id=prior_id,
            block_verification=verification,
        )
        previous.append(path)
        prior_verifications[prior_id] = verification
    block_dir = BLOCK_DIRS[block_id]
    result_path = paths[block_id]
    daily_path = block_dir / "DAILY_EXECUTION.json"
    if daily_path.exists():
        verification = validate_block_result(
            path=result_path, block_id=block_id, ledger_path=LEDGERS[block_id]
        )
        daily_record = _validate_daily_record(
            path=daily_path, block_id=block_id, block_verification=verification
        )
    else:
        if block_dir.exists() and any(block_dir.iterdir()):
            raise RuntimeError("partial confirmation block cannot repeat")
        live = live_reader()
        aug10._validate_model_catalog(model_catalog_reader())
        if float(live["balance_usd"]) + 1e-12 < DAILY_CAP_USD:
            raise RuntimeError("live balance is below confirmation $5 start gate")
        ledger = _initialize_ledger(
            path=LEDGERS[block_id], live=live, block_id=block_id, now=local_now
        )
        require_budget(
            ledger,
            projected_cost_usd=confirmation.RUN_BUDGET_USD,
            total_usage_usd=float(live["total_usage_usd"]),
            now=local_now,
        )
        mechanics_verification = development.verify_mechanics_result(MECHANICS_RESULT)
        projected = (
            mechanics_verification["cost_usd"]
            / mechanics_verification["request_count"]
            * confirmation.MAX_REQUESTS_PER_BLOCK
            * 1.5
        )
        if projected > confirmation.RUN_BUDGET_USD + 1e-12:
            raise RuntimeError("mechanics cost projection exceeds confirmation cap")
        try:
            result = block_runner(
                output_dir=block_dir,
                run_id=BLOCK_RUN_IDS[block_id],
                block_id=block_id,
                mechanics_result=MECHANICS_RESULT,
                protocol_manifest=confirmation.PROTOCOL_MANIFEST,
            )
        except Exception as exc:
            try:
                checkpoint(
                    LEDGERS[block_id],
                    _reconcile(
                        ledger=ledger,
                        block_id=block_id,
                        measured_cost_usd=0.0,
                        live_after=live_reader(),
                        status="failed_closed_posted_spend_reconciled",
                    ),
                )
            except Exception:
                pass
            block_dir.mkdir(parents=True, exist_ok=True)
            checkpoint(
                block_dir / "FAILURE.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "interface_version": INTERFACE_VERSION,
                    "status": "failed_closed",
                    "block_id": block_id,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            raise
        local = _reconcile(
            ledger=ledger,
            block_id=block_id,
            measured_cost_usd=float(result["usage"]["run_cost_usd"]),
            live_after=live,
            status=result["status"],
        )
        checkpoint(LEDGERS[block_id], local)
        final = _reconcile(
            ledger=local,
            block_id=block_id,
            measured_cost_usd=0.0,
            live_after=live_reader(),
            status=result["status"],
        )
        checkpoint(LEDGERS[block_id], final)
        checkpoint(
            block_dir / "EXECUTION.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": confirmation.INTERFACE_VERSION,
                "status": "complete_reconciled",
                "block_id": block_id,
                "result_sha256": _sha256(result_path),
                "ledger_sha256": _sha256(LEDGERS[block_id]),
                "recorded_daily_spend_usd": final["recorded_actual_spend_usd"],
            },
        )
        verification = validate_block_result(
            path=result_path, block_id=block_id, ledger_path=LEDGERS[block_id]
        )
        daily_record = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "block_complete_verified",
            "block_id": block_id,
            "date": confirmation.BLOCK_DATES[block_id],
            "protocol_manifest_sha256": manifest["manifest_sha256"],
            "execution_bindings": execution_bindings,
            "development_authorization": development_authorization,
            "prior_block_verifications": prior_verifications,
            "block_verification": verification,
            "combined_endpoint_accessed": False,
            "sealed_test_accessed": False,
            "reserve_accessed": False,
        }
        checkpoint(daily_path, daily_record)
    if block_id != confirmation.BLOCK_ORDER[-1]:
        if COMBINED_RESULT.exists():
            raise RuntimeError("combined confirmation opened before block D")
        return daily_record
    all_paths = [paths[item] for item in confirmation.BLOCK_ORDER]
    all_verifications = {
        item: validate_block_result(
            path=paths[item], block_id=item, ledger_path=LEDGERS[item]
        )
        for item in confirmation.BLOCK_ORDER
    }
    if not COMBINED_RESULT.exists():
        analyzer(block_results=all_paths, output_path=COMBINED_RESULT)
    combined_verification = confirmation.verify_combined_result(
        result_path=COMBINED_RESULT, block_results=all_paths
    )
    if daily_record["combined_endpoint_accessed"] is True:
        if (
            daily_record.get("combined_verification") != combined_verification
            or daily_record.get("all_block_verifications") != all_verifications
        ):
            raise RuntimeError("banked combined confirmation changed")
        return daily_record
    daily_record["status"] = "complete_combined_verified"
    daily_record["combined_endpoint_accessed"] = True
    daily_record["all_block_verifications"] = all_verifications
    daily_record["combined_verification"] = combined_verification
    daily_record["claim_tier"] = combined_verification["claim_tier"]
    checkpoint(daily_path, daily_record)
    return daily_record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block", choices=confirmation.BLOCK_ORDER, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    function = preflight_daily_block if args.preflight else execute_daily_block
    print(json.dumps(function(block_id=args.block), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
