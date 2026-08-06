#!/usr/bin/env python3
"""Execute one exact daily block of the staged diversity confirmation."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen
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
CLAIM_PLAN = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_TWO_DRAW_DIVERSITY_BONUS_CONFIRMATION64_CLAIM_PLAN.md"
)
CLAIM_PLAN_SHA256 = (
    "c90ede19c9438917e2d8ac198211d2da6c5d0bae788bf77d5a9a7b283edf4b8a"
)
MODELS_URL = "https://openrouter.ai/api/v1/models"
PLANNING_MODEL_ID = "qwen/qwen3.7-plus"
TARGET_MODEL_ID = "google/gemini-2.5-flash"
MAX_OUTPUT_TOKENS = 4_200
PROTOCOL_INPUT_SHA256 = (
    "041994f4de92b573c511414a293c049655a6adec6321189f522246b0f1ea6eba"
)


class PreExecutionGateError(RuntimeError):
    """Raised when a pristine paid block fails its read-only launch gate."""


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


def read_openrouter_model_catalog() -> dict[str, Any]:
    headers = {}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(MODELS_URL, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _protocol_inputs() -> dict[str, Any]:
    return {
        "run_id": RUN_ID,
        "block_dates": BLOCK_DATES,
        "expected_requests_per_block": staged.EXPECTED_REQUESTS_PER_BLOCK,
        "expected_requests_total": staged.EXPECTED_REQUESTS_TOTAL,
        "daily_cap_usd": staged.DAILY_CAP_USD,
        "minimum_starting_balance_usd": staged.MIN_STARTING_BALANCE_USD,
        "combined_bootstrap_seed": staged.COMBINED_BOOTSTRAP_SEED,
        "preregistration_sha256": staged.PREREGISTRATION_SHA256,
        "planning_model": PLANNING_MODEL_ID,
        "target_model": TARGET_MODEL_ID,
        "reasoning": False,
        "blocks": {
            block: {
                name: list(spec[name])
                if isinstance(spec[name], tuple)
                else spec[name]
                for name in (
                    "tree_seeds",
                    "target_seeds",
                    "validation_seed_start",
                    "source_bootstrap_seed",
                )
            }
            for block, spec in staged.BLOCKS.items()
        },
    }


def _verify_protocol_inputs() -> dict[str, Any]:
    if audit.sha256_file(EXECUTION_AMENDMENT) != EXECUTION_AMENDMENT_SHA256:
        raise ValueError("diversity execution amendment hash changed")
    if audit.sha256_file(CLAIM_PLAN) != CLAIM_PLAN_SHA256:
        raise ValueError("diversity claim plan hash changed")
    staged._validate_preregistration()
    staged.base.validate_predecessors()
    from scripts import number_game_qwen_planner_depth_three as qwen

    if (
        qwen.PLANNING_MODEL_ID != PLANNING_MODEL_ID
        or qwen.TARGET_MODEL_ID != TARGET_MODEL_ID
    ):
        raise RuntimeError("frozen Number Game model routing changed")
    protocol = _protocol_inputs()
    digest = hashlib.sha256(
        json.dumps(
            protocol, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    if digest != PROTOCOL_INPUT_SHA256:
        raise RuntimeError("frozen diversity protocol inputs changed")
    return {
        "verified": True,
        "protocol_input_sha256": digest,
        "execution_amendment_sha256": EXECUTION_AMENDMENT_SHA256,
        "claim_plan_sha256": CLAIM_PLAN_SHA256,
        "preregistration_sha256": staged.PREREGISTRATION_SHA256,
        "expected_requests_per_block": staged.EXPECTED_REQUESTS_PER_BLOCK,
        "expected_requests_total": staged.EXPECTED_REQUESTS_TOTAL,
        "planning_model": PLANNING_MODEL_ID,
        "target_model": TARGET_MODEL_ID,
    }


def _validate_model_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    records = {model.get("id"): model for model in catalog.get("data", [])}
    verified = {}
    for model_id in (PLANNING_MODEL_ID, TARGET_MODEL_ID):
        model = records.get(model_id)
        if model is None:
            raise RuntimeError(f"OpenRouter does not expose exact model {model_id}")
        modalities = set(
            (model.get("architecture") or {}).get("input_modalities") or []
        )
        supported = set(model.get("supported_parameters") or [])
        max_completion = int(
            (model.get("top_provider") or {}).get("max_completion_tokens") or 0
        )
        pricing = model.get("pricing") or {}
        prompt_price = float(pricing.get("prompt", math.nan))
        completion_price = float(pricing.get("completion", math.nan))
        if "text" not in modalities:
            raise RuntimeError(f"{model_id} no longer supports text input")
        if not ({"response_format", "structured_outputs"} & supported):
            raise RuntimeError(f"{model_id} no longer supports structured output")
        if max_completion < MAX_OUTPUT_TOKENS:
            raise RuntimeError(f"{model_id} completion limit is too small")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (prompt_price, completion_price)
        ):
            raise RuntimeError(f"{model_id} pricing is missing or invalid")
        verified[model_id] = {
            "context_length": int(model.get("context_length") or 0),
            "max_completion_tokens": max_completion,
            "input_modalities": sorted(modalities),
            "supports_structured_output": True,
            "prompt_usd_per_million_tokens": round(
                prompt_price * 1_000_000, 12
            ),
            "completion_usd_per_million_tokens": round(
                completion_price * 1_000_000, 12
            ),
        }
    return verified


def _verify_aug7_predecessor(path: Path) -> dict[str, Any]:
    from scripts import number_game_budget_model_aug7_execute as aug7

    if path != aug7.CONTROL_RUN_DIR / "CONTROL_DAILY_EXECUTION.json":
        raise RuntimeError("August 7 control path changed")
    verification = aug7._verify_control_component(
        run_dir=path.parent,
        daily_ledger=aug7.DAILY_LEDGER,
    )
    if (
        verification.get("status") != "complete_verified"
        or verification.get("request_count") != 3_072
    ):
        raise RuntimeError("August 7 control did not independently verify")
    return verification


def _verify_block_a_predecessor(run_dir: Path) -> dict[str, Any]:
    from scripts import number_game_two_draw_diversity_bonus_confirmation64_verify as verify

    execution_path = run_dir / "BLOCK_A_DAILY_EXECUTION.json"
    stage_path = staged.block_stage_path(run_dir, "a")
    source_dir = staged.block_directory(run_dir, "a") / "source"
    stored_path = staged.block_a_verification_path(run_dir)
    ledger_path = LEDGER_PATHS["a"]
    execution = _load(execution_path)
    stage = _load(stage_path)
    source_result = _load(source_dir / "RESULT.json")
    source_artifacts = verify._source_hashes(source_dir)
    stored = _load(stored_path)
    ledger = _load(ledger_path)
    checks = verify.block_a_authorization_checks(
        stage=stage,
        source_result=source_result,
        source_artifacts=source_artifacts,
    )
    authorization = ledger.get("first_authorized_block") or {}
    block_record = ledger.get("diversity_bonus_confirmation64_block_a") or {}
    cost = float(execution.get("measured_cost_usd", math.nan))
    if (
        execution.get("status") != "complete"
        or execution.get("run_id") != RUN_ID
        or execution.get("block") != "a"
        or execution.get("block_stage_status") != "block_b_authorized"
        or execution.get("verification_status") != "verified"
        or execution.get("measured_requests")
        != staged.EXPECTED_REQUESTS_PER_BLOCK
        or not math.isfinite(cost)
        or not 0.0 <= cost <= staged.DAILY_CAP_USD
        or ledger.get("date") != BLOCK_DATES["a"]
        or ledger.get("timezone") != "Europe/London"
        or float(ledger.get("daily_cap_usd", 0.0)) != staged.DAILY_CAP_USD
        or float(ledger.get("recorded_actual_spend_usd", math.inf))
        > staged.DAILY_CAP_USD + 1e-12
        or authorization.get("interface_version") != staged.INTERFACE_VERSION
        or authorization.get("run_id") != RUN_ID
        or authorization.get("block") != "a"
        or authorization.get("status") != "complete"
        or not math.isclose(
            float(authorization.get("actual_cost_usd", math.nan)),
            cost,
            rel_tol=0.0,
            abs_tol=1e-12,
        )
        or block_record.get("status") != "complete"
        or not all(checks.values())
        or stored.get("interface_version") != verify.BLOCK_A_INTERFACE_VERSION
        or stored.get("status") != "verified"
        or stored.get("model_calls") != 0
        or float(stored.get("cost_usd", math.nan)) != 0.0
        or stored.get("checks") != checks
        or stored.get("failed_checks") != []
        or stored.get("block_a_stage_sha256") != audit.sha256_file(stage_path)
        or stored.get("block_a_source_artifacts") != source_artifacts
    ):
        raise RuntimeError("Block A predecessor does not independently replay")
    return {
        "verified": True,
        "status": execution["status"],
        "request_count": int(execution["measured_requests"]),
        "cost_usd": cost,
        "execution_sha256": audit.sha256_file(execution_path),
        "authorization_verification_sha256": audit.sha256_file(stored_path),
    }


def _verify_preflight_paths(
    *, block: str, run_dir: Path, ledger_path: Path
) -> dict[str, str]:
    if ledger_path.exists():
        raise RuntimeError(f"Block {block.upper()} daily ledger already exists")
    execution_path = run_dir / f"BLOCK_{block.upper()}_DAILY_EXECUTION.json"
    failure_path = run_dir / f"BLOCK_{block.upper()}_RUNNER_FAILURE.json"
    if execution_path.exists() or failure_path.exists():
        raise RuntimeError(f"Block {block.upper()} already has a terminal artifact")
    if _partial_block_exists(run_dir=run_dir, block=block):
        raise RuntimeError(f"partial Block {block.upper()} artifacts exist")
    if block == "a" and run_dir.exists() and any(run_dir.iterdir()):
        raise RuntimeError("Block A run directory is not pristine")
    return {
        "run_directory": "absent" if not run_dir.exists() else "present",
        "target_execution": "absent",
        "target_failure": "absent",
        "target_daily_ledger": "absent",
    }


def preflight_formal_block(
    *,
    block: str,
    run_dir: Path = RUN_DIR,
    ledger_path: Path | None = None,
    control_execution_path: Path = CONTROL_EXECUTION,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        read_openrouter_model_catalog
    ),
    protocol_validator: Callable[[], dict[str, Any]] = _verify_protocol_inputs,
    aug7_validator: Callable[[Path], dict[str, Any]] = _verify_aug7_predecessor,
    block_a_validator: Callable[[Path], dict[str, Any]] = (
        _verify_block_a_predecessor
    ),
) -> dict[str, Any]:
    """Check a future daily block without writing files or calling a model."""
    if block not in BLOCK_DATES:
        raise ValueError(f"unknown block: {block}")
    ledger_path = ledger_path or LEDGER_PATHS[block]
    paths = _verify_preflight_paths(
        block=block, run_dir=run_dir, ledger_path=ledger_path
    )
    protocol = protocol_validator()
    if block == "a":
        if control_execution_path.exists():
            predecessor = aug7_validator(control_execution_path)
            predecessor["kind"] = "verified_aug7_control"
        else:
            predecessor = {
                "verified": False,
                "status": "waiting_for_aug7_control",
                "kind": "aug7_control",
                "path": str(control_execution_path),
            }
    else:
        block_a_execution = run_dir / "BLOCK_A_DAILY_EXECUTION.json"
        if block_a_execution.exists():
            predecessor = block_a_validator(run_dir)
            predecessor["kind"] = "verified_block_a"
        elif run_dir.exists() and any(run_dir.iterdir()):
            raise RuntimeError("Block A is partial or lacks daily completion")
        else:
            predecessor = {
                "verified": False,
                "status": "waiting_for_block_a",
                "kind": "block_a",
                "path": str(block_a_execution),
            }

    models = _validate_model_catalog(model_catalog_reader())
    live = live_reader()
    credit_values = [
        float(live[name])
        for name in ("total_credits_usd", "total_usage_usd", "balance_usd")
    ]
    if not all(math.isfinite(value) for value in credit_values):
        raise RuntimeError("live OpenRouter credit values are non-finite")
    if float(live["balance_usd"]) + 1e-12 < staged.MIN_STARTING_BALANCE_USD:
        raise RuntimeError("live OpenRouter balance is below the $5 start gate")
    ready = predecessor["verified"] is True
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "ready_without_paid_calls" if ready else predecessor["status"]
        ),
        "block": block,
        "execution_date": BLOCK_DATES[block],
        "protocol": protocol,
        "predecessor": predecessor,
        "execution_paths": paths,
        "models": models,
        "live_credits": live,
        "budget": {
            "account_wide_daily_cap_usd": staged.DAILY_CAP_USD,
            "block_maximum_cost_usd": staged.DAILY_CAP_USD,
            "minimum_starting_balance_usd": staged.MIN_STARTING_BALANCE_USD,
            "expected_requests": staged.EXPECTED_REQUESTS_PER_BLOCK,
            "additional_paid_blocks_authorized": False,
            "unspent_allowance_does_not_roll_over": True,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


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

    claim_path = run_dir / report.CLAIM_REPORT_NAME
    report.write_report(
        run_dir=run_dir,
        output_path=report.REPORT_PATH,
        claim_path=claim_path,
    )
    return {
        "path": str(report.REPORT_PATH),
        "sha256": audit.sha256_file(report.REPORT_PATH),
        "claim_path": str(claim_path),
        "claim_sha256": audit.sha256_file(claim_path),
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
    fresh_preflight: Callable[..., dict[str, Any]] = preflight_formal_block,
) -> dict[str, Any]:
    if audit.sha256_file(EXECUTION_AMENDMENT) != EXECUTION_AMENDMENT_SHA256:
        raise ValueError("diversity execution amendment hash changed")
    if audit.sha256_file(CLAIM_PLAN) != CLAIM_PLAN_SHA256:
        raise ValueError("diversity claim plan hash changed")
    _validate_control_predecessor(control_execution_path)
    ledger_path = ledger_path or LEDGER_PATHS[block]
    execution_path = run_dir / f"BLOCK_{block.upper()}_DAILY_EXECUTION.json"
    failure_path = run_dir / f"BLOCK_{block.upper()}_RUNNER_FAILURE.json"
    if execution_path.exists():
        execution = _load(execution_path)
        if execution.get("status") != "complete":
            raise RuntimeError("banked daily execution is not complete")
        if block == "b":
            verified_report = reporter(run_dir=run_dir)
            if "verified_report" in execution:
                if execution["verified_report"] != verified_report:
                    raise RuntimeError("banked verified report record changed")
            else:
                execution["verified_report"] = verified_report
                checkpoint(execution_path, execution)
        return execution
    local_now = _validate_date(block=block, now=now)
    if failure_path.exists():
        raise RuntimeError(f"Block {block.upper()} already failed closed")
    if _partial_block_exists(run_dir=run_dir, block=block):
        raise RuntimeError(
            f"partial Block {block.upper()} artifacts exist; refusing rerun"
        )
    if ledger_path.exists():
        live_opening = live_reader()
    else:
        try:
            preflight = fresh_preflight(
                block=block,
                run_dir=run_dir,
                ledger_path=ledger_path,
                control_execution_path=control_execution_path,
                live_reader=live_reader,
            )
        except Exception as exc:
            raise PreExecutionGateError(str(exc)) from exc
        if preflight.get("status") != "ready_without_paid_calls":
            raise PreExecutionGateError(
                f"fresh Block {block.upper()} preflight did not authorize execution"
            )
        live_opening = preflight.get("live_credits")
        if not isinstance(live_opening, dict):
            raise PreExecutionGateError(
                "fresh preflight omitted the live credit snapshot"
            )
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
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    function = preflight_formal_block if args.preflight else execute_formal_block
    execution = function(block=args.block)
    print(json.dumps(execution, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
