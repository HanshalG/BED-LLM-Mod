#!/usr/bin/env python3
"""Execute the Tau2 MMS Gemma26B thinking calibration exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_mms_gemma26b_thinking_calibration as calibration
from scripts import tau2_mms_gemma26b_thinking_verify as verifier
from scripts import tau2_native_prerequisite_semantic_aug13_execute as account
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits

SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-mms-gemma26b-thinking-aug13-execute-1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
OPENING_USAGE_USD = 220.134128880
RUN_CAP_USD = 0.12
MIN_COVERED_PROMPT_TOKENS = 10_000
PREDECESSOR_RESULT = REPO_ROOT / "results/nonmyopic/tau2_mms_documented_split_resilient_calibration/DAILY_FAILURE_20260813.json"
PREDECESSOR_LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-tau2-mms-documented-split-resilient.json"
PREDECESSOR_REPORT = REPO_ROOT / "results/nonmyopic/TAU2_MMS_DOCUMENTED_SPLIT_RESILIENT_TERMINAL_RESULT_20260813.md"
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/tau2_mms_gemma26b_thinking_calibration"
RUN_DIR = OUTPUT_ROOT / "calibration-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-tau2-mms-gemma26b-thinking.json"
DAILY_RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"
DAILY_FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
EXECUTION_BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"
SOURCE_AUDIT_RESULT = OUTPUT_ROOT / "SOURCE_AUDIT.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def pristine(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def validate_date(now: datetime | None = None) -> datetime:
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"Gemma26B thinking MMS can run only on {DATE}")
    return local


def validate_catalog(catalog: dict[str, Any]) -> dict[str, Any]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == calibration.MODEL_ID]
    if len(rows) != 1:
        raise RuntimeError("exact Gemma 26B endpoint unavailable")
    model = rows[0]
    architecture = model.get("architecture") or {}
    inputs = set(architecture.get("input_modalities") or [])
    outputs = set(architecture.get("output_modalities") or [])
    supported = set(model.get("supported_parameters") or [])
    if "text" not in inputs or "text" not in outputs or "seed" not in supported or "reasoning" not in supported or not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("Gemma26B thinking route changed")
    try:
        prompt = float(model["pricing"]["prompt"])
        completion = float(model["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Gemma26B thinking pricing malformed") from exc
    residual = calibration.MAX_REQUEST_COST_USD - completion * calibration.ROOT_MAX_TOKENS
    covered = math.inf if prompt == 0 else residual / prompt
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)) or residual < 0 or covered < MIN_COVERED_PROMPT_TOKENS:
        raise RuntimeError("Gemma26B thinking request reservation no longer covers prompt")
    return {"id": calibration.MODEL_ID, "input_modalities": sorted(inputs), "output_modalities": sorted(outputs), "seed_supported": True, "reasoning_supported": True, "structured_output_supported": True, "prompt_usd_per_million_tokens": prompt * 1e6, "completion_usd_per_million_tokens": completion * 1e6, "maximum_request_cost_usd": calibration.MAX_REQUEST_COST_USD, "covered_prompt_tokens_at_live_price": covered}


def validate_predecessor() -> dict[str, Any]:
    result = load_object(PREDECESSOR_RESULT)
    ledger = load_object(PREDECESSOR_LEDGER)
    if (
        result.get("status") != "failed_closed"
        or result.get("authorizes") != "nothing"
        or result.get("development_confirmation_reserve_opened") is not False
        or result.get("repair_or_task_success_endpoints_opened") is not False
        or ledger.get("stage", {}).get("status") != "failed_closed"
        or float(ledger.get("opening_total_usage_usd", math.nan)) != OPENING_USAGE_USD
        or float(ledger.get("recorded_actual_spend_usd", -1.0)) < float(result.get("actual_cost_usd", math.inf)) - 1e-12
    ):
        raise RuntimeError("Gemma26B thinking predecessor malformed")
    return {"result_sha256": sha256_file(PREDECESSOR_RESULT), "ledger_sha256": sha256_file(PREDECESSOR_LEDGER), "report_sha256": sha256_file(PREDECESSOR_REPORT), "recorded_spend_usd": float(ledger["recorded_actual_spend_usd"])}


def validate_bindings() -> dict[str, Any]:
    predecessor = validate_predecessor()
    binding = load_object(EXECUTION_BINDING)
    expected = {
        "protocol": calibration.PROTOCOL,
        "calibration_manifest": calibration.MANIFEST,
        "source_audit": Path(calibration.source_audit.__file__).resolve(),
        "source_audit_result": SOURCE_AUDIT_RESULT,
        "tau2_user_tools": calibration.source_audit.USER_TOOLS,
        "predecessor_result": PREDECESSOR_RESULT,
        "predecessor_ledger": PREDECESSOR_LEDGER,
        "predecessor_report": PREDECESSOR_REPORT,
        "producer": Path(calibration.__file__).resolve(),
        "independent_verifier": Path(verifier.__file__).resolve(),
        "dated_wrapper": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Gemma26B thinking MMS {name} binding changed")
    if (
        binding.get("date") != DATE
        or binding.get("opening_total_usage_usd") != OPENING_USAGE_USD
        or binding.get("daily_cap_usd") != DAILY_CAP_USD
        or binding.get("run_cap_usd") != RUN_CAP_USD
        or binding.get("accepted_requests_authorized") != 12
        or binding.get("maximum_http_attempts") != 12
        or binding.get("maximum_provider_error_retries") != 0
        or binding.get("thinking_max_tokens") != 8192
        or binding.get("final_answer_max_tokens") != 512
        or binding.get("forced_final_continuation_enabled") is not False
        or binding.get("concurrency") != 2
        or binding.get("development_calls_authorized") is not False
        or binding.get("endpoint_outcomes_authorized") is not False
    ):
        raise RuntimeError("Gemma26B thinking execution metadata changed")
    source_result = calibration.source_audit.audit()
    banked_source_result = load_object(SOURCE_AUDIT_RESULT)
    if source_result != banked_source_result or source_result.get("status") != "source_pass" or source_result.get("authorizes") != "implementation_only":
        raise RuntimeError("Gemma26B thinking source audit no longer passes")
    return {"execution_binding_sha256": sha256_file(EXECUTION_BINDING), "producer_sha256": sha256_file(Path(calibration.__file__).resolve()), "verifier_sha256": sha256_file(Path(verifier.__file__).resolve()), "wrapper_sha256": sha256_file(Path(__file__).resolve()), "predecessor": predecessor}


def preflight(*, now: datetime | None = None, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader: Callable[[], dict[str, Any]] = account.read_model_catalog) -> dict[str, Any]:
    local = validate_date(now)
    bindings = validate_bindings()
    if DAILY_RESULT.exists() or DAILY_FAILURE.exists():
        raise RuntimeError("Gemma26B thinking calibration already terminal")
    for path in (RUN_DIR, LEDGER):
        if not pristine(path):
            raise RuntimeError(f"Gemma26B thinking path not pristine: {path}")
    model = validate_catalog(catalog_reader())
    live = account.validate_live(live_reader())
    spent = max(account.prior_spend(live), bindings["predecessor"]["recorded_spend_usd"])
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < RUN_CAP_USD:
        raise RuntimeError("Gemma26B thinking budget unavailable")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "ready_without_paid_calls", "date": local.date().isoformat(), "bindings": bindings, "model": model, "live_credits": live, "budget": {"daily_cap_usd": DAILY_CAP_USD, "opening_total_usage_boundary_usd": OPENING_USAGE_USD, "prior_account_spend_usd": spent, "run_cap_usd": RUN_CAP_USD, "remaining_after_full_cap_usd": DAILY_CAP_USD - spent - RUN_CAP_USD}, "model_calls_made": 0, "files_written": 0}


def initial_ledger(ready: dict[str, Any], live: dict[str, float]) -> dict[str, Any]:
    clean = account.validate_live(live)
    prior = max(account.prior_spend(clean), ready["bindings"]["predecessor"]["recorded_spend_usd"])
    if prior + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("usage raced beyond Gemma26B thinking reservation")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "date": DATE, "timezone": TIMEZONE, "daily_cap_usd": DAILY_CAP_USD, "opening_total_credits_usd": 245.0, "opening_total_usage_usd": OPENING_USAGE_USD, "opening_balance_usd": 245.0 - OPENING_USAGE_USD, "execution_opening_total_credits_usd": clean["total_credits_usd"], "execution_opening_total_usage_usd": clean["total_usage_usd"], "execution_opening_balance_usd": clean["balance_usd"], "recorded_actual_spend_usd": prior, "account_wide_usage_counts_against_cap": True, "unspent_allowance_does_not_roll_over": True, "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"], "predecessor_result_sha256": ready["bindings"]["predecessor"]["result_sha256"], "predecessor_ledger_sha256": ready["bindings"]["predecessor"]["ledger_sha256"], "predecessor_report_sha256": ready["bindings"]["predecessor"]["report_sha256"], "stage": {"status": "authorized_pending", "model": calibration.MODEL_ID, "expected_accepted_requests": 12, "maximum_http_attempts": 12, "maximum_provider_error_retries": 0, "thinking_max_tokens": 8192, "final_answer_max_tokens": 512, "forced_final_continuation_enabled": False, "maximum_cost_usd": RUN_CAP_USD}}


def adapter_cost(adapter: Any) -> float:
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: dict[str, Any], *, status: str, local_cost: float, live: dict[str, float] | None) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    local_total = local_cost + float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
    posted = account.prior_spend(live) if live is not None else 0.0
    recorded = max(float(ledger["recorded_actual_spend_usd"]), local_total, posted)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("Gemma26B thinking reconciliation exceeds cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live is not None:
        updated.update({"closing_total_credits_usd": float(live["total_credits_usd"]), "closing_total_usage_usd": float(live["total_usage_usd"]), "closing_balance_usd": float(live["balance_usd"])})
    return updated


def execute(*, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader: Callable[[], dict[str, Any]] = account.read_model_catalog) -> dict[str, Any]:
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader)
    ledger = initial_ledger(ready, account.validate_live(live_reader()))
    checkpoint(LEDGER, ledger)
    adapter = None
    try:
        adapter = calibration.build_adapter(run_id="tau2-mms-gemma26b-thinking-20260813", output_dir=RUN_DIR)
        result = calibration.run(output_dir=RUN_DIR, adapter=adapter, daily_budget_status=ready["budget"])
        verification = verifier.verify(RUN_DIR, output_path=RUN_DIR / "VERIFICATION.json")
        try:
            closing = account.validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status=result["status"], local_cost=adapter_cost(adapter), live=closing)
        checkpoint(LEDGER, ledger)
        terminal = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": result["status"], "authorizes": result["authorizes"], "result_sha256": sha256_file(RUN_DIR / "RESULT.json"), "verification_sha256": sha256_file(RUN_DIR / "VERIFICATION.json"), "partial_bank_sha256": sha256_file(RUN_DIR / "private/PARTIAL_RAW_RESPONSES.json"), "ledger_sha256": sha256_file(LEDGER), "actual_cost_usd": adapter_cost(adapter), "verification": verification}
        checkpoint(DAILY_RESULT, terminal)
        return terminal
    except Exception as exc:
        try:
            closing = account.validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status="failed_closed", local_cost=adapter_cost(adapter), live=closing)
        checkpoint(LEDGER, ledger)
        partial = RUN_DIR / "private/PARTIAL_RAW_RESPONSES.json"
        failure = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "failed_closed", "authorizes": "nothing", "error_type": type(exc).__name__, "error": str(exc), "actual_cost_usd": adapter_cost(adapter), "partial_bank_sha256": sha256_file(partial) if partial.exists() else None, "ledger_sha256": sha256_file(LEDGER), "development_confirmation_reserve_opened": False, "repair_or_task_success_endpoints_opened": False}
        checkpoint(DAILY_FAILURE, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    print(json.dumps(preflight() if args.preflight else execute(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
