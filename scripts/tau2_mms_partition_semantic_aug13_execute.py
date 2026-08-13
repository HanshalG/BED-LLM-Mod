#!/usr/bin/env python3
"""Execute the fresh Tau2 MMS partition semantic calibration exactly once."""

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
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import tau2_mms_partition_semantic_calibration as calibration
from scripts import tau2_mms_partition_semantic_verify as verifier
from scripts import tau2_native_prerequisite_semantic_aug13_execute as account
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits

SCHEMA_VERSION = 1; INTERFACE_VERSION = "tau2-mms-partition-semantic-aug13-execute-1"
DATE = "2026-08-13"; TIMEZONE = "Europe/London"; DAILY_CAP_USD = 5.0; OPENING_USAGE_USD = 220.134128880; RUN_CAP_USD = 0.06; MIN_COVERED_PROMPT_TOKENS = 10_000
PREDECESSOR_RESULT = REPO_ROOT / "results/nonmyopic/tau2_mms_checkpointed_semantic_calibration/DAILY_RESULT_20260813.json"
PREDECESSOR_LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-tau2-mms-checkpointed-semantic.json"
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/tau2_mms_partition_semantic_calibration"; RUN_DIR = OUTPUT_ROOT / "calibration-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-tau2-mms-partition-semantic.json"; DAILY_RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"; DAILY_FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"; EXECUTION_BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"


def sha256_file(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load_object(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict): raise RuntimeError(f"expected object: {path}")
    return value
def pristine(path: Path): return not path.exists() or (path.is_dir() and not any(path.iterdir()))
def validate_date(now=None):
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE: raise RuntimeError(f"partition MMS can run only on {DATE}")
    return local


def validate_catalog(catalog):
    rows = [row for row in catalog.get("data", []) if row.get("id") == calibration.MODEL_ID]
    if len(rows) != 1: raise RuntimeError("exact DeepSeek endpoint unavailable")
    model = rows[0]; architecture = model.get("architecture") or {}; inputs = set(architecture.get("input_modalities") or []); outputs = set(architecture.get("output_modalities") or []); supported = set(model.get("supported_parameters") or [])
    if inputs != {"text"} or "text" not in outputs or "seed" not in supported or not ({"response_format", "structured_outputs"} & supported): raise RuntimeError("DeepSeek partition route changed")
    try: prompt = float(model["pricing"]["prompt"]); completion = float(model["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc: raise RuntimeError("DeepSeek partition pricing malformed") from exc
    residual = calibration.MAX_REQUEST_COST_USD - completion * calibration.MAX_TOKENS; covered = math.inf if prompt == 0 else residual / prompt
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)) or residual < 0 or covered < MIN_COVERED_PROMPT_TOKENS: raise RuntimeError("partition request reservation no longer covers prompt")
    return {"id": calibration.MODEL_ID, "input_modalities": sorted(inputs), "output_modalities": sorted(outputs), "seed_supported": True, "structured_output_supported": True, "prompt_usd_per_million_tokens": prompt * 1e6, "completion_usd_per_million_tokens": completion * 1e6, "maximum_request_cost_usd": calibration.MAX_REQUEST_COST_USD, "covered_prompt_tokens_at_live_price": covered}


def validate_predecessor():
    result = load_object(PREDECESSOR_RESULT); ledger = load_object(PREDECESSOR_LEDGER)
    if result.get("status") != "mms_checkpointed_semantic_null" or result.get("authorizes") != "nothing" or ledger.get("stage", {}).get("status") != "mms_checkpointed_semantic_null" or float(ledger.get("opening_total_usage_usd", math.nan)) != OPENING_USAGE_USD or float(ledger.get("recorded_actual_spend_usd", -1)) < float(result.get("actual_cost_usd", math.inf)) - 1e-12: raise RuntimeError("partition predecessor malformed")
    return {"result_sha256": sha256_file(PREDECESSOR_RESULT), "ledger_sha256": sha256_file(PREDECESSOR_LEDGER), "recorded_spend_usd": float(ledger["recorded_actual_spend_usd"])}


def validate_bindings():
    predecessor = validate_predecessor(); binding = load_object(EXECUTION_BINDING)
    expected = {"protocol": calibration.PROTOCOL, "calibration_manifest": calibration.MANIFEST, "source_protocol": calibration.absolute.source.PROTOCOL, "source_manifest": calibration.absolute.source.OUTPUT_DIR / "SOURCE_MANIFEST.json", "source_selector": Path(calibration.absolute.source.__file__).resolve(), "predecessor_result": PREDECESSOR_RESULT, "predecessor_ledger": PREDECESSOR_LEDGER, "producer": Path(calibration.__file__).resolve(), "independent_verifier": Path(verifier.__file__).resolve(), "dated_wrapper": Path(__file__).resolve()}
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != sha256_file(path): raise RuntimeError(f"partition MMS {name} binding changed")
    if binding.get("date") != DATE or binding.get("opening_total_usage_usd") != OPENING_USAGE_USD or binding.get("daily_cap_usd") != DAILY_CAP_USD or binding.get("run_cap_usd") != RUN_CAP_USD or binding.get("accepted_requests_authorized") != 6 or binding.get("maximum_http_attempts") != 6 or binding.get("concurrency") != 2 or binding.get("development_calls_authorized") is not False or binding.get("endpoint_outcomes_authorized") is not False: raise RuntimeError("partition execution metadata changed")
    calibration.selected_episodes(); return {"execution_binding_sha256": sha256_file(EXECUTION_BINDING), "producer_sha256": sha256_file(Path(calibration.__file__).resolve()), "verifier_sha256": sha256_file(Path(verifier.__file__).resolve()), "wrapper_sha256": sha256_file(Path(__file__).resolve()), "predecessor": predecessor}


def preflight(*, now=None, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader=account.read_model_catalog):
    local = validate_date(now); bindings = validate_bindings()
    if DAILY_RESULT.exists() or DAILY_FAILURE.exists(): raise RuntimeError("partition calibration already terminal")
    for path in (RUN_DIR, LEDGER):
        if not pristine(path): raise RuntimeError(f"partition path not pristine: {path}")
    model = validate_catalog(catalog_reader()); live = account.validate_live(live_reader()); spent = max(account.prior_spend(live), bindings["predecessor"]["recorded_spend_usd"])
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < RUN_CAP_USD: raise RuntimeError("partition budget unavailable")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "ready_without_paid_calls", "date": local.date().isoformat(), "bindings": bindings, "model": model, "live_credits": live, "budget": {"daily_cap_usd": DAILY_CAP_USD, "opening_total_usage_boundary_usd": OPENING_USAGE_USD, "prior_account_spend_usd": spent, "run_cap_usd": RUN_CAP_USD, "remaining_after_full_cap_usd": DAILY_CAP_USD - spent - RUN_CAP_USD}, "model_calls_made": 0, "files_written": 0}


def initial_ledger(ready, live):
    clean = account.validate_live(live); prior = max(account.prior_spend(clean), ready["bindings"]["predecessor"]["recorded_spend_usd"])
    if prior + RUN_CAP_USD > DAILY_CAP_USD + 1e-12: raise RuntimeError("usage raced beyond partition reservation")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "date": DATE, "timezone": TIMEZONE, "daily_cap_usd": DAILY_CAP_USD, "opening_total_credits_usd": 245.0, "opening_total_usage_usd": OPENING_USAGE_USD, "opening_balance_usd": 245.0 - OPENING_USAGE_USD, "execution_opening_total_credits_usd": clean["total_credits_usd"], "execution_opening_total_usage_usd": clean["total_usage_usd"], "execution_opening_balance_usd": clean["balance_usd"], "recorded_actual_spend_usd": prior, "account_wide_usage_counts_against_cap": True, "unspent_allowance_does_not_roll_over": True, "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"], "predecessor_result_sha256": ready["bindings"]["predecessor"]["result_sha256"], "predecessor_ledger_sha256": ready["bindings"]["predecessor"]["ledger_sha256"], "stage": {"status": "authorized_pending", "model": calibration.MODEL_ID, "expected_accepted_requests": 6, "maximum_http_attempts": 6, "maximum_cost_usd": RUN_CAP_USD}}


def adapter_cost(adapter): return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))
def reconcile(ledger, *, status, local_cost, live):
    updated = json.loads(json.dumps(ledger)); local_total = local_cost + float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD; posted = account.prior_spend(live) if live is not None else 0.0; recorded = max(float(ledger["recorded_actual_spend_usd"]), local_total, posted)
    if recorded > DAILY_CAP_USD + 1e-12: raise RuntimeError("partition reconciliation exceeds cap")
    updated["recorded_actual_spend_usd"] = recorded; updated["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live is not None: updated.update({"closing_total_credits_usd": float(live["total_credits_usd"]), "closing_total_usage_usd": float(live["total_usage_usd"]), "closing_balance_usd": float(live["balance_usd"])})
    return updated


def execute(*, live_reader=read_live_credits, catalog_reader=account.read_model_catalog):
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader); ledger = initial_ledger(ready, account.validate_live(live_reader())); checkpoint(LEDGER, ledger); adapter = None
    try:
        adapter = calibration.build_adapter(run_id="tau2-mms-partition-semantic-20260813", output_dir=RUN_DIR); result = calibration.run(output_dir=RUN_DIR, adapter=adapter, daily_budget_status=ready["budget"]); verification = verifier.verify(RUN_DIR, output_path=RUN_DIR / "VERIFICATION.json")
        try: closing = account.validate_live(live_reader())
        except Exception: closing = None
        ledger = reconcile(ledger, status=result["status"], local_cost=adapter_cost(adapter), live=closing); checkpoint(LEDGER, ledger); terminal = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": result["status"], "authorizes": result["authorizes"], "result_sha256": sha256_file(RUN_DIR / "RESULT.json"), "verification_sha256": sha256_file(RUN_DIR / "VERIFICATION.json"), "partial_bank_sha256": sha256_file(RUN_DIR / "private/PARTIAL_RAW_RESPONSES.json"), "ledger_sha256": sha256_file(LEDGER), "actual_cost_usd": adapter_cost(adapter), "verification": verification}; checkpoint(DAILY_RESULT, terminal); return terminal
    except Exception as exc:
        try: closing = account.validate_live(live_reader())
        except Exception: closing = None
        ledger = reconcile(ledger, status="failed_closed", local_cost=adapter_cost(adapter), live=closing); checkpoint(LEDGER, ledger); partial = RUN_DIR / "private/PARTIAL_RAW_RESPONSES.json"; failure = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "failed_closed", "authorizes": "nothing", "error_type": type(exc).__name__, "error": str(exc), "actual_cost_usd": adapter_cost(adapter), "partial_bank_sha256": sha256_file(partial) if partial.exists() else None, "ledger_sha256": sha256_file(LEDGER), "development_confirmation_reserve_opened": False, "repair_or_task_success_endpoints_opened": False}; checkpoint(DAILY_FAILURE, failure); raise


def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"): raise RuntimeError("OPENROUTER_API_KEY is required")
    print(json.dumps(preflight() if args.preflight else execute(), indent=2, sort_keys=True)); return 0
if __name__ == "__main__": raise SystemExit(main())
