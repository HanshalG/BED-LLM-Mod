#!/usr/bin/env python3
"""Execute the Aug 13 Tau2 semantic mechanics gate exactly once."""

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

from scripts import tau2_native_prerequisite_semantic_mechanics as mechanics
from scripts import tau2_native_prerequisite_semantic_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "tau2-native-prerequisite-semantic-aug13-execute-1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
OPENING_USAGE_USD = 220.134128880
OPENING_CREDITS_USD = 245.0
RUN_CAP_USD = mechanics.RUN_CAP_USD
MAX_REQUEST_COST_USD = mechanics.MAX_REQUEST_COST_USD
MIN_COVERED_PROMPT_TOKENS = 12_000
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/tau2_native_prerequisite_semantic_mechanics"
RUN_DIR = OUTPUT_ROOT / "mechanics-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-tau2-native-semantic-mechanics.json"
DAILY_RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"
DAILY_FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
EXECUTION_BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def read_model_catalog() -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    request = Request(
        MODELS_URL,
        headers={"Authorization": f"Bearer {api_key}"},
    )
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def pristine(path: Path) -> bool:
    if not path.exists():
        return True
    return path.is_dir() and not any(path.iterdir())


def validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"Tau2 semantic mechanics can run only on {DATE}")
    return local


def validate_live(live: Mapping[str, float]) -> dict[str, float]:
    try:
        credits = float(live["total_credits_usd"])
        usage = float(live["total_usage_usd"])
        balance = float(live["balance_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter account values are malformed") from exc
    if not all(math.isfinite(value) and value >= 0.0 for value in (credits, usage, balance)):
        raise RuntimeError("OpenRouter account values are malformed")
    if abs((credits - usage) - balance) > 1e-6:
        raise RuntimeError("OpenRouter account values are inconsistent")
    if usage + 1e-12 < OPENING_USAGE_USD:
        raise RuntimeError("OpenRouter usage is below the frozen Aug 13 boundary")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live: Mapping[str, float]) -> float:
    clean = validate_live(live)
    return clean["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == mechanics.MODEL_ID]
    if len(rows) != 1:
        raise RuntimeError("exact DeepSeek V4 Flash 0731 endpoint is unavailable")
    model = rows[0]
    architecture = model.get("architecture") or {}
    inputs = set(architecture.get("input_modalities") or [])
    outputs = set(architecture.get("output_modalities") or [])
    supported = set(model.get("supported_parameters") or [])
    if inputs != {"text"} or "text" not in outputs:
        raise RuntimeError("DeepSeek 0731 modality changed")
    if "seed" not in supported or not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("DeepSeek 0731 seeded structured output is unavailable")
    try:
        prompt = float(model["pricing"]["prompt"])
        completion = float(model["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DeepSeek 0731 pricing is malformed") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)):
        raise RuntimeError("DeepSeek 0731 pricing is malformed")
    residual = MAX_REQUEST_COST_USD - completion * mechanics.MAX_TOKENS
    covered_prompt = math.inf if prompt == 0 else residual / prompt
    if residual < 0 or covered_prompt + 1e-9 < MIN_COVERED_PROMPT_TOKENS:
        raise RuntimeError("per-request reservation no longer covers the frozen prompt")
    return {
        "id": mechanics.MODEL_ID,
        "input_modalities": sorted(inputs),
        "output_modalities": sorted(outputs),
        "seed_supported": True,
        "structured_output_supported": True,
        "prompt_usd_per_million_tokens": prompt * 1_000_000,
        "completion_usd_per_million_tokens": completion * 1_000_000,
        "maximum_request_cost_usd": MAX_REQUEST_COST_USD,
        "covered_prompt_tokens_at_live_price": covered_prompt,
    }


def validate_bindings() -> dict[str, Any]:
    if not EXECUTION_BINDING.is_file():
        raise RuntimeError("Tau2 semantic execution binding is absent")
    binding = load_object(EXECUTION_BINDING)
    expected = {
        "semantic_protocol": mechanics.PROTOCOL,
        "source_protocol": mechanics.SOURCE_PROTOCOL,
        "source_manifest": mechanics.SOURCE_MANIFEST,
        "source_result": mechanics.SOURCE_RESULT,
        "opportunity_result": mechanics.OPPORTUNITY_RESULT,
        "source_selector": Path(mechanics.source.__file__).resolve(),
        "opportunity_audit": Path(mechanics.exact.__file__).resolve(),
        "producer": Path(mechanics.__file__).resolve(),
        "independent_verifier": Path(verifier.__file__).resolve(),
        "dated_wrapper": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"Tau2 semantic {name} binding changed")
    if (
        binding.get("date") != DATE
        or binding.get("opening_total_usage_usd") != OPENING_USAGE_USD
        or binding.get("daily_cap_usd") != DAILY_CAP_USD
        or binding.get("run_cap_usd") != RUN_CAP_USD
        or binding.get("accepted_requests_authorized") != mechanics.EXPECTED_REQUESTS
        or binding.get("maximum_http_attempts") != mechanics.EXPECTED_REQUESTS
        or binding.get("development_calls_authorized") is not False
        or binding.get("endpoint_outcomes_authorized") is not False
    ):
        raise RuntimeError("Tau2 semantic execution metadata changed")
    mechanics.validate_bindings()
    return {
        "execution_binding_sha256": sha256_file(EXECUTION_BINDING),
        "producer_sha256": sha256_file(Path(mechanics.__file__).resolve()),
        "verifier_sha256": sha256_file(Path(verifier.__file__).resolve()),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
    }


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = read_model_catalog,
) -> dict[str, Any]:
    local = validate_date(now)
    bindings = validate_bindings()
    if DAILY_RESULT.exists() or DAILY_FAILURE.exists():
        raise RuntimeError("Tau2 semantic mechanics is already terminal")
    for path in (RUN_DIR, LEDGER):
        if not pristine(path):
            raise RuntimeError(f"Tau2 semantic path is not pristine: {path}")
    model = validate_catalog(catalog_reader())
    live = validate_live(live_reader())
    spent = prior_spend(live)
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("Tau2 semantic stage exceeds the account-wide daily cap")
    if live["balance_usd"] + 1e-12 < RUN_CAP_USD:
        raise RuntimeError("OpenRouter balance is below the Tau2 semantic stage cap")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": local.date().isoformat(),
        "bindings": bindings,
        "model": model,
        "live_credits": live,
        "budget": {
            "daily_cap_usd": DAILY_CAP_USD,
            "opening_total_usage_boundary_usd": OPENING_USAGE_USD,
            "prior_account_spend_usd": spent,
            "run_cap_usd": RUN_CAP_USD,
            "remaining_after_full_cap_usd": DAILY_CAP_USD - spent - RUN_CAP_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def initial_ledger(ready: Mapping[str, Any], live: Mapping[str, float]) -> dict[str, Any]:
    clean = validate_live(live)
    spent = prior_spend(clean)
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("usage raced beyond the Tau2 semantic full reservation")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": OPENING_CREDITS_USD,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "opening_balance_usd": OPENING_CREDITS_USD - OPENING_USAGE_USD,
        "execution_opening_total_credits_usd": clean["total_credits_usd"],
        "execution_opening_total_usage_usd": clean["total_usage_usd"],
        "execution_opening_balance_usd": clean["balance_usd"],
        "recorded_actual_spend_usd": spent,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"],
        "stage": {
            "status": "authorized_pending",
            "model": mechanics.MODEL_ID,
            "expected_accepted_requests": mechanics.EXPECTED_REQUESTS,
            "maximum_http_attempts": mechanics.EXPECTED_REQUESTS,
            "maximum_cost_usd": RUN_CAP_USD,
        },
    }


def adapter_cost(adapter: Any | None) -> float:
    if adapter is None:
        return 0.0
    return float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: Mapping[str, Any], *, status: str, local_cost: float, live: Mapping[str, float] | None) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    execution_opening = float(ledger["execution_opening_total_usage_usd"])
    local_total = local_cost + execution_opening - OPENING_USAGE_USD
    posted = prior_spend(live) if live is not None else 0.0
    recorded = max(float(ledger["recorded_actual_spend_usd"]), local_total, posted)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("Tau2 semantic reconciliation exceeds the daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live is not None:
        updated["closing_total_credits_usd"] = float(live["total_credits_usd"])
        updated["closing_total_usage_usd"] = float(live["total_usage_usd"])
        updated["closing_balance_usd"] = float(live["balance_usd"])
    return updated


def execute(
    *,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = read_model_catalog,
) -> dict[str, Any]:
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader)
    race_live = validate_live(live_reader())
    ledger = initial_ledger(ready, race_live)
    checkpoint(LEDGER, ledger)
    adapter = None
    try:
        adapter = mechanics.build_adapter(run_id="tau2-native-semantic-mechanics-20260813", output_dir=RUN_DIR)
        result = mechanics.run(output_dir=RUN_DIR, adapter=adapter, daily_budget_status=ready["budget"])
        verification = verifier.verify(RUN_DIR, output_path=RUN_DIR / "VERIFICATION.json")
        closing = None
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status=result["status"], local_cost=adapter_cost(adapter), live=closing)
        checkpoint(LEDGER, ledger)
        terminal = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": result["status"],
            "authorizes": result["authorizes"],
            "result_sha256": sha256_file(RUN_DIR / "RESULT.json"),
            "verification_sha256": sha256_file(RUN_DIR / "VERIFICATION.json"),
            "ledger_sha256": sha256_file(LEDGER),
            "actual_cost_usd": adapter_cost(adapter),
            "verification": verification,
        }
        checkpoint(DAILY_RESULT, terminal)
        return terminal
    except Exception as exc:
        local_cost = adapter_cost(adapter)
        closing = None
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status="failed_closed", local_cost=local_cost, live=closing)
        checkpoint(LEDGER, ledger)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "authorizes": "nothing",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_cost_usd": local_cost,
            "ledger_sha256": sha256_file(LEDGER),
            "development_confirmation_reserve_opened": False,
            "repair_or_task_success_endpoints_opened": False,
        }
        checkpoint(DAILY_FAILURE, failure)
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("OPENROUTER_API_KEY is required")
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
