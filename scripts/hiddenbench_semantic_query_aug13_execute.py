#!/usr/bin/env python3
"""Execute the Aug 13 HiddenBench semantic-query serving gate once."""

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

from scripts import hiddenbench_semantic_query_serving as serving
from scripts import hiddenbench_semantic_query_serving_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits


INTERFACE_VERSION = "hiddenbench-semantic-query-aug13-execute-v1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
OPENING_CREDITS_USD = 245.0
OPENING_USAGE_USD = 220.134128880
RUN_CAP_USD = serving.RUN_BUDGET_USD
MIN_COVERED_PROMPT_TOKENS = 8_000
MAX_PROMPT_PRICE = 0.09 / 1_000_000
MAX_COMPLETION_PRICE = 0.18 / 1_000_000
MODELS_URL = "https://openrouter.ai/api/v1/models"
SOURCE_PATH = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")
SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/hiddenbench_semantic_query_serving"
RUN_DIR = OUTPUT_ROOT / "serving-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-hiddenbench-semantic-query.json"
DAILY_RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"
DAILY_FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
EXECUTION_BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def read_model_catalog() -> dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    request = Request(MODELS_URL, headers={"Authorization": f"Bearer {api_key}"})
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def validate_live(live: Mapping[str, float]) -> dict[str, float]:
    try:
        credits = float(live["total_credits_usd"])
        usage = float(live["total_usage_usd"])
        balance = float(live["balance_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter account values are malformed") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance)) or abs(credits - usage - balance) > 1e-6 or usage + 1e-12 < OPENING_USAGE_USD:
        raise RuntimeError("OpenRouter account values are invalid for Aug 13")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live: Mapping[str, float]) -> float:
    return validate_live(live)["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == serving.MODEL_ID]
    if len(rows) != 1:
        raise RuntimeError("exact DeepSeek V4 Flash 0731 endpoint is unavailable")
    model = rows[0]
    architecture = model.get("architecture") or {}
    inputs = set(architecture.get("input_modalities") or [])
    outputs = set(architecture.get("output_modalities") or [])
    supported = set(model.get("supported_parameters") or [])
    if inputs != {"text"} or "text" not in outputs or "seed" not in supported or not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("DeepSeek 0731 structured text interface changed")
    try:
        prompt = float(model["pricing"]["prompt"])
        completion = float(model["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DeepSeek 0731 pricing is malformed") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)) or prompt > MAX_PROMPT_PRICE + 1e-15 or completion > MAX_COMPLETION_PRICE + 1e-15:
        raise RuntimeError("DeepSeek 0731 price increased beyond the frozen ceiling")
    residual = serving.MAX_REQUEST_COST_USD - completion * serving.MAX_TOKENS
    covered_prompt = math.inf if prompt == 0 else residual / prompt
    if residual < 0 or covered_prompt + 1e-9 < MIN_COVERED_PROMPT_TOKENS:
        raise RuntimeError("per-request reservation no longer covers the frozen prompt")
    return {"id": serving.MODEL_ID, "input_modalities": sorted(inputs), "output_modalities": sorted(outputs), "seed_supported": True, "structured_output_supported": True, "prompt_usd_per_million_tokens": prompt * 1_000_000, "completion_usd_per_million_tokens": completion * 1_000_000, "covered_prompt_tokens_at_live_price": covered_prompt}


def pristine(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"HiddenBench serving can run only on {DATE}")
    return local


def validate_bindings() -> dict[str, Any]:
    if not EXECUTION_BINDING.is_file():
        raise RuntimeError("HiddenBench execution binding is absent")
    binding = load_object(EXECUTION_BINDING)
    expected = {
        "protocol": REPO_ROOT / "results/nonmyopic/HIDDENBENCH_SEMANTIC_QUERY_SERVING_PROTOCOL_20260813.md",
        "source_protocol": REPO_ROOT / "results/nonmyopic/HIDDENBENCH_ADAPTIVE_ELICITATION_SOURCE_PROTOCOL_20260813.md",
        "source_manifest": REPO_ROOT / "results/nonmyopic/hiddenbench_adaptive_elicitation_source/MANIFEST.json",
        "source_result": REPO_ROOT / "results/nonmyopic/HIDDENBENCH_ADAPTIVE_ELICITATION_SOURCE_RESULT_20260813.md",
        "source_audit": REPO_ROOT / "results/nonmyopic/hiddenbench_adaptive_elicitation_source/SOURCE_AUDIT.json",
        "custodian": Path(serving.__file__).with_name("hiddenbench_semantic_query_custodian.py"),
        "producer": Path(serving.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"HiddenBench {name} binding changed")
    metadata = {
        "date": DATE,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "daily_cap_usd": DAILY_CAP_USD,
        "run_cap_usd": RUN_CAP_USD,
        "accepted_requests_authorized": 10,
        "maximum_http_attempts": 10,
        "maximum_retries": 0,
        "registered_answers_authorized": False,
        "opportunity_endpoints_authorized": False,
    }
    if any(binding.get(key) != value for key, value in metadata.items()):
        raise RuntimeError("HiddenBench execution metadata changed")
    return {"execution_binding_sha256": sha256_file(EXECUTION_BINDING), "producer_sha256": sha256_file(Path(serving.__file__).resolve()), "verifier_sha256": sha256_file(Path(verifier.__file__).resolve()), "wrapper_sha256": sha256_file(Path(__file__).resolve())}


def preflight(*, now: datetime | None = None, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader: Callable[[], dict[str, Any]] = read_model_catalog) -> dict[str, Any]:
    local = validate_date(now)
    bindings = validate_bindings()
    if DAILY_RESULT.exists() or DAILY_FAILURE.exists():
        raise RuntimeError("HiddenBench semantic-query serving is already terminal")
    for path in (RUN_DIR, LEDGER):
        if not pristine(path):
            raise RuntimeError(f"HiddenBench serving path is not pristine: {path}")
    if not SOURCE_PATH.is_file() or sha256_file(SOURCE_PATH) != SOURCE_SHA256:
        raise RuntimeError("bound HiddenBench source is unavailable")
    model = validate_catalog(catalog_reader())
    live = validate_live(live_reader())
    spent = prior_spend(live)
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < RUN_CAP_USD:
        raise RuntimeError("HiddenBench serving full reservation is unavailable")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "ready_without_paid_calls", "date": local.date().isoformat(), "bindings": bindings, "model": model, "live_credits": live, "budget": {"daily_cap_usd": DAILY_CAP_USD, "opening_total_usage_boundary_usd": OPENING_USAGE_USD, "prior_account_spend_usd": spent, "run_cap_usd": RUN_CAP_USD, "remaining_after_full_cap_usd": DAILY_CAP_USD - spent - RUN_CAP_USD}, "model_calls_made": 0, "files_written": 0}


def initial_ledger(ready: Mapping[str, Any], live: Mapping[str, float]) -> dict[str, Any]:
    clean = validate_live(live)
    spent = prior_spend(clean)
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("usage raced beyond the HiddenBench full reservation")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "date": DATE, "timezone": TIMEZONE, "daily_cap_usd": DAILY_CAP_USD, "opening_total_credits_usd": OPENING_CREDITS_USD, "opening_total_usage_usd": OPENING_USAGE_USD, "opening_balance_usd": OPENING_CREDITS_USD - OPENING_USAGE_USD, "execution_opening_total_credits_usd": clean["total_credits_usd"], "execution_opening_total_usage_usd": clean["total_usage_usd"], "execution_opening_balance_usd": clean["balance_usd"], "recorded_actual_spend_usd": spent, "account_wide_usage_counts_against_cap": True, "unspent_allowance_does_not_roll_over": True, "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"], "stage": {"status": "authorized_pending", "model": serving.MODEL_ID, "expected_accepted_requests": 10, "maximum_http_attempts": 10, "maximum_retries": 0, "maximum_cost_usd": RUN_CAP_USD}}


def adapter_cost(adapter: Any | None) -> float:
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: Mapping[str, Any], *, status: str, local_cost: float, live: Mapping[str, float] | None) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    local_total = local_cost + float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
    posted = prior_spend(live) if live is not None else 0.0
    recorded = max(float(ledger["recorded_actual_spend_usd"]), local_total, posted)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("HiddenBench reconciliation exceeds the daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live is not None:
        updated.update({"closing_total_credits_usd": float(live["total_credits_usd"]), "closing_total_usage_usd": float(live["total_usage_usd"]), "closing_balance_usd": float(live["balance_usd"])})
    return updated


def execute(*, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader: Callable[[], dict[str, Any]] = read_model_catalog) -> dict[str, Any]:
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader)
    ledger = initial_ledger(ready, validate_live(live_reader()))
    checkpoint(LEDGER, ledger)
    adapter = None
    try:
        adapter = serving.build_adapter("hiddenbench-semantic-query-serving-20260813", RUN_DIR)
        result = serving.run_serving(source_path=SOURCE_PATH, output_dir=RUN_DIR, adapter=adapter)
        verification = verifier.verify(RUN_DIR, output_path=RUN_DIR / "VERIFICATION.json")
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status=result["status"], local_cost=adapter_cost(adapter), live=closing)
        checkpoint(LEDGER, ledger)
        terminal = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": result["status"], "authorizes": result["authorizes"], "result_sha256": sha256_file(RUN_DIR / "RESULT.json"), "verification_sha256": sha256_file(RUN_DIR / "VERIFICATION.json"), "raw_response_sha256": sha256_file(RUN_DIR / "private/RAW_RESPONSES.json"), "ledger_sha256": sha256_file(LEDGER), "actual_cost_usd": adapter_cost(adapter), "verification_status": verification["status"]}
        checkpoint(DAILY_RESULT, terminal)
        return terminal
    except Exception as exc:
        local_cost = adapter_cost(adapter)
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        ledger = reconcile(ledger, status="failed_closed", local_cost=local_cost, live=closing)
        checkpoint(LEDGER, ledger)
        failure = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "failed_closed", "authorizes": "nothing", "error_type": type(exc).__name__, "error": str(exc), "actual_cost_usd": local_cost, "ledger_sha256": sha256_file(LEDGER), "registered_answers_opened": False, "opportunity_endpoints_opened": False}
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
