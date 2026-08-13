#!/usr/bin/env python3
"""Execute the Aug 13 HiddenBench dynamic-belief V3 mechanics once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Callable, Mapping
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import hiddenbench_dynamic_belief_v3_endpoint_score as endpoint_score
from scripts import hiddenbench_dynamic_belief_v3_pass_token as pass_token
from scripts import hiddenbench_dynamic_belief_v3_serving as serving
from scripts import hiddenbench_dynamic_belief_v3_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits


INTERFACE_VERSION = "hiddenbench-dynamic-belief-v3-aug13-execute-v1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
OPENING_CREDITS_USD = 245.0
OPENING_USAGE_USD = 220.134128880
RUN_CAP_USD = 0.040
MAX_PROMPT_PRICE = 0.09 / 1_000_000
MAX_COMPLETION_PRICE = 0.18 / 1_000_000
MIN_COVERED_PROMPT_TOKENS = 25_000
SOURCE = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")
SOURCE_SHA256 = "2815afffca4e470d1dfbc81e625160447df1109ce371968181c9e1e6b90443a3"
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/hiddenbench_dynamic_belief_v3_mechanics"
RUN_DIR = OUTPUT_ROOT / "mechanics-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-hiddenbench-dynamic-belief-v3.json"
EXECUTION_BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"
DAILY_RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"
DAILY_FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
SOURCE_MANIFEST = REPO_ROOT / "results/nonmyopic/hiddenbench_dynamic_belief_v3_source/MANIFEST.json"
SOURCE_AUDIT = REPO_ROOT / "results/nonmyopic/hiddenbench_dynamic_belief_v3_source/SOURCE_AUDIT.json"
RAW = RUN_DIR / "private/RAW_RESPONSES.json"
LABEL_FREE = RUN_DIR / "LABEL_FREE_RESULT.json"
VERIFICATION = RUN_DIR / "VERIFICATION.json"
PASS_TOKEN = RUN_DIR / "LABEL_FREE_PASS_TOKEN.json"
ENDPOINTS = RUN_DIR / "private/ENDPOINTS.json"
ENDPOINT_RESULT = RUN_DIR / "ENDPOINT_RESULT.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def read_catalog() -> dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY is required")
    request = Request(MODELS_URL, headers={"Authorization": f"Bearer {key}"})
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
        raise RuntimeError("OpenRouter account values are invalid")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live: Mapping[str, float]) -> float:
    return validate_live(live)["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == serving.MODEL_ID]
    if len(rows) != 1:
        raise RuntimeError("exact DeepSeek V4 Flash 0731 endpoint is unavailable")
    model = rows[0]
    architecture = model.get("architecture") or {}
    supported = set(model.get("supported_parameters") or [])
    if set(architecture.get("input_modalities") or []) != {"text"} or "text" not in set(architecture.get("output_modalities") or []) or "seed" not in supported or not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("DeepSeek structured text interface changed")
    try:
        prompt = float(model["pricing"]["prompt"])
        completion = float(model["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DeepSeek pricing is malformed") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)) or prompt > MAX_PROMPT_PRICE + 1e-15 or completion > MAX_COMPLETION_PRICE + 1e-15:
        raise RuntimeError("DeepSeek price increased beyond the frozen ceiling")
    residual = serving.MAX_REQUEST_COST_USD - completion * serving.MAX_TOKENS
    covered_prompt = math.inf if prompt == 0 else residual / prompt
    if residual < 0 or covered_prompt + 1e-9 < MIN_COVERED_PROMPT_TOKENS:
        raise RuntimeError("per-request cap no longer covers the frozen payload")
    return {"id": serving.MODEL_ID, "prompt_usd_per_million_tokens": prompt * 1_000_000, "completion_usd_per_million_tokens": completion * 1_000_000, "covered_prompt_tokens_at_live_price": covered_prompt, "seed_supported": True, "structured_output_supported": True}


def validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"HiddenBench V3 can run only on {DATE}")
    return local


def pristine(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def validate_bindings() -> dict[str, Any]:
    binding = load_object(EXECUTION_BINDING)
    expected = {
        "protocol": REPO_ROOT / "results/nonmyopic/HIDDENBENCH_DYNAMIC_BELIEF_V3_MECHANICS_PROTOCOL_20260813.md",
        "source_manifest": SOURCE_MANIFEST,
        "source_audit": SOURCE_AUDIT,
        "source_selector": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_source.py",
        "serving_custodian": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_serving_custodian.py",
        "codec": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_codec.py",
        "math": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_math.py",
        "producer": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_serving.py",
        "verifier": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_verify.py",
        "pass_token": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_pass_token.py",
        "endpoint_custodian": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_endpoint_custodian.py",
        "endpoint_scorer": REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_endpoint_score.py",
        "wrapper": Path(__file__).resolve(),
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"V3 {name} binding changed")
    required = {"date": DATE, "opening_total_usage_usd": OPENING_USAGE_USD, "daily_cap_usd": DAILY_CAP_USD, "run_cap_usd": RUN_CAP_USD, "accepted_requests_authorized": 10, "maximum_http_attempts": 10, "maximum_retries": 0, "endpoint_requires_pass_token": True, "development_calls_authorized": False}
    if any(binding.get(key) != value for key, value in required.items()):
        raise RuntimeError("V3 execution metadata changed")
    return {"execution_binding_sha256": sha256_file(EXECUTION_BINDING), "producer_sha256": sha256_file(Path(serving.__file__).resolve()), "verifier_sha256": sha256_file(Path(verifier.__file__).resolve()), "wrapper_sha256": sha256_file(Path(__file__).resolve())}


def preflight(*, now: datetime | None = None, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader: Callable[[], dict[str, Any]] = read_catalog) -> dict[str, Any]:
    local = validate_date(now); bindings = validate_bindings()
    if DAILY_RESULT.exists() or DAILY_FAILURE.exists(): raise RuntimeError("HiddenBench V3 is already terminal")
    for path in (RUN_DIR, LEDGER):
        if not pristine(path): raise RuntimeError(f"V3 path is not pristine: {path}")
    if not SOURCE.is_file() or sha256_file(SOURCE) != SOURCE_SHA256: raise RuntimeError("bound HiddenBench source is unavailable")
    model = validate_catalog(catalog_reader()); live = validate_live(live_reader()); spent = prior_spend(live)
    if spent + RUN_CAP_USD > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < RUN_CAP_USD: raise RuntimeError("V3 full reservation is unavailable")
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "ready_without_paid_calls", "date": local.date().isoformat(), "bindings": bindings, "model": model, "live_credits": live, "budget": {"daily_cap_usd": DAILY_CAP_USD, "opening_total_usage_boundary_usd": OPENING_USAGE_USD, "prior_account_spend_usd": spent, "run_cap_usd": RUN_CAP_USD, "remaining_after_full_cap_usd": DAILY_CAP_USD-spent-RUN_CAP_USD}, "model_calls_made": 0, "files_written": 0}


def initial_ledger(ready: Mapping[str, Any], live: Mapping[str, float]) -> dict[str, Any]:
    clean=validate_live(live); spent=prior_spend(clean)
    if spent+RUN_CAP_USD>DAILY_CAP_USD+1e-12: raise RuntimeError("usage raced beyond V3 reservation")
    return {"schema_version":1,"interface_version":INTERFACE_VERSION,"date":DATE,"timezone":TIMEZONE,"daily_cap_usd":DAILY_CAP_USD,"opening_total_credits_usd":OPENING_CREDITS_USD,"opening_total_usage_usd":OPENING_USAGE_USD,"execution_opening_total_credits_usd":clean["total_credits_usd"],"execution_opening_total_usage_usd":clean["total_usage_usd"],"execution_opening_balance_usd":clean["balance_usd"],"recorded_actual_spend_usd":spent,"account_wide_usage_counts_against_cap":True,"unspent_allowance_does_not_roll_over":True,"execution_binding_sha256":ready["bindings"]["execution_binding_sha256"],"stage":{"status":"authorized_pending","model":serving.MODEL_ID,"expected_accepted_requests":10,"maximum_http_attempts":10,"maximum_retries":0,"maximum_cost_usd":RUN_CAP_USD}}


def adapter_cost(adapter: Any | None) -> float:
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: Mapping[str, Any], *, status: str, local_cost: float, live: Mapping[str, float] | None) -> dict[str, Any]:
    updated=json.loads(json.dumps(ledger)); local_total=local_cost+float(ledger["execution_opening_total_usage_usd"])-OPENING_USAGE_USD; posted=prior_spend(live) if live is not None else 0.0; recorded=max(float(ledger["recorded_actual_spend_usd"]),local_total,posted)
    if recorded>DAILY_CAP_USD+1e-12: raise RuntimeError("V3 reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"]=recorded; updated["stage"].update({"status":status,"actual_cost_usd":local_cost})
    if live is not None: updated.update({"closing_total_credits_usd":float(live["total_credits_usd"]),"closing_total_usage_usd":float(live["total_usage_usd"]),"closing_balance_usd":float(live["balance_usd"])})
    return updated


def endpoint_command() -> list[str]:
    return [sys.executable, str(REPO_ROOT / "scripts/hiddenbench_dynamic_belief_v3_endpoint_custodian.py"), "--source", str(SOURCE), "--pass-token", str(PASS_TOKEN), "--execution-binding", str(EXECUTION_BINDING), "--source-manifest", str(SOURCE_MANIFEST), "--source-audit", str(SOURCE_AUDIT), "--raw-responses", str(RAW), "--label-free-result", str(LABEL_FREE), "--verification", str(VERIFICATION), "--output", str(ENDPOINTS)]


def execute(*, live_reader: Callable[[], dict[str, float]] = read_live_credits, catalog_reader: Callable[[], dict[str, Any]] = read_catalog) -> dict[str, Any]:
    ready=preflight(live_reader=live_reader,catalog_reader=catalog_reader); ledger=initial_ledger(ready,validate_live(live_reader())); checkpoint(LEDGER,ledger); adapter=None
    try:
        adapter=serving.build_adapter("hiddenbench-dynamic-belief-v3-20260813",RUN_DIR)
        label_free=serving.run_serving(source_path=SOURCE,output_dir=RUN_DIR,adapter=adapter)
        if label_free["status"]!="serving_pass":
            try: closing=validate_live(live_reader())
            except Exception: closing=None
            ledger=reconcile(ledger,status=label_free["status"],local_cost=adapter_cost(adapter),live=closing); checkpoint(LEDGER,ledger)
            terminal={"schema_version":1,"interface_version":INTERFACE_VERSION,"status":label_free["status"],"authorizes":"nothing","label_free_result_sha256":sha256_file(LABEL_FREE),"raw_response_sha256":sha256_file(RAW),"ledger_sha256":sha256_file(LEDGER),"actual_cost_usd":adapter_cost(adapter),"verification_opened":False,"pass_token_opened":False,"registered_answers_opened":False,"endpoint_result_opened":False}; checkpoint(DAILY_RESULT,terminal); return terminal
        verification=verifier.verify(RUN_DIR,output_path=VERIFICATION)
        token=pass_token.create_token(execution_binding=EXECUTION_BINDING,source_manifest=SOURCE_MANIFEST,source_audit=SOURCE_AUDIT,raw_responses=RAW,label_free_result=LABEL_FREE,verification=VERIFICATION); checkpoint(PASS_TOKEN,token)
        subprocess.run(endpoint_command(),check=True)
        endpoint=endpoint_score.score(raw=load_object(RAW),label_free=label_free,endpoints=load_object(ENDPOINTS)); checkpoint(ENDPOINT_RESULT,endpoint)
        try: closing=validate_live(live_reader())
        except Exception: closing=None
        ledger=reconcile(ledger,status=endpoint["status"],local_cost=adapter_cost(adapter),live=closing); checkpoint(LEDGER,ledger)
        terminal={"schema_version":1,"interface_version":INTERFACE_VERSION,"status":endpoint["status"],"authorizes":endpoint["authorizes"],"label_free_result_sha256":sha256_file(LABEL_FREE),"verification_sha256":sha256_file(VERIFICATION),"pass_token_sha256":sha256_file(PASS_TOKEN),"raw_response_sha256":sha256_file(RAW),"endpoint_result_sha256":sha256_file(ENDPOINT_RESULT),"ledger_sha256":sha256_file(LEDGER),"actual_cost_usd":adapter_cost(adapter)}; checkpoint(DAILY_RESULT,terminal); return terminal
    except Exception as exc:
        try: closing=validate_live(live_reader())
        except Exception: closing=None
        ledger=reconcile(ledger,status="failed_closed",local_cost=adapter_cost(adapter),live=closing); checkpoint(LEDGER,ledger)
        failure={"schema_version":1,"interface_version":INTERFACE_VERSION,"status":"failed_closed","authorizes":"nothing","error_type":type(exc).__name__,"error":str(exc),"actual_cost_usd":adapter_cost(adapter),"ledger_sha256":sha256_file(LEDGER),"label_free_result_exists":LABEL_FREE.exists(),"verification_exists":VERIFICATION.exists(),"pass_token_exists":PASS_TOKEN.exists(),"registered_answers_opened":ENDPOINTS.exists(),"endpoint_result_opened":ENDPOINT_RESULT.exists()}; checkpoint(DAILY_FAILURE,failure); raise


def main() -> int:
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument("--preflight",action="store_true"); args=parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"): raise RuntimeError("OPENROUTER_API_KEY is required")
    result=preflight() if args.preflight else execute(); print(json.dumps(result,indent=2,sort_keys=True)); return 0


if __name__=="__main__": raise SystemExit(main())
