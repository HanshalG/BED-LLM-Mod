#!/usr/bin/env python3
"""Execute the fresh Aug 13 Number Game bitmask semantic gate once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from typing import Any, Callable, Mapping, Sequence
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path: sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_bitmask_semantic_gate as serving
from scripts import number_game_bitmask_semantic_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_bitmask_semantic_codec import HISTORIES, proposal_messages
from scripts.openrouter_daily_budget import read_live_credits


INTERFACE_VERSION = "number-game-bitmask-semantic-aug13-execute-1"
DATE = "2026-08-13"; TIMEZONE = "Europe/London"; DAILY_CAP_USD = 5.0
OPENING_USAGE_USD = 220.134128880; STAGE_CAP_USD = 0.08
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/number_game_bitmask_semantic_gate"
RUN_DIR = OUTPUT_ROOT / "mechanics-20260813"; BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"
RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"; FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-bitmask-semantic.json"
RAW = RUN_DIR / "private/RAW_RESPONSES.json"; LABEL = RUN_DIR / "LABEL_FREE_RESULT.json"; VERIFY = RUN_DIR / "VERIFICATION.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICE_CEILINGS = {serving.PROPOSAL_MODEL_ID: {"prompt": 0.32e-6, "completion": 1.28e-6}, serving.AUDIT_MODEL_ID: {"prompt": 0.10e-6, "completion": 0.60e-6}}


def digest(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text());
    if not isinstance(value, dict): raise RuntimeError("expected object")
    return value


def read_catalog() -> dict[str, Any]:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key: raise RuntimeError("OPENROUTER_API_KEY required")
    with urlopen(Request(MODELS_URL, headers={"Authorization": f"Bearer {key}"}), timeout=30) as response: value = json.load(response)
    if not isinstance(value, dict): raise RuntimeError("catalog malformed")
    return value


def validate_live(live: Mapping[str, float]) -> dict[str, float]:
    try: credits, usage, balance = (float(live[key]) for key in ("total_credits_usd", "total_usage_usd", "balance_usd"))
    except (KeyError, TypeError, ValueError) as exc: raise RuntimeError("account values malformed") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance)) or abs(credits - usage - balance) > 1e-6 or usage + 1e-12 < OPENING_USAGE_USD: raise RuntimeError("account values invalid")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live: Mapping[str, float]) -> float:
    return validate_live(live)["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any], model: str) -> dict[str, Any]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == model]
    if len(rows) != 1: raise RuntimeError(f"exact route unavailable: {model}")
    row = rows[0]; architecture = row.get("architecture") or {}; supported = set(row.get("supported_parameters") or [])
    if "text" not in set(architecture.get("input_modalities") or []) or "text" not in set(architecture.get("output_modalities") or []) or "seed" not in supported or not ({"response_format", "structured_outputs"} & supported): raise RuntimeError(f"structured route changed: {model}")
    try: prompt, completion = float(row["pricing"]["prompt"]), float(row["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc: raise RuntimeError("catalog price malformed") from exc
    ceiling = PRICE_CEILINGS[model]
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)) or prompt > ceiling["prompt"] + 1e-15 or completion > ceiling["completion"] + 1e-15: raise RuntimeError("catalog price increased")
    return {"id": model, "prompt_price_usd_per_token": prompt, "completion_price_usd_per_token": completion, "seed_supported": True, "structured_output_supported": True}


def phase_exposure(messages: Sequence[Sequence[dict[str, str]]], max_tokens: int, prices: Mapping[str, Any]) -> dict[str, Any]:
    values = [len(json.dumps(row, sort_keys=True, separators=(",", ":")).encode()) * float(prices["prompt_price_usd_per_token"]) + max_tokens * float(prices["completion_price_usd_per_token"]) for row in messages]
    if len(values) != 10 or any(not math.isfinite(value) or value <= 0 for value in values): raise RuntimeError("exposure malformed")
    return {"exact_phase_exposure_usd": sum(values), "maximum_request_exposure_usd": max(values), "tracker_reservation_usd": 10 * max(values), "request_exposures_usd": values, "max_output_tokens_per_request": max_tokens}


def authorize(*, model: str, messages, max_tokens: int, accepted_cost: float, live_reader, catalog_reader):
    prices = validate_catalog(catalog_reader(), model); exposure = phase_exposure(messages, max_tokens, prices); live = validate_live(live_reader()); reserved = exposure["tracker_reservation_usd"]
    if accepted_cost + reserved > STAGE_CAP_USD + 1e-12: raise RuntimeError("stage cap unavailable")
    if prior_spend(live) + reserved > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < reserved: raise RuntimeError("daily allowance unavailable")
    return {"model": prices, "exposure": exposure, "live": live}


def validate_bindings():
    binding = load(BINDING)
    expected = {"protocol": serving.PROTOCOL, "codec": REPO_ROOT / "scripts/number_game_bitmask_semantic_codec.py", "producer": Path(serving.__file__).resolve(), "verifier": Path(verifier.__file__).resolve(), "wrapper": Path(__file__).resolve(), "tests": REPO_ROOT / "tests/test_number_game_bitmask_semantic_codec.py"}
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != digest(path): raise RuntimeError(f"binding changed: {name}")
    required = {"date": DATE, "opening_total_usage_usd": OPENING_USAGE_USD, "daily_cap_usd": DAILY_CAP_USD, "stage_cap_usd": STAGE_CAP_USD, "proposal_requests": 10, "audit_requests": 10, "maximum_http_attempts": 20, "maximum_retries": 0, "prior_interface_closed": True, "targets_authorized": False, "endpoints_authorized": False}
    if any(binding.get(key) != value for key, value in required.items()): raise RuntimeError("binding metadata changed")
    return {"execution_binding_sha256": digest(BINDING)}


def pristine(path): return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def preflight(*, now=None, live_reader=read_live_credits, catalog_reader=read_catalog):
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE: raise RuntimeError("wrong execution date")
    bindings = validate_bindings()
    if RESULT.exists() or FAILURE.exists() or not pristine(RUN_DIR) or not pristine(LEDGER): raise RuntimeError("bitmask path is not pristine")
    messages = [proposal_messages(HISTORIES[index // 2]) for index in range(10)]
    proposal = authorize(model=serving.PROPOSAL_MODEL_ID, messages=messages, max_tokens=serving.PROPOSAL_MAX_TOKENS, accepted_cost=0.0, live_reader=live_reader, catalog_reader=catalog_reader)
    return {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "ready_without_paid_calls", "date": DATE, "bindings": bindings, "proposal_authorization": proposal, "budget": {"opening_total_usage_usd": OPENING_USAGE_USD, "prior_account_spend_usd": prior_spend(proposal["live"]), "daily_cap_usd": DAILY_CAP_USD, "stage_cap_usd": STAGE_CAP_USD}, "model_calls_made": 0, "files_written": 0}


def adapter_cost(adapter): return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger, status, local_cost, live):
    out = json.loads(json.dumps(ledger)); posted = prior_spend(live) if live else 0.0; local = local_cost + float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD; recorded = max(float(ledger["recorded_actual_spend_usd"]), posted, local)
    if recorded > DAILY_CAP_USD + 1e-12: raise RuntimeError("daily cap exceeded")
    out["recorded_actual_spend_usd"] = recorded; out["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live: out.update({"closing_total_credits_usd": live["total_credits_usd"], "closing_total_usage_usd": live["total_usage_usd"], "closing_balance_usd": live["balance_usd"]})
    return out


def execute(*, live_reader=read_live_credits, catalog_reader=read_catalog):
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader); clean = validate_live(live_reader()); proposal_auth = ready["proposal_authorization"]
    ledger = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "date": DATE, "timezone": TIMEZONE, "opening_total_usage_usd": OPENING_USAGE_USD, "execution_opening_total_credits_usd": clean["total_credits_usd"], "execution_opening_total_usage_usd": clean["total_usage_usd"], "execution_opening_balance_usd": clean["balance_usd"], "recorded_actual_spend_usd": prior_spend(clean), "daily_cap_usd": DAILY_CAP_USD, "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"], "proposal_authorization": proposal_auth, "stage": {"status": "authorized_pending", "maximum_cost_usd": STAGE_CAP_USD, "maximum_http_attempts": 20, "maximum_retries": 0}}
    checkpoint(LEDGER, ledger); proposal_adapter = None; audit_adapter = None
    try:
        pcap = float(proposal_auth["exposure"]["maximum_request_exposure_usd"])
        def authorize_request(cap):
            live = validate_live(live_reader())
            if prior_spend(live) + cap > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < cap: raise RuntimeError("request reservation lost")
        proposal_adapter = serving.build_adapter(model=serving.PROPOSAL_MODEL_ID, run_id="number-game-bitmask-semantic-20260813", output_dir=RUN_DIR, phase_exposure=float(proposal_auth["exposure"]["tracker_reservation_usd"]), request_cap=pcap, max_tokens=serving.PROPOSAL_MAX_TOKENS, authorize=lambda: authorize_request(pcap))
        def audit_factory(messages):
            nonlocal audit_adapter, ledger
            auth = authorize(model=serving.AUDIT_MODEL_ID, messages=messages, max_tokens=serving.AUDIT_MAX_TOKENS, accepted_cost=adapter_cost(proposal_adapter), live_reader=live_reader, catalog_reader=catalog_reader); ledger["audit_authorization"] = auth; checkpoint(LEDGER, ledger); cap = float(auth["exposure"]["maximum_request_exposure_usd"])
            audit_adapter = serving.build_adapter(model=serving.AUDIT_MODEL_ID, run_id="number-game-bitmask-semantic-20260813", output_dir=RUN_DIR, phase_exposure=float(auth["exposure"]["tracker_reservation_usd"]), request_cap=cap, max_tokens=serving.AUDIT_MAX_TOKENS, authorize=lambda: authorize_request(cap)); return audit_adapter
        label = serving.run_gate(output_dir=RUN_DIR, proposal_adapter=proposal_adapter, audit_factory=audit_factory); verification = verifier.verify(RUN_DIR, output=VERIFY)
        try: closing = validate_live(live_reader())
        except Exception: closing = None
        cost = adapter_cost(proposal_adapter) + adapter_cost(audit_adapter); ledger = reconcile(ledger, label["status"], cost, closing); checkpoint(LEDGER, ledger)
        terminal = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": label["status"], "decision": label["decision"], "authorizes": label["authorizes"], "label_free_result_sha256": digest(LABEL), "raw_response_sha256": digest(RAW), "verification_sha256": digest(VERIFY), "ledger_sha256": digest(LEDGER), "actual_cost_usd": cost, "targets_opened": False, "endpoints_opened": False}; checkpoint(RESULT, terminal); return terminal
    except Exception as exc:
        try: closing = validate_live(live_reader())
        except Exception: closing = None
        cost = adapter_cost(proposal_adapter) + adapter_cost(audit_adapter); ledger = reconcile(ledger, "failed_closed", cost, closing); checkpoint(LEDGER, ledger)
        failure = {"schema_version": 1, "interface_version": INTERFACE_VERSION, "status": "failed_closed", "authorizes": "nothing", "error_type": type(exc).__name__, "error": str(exc), "actual_cost_usd": cost, "ledger_sha256": digest(LEDGER), "raw_exists": RAW.exists(), "label_exists": LABEL.exists(), "verification_exists": VERIFY.exists(), "targets_opened": False, "endpoints_opened": False}; checkpoint(FAILURE, failure); raise


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--preflight", action="store_true"); args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"): raise RuntimeError("OPENROUTER_API_KEY required")
    result = preflight() if args.preflight else execute(); print(json.dumps(result, indent=2, sort_keys=True)); return 0


if __name__ == "__main__": raise SystemExit(main())
