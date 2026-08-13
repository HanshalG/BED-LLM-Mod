#!/usr/bin/env python3
"""Execute the Aug 13 Number Game extension-native semantic gate once."""

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
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_extension_native_semantic_gate as serving
from scripts import number_game_extension_native_semantic_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_extension_native_semantic_codec import HISTORIES, proposal_messages
from scripts.openrouter_daily_budget import read_live_credits


INTERFACE_VERSION = "number-game-extension-native-semantic-aug13-execute-1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
OPENING_CREDITS_USD = 245.0
OPENING_USAGE_USD = 220.134128880
STAGE_CAP_USD = 0.08
PRICE_CEILINGS = {
    serving.PROPOSAL_MODEL_ID: {"prompt": 0.32 / 1_000_000, "completion": 1.28 / 1_000_000},
    serving.AUDIT_MODEL_ID: {"prompt": 0.10 / 1_000_000, "completion": 0.60 / 1_000_000},
}
OUTPUT_ROOT = REPO_ROOT / "results/nonmyopic/number_game_extension_native_semantic_gate"
RUN_DIR = OUTPUT_ROOT / "mechanics-20260813"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-extension-native-semantic.json"
EXECUTION_BINDING = OUTPUT_ROOT / "EXECUTION_BINDING.json"
DAILY_RESULT = OUTPUT_ROOT / "DAILY_RESULT_20260813.json"
DAILY_FAILURE = OUTPUT_ROOT / "DAILY_FAILURE_20260813.json"
RAW = RUN_DIR / "private/RAW_RESPONSES.json"
LABEL_FREE = RUN_DIR / "LABEL_FREE_RESULT.json"
VERIFICATION = RUN_DIR / "VERIFICATION.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


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
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("OpenRouter catalog is malformed")
    return value


def validate_live(live: Mapping[str, float]) -> dict[str, float]:
    try:
        credits = float(live["total_credits_usd"])
        usage = float(live["total_usage_usd"])
        balance = float(live["balance_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter account values are malformed") from exc
    if (
        not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance))
        or abs(credits - usage - balance) > 1e-6
        or usage + 1e-12 < OPENING_USAGE_USD
    ):
        raise RuntimeError("OpenRouter account values are invalid")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live: Mapping[str, float]) -> float:
    return validate_live(live)["total_usage_usd"] - OPENING_USAGE_USD


def validate_catalog(catalog: Mapping[str, Any], model_id: str) -> dict[str, float | str | bool]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == model_id]
    if len(rows) != 1:
        raise RuntimeError(f"exact model route is unavailable: {model_id}")
    model = rows[0]
    architecture = model.get("architecture") or {}
    supported = set(model.get("supported_parameters") or [])
    if (
        "text" not in set(architecture.get("input_modalities") or [])
        or "text" not in set(architecture.get("output_modalities") or [])
        or "seed" not in supported
        or not ({"response_format", "structured_outputs"} & supported)
    ):
        raise RuntimeError(f"structured text route changed: {model_id}")
    try:
        prompt = float(model["pricing"]["prompt"])
        completion = float(model["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(f"catalog pricing is malformed: {model_id}") from exc
    ceilings = PRICE_CEILINGS[model_id]
    if (
        not all(math.isfinite(value) and value >= 0 for value in (prompt, completion))
        or prompt > ceilings["prompt"] + 1e-15
        or completion > ceilings["completion"] + 1e-15
    ):
        raise RuntimeError(f"catalog price increased beyond frozen ceiling: {model_id}")
    return {
        "id": model_id,
        "prompt_price_usd_per_token": prompt,
        "completion_price_usd_per_token": completion,
        "seed_supported": True,
        "structured_output_supported": True,
    }


def message_bytes(messages: Sequence[dict[str, str]]) -> int:
    return len(canonical_json(messages).encode("utf-8"))


def phase_exposure(messages: Sequence[Sequence[dict[str, str]]], *, max_tokens: int, prices: Mapping[str, Any]) -> dict[str, Any]:
    input_price = float(prices["prompt_price_usd_per_token"])
    output_price = float(prices["completion_price_usd_per_token"])
    request_exposures = [message_bytes(item) * input_price + max_tokens * output_price for item in messages]
    if len(request_exposures) != 10 or any(not math.isfinite(value) or value <= 0 for value in request_exposures):
        raise RuntimeError("phase exposure is malformed")
    return {
        "message_bytes": [message_bytes(item) for item in messages],
        "request_exposures_usd": request_exposures,
        "maximum_request_exposure_usd": max(request_exposures),
        "tracker_reservation_usd": 10 * max(request_exposures),
        "exact_phase_exposure_usd": sum(request_exposures),
        "max_output_tokens_per_request": max_tokens,
    }


def validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"semantic-support gate can run only on {DATE}")
    return local


def pristine(path: Path) -> bool:
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def validate_bindings() -> dict[str, Any]:
    binding = load_object(EXECUTION_BINDING)
    expected = {
        "protocol": serving.PROTOCOL_PATH,
        "budget_amendment": serving.AMENDMENT_PATH,
        "codec": REPO_ROOT / "scripts/number_game_extension_native_semantic_codec.py",
        "producer": Path(serving.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
        "tests": REPO_ROOT / "tests/test_number_game_extension_native_semantic_gate.py",
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != sha256_file(path):
            raise RuntimeError(f"execution binding changed: {name}")
    required = {
        "date": DATE,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "daily_cap_usd": DAILY_CAP_USD,
        "stage_cap_usd": STAGE_CAP_USD,
        "proposal_requests": 10,
        "audit_requests": 10,
        "maximum_http_attempts": 20,
        "maximum_retries": 0,
        "number_game_targets_authorized": False,
        "policy_endpoints_authorized": False,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise RuntimeError("execution metadata changed")
    return {"execution_binding_sha256": sha256_file(EXECUTION_BINDING)}


def authorize_phase(
    *,
    model_id: str,
    messages: Sequence[Sequence[dict[str, str]]],
    max_tokens: int,
    accepted_cost_usd: float,
    live_reader: Callable[[], dict[str, float]],
    catalog_reader: Callable[[], dict[str, Any]],
) -> dict[str, Any]:
    prices = validate_catalog(catalog_reader(), model_id)
    exposure = phase_exposure(messages, max_tokens=max_tokens, prices=prices)
    live = validate_live(live_reader())
    reserved = float(exposure["tracker_reservation_usd"])
    if accepted_cost_usd + reserved > STAGE_CAP_USD + 1e-12:
        raise RuntimeError("phase reservation exceeds the frozen stage cap")
    if prior_spend(live) + reserved > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < reserved:
        raise RuntimeError("phase reservation exceeds live account allowance")
    return {"model": prices, "exposure": exposure, "live": live}


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = read_catalog,
) -> dict[str, Any]:
    local = validate_date(now)
    bindings = validate_bindings()
    if DAILY_RESULT.exists() or DAILY_FAILURE.exists():
        raise RuntimeError("semantic-support gate is already terminal")
    for path in (RUN_DIR, LEDGER):
        if not pristine(path):
            raise RuntimeError(f"execution path is not pristine: {path}")
    messages = [proposal_messages(HISTORIES[index // 2]) for index in range(10)]
    authorization = authorize_phase(model_id=serving.PROPOSAL_MODEL_ID, messages=messages, max_tokens=serving.PROPOSAL_MAX_TOKENS, accepted_cost_usd=0.0, live_reader=live_reader, catalog_reader=catalog_reader)
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": local.date().isoformat(),
        "bindings": bindings,
        "proposal_authorization": authorization,
        "budget": {"daily_cap_usd": DAILY_CAP_USD, "opening_total_usage_boundary_usd": OPENING_USAGE_USD, "prior_account_spend_usd": prior_spend(authorization["live"]), "stage_cap_usd": STAGE_CAP_USD, "remaining_after_proposal_reservation_usd": DAILY_CAP_USD - prior_spend(authorization["live"]) - authorization["exposure"]["tracker_reservation_usd"]},
        "model_calls_made": 0,
        "files_written": 0,
    }


def initial_ledger(ready: Mapping[str, Any], live: Mapping[str, float]) -> dict[str, Any]:
    clean = validate_live(live)
    proposal_reservation = float(ready["proposal_authorization"]["exposure"]["tracker_reservation_usd"])
    if prior_spend(clean) + proposal_reservation > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("usage raced beyond proposal reservation")
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": OPENING_CREDITS_USD,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "execution_opening_total_credits_usd": clean["total_credits_usd"],
        "execution_opening_total_usage_usd": clean["total_usage_usd"],
        "execution_opening_balance_usd": clean["balance_usd"],
        "recorded_actual_spend_usd": prior_spend(clean),
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"],
        "proposal_authorization": ready["proposal_authorization"],
        "stage": {"status": "authorized_pending", "maximum_cost_usd": STAGE_CAP_USD, "maximum_http_attempts": 20, "maximum_retries": 0},
    }


def adapter_cost(adapter: Any | None) -> float:
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger: Mapping[str, Any], *, status: str, local_cost: float, live: Mapping[str, float] | None) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    local_total = local_cost + float(ledger["execution_opening_total_usage_usd"]) - OPENING_USAGE_USD
    posted = prior_spend(live) if live is not None else 0.0
    recorded = max(float(ledger["recorded_actual_spend_usd"]), local_total, posted)
    if recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("semantic-support reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live is not None:
        updated.update({"closing_total_credits_usd": float(live["total_credits_usd"]), "closing_total_usage_usd": float(live["total_usage_usd"]), "closing_balance_usd": float(live["balance_usd"])})
    return updated


def execute(
    *,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = read_catalog,
) -> dict[str, Any]:
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader)
    ledger = initial_ledger(ready, validate_live(live_reader()))
    checkpoint(LEDGER, ledger)
    proposal_adapter = None
    audit_adapter = None
    try:
        proposal_auth = ready["proposal_authorization"]
        proposal_request_cap = float(proposal_auth["exposure"]["maximum_request_exposure_usd"])

        def authorize_proposal_request() -> None:
            live = validate_live(live_reader())
            if prior_spend(live) + proposal_request_cap > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < proposal_request_cap:
                raise RuntimeError("proposal request lost its live account reservation")

        proposal_adapter = serving.build_adapter(
            model_id=serving.PROPOSAL_MODEL_ID,
            run_id="number-game-extension-native-semantic-20260813",
            output_dir=RUN_DIR,
            phase_exposure_usd=float(proposal_auth["exposure"]["tracker_reservation_usd"]),
            maximum_request_cost_usd=proposal_request_cap,
            max_tokens=serving.PROPOSAL_MAX_TOKENS,
            pre_dispatch_authorizer=authorize_proposal_request,
        )

        def audit_factory(messages: Sequence[list[dict[str, str]]]):
            nonlocal audit_adapter, ledger
            proposal_cost = adapter_cost(proposal_adapter)
            audit_auth = authorize_phase(model_id=serving.AUDIT_MODEL_ID, messages=messages, max_tokens=serving.AUDIT_MAX_TOKENS, accepted_cost_usd=proposal_cost, live_reader=live_reader, catalog_reader=catalog_reader)
            ledger["audit_authorization"] = audit_auth
            checkpoint(LEDGER, ledger)
            audit_request_cap = float(audit_auth["exposure"]["maximum_request_exposure_usd"])

            def authorize_audit_request() -> None:
                live = validate_live(live_reader())
                if prior_spend(live) + audit_request_cap > DAILY_CAP_USD + 1e-12 or live["balance_usd"] + 1e-12 < audit_request_cap:
                    raise RuntimeError("audit request lost its live account reservation")

            audit_adapter = serving.build_adapter(
                model_id=serving.AUDIT_MODEL_ID,
                run_id="number-game-extension-native-semantic-20260813",
                output_dir=RUN_DIR,
                phase_exposure_usd=float(audit_auth["exposure"]["tracker_reservation_usd"]),
                maximum_request_cost_usd=audit_request_cap,
                max_tokens=serving.AUDIT_MAX_TOKENS,
                pre_dispatch_authorizer=authorize_audit_request,
            )
            return audit_adapter

        label_free = serving.run_gate(output_dir=RUN_DIR, proposal_adapter=proposal_adapter, audit_adapter_factory=audit_factory)
        verification = verifier.verify(RUN_DIR, output_path=VERIFICATION)
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        total_cost = adapter_cost(proposal_adapter) + adapter_cost(audit_adapter)
        ledger = reconcile(ledger, status=label_free["status"], local_cost=total_cost, live=closing)
        checkpoint(LEDGER, ledger)
        terminal = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": label_free["status"],
            "decision": label_free["decision"],
            "authorizes": label_free["authorizes"],
            "label_free_result_sha256": sha256_file(LABEL_FREE),
            "raw_response_sha256": sha256_file(RAW),
            "verification_sha256": sha256_file(VERIFICATION),
            "ledger_sha256": sha256_file(LEDGER),
            "actual_cost_usd": total_cost,
            "number_game_targets_opened": False,
            "policy_endpoints_opened": False,
        }
        checkpoint(DAILY_RESULT, terminal)
        return terminal
    except Exception as exc:
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        total_cost = adapter_cost(proposal_adapter) + adapter_cost(audit_adapter)
        ledger = reconcile(ledger, status="failed_closed", local_cost=total_cost, live=closing)
        checkpoint(LEDGER, ledger)
        failure = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "authorizes": "nothing",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_cost_usd": total_cost,
            "ledger_sha256": sha256_file(LEDGER),
            "raw_response_exists": RAW.exists(),
            "label_free_result_exists": LABEL_FREE.exists(),
            "verification_exists": VERIFICATION.exists(),
            "number_game_targets_opened": False,
            "policy_endpoints_opened": False,
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
