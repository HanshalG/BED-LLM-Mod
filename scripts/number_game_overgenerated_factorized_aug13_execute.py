#!/usr/bin/env python3
"""Execute the fresh Aug 13 overgenerated factorized gate once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import sys
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import number_game_overgenerated_factorized_gate as serving
from scripts import number_game_overgenerated_factorized_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_overgenerated_factorized_codec import HISTORIES, proposal_messages
from scripts.openrouter_daily_budget import read_live_credits


INTERFACE_VERSION = "number-game-overgenerated-factorized-aug13-execute-1"
DATE = "2026-08-13"
TIMEZONE = "Europe/London"
OPENING = 220.134128880
DAILY_CAP = 5.0
STAGE_CAP = 0.08
ROOT = REPO_ROOT / "results/nonmyopic/number_game_overgenerated_factorized_gate"
RUN = ROOT / "mechanics-20260813"
BINDING = ROOT / "EXECUTION_BINDING.json"
RESULT = ROOT / "DAILY_RESULT_20260813.json"
FAILURE = ROOT / "DAILY_FAILURE_20260813.json"
LEDGER = REPO_ROOT / "results/nonmyopic/openrouter_daily_budget/2026-08-13-number-game-overgenerated-factorized.json"
RAW = RUN / "private/RAW_RESPONSES.json"
LABEL = RUN / "LABEL_FREE_RESULT.json"
VERIFY = RUN / "VERIFICATION.json"
MODELS_URL = "https://openrouter.ai/api/v1/models"
PRICE_CEILINGS = {
    serving.PROPOSAL_MODEL_ID: {"prompt": .32e-6, "completion": 1.28e-6},
    serving.TRANSLATION_MODEL_ID: {"prompt": .08e-6, "completion": .18e-6},
    serving.AUDIT_MODEL_ID: {"prompt": .10e-6, "completion": .60e-6},
}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path):
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise RuntimeError("expected object")
    return value


def read_catalog():
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("key required")
    request = Request(MODELS_URL, headers={"Authorization": f"Bearer {key}"})
    with urlopen(request, timeout=30) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("catalog malformed")
    return value


def validate_live(live):
    try:
        credits, usage, balance = (float(live[key]) for key in ("total_credits_usd", "total_usage_usd", "balance_usd"))
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("account values malformed") from exc
    if (
        not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance))
        or abs(credits - usage - balance) > 1e-6
        or usage + 1e-12 < OPENING
    ):
        raise RuntimeError("account values invalid")
    return {"total_credits_usd": credits, "total_usage_usd": usage, "balance_usd": balance}


def prior_spend(live):
    return validate_live(live)["total_usage_usd"] - OPENING


def validate_catalog(catalog, model):
    rows = [row for row in catalog.get("data", []) if row.get("id") == model]
    if len(rows) != 1:
        raise RuntimeError(f"exact route unavailable: {model}")
    row = rows[0]
    architecture = row.get("architecture") or {}
    supported = set(row.get("supported_parameters") or [])
    if (
        "text" not in set(architecture.get("input_modalities") or [])
        or "text" not in set(architecture.get("output_modalities") or [])
        or "seed" not in supported
        or not ({"response_format", "structured_outputs"} & supported)
    ):
        raise RuntimeError(f"structured route changed: {model}")
    try:
        prompt, completion = float(row["pricing"]["prompt"]), float(row["pricing"]["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("catalog price malformed") from exc
    ceiling = PRICE_CEILINGS[model]
    if (
        not all(math.isfinite(value) and value >= 0 for value in (prompt, completion))
        or prompt > ceiling["prompt"] + 1e-15
        or completion > ceiling["completion"] + 1e-15
    ):
        raise RuntimeError("catalog price increased")
    return {
        "id": model,
        "prompt_price_usd_per_token": prompt,
        "completion_price_usd_per_token": completion,
        "seed_supported": True,
        "structured_output_supported": True,
    }


def phase_exposure(messages, max_tokens, prices):
    values = [
        len(json.dumps(message, sort_keys=True, separators=(",", ":")).encode()) * float(prices["prompt_price_usd_per_token"])
        + max_tokens * float(prices["completion_price_usd_per_token"])
        for message in messages
    ]
    if not values or any(not math.isfinite(value) or value <= 0 for value in values):
        raise RuntimeError("phase exposure malformed")
    return {
        "request_count": len(values),
        "exact_phase_exposure_usd": sum(values),
        "maximum_request_exposure_usd": max(values),
        "tracker_reservation_usd": len(values) * max(values),
        "request_exposures_usd": values,
        "max_output_tokens_per_request": max_tokens,
    }


def authorize(model, messages, max_tokens, accepted_cost, live_reader, catalog_reader):
    prices = validate_catalog(catalog_reader(), model)
    exposure = phase_exposure(messages, max_tokens, prices)
    live = validate_live(live_reader())
    reserve = exposure["tracker_reservation_usd"]
    if accepted_cost + reserve > STAGE_CAP + 1e-12:
        raise RuntimeError("overgenerated factorized stage cap unavailable")
    if prior_spend(live) + reserve > DAILY_CAP + 1e-12 or live["balance_usd"] + 1e-12 < reserve:
        raise RuntimeError("account allowance unavailable")
    return {"model": prices, "exposure": exposure, "live": live}


def validate_bindings():
    binding = load(BINDING)
    expected = {
        "protocol": serving.PROTOCOL,
        "codec": REPO_ROOT / "scripts/number_game_overgenerated_factorized_codec.py",
        "producer": Path(serving.__file__).resolve(),
        "verifier": Path(verifier.__file__).resolve(),
        "wrapper": Path(__file__).resolve(),
        "tests": REPO_ROOT / "tests/test_number_game_overgenerated_factorized_codec.py",
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get("sha256") != digest(path):
            raise RuntimeError(f"binding changed: {name}")
    required = {
        "date": DATE,
        "opening_total_usage_usd": OPENING,
        "daily_cap_usd": DAILY_CAP,
        "stage_cap_usd": STAGE_CAP,
        "proposal_requests": 40,
        "translation_requests": 30,
        "audit_requests": 10,
        "maximum_http_attempts": 80,
        "maximum_retries": 0,
        "targets_authorized": False,
        "endpoints_authorized": False,
    }
    if any(binding.get(key) != value for key, value in required.items()):
        raise RuntimeError("binding metadata changed")
    return {"execution_binding_sha256": digest(BINDING)}


def pristine(path):
    return not path.exists() or (path.is_dir() and not any(path.iterdir()))


def proposal_batch():
    return [proposal_messages(HISTORIES[draw // 2], shard) for draw in range(10) for shard in range(4)]


def preflight(*, now=None, live_reader=read_live_credits, catalog_reader=read_catalog):
    local = now.astimezone(ZoneInfo(TIMEZONE)) if now else datetime.now(ZoneInfo(TIMEZONE))
    if local.date().isoformat() != DATE:
        raise RuntimeError("wrong execution date")
    bindings = validate_bindings()
    if RESULT.exists() or FAILURE.exists() or not pristine(RUN) or not pristine(LEDGER):
        raise RuntimeError("overgenerated factorized path is not pristine")
    proposal = authorize(serving.PROPOSAL_MODEL_ID, proposal_batch(), serving.PROPOSAL_MAX_TOKENS, 0.0, live_reader, catalog_reader)
    return {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "bindings": bindings,
        "proposal_authorization": proposal,
        "budget": {
            "opening_total_usage_usd": OPENING,
            "prior_account_spend_usd": prior_spend(proposal["live"]),
            "daily_cap_usd": DAILY_CAP,
            "stage_cap_usd": STAGE_CAP,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def adapter_cost(adapter):
    return 0.0 if adapter is None else float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(ledger, status, local_cost, live):
    out = json.loads(json.dumps(ledger))
    recorded = max(
        float(ledger["recorded_actual_spend_usd"]),
        prior_spend(live) if live else 0.0,
        local_cost + float(ledger["execution_opening_total_usage_usd"]) - OPENING,
    )
    if recorded > DAILY_CAP + 1e-12:
        raise RuntimeError("daily cap exceeded")
    out["recorded_actual_spend_usd"] = recorded
    out["stage"].update({"status": status, "actual_cost_usd": local_cost})
    if live:
        out.update({
            "closing_total_credits_usd": live["total_credits_usd"],
            "closing_total_usage_usd": live["total_usage_usd"],
            "closing_balance_usd": live["balance_usd"],
        })
    return out


def execute(*, live_reader=read_live_credits, catalog_reader=read_catalog):
    ready = preflight(live_reader=live_reader, catalog_reader=catalog_reader)
    live = validate_live(live_reader())
    proposal_authorization = ready["proposal_authorization"]
    ledger = {
        "schema_version": 1,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "opening_total_usage_usd": OPENING,
        "execution_opening_total_credits_usd": live["total_credits_usd"],
        "execution_opening_total_usage_usd": live["total_usage_usd"],
        "execution_opening_balance_usd": live["balance_usd"],
        "recorded_actual_spend_usd": prior_spend(live),
        "daily_cap_usd": DAILY_CAP,
        "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"],
        "proposal_authorization": proposal_authorization,
        "stage": {"status": "authorized_pending", "maximum_cost_usd": STAGE_CAP, "maximum_http_attempts": 80, "maximum_retries": 0},
    }
    checkpoint(LEDGER, ledger)
    proposal = translation = audit = None
    try:
        proposal_cap = float(proposal_authorization["exposure"]["maximum_request_exposure_usd"])

        def request_authorizer(cap):
            current = validate_live(live_reader())
            if prior_spend(current) + cap > DAILY_CAP + 1e-12 or current["balance_usd"] + 1e-12 < cap:
                raise RuntimeError("request allowance lost")

        proposal = serving.build_adapter(
            model=serving.PROPOSAL_MODEL_ID,
            run_id="number-game-overgenerated-factorized-20260813",
            output_dir=RUN,
            phase_exposure=float(proposal_authorization["exposure"]["tracker_reservation_usd"]),
            request_cap=proposal_cap,
            max_tokens=serving.PROPOSAL_MAX_TOKENS,
            temperature=.7,
            authorize=lambda: request_authorizer(proposal_cap),
        )

        def translation_factory(messages):
            nonlocal translation, ledger
            auth = authorize(serving.TRANSLATION_MODEL_ID, messages, serving.TRANSLATION_MAX_TOKENS, adapter_cost(proposal), live_reader, catalog_reader)
            ledger["translation_authorization"] = auth
            checkpoint(LEDGER, ledger)
            cap = float(auth["exposure"]["maximum_request_exposure_usd"])
            translation = serving.build_adapter(
                model=serving.TRANSLATION_MODEL_ID,
                run_id="number-game-overgenerated-factorized-20260813",
                output_dir=RUN,
                phase_exposure=float(auth["exposure"]["tracker_reservation_usd"]),
                request_cap=cap,
                max_tokens=serving.TRANSLATION_MAX_TOKENS,
                temperature=0.0,
                authorize=lambda: request_authorizer(cap),
            )
            return translation

        def audit_factory(messages):
            nonlocal audit, ledger
            accepted = adapter_cost(proposal) + adapter_cost(translation)
            auth = authorize(serving.AUDIT_MODEL_ID, messages, serving.AUDIT_MAX_TOKENS, accepted, live_reader, catalog_reader)
            ledger["audit_authorization"] = auth
            checkpoint(LEDGER, ledger)
            cap = float(auth["exposure"]["maximum_request_exposure_usd"])
            audit = serving.build_adapter(
                model=serving.AUDIT_MODEL_ID,
                run_id="number-game-overgenerated-factorized-20260813",
                output_dir=RUN,
                phase_exposure=float(auth["exposure"]["tracker_reservation_usd"]),
                request_cap=cap,
                max_tokens=serving.AUDIT_MAX_TOKENS,
                temperature=0.0,
                authorize=lambda: request_authorizer(cap),
            )
            return audit

        label = serving.run_gate(
            output_dir=RUN,
            proposal_adapter=proposal,
            translation_factory=translation_factory,
            audit_factory=audit_factory,
        )
        verifier.verify(RUN, output=VERIFY)
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        actual = adapter_cost(proposal) + adapter_cost(translation) + adapter_cost(audit)
        ledger = reconcile(ledger, label["status"], actual, closing)
        checkpoint(LEDGER, ledger)
        terminal = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": label["status"],
            "decision": label["decision"],
            "authorizes": label["authorizes"],
            "label_sha256": digest(LABEL),
            "raw_sha256": digest(RAW),
            "verification_sha256": digest(VERIFY),
            "ledger_sha256": digest(LEDGER),
            "actual_cost_usd": actual,
            "targets_opened": False,
            "endpoints_opened": False,
        }
        checkpoint(RESULT, terminal)
        return terminal
    except Exception as exc:
        try:
            closing = validate_live(live_reader())
        except Exception:
            closing = None
        actual = adapter_cost(proposal) + adapter_cost(translation) + adapter_cost(audit)
        ledger = reconcile(ledger, "failed_closed", actual, closing)
        checkpoint(LEDGER, ledger)
        failure = {
            "schema_version": 1,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "authorizes": "nothing",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "actual_cost_usd": actual,
            "ledger_sha256": digest(LEDGER),
            "raw_exists": RAW.exists(),
            "label_exists": LABEL.exists(),
            "verification_exists": VERIFY.exists(),
            "targets_opened": False,
            "endpoints_opened": False,
        }
        checkpoint(FAILURE, failure)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    if not os.environ.get("OPENROUTER_API_KEY"):
        raise RuntimeError("key required")
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
