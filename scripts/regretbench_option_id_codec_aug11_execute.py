#!/usr/bin/env python3
"""Execute the Aug 11 option-ID codec v1 exact-20 smoke once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_aug10_execute as catalog_api
from scripts import regretbench_option_id_codec_smoke as smoke
from scripts import regretbench_option_id_codec_smoke_verify as verifier
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-option-id-codec-aug11-execute-1"
DATE = "2026-08-11"
TIMEZONE = "Europe/London"
DAILY_CAP_USD = 5.0
RUN_CAP_USD = smoke.RUN_BUDGET_USD
MAX_REQUEST_COST_USD = 0.002
MIN_RESERVED_PROMPT_TOKENS = 8_192
BOUNDARY_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/2026-08-11.json"
)
BOUNDARY_LEDGER_SHA256 = (
    "8ab4fc46b62d896393ed68514473560c53ccb1dc2978e01b53f50cf4e6c4c301"
)
EARLIER_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-11-regretbench-factorized-v2-smoke.json"
)
EARLIER_LEDGER_SHA256 = (
    "8d8b48c4514c41cfb679908289aecc3942625611605ab3190d0e2a79fb5849cf"
)
PREDECESSOR_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-11-regretbench-proposal-evaluator-v1-smoke.json"
)
PREDECESSOR_LEDGER_SHA256 = (
    "307cbb01ec7602e5b43cc999ce897878bbb2a5be227de5b173249680e0140f5a"
)
OPENING_USAGE_USD = 220.129680012
EXECUTION_BINDING = REPO_ROOT / (
    "results/nonmyopic/regretbench_option_id_codec_smoke/"
    "EXECUTION_BINDING.json"
)
ROOT = REPO_ROOT / "results/nonmyopic/regretbench_option_id_codec_smoke"
RUN_DIR = ROOT / "smoke-20260811"
STAGE_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/"
    "2026-08-11-regretbench-option-id-codec-smoke.json"
)
DAILY_RESULT = ROOT / "DAILY_RESULT_20260811.json"
DAILY_FAILURE = ROOT / "DAILY_FAILURE_20260811.json"


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pristine(path: Path) -> bool:
    if not path.exists():
        return True
    if path.is_file():
        return False
    return not any(path.iterdir())


def validate_date(now: datetime | None = None) -> datetime:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"option-ID codec smoke can run only on {DATE}")
    return local


def validate_bindings() -> dict[str, Any]:
    smoke.validate_bindings()
    if not BOUNDARY_LEDGER.is_file() or sha256_file(BOUNDARY_LEDGER) != BOUNDARY_LEDGER_SHA256:
        raise RuntimeError("Aug 11 account boundary ledger changed")
    boundary = load_object(BOUNDARY_LEDGER)
    if (
        boundary.get("date") != DATE
        or boundary.get("timezone") != TIMEZONE
        or float(boundary.get("daily_cap_usd", 0.0)) != DAILY_CAP_USD
        or float(boundary.get("opening_total_usage_usd", math.nan)) != OPENING_USAGE_USD
        or boundary.get("account_wide_usage_counts_against_cap") is not True
    ):
        raise RuntimeError("Aug 11 account boundary is malformed")
    if (
        not EARLIER_LEDGER.is_file()
        or sha256_file(EARLIER_LEDGER) != EARLIER_LEDGER_SHA256
        or load_object(EARLIER_LEDGER).get("stage", {}).get("status")
        != "failed_closed"
    ):
        raise RuntimeError("factorized-v2 predecessor ledger changed")
    if (
        not PREDECESSOR_LEDGER.is_file()
        or sha256_file(PREDECESSOR_LEDGER) != PREDECESSOR_LEDGER_SHA256
        or load_object(PREDECESSOR_LEDGER).get("stage", {}).get("status")
        != "failed_closed"
    ):
        raise RuntimeError("proposal-evaluator predecessor ledger changed")
    binding = load_object(EXECUTION_BINDING)
    expected = {
        "source_protocol": smoke.SOURCE_PROTOCOL,
        "smoke_protocol": smoke.CODEC_PROTOCOL,
        "source_manifest": smoke.SOURCE_MANIFEST,
        "source_result": smoke.SOURCE_RESULT,
        "public_tasks": smoke.PUBLIC_TASKS,
        "source_audit": REPO_ROOT
        / "scripts/regretbench_option_id_codec_source_audit.py",
        "producer": Path(smoke.__file__).resolve(),
        "independent_verifier": Path(verifier.__file__).resolve(),
        "dated_budget_wrapper": Path(__file__).resolve(),
        "account_boundary": BOUNDARY_LEDGER,
        "earlier_stage_ledger": EARLIER_LEDGER,
        "predecessor_stage_ledger": PREDECESSOR_LEDGER,
    }
    for name, path in expected.items():
        row = binding.get(name) or {}
        if row.get("path") != str(path.relative_to(REPO_ROOT)) or row.get(
            "sha256"
        ) != sha256_file(path):
            raise RuntimeError(f"option-ID codec {name} binding changed")
    if (
        binding.get("accepted_requests_authorized") != smoke.EXPECTED_REQUESTS
        or binding.get("maximum_http_attempts")
        != smoke.EXPECTED_REQUESTS + smoke.MAX_RETRIES
        or float(binding.get("run_cap_usd", math.nan)) != RUN_CAP_USD
        or binding.get("development_calls_authorized") is not False
        or binding.get("confirmation_calls_authorized") is not False
        or binding.get("endpoint_outcomes_authorized") is not False
    ):
        raise RuntimeError("option-ID codec execution binding metadata changed")
    return {
        "status": "verified_frozen_execution",
        "execution_binding_sha256": sha256_file(EXECUTION_BINDING),
        "producer_sha256": sha256_file(Path(smoke.__file__).resolve()),
        "verifier_sha256": sha256_file(Path(verifier.__file__).resolve()),
        "wrapper_sha256": sha256_file(Path(__file__).resolve()),
    }


def validate_model_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    rows = [row for row in catalog.get("data", []) if row.get("id") == smoke.MODEL_ID]
    if len(rows) != 1:
        raise RuntimeError("exact DeepSeek 0731 endpoint is unavailable")
    model = rows[0]
    architecture = model.get("architecture") or {}
    inputs = set(architecture.get("input_modalities") or [])
    outputs = set(architecture.get("output_modalities") or [])
    supported = set(model.get("supported_parameters") or [])
    if inputs != {"text"} or "text" not in outputs:
        raise RuntimeError("DeepSeek 0731 modality changed")
    if "seed" not in supported or not ({"response_format", "structured_outputs"} & supported):
        raise RuntimeError("DeepSeek 0731 seeded structured output is unavailable")
    pricing = model.get("pricing") or {}
    try:
        prompt = float(pricing["prompt"])
        completion = float(pricing["completion"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("DeepSeek 0731 live pricing is invalid") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (prompt, completion)):
        raise RuntimeError("DeepSeek 0731 live pricing is invalid")
    maximum_output = max(
        smoke.PROPOSAL_MAX_TOKENS,
        smoke.EVALUATOR_MAX_TOKENS,
        smoke.CODEC_MAX_TOKENS,
    )
    residual = MAX_REQUEST_COST_USD - completion * maximum_output
    covered_prompt = math.inf if prompt == 0 else residual / prompt
    if residual < 0 or covered_prompt + 1e-9 < MIN_RESERVED_PROMPT_TOKENS:
        raise RuntimeError("per-request reservation no longer covers frozen request")
    return {
        "id": model["id"],
        "input_modalities": sorted(inputs),
        "output_modalities": sorted(outputs),
        "seed_supported": True,
        "structured_output_supported": True,
        "prompt_usd_per_million_tokens": prompt * 1_000_000,
        "completion_usd_per_million_tokens": completion * 1_000_000,
        "maximum_request_cost_usd": MAX_REQUEST_COST_USD,
        "covered_prompt_tokens_at_live_price": covered_prompt,
    }


def validate_live(live: Mapping[str, float]) -> None:
    try:
        credits = float(live["total_credits_usd"])
        usage = float(live["total_usage_usd"])
        balance = float(live["balance_usd"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("OpenRouter account values are malformed") from exc
    if not all(math.isfinite(value) and value >= 0 for value in (credits, usage, balance)):
        raise RuntimeError("OpenRouter account values are malformed")
    if abs((credits - usage) - balance) > 1e-6:
        raise RuntimeError("OpenRouter account values are inconsistent")


def prior_spend(live: Mapping[str, float]) -> float:
    validate_live(live)
    usage = float(live["total_usage_usd"])
    if usage + 1e-12 < OPENING_USAGE_USD:
        raise RuntimeError("live usage is below the frozen Aug 11 boundary")
    return usage - OPENING_USAGE_USD


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = catalog_api.read_openrouter_model_catalog,
) -> dict[str, Any]:
    local = validate_date(now)
    bindings = validate_bindings()
    if DAILY_FAILURE.exists():
        raise RuntimeError("option-ID codec smoke already failed closed")
    if DAILY_RESULT.exists():
        raise RuntimeError("option-ID codec smoke already has a terminal result")
    for path in (RUN_DIR, STAGE_LEDGER):
        if not pristine(path):
            raise RuntimeError(f"option-ID codec path is not pristine: {path}")
    model = validate_model_catalog(catalog_reader())
    live = live_reader()
    prior = prior_spend(live)
    if prior + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("option-ID codec smoke exceeds remaining account-wide day")
    if float(live["balance_usd"]) + 1e-12 < RUN_CAP_USD:
        raise RuntimeError("OpenRouter balance is below option-ID codec smoke cap")
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
            "prior_account_spend_usd": prior,
            "run_cap_usd": RUN_CAP_USD,
            "remaining_after_full_cap_usd": DAILY_CAP_USD - prior - RUN_CAP_USD,
        },
        "model_calls_made": 0,
        "files_written": 0,
    }


def initial_ledger(ready: Mapping[str, Any], live: Mapping[str, float]) -> dict[str, Any]:
    prior = prior_spend(live)
    if prior + RUN_CAP_USD > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("usage changed beyond the option-ID codec reservation")
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "date": DATE,
        "timezone": TIMEZONE,
        "daily_cap_usd": DAILY_CAP_USD,
        "opening_total_credits_usd": 245.0,
        "opening_total_usage_usd": OPENING_USAGE_USD,
        "opening_balance_usd": 245.0 - OPENING_USAGE_USD,
        "execution_opening_total_credits_usd": float(live["total_credits_usd"]),
        "execution_opening_total_usage_usd": float(live["total_usage_usd"]),
        "execution_opening_balance_usd": float(live["balance_usd"]),
        "recorded_actual_spend_usd": prior,
        "account_wide_usage_counts_against_cap": True,
        "unspent_allowance_does_not_roll_over": True,
        "account_boundary_sha256": BOUNDARY_LEDGER_SHA256,
        "earlier_stage_ledger_sha256": EARLIER_LEDGER_SHA256,
        "predecessor_stage_ledger_sha256": PREDECESSOR_LEDGER_SHA256,
        "execution_binding_sha256": ready["bindings"]["execution_binding_sha256"],
        "stage": {
            "status": "authorized_pending",
            "model": smoke.MODEL_ID,
            "expected_accepted_requests": smoke.EXPECTED_REQUESTS,
            "maximum_http_attempts": smoke.EXPECTED_REQUESTS + smoke.MAX_RETRIES,
            "maximum_cost_usd": RUN_CAP_USD,
        },
    }


def budget_status(ledger: Mapping[str, Any], live: Mapping[str, float], *, now=None):
    status = require_budget(
        dict(ledger),
        projected_cost_usd=RUN_CAP_USD,
        total_usage_usd=float(live["total_usage_usd"]),
        now=now,
    )
    status.update(live)
    return status


def adapter_cost(adapter: Any) -> float:
    return float(adapter.usage_snapshot().get("adapter_cost_usd", 0.0))


def reconcile(
    ledger: Mapping[str, Any],
    *,
    status: str,
    measured: float,
    live: Mapping[str, float],
    enforce_cap: bool = True,
    live_read_fallback: bool = False,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    posted = prior_spend(live)
    recorded = max(
        posted,
        measured
        + float(ledger["execution_opening_total_usage_usd"])
        - OPENING_USAGE_USD,
    )
    if enforce_cap and recorded > DAILY_CAP_USD + 1e-12:
        raise RuntimeError("option-ID codec reconciliation exceeds daily cap")
    updated["recorded_actual_spend_usd"] = recorded
    updated["stage"].update({"status": status, "actual_cost_usd": measured})
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live["total_credits_usd"]),
        "live_total_usage_usd": float(live["total_usage_usd"]),
        "live_balance_usd": float(live["balance_usd"]),
        "posted_spend_since_boundary_usd": posted,
        "locally_measured_stage_cost_usd": measured,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": DAILY_CAP_USD - recorded,
        "live_read_fallback": live_read_fallback,
    }
    return updated


def failure_live(
    live_reader: Callable[[], dict[str, float]], fallback: Mapping[str, float]
) -> tuple[dict[str, float], bool]:
    try:
        live = live_reader()
        validate_live(live)
        return live, False
    except Exception:
        return dict(fallback), True


def validate_completed() -> dict[str, Any]:
    validate_bindings()
    daily = load_object(DAILY_RESULT)
    ledger = load_object(STAGE_LEDGER)
    replay = verifier.replay(RUN_DIR)
    saved = load_object(RUN_DIR / "VERIFICATION.json")
    if (
        daily.get("status") != "complete_reconciled"
        or daily.get("smoke_result_sha256") != sha256_file(RUN_DIR / "RESULT.json")
        or daily.get("smoke_verification_sha256")
        != sha256_file(RUN_DIR / "VERIFICATION.json")
        or daily.get("stage_ledger_sha256") != sha256_file(STAGE_LEDGER)
        or canonical_json(saved) != canonical_json(replay)
        or replay.get("status") != "verified"
        or daily.get("development_opened") is not False
        or daily.get("confirmation_opened") is not False
        or daily.get("endpoint_outcomes_opened") is not False
        or daily.get("mechanics_opened") is not False
        or ledger.get("stage", {}).get("status")
        not in {"passed", "codec_failed"}
    ):
        raise RuntimeError("completed option-ID codec smoke does not replay exactly")
    return daily


def validate_failed() -> dict[str, Any]:
    validate_bindings()
    failure = load_object(DAILY_FAILURE)
    ledger = load_object(STAGE_LEDGER)
    if (
        failure.get("status") != "failed_closed"
        or failure.get("authorizes") != "nothing"
        or failure.get("development_opened") is not False
        or failure.get("confirmation_opened") is not False
        or failure.get("endpoint_outcomes_opened") is not False
        or failure.get("mechanics_opened") is not False
        or failure.get("stage_ledger_sha256") != sha256_file(STAGE_LEDGER)
        or ledger.get("stage", {}).get("status") != "failed_closed"
    ):
        raise RuntimeError("failed option-ID codec smoke does not replay exactly")
    return failure


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] = catalog_api.read_openrouter_model_catalog,
    adapter_builder=None,
) -> dict[str, Any]:
    if DAILY_RESULT.is_file():
        return validate_completed()
    if DAILY_FAILURE.is_file():
        return validate_failed()
    ready = preflight(now=now, live_reader=live_reader, catalog_reader=catalog_reader)
    race_live = live_reader()
    ledger = initial_ledger(ready, race_live)
    checkpoint(STAGE_LEDGER, ledger)
    builder = adapter_builder or smoke.build_adapter
    adapter = builder(
        run_id="regretbench-option-id-codec-smoke-20260811",
        output_dir=RUN_DIR,
    )
    try:
        result = smoke.run_smoke(
            output_dir=RUN_DIR,
            adapter=adapter,
            daily_budget_status=budget_status(ledger, race_live, now=now),
        )
        replay = verifier.replay(RUN_DIR)
        checkpoint(RUN_DIR / "VERIFICATION.json", replay)
        if replay.get("status") != "verified":
            raise RuntimeError("option-ID codec independent replay failed")
        ledger = reconcile(
            ledger,
            status=result["status"],
            measured=float(result["usage"]["run_cost_usd"]),
            live=live_reader(),
        )
        checkpoint(STAGE_LEDGER, ledger)
    except Exception as exc:
        final_live, used_fallback = failure_live(live_reader, race_live)
        ledger = reconcile(
            ledger,
            status="failed_closed",
            measured=adapter_cost(adapter),
            live=final_live,
            enforce_cap=False,
            live_read_fallback=used_fallback,
        )
        checkpoint(STAGE_LEDGER, ledger)
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "authorizes": "nothing",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "endpoint_outcomes_opened": False,
            "mechanics_opened": False,
            "development_opened": False,
            "confirmation_opened": False,
            "stage_ledger_sha256": sha256_file(STAGE_LEDGER),
        }
        checkpoint(DAILY_FAILURE, failure)
        raise
    daily = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete_reconciled",
        "smoke_status": result["status"],
        "authorizes": result["authorizes"],
        "smoke_result_sha256": sha256_file(RUN_DIR / "RESULT.json"),
        "smoke_verification_sha256": sha256_file(RUN_DIR / "VERIFICATION.json"),
        "independent_replay_passed": True,
        "endpoint_outcomes_opened": False,
        "mechanics_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
        "stage_ledger_sha256": sha256_file(STAGE_LEDGER),
        "recorded_daily_spend_usd": ledger["recorded_actual_spend_usd"],
    }
    checkpoint(DAILY_RESULT, daily)
    return daily


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight() if args.preflight else execute()
    except Exception as exc:
        result = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
