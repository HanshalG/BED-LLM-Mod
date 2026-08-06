#!/usr/bin/env python3
"""Execute the frozen August 7 Number Game budget sequence exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime
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
from scripts import number_game_budget_model_reliability128 as reliability
from scripts import number_game_budget_model_stress3584 as stress
from scripts import number_game_qwen_fully_fresh_control_daily_execute as control
from scripts import number_game_qwen_fully_fresh_claim_report as claim
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-budget-model-aug7-execute-1"
EXPECTED_DATE = "2026-08-07"
TIMEZONE = "Europe/London"
CONTROL_RUN_ID = (
    "number-game-qwen-fully-fresh-daily-stages-20260806T000200Z"
)
CONTROL_RUN_DIR = REPO_ROOT / (
    "results/nonmyopic/number_game_qwen_fully_fresh_daily_stages/"
    f"{CONTROL_RUN_ID}"
)
DAILY_LEDGER = REPO_ROOT / (
    "results/nonmyopic/openrouter_daily_budget/2026-08-07.json"
)
OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/number_game_budget_model_aug7_execution/"
    "number-game-budget-model-aug7-20260807"
)
RELIABILITY_RUNS = {
    "openai/gpt-5.6-luna": (
        "number-game-budget-model-reliability128-luna-20260807"
    ),
    "deepseek/deepseek-v4-flash-0731": (
        "number-game-budget-model-reliability128-deepseek0731-20260807"
    ),
}
RELIABILITY_ROOT = REPO_ROOT / (
    "results/nonmyopic/number_game_budget_model_reliability128"
)
STRESS_RUN_ID = "number-game-budget-model-stress3584-20260807"
STRESS_OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/number_game_budget_model_stress3584/"
    f"{STRESS_RUN_ID}"
)
EXPECTED_CONTROL_COST_USD = 3.21
MINIMUM_STARTING_BALANCE_USD = 5.0
MODELS_URL = "https://openrouter.ai/api/v1/models"
QWEN_MODEL_ID = control.staged.base.control.MODEL_ID
MODEL_MAX_OUTPUT_TOKENS = {
    QWEN_MODEL_ID: control.staged.base.control.MAX_TOKENS,
    **{model: reliability.MAX_TOKENS for model in RELIABILITY_RUNS},
}


class PreExecutionGateError(RuntimeError):
    """A zero-call launch gate failed before any artifact should be written."""


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_openrouter_model_catalog() -> dict[str, Any]:
    headers = {}
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = Request(MODELS_URL, headers=headers)
    with urlopen(request, timeout=30) as response:
        return json.load(response)


def _validate_model_catalog(catalog: Mapping[str, Any]) -> dict[str, Any]:
    records = {model.get("id"): model for model in catalog.get("data", [])}
    verified = {}
    for model_id, required_tokens in MODEL_MAX_OUTPUT_TOKENS.items():
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
        if max_completion < required_tokens:
            raise RuntimeError(f"{model_id} completion limit is too small")
        if not all(
            math.isfinite(value) and value >= 0.0
            for value in (prompt_price, completion_price)
        ):
            raise RuntimeError(f"{model_id} pricing is missing or invalid")
        verified[model_id] = {
            "context_length": int(model.get("context_length") or 0),
            "max_completion_tokens": max_completion,
            "required_completion_tokens": required_tokens,
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


def _validate_date(ledger: dict[str, Any], now: datetime | None) -> None:
    timezone = ZoneInfo(TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if ledger.get("date") != EXPECTED_DATE:
        raise RuntimeError("the frozen sequence requires the August 7 ledger")
    if ledger.get("timezone") != TIMEZONE:
        raise RuntimeError("the frozen sequence requires Europe/London")
    if local_now.date().isoformat() != EXPECTED_DATE:
        raise RuntimeError("the frozen sequence can run only on August 7")
    if float(ledger.get("daily_cap_usd", 0.0)) != 5.0:
        raise RuntimeError("the frozen sequence requires the exact $5 daily cap")


def _validate_ledger_boundary(ledger: dict[str, Any]) -> None:
    recorded = float(ledger.get("recorded_actual_spend_usd", math.inf))
    if (
        ledger.get("date") != EXPECTED_DATE
        or ledger.get("timezone") != TIMEZONE
        or float(ledger.get("daily_cap_usd", 0.0)) != 5.0
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("unspent_allowance_does_not_roll_over") is not True
        or not math.isfinite(recorded)
        or not 0.0 <= recorded <= 5.0
    ):
        raise RuntimeError("August 7 daily ledger boundary changed")
    reconciliation = ledger.get("reconciliation")
    if reconciliation is not None:
        remaining = float(
            reconciliation.get("remaining_daily_allowance_usd", math.inf)
        )
        if not math.isfinite(remaining) or remaining != max(0.0, 5.0 - recorded):
            raise RuntimeError("August 7 daily ledger reconciliation changed")


def _validate_pristine_ledger(ledger: dict[str, Any]) -> None:
    _validate_ledger_boundary(ledger)
    first = ledger.get("first_authorized_block") or {}
    opening_fields = (
        "opening_total_credits_usd",
        "opening_total_usage_usd",
        "opening_balance_usd",
    )
    if any(
        not math.isfinite(float(ledger.get(field, math.inf)))
        for field in opening_fields
    ):
        raise RuntimeError("August 7 opening credit boundary changed")
    if (
        ledger.get("opening_baseline_frozen_on_previous_closed_day") is not True
        or not math.isclose(
            float(ledger["opening_balance_usd"]),
            float(ledger["opening_total_credits_usd"])
            - float(ledger["opening_total_usage_usd"]),
            rel_tol=0.0,
            abs_tol=1e-9,
        )
        or float(ledger["recorded_actual_spend_usd"]) != 0.0
        or ledger.get("additional_paid_blocks_authorized") is not False
        or ledger.get("authorized_tail_blocks") not in (None, [])
        or float(first.get("maximum_cost_usd", 0.0))
        != control.staged.CONTROL_DAILY_CAP_USD
        or float(first.get("expected_cost_usd", 0.0))
        != EXPECTED_CONTROL_COST_USD
        or first.get("actual_cost_usd") is not None
        or first.get("status") != "authorized_pending_later_day"
    ):
        raise RuntimeError("August 7 pristine ledger authorization changed")


def _verify_source_readiness(run_dir: Path) -> dict[str, Any]:
    control.staged.base.validate_predecessors()
    _, authorization, source_hashes = control.staged.validate_control_handoff(
        output_dir=run_dir,
        control_calendar_date=EXPECTED_DATE,
    )
    source_stage = _load(run_dir / "SOURCE_STAGE.json")
    if (
        source_stage.get("schema_version") != control.staged.SCHEMA_VERSION
        or source_stage.get("interface_version")
        != control.staged.INTERFACE_VERSION
        or source_stage.get("status") != "control_authorized"
        or source_stage.get("run_id") != CONTROL_RUN_ID
        or source_stage.get("source_calendar_date") != "2026-08-06"
        or source_stage.get("source_artifacts") != source_hashes
        or authorization.get("source_artifacts") != source_hashes
        or authorization.get("source_science_was_not_an_authorization_input")
        is not True
    ):
        raise RuntimeError("frozen source-to-control handoff changed")
    return {
        "verified": True,
        "source_artifacts": source_hashes,
        "control_authorization_sha256": reliability.sha256_file(
            run_dir / "CONTROL_AUTHORIZATION.json"
        ),
        "predecessors": {
            "verified": True,
            "source_result_sha256": (
                control.staged.base.SOURCE_PREDECESSOR_RESULT_SHA256
            ),
            "control_result_sha256": (
                control.staged.base.CONTROL_PREDECESSOR_RESULT_SHA256
            ),
        },
    }


def _verify_frozen_case_inputs() -> dict[str, Any]:
    reliability_cases = reliability.build_cases()
    stress_cases = stress.build_stress_cases()
    if reliability.sha256_file(stress.PREREGISTRATION) != (
        stress.PREREGISTRATION_SHA256
    ):
        raise RuntimeError("stress preregistration hash changed")
    if (
        len(reliability_cases) != reliability.EXPECTED_INITIAL_REQUESTS
        or len(stress_cases) != stress.EXPECTED_INITIAL_REQUESTS
        or stress.case_manifest_sha256(stress_cases)
        != stress.CASE_MANIFEST_SHA256
    ):
        raise RuntimeError("frozen budget-model case inventory changed")
    return {
        "reliability_case_count_per_model": len(reliability_cases),
        "stress_case_count": len(stress_cases),
        "stress_case_manifest_sha256": stress.CASE_MANIFEST_SHA256,
        "stress_preregistration_sha256": stress.PREREGISTRATION_SHA256,
        "source_trees_sha256": reliability.SOURCE_TREES_SHA256,
        "reliability_protocols": {
            model: reliability._protocol(model)
            for model in RELIABILITY_RUNS
        },
    }


def _require_empty_or_absent(path: Path, *, label: str) -> None:
    if not path.exists():
        return
    if path.is_file() or any(path.iterdir()):
        raise RuntimeError(f"partial or terminal {label} artifacts already exist")


def _verify_pristine_execution_paths(
    *,
    control_run_dir: Path,
    reliability_root: Path,
    stress_output_dir: Path,
    output_dir: Path,
) -> dict[str, Any]:
    forbidden_control_paths = (
        control_run_dir / "control",
        control_run_dir / "CONTROL_STAGE.json",
        control_run_dir / "RESULT.json",
        control_run_dir / "CONTROL_VERIFICATION.json",
        control_run_dir / "CONTROL_DAILY_EXECUTION.json",
        control_run_dir / "CONTROL_DAILY_EXECUTION_FAILURE.json",
    )
    existing = [str(path) for path in forbidden_control_paths if path.exists()]
    if existing:
        raise RuntimeError(
            "partial or terminal paid control artifacts already exist: "
            + ", ".join(existing)
        )
    for model, run_id in RELIABILITY_RUNS.items():
        _require_empty_or_absent(
            reliability_root / run_id,
            label=f"reliability {model}",
        )
    _require_empty_or_absent(stress_output_dir, label="stress")
    _require_empty_or_absent(output_dir, label="August 7 wrapper")
    return {
        "paid_control": "not_started",
        "reliability": {
            model: "not_started" for model in RELIABILITY_RUNS
        },
        "stress": "not_started",
        "wrapper": "not_started",
    }


def preflight_aug7_sequence(
    *,
    output_dir: Path = OUTPUT_DIR,
    control_run_dir: Path = CONTROL_RUN_DIR,
    daily_ledger: Path = DAILY_LEDGER,
    reliability_root: Path = RELIABILITY_ROOT,
    stress_output_dir: Path = STRESS_OUTPUT_DIR,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    source_validator: Callable[[Path], dict[str, Any]] = (
        _verify_source_readiness
    ),
    case_validator: Callable[[], dict[str, Any]] = _verify_frozen_case_inputs,
    catalog_reader: Callable[[], dict[str, Any]] = read_openrouter_model_catalog,
) -> dict[str, Any]:
    """Validate tomorrow's frozen sequence without writing or model calls."""
    ledger = _load(daily_ledger)
    _validate_pristine_ledger(ledger)
    paths = _verify_pristine_execution_paths(
        control_run_dir=control_run_dir,
        reliability_root=reliability_root,
        stress_output_dir=stress_output_dir,
        output_dir=output_dir,
    )
    source = source_validator(control_run_dir)
    cases = case_validator()
    models = _validate_model_catalog(catalog_reader())
    live = live_reader()
    live_usage = float(live["total_usage_usd"])
    live_balance = float(live["balance_usd"])
    opening_usage = float(ledger["opening_total_usage_usd"])
    if not all(
        math.isfinite(float(live[field]))
        for field in (
            "total_credits_usd",
            "total_usage_usd",
            "balance_usd",
        )
    ):
        raise RuntimeError("live OpenRouter credit values are non-finite")
    if live_usage > opening_usage + 1e-9:
        raise RuntimeError(
            "OpenRouter usage advanced beyond the frozen August 7 opening "
            "baseline"
        )
    if live_balance + 1e-12 < MINIMUM_STARTING_BALANCE_USD:
        raise RuntimeError("live OpenRouter balance is below the $5 start gate")

    reliability_cap = len(RELIABILITY_RUNS) * reliability.RUN_BUDGET_USD
    full_tail_cap = reliability_cap + stress.RUN_BUDGET_USD
    expected_full_cost = EXPECTED_CONTROL_COST_USD + full_tail_cap
    budget = {
        "daily_cap_usd": 5.0,
        "control_maximum_cost_usd": control.staged.CONTROL_DAILY_CAP_USD,
        "control_expected_cost_usd": EXPECTED_CONTROL_COST_USD,
        "reliability_maximum_cost_usd": reliability_cap,
        "stress_maximum_cost_usd": stress.RUN_BUDGET_USD,
        "expected_full_sequence_cost_usd": expected_full_cost,
        "expected_full_sequence_slack_usd": 5.0 - expected_full_cost,
        "maximum_control_cost_for_reliability_usd": 5.0 - reliability_cap,
        "maximum_control_cost_for_full_stress_usd": 5.0 - full_tail_cap,
        "full_stress_is_conditional_on_measured_control_cost": True,
        "expected_full_sequence_fits": expected_full_cost <= 5.0,
        "no_reserve_and_no_rollover": True,
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": EXPECTED_DATE,
        "daily_ledger": str(daily_ledger),
        "daily_ledger_sha256": reliability.sha256_file(daily_ledger),
        "live_credits": live,
        "usage_headroom_to_frozen_opening_usd": opening_usage - live_usage,
        "source": source,
        "cases": cases,
        "models": models,
        "execution_paths": paths,
        "budget": budget,
        "model_calls_made": 0,
        "files_written": 0,
    }


def _verify_control_component(
    *, run_dir: Path, daily_ledger: Path
) -> dict[str, Any]:
    execution_path = run_dir / "CONTROL_DAILY_EXECUTION.json"
    execution = _load(execution_path)
    replay = control.verify_completed_control(
        run_dir=run_dir,
        write_outputs=False,
    )
    stored_verification_path = run_dir / "CONTROL_VERIFICATION.json"
    stored_verification = _load(stored_verification_path)
    if (
        execution.get("schema_version") != control.SCHEMA_VERSION
        or execution.get("interface_version") != control.INTERFACE_VERSION
        or execution.get("status") != "complete_verified"
        or execution.get("run_id") != CONTROL_RUN_ID
        or execution.get("daily_ledger") != str(daily_ledger)
        or execution.get("measured_control_requests")
        != reliability.EXPECTED_QWEN_CONTROL_REQUESTS
        or execution.get("verification_status") != "verified"
        or execution.get("verification_sha256")
        != reliability.sha256_file(stored_verification_path)
        or execution.get("composite_result_sha256")
        != reliability.sha256_file(run_dir / "RESULT.json")
        or stored_verification != replay
    ):
        raise RuntimeError("banked control execution does not independently replay")
    return {
        "verified": True,
        "status": execution["status"],
        "decision": execution.get("decision"),
        "artifact_sha256": reliability.sha256_file(execution_path),
        "verification_sha256": reliability.sha256_file(
            stored_verification_path
        ),
        "composite_result_sha256": reliability.sha256_file(
            run_dir / "RESULT.json"
        ),
        "request_count": int(execution["measured_control_requests"]),
        "cost_usd": float(execution["measured_control_cost_usd"]),
    }


def _record_incomplete_control(path: Path) -> dict[str, Any]:
    payload = _load(path)
    if (
        payload.get("interface_version") != control.INTERFACE_VERSION
        or payload.get("status") != "control_incomplete_or_mechanics_failed"
    ):
        raise RuntimeError("incomplete control artifact shape changed")
    return {
        "verified": False,
        "status": payload["status"],
        "decision": payload.get("decision"),
        "artifact_sha256": reliability.sha256_file(path),
    }


def _artifact_path(output_dir: Path) -> Path | None:
    existing = [
        path
        for path in (output_dir / "RESULT.json", output_dir / "FAILURE.json")
        if path.exists()
    ]
    if len(existing) > 1:
        raise RuntimeError(f"ambiguous result artifacts in {output_dir}")
    return existing[0] if existing else None


def _validate_reliability_artifact(path: Path, model: str) -> dict[str, Any]:
    payload = _load(path)
    artifact_model = payload.get("model") or (
        payload.get("protocol") or {}
    ).get("model")
    artifact_interface = payload.get("interface_version") or (
        payload.get("protocol") or {}
    ).get("interface_version")
    if artifact_model != model:
        raise RuntimeError(f"reliability artifact model mismatch for {model}")
    if artifact_interface != reliability.INTERFACE_VERSION:
        raise RuntimeError(f"reliability artifact interface mismatch for {model}")
    if payload.get("status") not in {"passed", "gated_null", "failed_closed"}:
        raise RuntimeError(f"invalid reliability status for {model}")
    if (
        payload.get("status") == "failed_closed"
        and payload.get("ledger_reconciliation_error")
    ):
        raise RuntimeError(f"unreconciled failed reliability gate for {model}")
    return payload


def _verify_reliability_component(path: Path, model: str) -> dict[str, Any]:
    payload = _validate_reliability_artifact(path, model)
    if payload["status"] in {"passed", "gated_null"}:
        replay = reliability.replay_reliability_result(
            result_path=path,
            model_id=model,
        )
        if replay["status"] != payload["status"]:
            raise RuntimeError(f"reliability replay status changed for {model}")
        return {
            **replay,
            "decision": payload.get("decision"),
            "artifact_sha256": replay["result_sha256"],
        }
    return {
        "verified": True,
        "model": model,
        "status": "failed_closed",
        "decision": payload.get("decision"),
        "artifact_sha256": reliability.sha256_file(path),
        "raw_responses_sha256": None,
        "request_count": None,
        "cost_usd": None,
    }


def _validate_reliability_ledger(
    ledger: dict[str, Any],
    *,
    model: str,
    artifact_status: str,
) -> None:
    matches = [
        item
        for item in (ledger.get("authorized_tail_blocks") or [])
        if item.get("interface") == reliability.INTERFACE_VERSION
        and item.get("model") == model
    ]
    if len(matches) != 1:
        raise RuntimeError(f"missing exact ledger entry for {model}")
    allowed = (
        {"failed_closed", "failed_closed_posted_spend_reconciled"}
        if artifact_status == "failed_closed"
        else {artifact_status}
    )
    if matches[0].get("status") not in allowed:
        raise RuntimeError(f"ledger and artifact status differ for {model}")
    cost = float(matches[0].get("actual_cost_usd", 0.0))
    if (
        float(matches[0].get("maximum_cost_usd", 0.0))
        != reliability.RUN_BUDGET_USD
        or not math.isfinite(cost)
        or not 0.0 <= cost <= reliability.RUN_BUDGET_USD
    ):
        raise RuntimeError(f"reliability ledger cost changed for {model}")


def _stress_authorized(ledger: dict[str, Any]) -> bool:
    return any(
        item.get("interface") == stress.INTERFACE_VERSION
        and item.get("status") == "waiting_for_reliability_results"
        for item in (ledger.get("authorized_tail_blocks") or [])
    )


def _verify_stress_component(
    path: Path,
    reliability_paths: dict[str, Path],
) -> dict[str, Any]:
    payload = _load(path)
    if payload.get("status") in {"passed", "gated_null"}:
        replay = stress.replay_stress_result(
            result_path=path,
            reliability_paths=reliability_paths,
        )
        return {
            **replay,
            "decision": payload.get("decision"),
            "artifact_sha256": replay["result_sha256"],
        }
    if (
        payload.get("interface_version") != stress.INTERFACE_VERSION
        or payload.get("status") != "failed_closed"
    ):
        raise RuntimeError("invalid stress failure artifact")
    return {
        "verified": True,
        "status": "failed_closed",
        "decision": payload.get("decision"),
        "selected_model": None,
        "artifact_sha256": reliability.sha256_file(path),
        "raw_responses_sha256": None,
        "request_count": None,
        "cost_usd": None,
    }


def _validate_stress_ledger(
    ledger: dict[str, Any], *, verification: dict[str, Any]
) -> None:
    matches = [
        item
        for item in (ledger.get("authorized_tail_blocks") or [])
        if item.get("interface") == stress.INTERFACE_VERSION
    ]
    if len(matches) != 1:
        raise RuntimeError("missing exact stress ledger entry")
    entry = matches[0]
    allowed = (
        {"failed_closed", "failed_closed_posted_spend_reconciled"}
        if verification["status"] == "failed_closed"
        else {verification["status"], "not_authorized_no_model_passed"}
    )
    cost = float(entry.get("actual_cost_usd", 0.0))
    if (
        entry.get("status") not in allowed
        or float(entry.get("maximum_cost_usd", 0.0)) != stress.RUN_BUDGET_USD
        or not math.isfinite(cost)
        or not 0.0 <= cost <= stress.RUN_BUDGET_USD
        or (
            verification.get("selected_model") is not None
            and entry.get("model") != verification["selected_model"]
        )
    ):
        raise RuntimeError("stress ledger and artifact differ")


def _component_record(
    path: Path,
    payload: dict[str, Any],
    verification: dict[str, Any],
) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "artifact_sha256": reliability.sha256_file(path),
        "status": payload.get("status"),
        "decision": payload.get("decision"),
        "verification": verification,
    }


def validate_completed_sequence(
    *,
    final_path: Path,
    control_run_dir: Path,
    daily_ledger: Path,
    reliability_root: Path,
    stress_output_dir: Path,
    control_validator: Callable[..., dict[str, Any]],
    claim_reporter: Callable[..., dict[str, Any]],
    reliability_validator: Callable[..., dict[str, Any]],
    stress_validator: Callable[..., dict[str, Any]],
) -> dict[str, Any]:
    state = _load(final_path)
    ledger = _load(daily_ledger)
    _validate_ledger_boundary(ledger)
    if (
        state.get("schema_version") != SCHEMA_VERSION
        or state.get("interface_version") != INTERFACE_VERSION
        or state.get("date") != EXPECTED_DATE
        or state.get("daily_ledger") != str(daily_ledger)
        or state.get("status")
        not in {"complete", "failed_closed", "stopped_after_control"}
        or not isinstance(state.get("components"), dict)
        or ledger.get("additional_paid_blocks_authorized") is not False
    ):
        raise RuntimeError("completed August 7 wrapper shape changed")
    components = state["components"]
    control_path = control_run_dir / "CONTROL_DAILY_EXECUTION.json"
    control_execution = _load(control_path)
    if state["status"] == "stopped_after_control":
        verification = _record_incomplete_control(control_path)
        if (
            components
            != {
                "control": _component_record(
                    control_path,
                    control_execution,
                    verification,
                )
            }
        ):
            raise RuntimeError("stopped control wrapper changed")
        return state
    control_verification = control_validator(
        run_dir=control_run_dir,
        daily_ledger=daily_ledger,
    )
    expected_control = _component_record(
        control_path,
        control_execution,
        control_verification,
    )
    if components.get("control") != expected_control:
        raise RuntimeError("completed wrapper control component changed")
    claim_reporter(run_dir=control_run_dir)

    reliability_paths: dict[str, Path] = {}
    expected_component_names = {"control"}
    for model, run_id in RELIABILITY_RUNS.items():
        if model not in components:
            continue
        path = _artifact_path(reliability_root / run_id)
        if path is None:
            raise RuntimeError(f"completed wrapper artifact missing for {model}")
        if components[model].get("artifact") != str(path):
            raise RuntimeError(f"completed wrapper path changed for {model}")
        payload = _validate_reliability_artifact(path, model)
        verification = reliability_validator(path, model)
        _validate_reliability_ledger(
            ledger,
            model=model,
            artifact_status=str(payload["status"]),
        )
        if components[model] != _component_record(path, payload, verification):
            raise RuntimeError(f"completed wrapper component changed for {model}")
        reliability_paths[model] = path
        expected_component_names.add(model)

    stress_record = components.get("stress3584")
    if stress_record is not None:
        if set(reliability_paths) != set(RELIABILITY_RUNS):
            raise RuntimeError("stress result lacks both reliability predecessors")
        stress_path = _artifact_path(stress_output_dir)
        if stress_path is None or stress_record.get("artifact") != str(stress_path):
            raise RuntimeError("completed wrapper stress path changed")
        payload = _load(stress_path)
        verification = stress_validator(stress_path, reliability_paths)
        _validate_stress_ledger(ledger, verification=verification)
        if stress_record != _component_record(stress_path, payload, verification):
            raise RuntimeError("completed wrapper stress component changed")
        expected_component_names.add("stress3584")
        expected_status = (
            "failed_closed"
            if payload.get("status") == "failed_closed"
            else "complete"
        )
        expected_selected = (payload.get("selection") or {}).get(
            "selected_model"
        )
        if (
            state["status"] != expected_status
            or state.get("decision") != payload.get("decision")
            or state.get("selected_model") != expected_selected
        ):
            raise RuntimeError("completed wrapper stress summary changed")
    else:
        if state["status"] != "complete" or state.get("decision") not in {
            "reliability_not_authorized_by_control_headroom",
            "stress_not_authorized_by_control_headroom",
        }:
            raise RuntimeError("completed wrapper stopping decision changed")
        expected_reliability_count = (
            0
            if state["decision"]
            == "reliability_not_authorized_by_control_headroom"
            else len(RELIABILITY_RUNS)
        )
        if len(reliability_paths) != expected_reliability_count:
            raise RuntimeError("completed wrapper stopping point changed")
    if set(components) != expected_component_names:
        raise RuntimeError("completed wrapper component set changed")
    if float(state.get("recorded_actual_spend_usd", math.inf)) != float(
        ledger["recorded_actual_spend_usd"]
    ):
        raise RuntimeError("completed wrapper spend summary changed")
    return state


def execute_aug7_sequence(
    *,
    output_dir: Path = OUTPUT_DIR,
    control_run_dir: Path = CONTROL_RUN_DIR,
    daily_ledger: Path = DAILY_LEDGER,
    reliability_root: Path = RELIABILITY_ROOT,
    stress_output_dir: Path = STRESS_OUTPUT_DIR,
    now: datetime | None = None,
    control_runner: Callable[..., dict[str, Any]] = control.run_control_daily,
    reliability_runner: Callable[..., dict[str, Any]] = (
        reliability.execute_reliability
    ),
    stress_runner: Callable[..., dict[str, Any]] = stress.execute_stress,
    control_validator: Callable[..., dict[str, Any]] = (
        _verify_control_component
    ),
    claim_reporter: Callable[..., dict[str, Any]] = claim.write_claim_report,
    reliability_validator: Callable[..., dict[str, Any]] = (
        _verify_reliability_component
    ),
    stress_validator: Callable[..., dict[str, Any]] = _verify_stress_component,
    fresh_preflight: Callable[..., dict[str, Any]] = preflight_aug7_sequence,
) -> dict[str, Any]:
    """Run or resume components without repeating any banked model call."""
    final_path = output_dir / "RESULT.json"
    ledger = _load(daily_ledger)
    _validate_ledger_boundary(ledger)
    if final_path.exists():
        return validate_completed_sequence(
            final_path=final_path,
            control_run_dir=control_run_dir,
            daily_ledger=daily_ledger,
            reliability_root=reliability_root,
            stress_output_dir=stress_output_dir,
            reliability_validator=reliability_validator,
            control_validator=control_validator,
            claim_reporter=claim_reporter,
            stress_validator=stress_validator,
        )
    try:
        _validate_date(ledger, now)
    except Exception as exc:
        raise PreExecutionGateError(str(exc)) from exc
    control_execution_path = control_run_dir / "CONTROL_DAILY_EXECUTION.json"
    control_failure_path = control_run_dir / "CONTROL_DAILY_EXECUTION_FAILURE.json"
    if not control_execution_path.exists() and not control_failure_path.exists():
        try:
            preflight = fresh_preflight(
                output_dir=output_dir,
                control_run_dir=control_run_dir,
                daily_ledger=daily_ledger,
                reliability_root=reliability_root,
                stress_output_dir=stress_output_dir,
            )
        except Exception as exc:
            raise PreExecutionGateError(str(exc)) from exc
        if preflight.get("status") != "ready_without_paid_calls":
            raise PreExecutionGateError(
                "fresh August 7 preflight did not authorize execution"
            )
    output_dir.mkdir(parents=True, exist_ok=True)
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "in_progress",
        "date": EXPECTED_DATE,
        "daily_ledger": str(daily_ledger),
        "components": {},
    }

    if control_execution_path.exists():
        control_execution = _load(control_execution_path)
        if control_execution.get("status") != "complete_verified":
            raise RuntimeError("banked control execution is not verified")
    else:
        if control_failure_path.exists():
            raise RuntimeError("banked control execution already failed closed")
        if (control_run_dir / "control").exists() or (
            control_run_dir / "CONTROL_VERIFICATION.json"
        ).exists():
            raise RuntimeError(
                "partial control artifacts exist without a completed wrapper"
            )
        try:
            control_execution = control_runner(
                run_dir=control_run_dir,
                run_id=CONTROL_RUN_ID,
                ledger_path=daily_ledger,
            )
        except Exception as exc:
            checkpoint(
                control_run_dir / "CONTROL_DAILY_EXECUTION_FAILURE.json",
                {
                    "schema_version": SCHEMA_VERSION,
                    "interface_version": control.INTERFACE_VERSION,
                    "status": "failed_closed",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "daily_ledger": str(daily_ledger),
                    "daily_ledger_sha256": reliability.sha256_file(
                        daily_ledger
                    ),
                },
            )
            raise
    if control_execution.get("status") != "complete_verified":
        state["status"] = "stopped_after_control"
        state["components"]["control"] = _component_record(
            control_execution_path,
            control_execution,
            _record_incomplete_control(control_execution_path),
        )
        state["recorded_actual_spend_usd"] = _load(daily_ledger).get(
            "recorded_actual_spend_usd"
        )
        checkpoint(final_path, state)
        return state
    control_verification = control_validator(
        run_dir=control_run_dir,
        daily_ledger=daily_ledger,
    )
    state["components"]["control"] = _component_record(
        control_execution_path,
        control_execution,
        control_verification,
    )
    claim_reporter(run_dir=control_run_dir)
    checkpoint(output_dir / "EXECUTION_STATE.json", state)

    ledger = _load(daily_ledger)
    _validate_ledger_boundary(ledger)
    banked_tail_exists = any(
        _artifact_path(reliability_root / run_id) is not None
        for run_id in RELIABILITY_RUNS.values()
    ) or _artifact_path(stress_output_dir) is not None
    if (
        ledger.get("additional_paid_blocks_authorized") is not True
        and not banked_tail_exists
    ):
        state["status"] = "complete"
        state["decision"] = "reliability_not_authorized_by_control_headroom"
        state["recorded_actual_spend_usd"] = ledger.get(
            "recorded_actual_spend_usd"
        )
        checkpoint(final_path, state)
        return state

    reliability_paths: dict[str, Path] = {}
    for model, run_id in RELIABILITY_RUNS.items():
        model_output = reliability_root / run_id
        artifact = _artifact_path(model_output)
        if artifact is None:
            try:
                reliability_runner(
                    output_dir=model_output,
                    run_id=run_id,
                    model_id=model,
                    ledger_path=daily_ledger,
                    qwen_control_result=control_run_dir / "RESULT.json",
                    now=now,
                )
            except Exception:
                artifact = _artifact_path(model_output)
                if artifact is None:
                    raise
            else:
                artifact = _artifact_path(model_output)
                if artifact is None:
                    raise RuntimeError(f"reliability gate omitted artifact: {model}")
        payload = _validate_reliability_artifact(artifact, model)
        verification = reliability_validator(artifact, model)
        _validate_reliability_ledger(
            _load(daily_ledger),
            model=model,
            artifact_status=str(payload["status"]),
        )
        reliability_paths[model] = artifact
        state["components"][model] = _component_record(
            artifact,
            payload,
            verification,
        )
        checkpoint(output_dir / "EXECUTION_STATE.json", state)

    ledger = _load(daily_ledger)
    _validate_ledger_boundary(ledger)
    stress_artifact = _artifact_path(stress_output_dir)
    if stress_artifact is None and not _stress_authorized(ledger):
        state["status"] = "complete"
        state["decision"] = "stress_not_authorized_by_control_headroom"
        state["recorded_actual_spend_usd"] = ledger.get(
            "recorded_actual_spend_usd"
        )
        checkpoint(final_path, state)
        return state

    if stress_artifact is None:
        try:
            stress_runner(
                output_dir=stress_output_dir,
                run_id=STRESS_RUN_ID,
                ledger_path=daily_ledger,
                reliability_paths=reliability_paths,
                now=now,
            )
        except Exception as exc:
            stress_artifact = _artifact_path(stress_output_dir)
            if stress_artifact is None:
                stress_output_dir.mkdir(parents=True, exist_ok=True)
                stress_artifact = stress_output_dir / "FAILURE.json"
                checkpoint(
                    stress_artifact,
                    {
                        "schema_version": SCHEMA_VERSION,
                        "interface_version": stress.INTERFACE_VERSION,
                        "status": "failed_closed",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
        else:
            stress_artifact = _artifact_path(stress_output_dir)
            if stress_artifact is None:
                raise RuntimeError("stress gate omitted its result artifact")
    stress_payload = _load(stress_artifact)
    if stress_payload.get("status") not in {
        "passed",
        "gated_null",
        "failed_closed",
    }:
        raise RuntimeError("invalid stress artifact status")
    stress_verification = stress_validator(stress_artifact, reliability_paths)
    ledger = _load(daily_ledger)
    _validate_ledger_boundary(ledger)
    _validate_stress_ledger(ledger, verification=stress_verification)
    state["components"]["stress3584"] = _component_record(
        stress_artifact,
        stress_payload,
        stress_verification,
    )
    state["status"] = (
        "complete" if stress_payload.get("status") != "failed_closed"
        else "failed_closed"
    )
    state["decision"] = stress_payload.get("decision")
    state["selected_model"] = (
        stress_payload.get("selection") or {}
    ).get("selected_model")
    state["recorded_actual_spend_usd"] = ledger.get(
        "recorded_actual_spend_usd"
    )
    checkpoint(final_path, state)
    return state


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="validate the frozen sequence read-only before August 7",
    )
    args = parser.parse_args()
    if args.preflight:
        result = preflight_aug7_sequence()
        print(json.dumps(result, indent=2, sort_keys=True))
        return 0
    try:
        result = execute_aug7_sequence()
    except PreExecutionGateError:
        raise
    except Exception as exc:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        checkpoint(
            OUTPUT_DIR / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result.get("status") == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
