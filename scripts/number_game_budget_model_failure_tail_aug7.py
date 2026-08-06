#!/usr/bin/env python3
"""Use the unopened model-screen tail after the Aug 7 Qwen mechanics stop."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_budget_model_aug7_execute as aug7
from scripts import number_game_budget_model_reliability128 as reliability
from scripts import number_game_budget_model_stress3584 as stress
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-budget-model-failure-tail-aug7-1"
EXPECTED_DATE = "2026-08-07"
TIMEZONE = "Europe/London"
AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "NUMBER_GAME_AUG7_CONTROL_FAILURE_MODEL_SCREEN_AMENDMENT.md"
)
AMENDMENT_SHA256 = (
    "9c1ed384932cfe8bf3a907abbfe025b1fce4d99e6c7199b50dcc18823c2960fe"
)
CONTROL_WRAPPER = aug7.CONTROL_RUN_DIR / "CONTROL_DAILY_EXECUTION.json"
CONTROL_WRAPPER_SHA256 = (
    "21dd28010309b769bc094824d18c27f1c758a7768b6ffd31de9d6201aeb8f968"
)
CONTROL_FAILURE = aug7.CONTROL_RUN_DIR / "control/FAILURE.json"
CONTROL_FAILURE_SHA256 = (
    "bf2a119f23e8241a7c2f694a3d3009104b0f3151073d37407cb6ddd7a917c244"
)
CONTROLS = aug7.CONTROL_RUN_DIR / "control/CONTROLS.json"
CONTROLS_SHA256 = (
    "513def73b8e75f31eca123e593c2e9b0deab715fbd52ed6d3833020564237672"
)
INITIAL_LEDGER_SHA256 = (
    "af3c6f3ed1b77281e82b22e1600983c363d90fd66cc4bfdfdd14fde024ec1045"
)
OUTPUT_DIR = REPO_ROOT / (
    "results/nonmyopic/number_game_budget_model_failure_tail_aug7/"
    "number-game-budget-model-failure-tail-20260807"
)
MINIMUM_REMAINING_ALLOWANCE_USD = (
    2 * reliability.RUN_BUDGET_USD + stress.RUN_BUDGET_USD
)


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return reliability.sha256_file(path)


def _local_now(now: datetime | None = None) -> datetime:
    return (now or datetime.now(tz=ZoneInfo(TIMEZONE))).astimezone(
        ZoneInfo(TIMEZONE)
    )


def _reliability_paths(root: Path = aug7.RELIABILITY_ROOT) -> dict[str, Path]:
    return {
        model: root / run_id / "RESULT.json"
        for model, run_id in aug7.RELIABILITY_RUNS.items()
    }


def validate_control_failure_boundary() -> dict[str, Any]:
    if _sha256(AMENDMENT) != AMENDMENT_SHA256:
        raise ValueError("failure-tail authorization amendment changed")
    expected_hashes = {
        CONTROL_WRAPPER: CONTROL_WRAPPER_SHA256,
        CONTROL_FAILURE: CONTROL_FAILURE_SHA256,
        CONTROLS: CONTROLS_SHA256,
    }
    for path, expected in expected_hashes.items():
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"frozen control failure artifact changed: {path}")
    wrapper = _load(CONTROL_WRAPPER)
    failure = _load(CONTROL_FAILURE)
    usage = failure.get("usage") or {}
    protocol = failure.get("protocol") or {}
    if (
        wrapper.get("status") != "control_incomplete_or_mechanics_failed"
        or wrapper.get("decision") != "stop_at_control_mechanics"
        or failure.get("status") != "mechanics_failed"
        or usage.get("adapter_requests") != 3_072
        or usage.get("http_attempts") != 3_072
        or usage.get("retry_count") != 0
        or usage.get("provider_error_retries") != 0
        or usage.get("adapter_reasoning_tokens") != 0
        or usage.get("forced_exits") != 0
        or float(usage.get("run_cost_usd", -1.0)) != 2.6901734400000024
        or protocol.get("endpoint_accessed") is not False
        or protocol.get("controls_sha256") != CONTROLS_SHA256
    ):
        raise ValueError("Qwen failure is not the exact clean mechanics stop")
    return {
        "verified": True,
        "control_wrapper_sha256": CONTROL_WRAPPER_SHA256,
        "control_failure_sha256": CONTROL_FAILURE_SHA256,
        "controls_sha256": CONTROLS_SHA256,
        "accepted_requests": 3_072,
        "cost_usd": float(usage["run_cost_usd"]),
        "endpoint_accessed": False,
        "failure_values_used_for_authorization": False,
    }


def _authorized_entries() -> list[dict[str, Any]]:
    entries = [
        {
            "interface": reliability.INTERFACE_VERSION,
            "model": model,
            "status": "authorized_pending",
            "maximum_cost_usd": reliability.RUN_BUDGET_USD,
            "authorization_stage": "post_control_failure_model_screen",
            "authorization_amendment_sha256": AMENDMENT_SHA256,
            "efficacy_used_for_authorization": False,
        }
        for model in aug7.RELIABILITY_RUNS
    ]
    entries.append(
        {
            "interface": stress.INTERFACE_VERSION,
            "model": None,
            "status": "waiting_for_reliability_results",
            "selection_rule": stress.SELECTION_RULE,
            "maximum_cost_usd": stress.RUN_BUDGET_USD,
            "authorization_stage": "post_reliability_reconciled",
            "authorization_amendment_sha256": AMENDMENT_SHA256,
            "efficacy_used_for_authorization": False,
        }
    )
    return entries


def authorize_ledger(
    *,
    ledger_path: Path = aug7.DAILY_LEDGER,
    reliability_root: Path = aug7.RELIABILITY_ROOT,
    stress_output_dir: Path = aug7.STRESS_OUTPUT_DIR,
) -> dict[str, Any]:
    ledger = _load(ledger_path)
    existing = ledger.get("authorized_tail_blocks") or []
    if existing:
        if existing != _authorized_entries() and any(
            item.get("authorization_amendment_sha256") != AMENDMENT_SHA256
            for item in existing
        ):
            raise RuntimeError("daily ledger contains another tail authorization")
        return ledger
    if _sha256(ledger_path) != INITIAL_LEDGER_SHA256:
        raise ValueError("pre-amendment Aug 7 ledger changed")
    if (
        ledger.get("date") != EXPECTED_DATE
        or float(ledger.get("daily_cap_usd", 0.0)) != 5.0
        or ledger.get("additional_paid_blocks_authorized") is not False
        or float(ledger.get("recorded_actual_spend_usd", -1.0))
        != 2.6901734400000024
        or float(
            (ledger.get("reconciliation") or {}).get(
                "remaining_daily_allowance_usd", -1.0
            )
        )
        + 1e-12
        < MINIMUM_REMAINING_ALLOWANCE_USD
    ):
        raise ValueError("Aug 7 ledger cannot fund the exact failure tail")
    paths = [
        reliability_root / run_id
        for run_id in aug7.RELIABILITY_RUNS.values()
    ] + [stress_output_dir]
    if any(path.exists() and any(path.iterdir()) for path in paths):
        raise FileExistsError("one failure-tail paid path is not pristine")
    updated = json.loads(json.dumps(ledger))
    updated["additional_paid_blocks_authorized"] = True
    updated["authorized_tail_blocks"] = _authorized_entries()
    updated["tail_authorization_reason"] = (
        "independent_model_screen_after_clean_terminal_control_mechanics_stop"
    )
    updated["failure_tail_authorization_amendment_sha256"] = AMENDMENT_SHA256
    updated["control_status_is_immutable"] = True
    checkpoint(ledger_path, updated)
    return updated


def preflight_failure_tail(
    *,
    ledger_path: Path = aug7.DAILY_LEDGER,
    reliability_root: Path = aug7.RELIABILITY_ROOT,
    stress_output_dir: Path = aug7.STRESS_OUTPUT_DIR,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    model_catalog_reader: Callable[[], dict[str, Any]] = (
        aug7.read_openrouter_model_catalog
    ),
    now: datetime | None = None,
) -> dict[str, Any]:
    local_now = _local_now(now)
    if local_now.date().isoformat() != EXPECTED_DATE:
        raise RuntimeError("failure-tail preflight is restricted to Aug 7")
    boundary = validate_control_failure_boundary()
    ledger = _load(ledger_path)
    if ledger.get("authorized_tail_blocks"):
        raise RuntimeError("failure-tail preflight requires an unopened ledger")
    if _sha256(ledger_path) != INITIAL_LEDGER_SHA256:
        raise ValueError("pre-amendment Aug 7 ledger changed")
    remaining = float(
        (ledger.get("reconciliation") or {}).get(
            "remaining_daily_allowance_usd", -1.0
        )
    )
    if remaining + 1e-12 < MINIMUM_REMAINING_ALLOWANCE_USD:
        raise RuntimeError("remaining daily allowance cannot fund failure tail")
    paths = {
        model: reliability_root / run_id
        for model, run_id in aug7.RELIABILITY_RUNS.items()
    }
    paths["stress"] = stress_output_dir
    if any(path.exists() and any(path.iterdir()) for path in paths.values()):
        raise FileExistsError("one failure-tail paid path is not pristine")
    live = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=MINIMUM_REMAINING_ALLOWANCE_USD,
        total_usage_usd=float(live["total_usage_usd"]),
        now=local_now,
    )
    if float(live["balance_usd"]) + 1e-12 < MINIMUM_REMAINING_ALLOWANCE_USD:
        raise RuntimeError("OpenRouter balance is below the full failure tail")
    models = aug7._validate_model_catalog(model_catalog_reader())
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "ready_without_paid_calls",
        "date": EXPECTED_DATE,
        "authorization_amendment_sha256": AMENDMENT_SHA256,
        "control_failure_boundary": boundary,
        "daily_ledger_sha256": INITIAL_LEDGER_SHA256,
        "recorded_actual_spend_usd": float(ledger["recorded_actual_spend_usd"]),
        "remaining_daily_allowance_usd": remaining,
        "maximum_failure_tail_cost_usd": MINIMUM_REMAINING_ALLOWANCE_USD,
        "post_tail_minimum_slack_usd": (
            remaining - MINIMUM_REMAINING_ALLOWANCE_USD
        ),
        "live_credits": live,
        "models": {
            model: models[model]
            for model in (
                "openai/gpt-5.6-luna",
                "deepseek/deepseek-v4-flash-0731",
            )
        },
        "execution_paths": {
            key: "not_started" for key in paths
        },
        "model_calls_made": 0,
        "files_written": 0,
        "qwen_control_status_changed": False,
        "authorizes_aug8_diversity": False,
    }


def _validate_model_authorization(ledger: dict[str, Any], model: str) -> None:
    matches = [
        item
        for item in ledger.get("authorized_tail_blocks") or []
        if item.get("interface") == reliability.INTERFACE_VERSION
        and item.get("model") == model
    ]
    if (
        len(matches) != 1
        or matches[0].get("status") != "authorized_pending"
        or matches[0].get("authorization_amendment_sha256")
        != AMENDMENT_SHA256
        or float(matches[0].get("maximum_cost_usd", -1.0))
        != reliability.RUN_BUDGET_USD
    ):
        raise RuntimeError(f"{model} failure-tail authorization is not exact")


def execute_one_reliability(
    *,
    model: str,
    output_dir: Path,
    run_id: str,
    ledger_path: Path,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    runner: Callable[..., dict[str, Any]] = reliability.run_reliability,
    now: datetime | None = None,
) -> dict[str, Any]:
    result_path = output_dir / "RESULT.json"
    if result_path.is_file():
        return reliability.replay_reliability_result(
            result_path=result_path, model_id=model
        )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"partial reliability path cannot repeat: {output_dir}")
    ledger = _load(ledger_path)
    _validate_model_authorization(ledger, model)
    live_before = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=reliability.PROJECTED_COST_USD,
        total_usage_usd=float(live_before["total_usage_usd"]),
        now=now,
    )
    if float(live_before["balance_usd"]) + 1e-12 < reliability.RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the reliability gate")
    try:
        runner(output_dir=output_dir, run_id=run_id, model_id=model)
    except Exception as exc:
        live_after = live_reader()
        reconciled = reliability.reconcile_daily_ledger(
            ledger=ledger,
            model_id=model,
            measured_cost_usd=0.0,
            live_after=live_after,
            status="failed_closed_posted_spend_reconciled",
        )
        checkpoint(ledger_path, reconciled)
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "model": model,
                "error_type": type(exc).__name__,
                "error": str(exc),
                "authorization_amendment_sha256": AMENDMENT_SHA256,
            },
        )
        raise
    result = _load(result_path)
    local = reliability.reconcile_daily_ledger(
        ledger=ledger,
        model_id=model,
        measured_cost_usd=float(result["usage"]["run_cost_usd"]),
        live_after=live_before,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    live_after = live_reader()
    reconciled = reliability.reconcile_daily_ledger(
        ledger=local,
        model_id=model,
        measured_cost_usd=0.0,
        live_after=live_after,
        status=result["status"],
    )
    checkpoint(ledger_path, reconciled)
    return reliability.replay_reliability_result(
        result_path=result_path, model_id=model
    )


def execute_failure_tail(
    *,
    output_dir: Path = OUTPUT_DIR,
    ledger_path: Path = aug7.DAILY_LEDGER,
    reliability_root: Path = aug7.RELIABILITY_ROOT,
    stress_output_dir: Path = aug7.STRESS_OUTPUT_DIR,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    reliability_runner: Callable[..., dict[str, Any]] = reliability.run_reliability,
    stress_executor: Callable[..., dict[str, Any]] = stress.execute_stress,
    now: datetime | None = None,
) -> dict[str, Any]:
    local_now = _local_now(now)
    if local_now.date().isoformat() != EXPECTED_DATE:
        raise RuntimeError("failure-tail executor is restricted to Aug 7")
    result_path = output_dir / "RESULT.json"
    if result_path.is_file():
        return _load(result_path)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError("partial failure-tail wrapper cannot repeat")
    boundary = validate_control_failure_boundary()
    authorize_ledger(
        ledger_path=ledger_path,
        reliability_root=reliability_root,
        stress_output_dir=stress_output_dir,
    )
    reliability_verifications = {}
    failures = {}
    for model, run_id in aug7.RELIABILITY_RUNS.items():
        model_dir = reliability_root / run_id
        try:
            reliability_verifications[model] = execute_one_reliability(
                model=model,
                output_dir=model_dir,
                run_id=run_id,
                ledger_path=ledger_path,
                live_reader=live_reader,
                runner=reliability_runner,
                now=local_now,
            )
        except Exception as exc:
            failures[model] = f"{type(exc).__name__}: {exc}"
    stress_verification = None
    stress_result = None
    if len(reliability_verifications) == len(aug7.RELIABILITY_RUNS):
        stress_result = stress_executor(
            output_dir=stress_output_dir,
            run_id=aug7.STRESS_RUN_ID,
            ledger_path=ledger_path,
            reliability_paths=_reliability_paths(reliability_root),
            live_reader=live_reader,
            now=local_now,
        )
        stress_verification = stress.replay_stress_result(
            result_path=stress_output_dir / "RESULT.json"
        )
    final_ledger = _load(ledger_path)
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "complete"
            if not failures and stress_verification is not None
            else "failed_closed"
        ),
        "authorization_amendment_sha256": AMENDMENT_SHA256,
        "control_failure_boundary": boundary,
        "qwen_control_status_changed": False,
        "qwen_endpoint_accessed": False,
        "reliability_verifications": reliability_verifications,
        "reliability_failures": failures,
        "stress_status": (
            stress_result.get("status") if stress_result is not None else None
        ),
        "stress_verification": stress_verification,
        "daily_ledger_sha256": _sha256(ledger_path),
        "recorded_actual_spend_usd": float(
            final_ledger["recorded_actual_spend_usd"]
        ),
        "remaining_daily_allowance_usd": float(
            final_ledger["reconciliation"]["remaining_daily_allowance_usd"]
        ),
        "authorizes_aug8_diversity": False,
        "efficacy_used_for_model_selection": False,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint(result_path, result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight_failure_tail() if args.preflight else execute_failure_tail()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
