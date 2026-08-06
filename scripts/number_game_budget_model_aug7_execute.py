#!/usr/bin/env python3
"""Execute the frozen August 7 Number Game budget sequence exactly once."""

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
from scripts import number_game_budget_model_reliability128 as reliability
from scripts import number_game_budget_model_stress3584 as stress
from scripts import number_game_qwen_fully_fresh_control_daily_execute as control


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


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def _stress_authorized(ledger: dict[str, Any]) -> bool:
    return any(
        item.get("interface") == stress.INTERFACE_VERSION
        and item.get("status") == "waiting_for_reliability_results"
        for item in (ledger.get("authorized_tail_blocks") or [])
    )


def _component_record(path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "artifact": str(path),
        "artifact_sha256": reliability.sha256_file(path),
        "status": payload.get("status"),
        "decision": payload.get("decision"),
    }


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
) -> dict[str, Any]:
    """Run or resume components without repeating any banked model call."""
    final_path = output_dir / "RESULT.json"
    if final_path.exists():
        return _load(final_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    ledger = _load(daily_ledger)
    _validate_date(ledger, now)
    state: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "in_progress",
        "date": EXPECTED_DATE,
        "daily_ledger": str(daily_ledger),
        "components": {},
    }

    control_execution_path = control_run_dir / "CONTROL_DAILY_EXECUTION.json"
    if control_execution_path.exists():
        control_execution = _load(control_execution_path)
        if control_execution.get("status") != "complete_verified":
            raise RuntimeError("banked control execution is not verified")
    else:
        if (control_run_dir / "CONTROL_DAILY_EXECUTION_FAILURE.json").exists():
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
        state["components"]["control"] = control_execution
        checkpoint(final_path, state)
        return state
    state["components"]["control"] = _component_record(
        control_execution_path, control_execution
    )
    checkpoint(output_dir / "EXECUTION_STATE.json", state)

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
        _validate_reliability_ledger(
            _load(daily_ledger),
            model=model,
            artifact_status=str(payload["status"]),
        )
        reliability_paths[model] = artifact
        state["components"][model] = _component_record(artifact, payload)
        checkpoint(output_dir / "EXECUTION_STATE.json", state)

    ledger = _load(daily_ledger)
    if not _stress_authorized(ledger):
        state["status"] = "complete"
        state["decision"] = "stress_not_authorized_by_control_headroom"
        state["recorded_actual_spend_usd"] = ledger.get(
            "recorded_actual_spend_usd"
        )
        checkpoint(final_path, state)
        return state

    stress_artifact = _artifact_path(stress_output_dir)
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
    state["components"]["stress3584"] = _component_record(
        stress_artifact, stress_payload
    )
    ledger = _load(daily_ledger)
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
    argparse.ArgumentParser(description=__doc__).parse_args()
    try:
        result = execute_aug7_sequence()
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
