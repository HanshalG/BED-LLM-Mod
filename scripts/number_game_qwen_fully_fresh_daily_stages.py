#!/usr/bin/env python3
"""Run the fully fresh Qwen source/control study across budget days."""

from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime
import json
from pathlib import Path
import sys
from typing import Any, Callable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts import number_game_qwen_fully_fresh_source_control32 as base
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-fully-fresh-daily-stages-1"
SOURCE_DAILY_CAP_USD = 5.00
CONTROL_DAILY_CAP_USD = 4.25
COMPOSITE_DAILY_CAP_USD = SOURCE_DAILY_CAP_USD + CONTROL_DAILY_CAP_USD
SOURCE_MIN_STARTING_BALANCE_USD = 5.00
CONTROL_MIN_STARTING_BALANCE_USD = 5.00


@contextmanager
def configured_daily_limits() -> Iterator[None]:
    overrides = {
        "SOURCE_RUN_BUDGET_USD": SOURCE_DAILY_CAP_USD,
        "COMPOSITE_RUN_BUDGET_USD": COMPOSITE_DAILY_CAP_USD,
        "MIN_STARTING_BALANCE_USD": SOURCE_MIN_STARTING_BALANCE_USD,
    }
    originals = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        yield
    finally:
        for name, value in originals.items():
            setattr(base, name, value)


def _check_stage_budget(
    *,
    ledger: dict[str, Any],
    projected_cost_usd: float,
    total_usage_usd: float,
    balance_usd: float,
    minimum_balance_usd: float,
    now: datetime | None = None,
) -> dict[str, Any]:
    status = require_budget(
        ledger,
        projected_cost_usd=projected_cost_usd,
        total_usage_usd=total_usage_usd,
        now=now,
    )
    if balance_usd + 1e-12 < minimum_balance_usd:
        raise RuntimeError(
            f"OpenRouter balance ${balance_usd:.6f} is below the frozen "
            f"${minimum_balance_usd:.2f} stage gate"
        )
    return status


def _source_paths(output_dir: Path) -> dict[str, Path]:
    source_dir = output_dir / "source"
    return {
        "dir": source_dir,
        "result": source_dir / "RESULT.json",
        "trees": source_dir / "TREES.json",
        "targets": source_dir / "TARGETS.json",
    }


def build_control_authorization(
    *,
    source_result: dict[str, Any],
    source_hashes: dict[str, str],
    source_calendar_date: str,
) -> dict[str, Any]:
    structural = base.source_control_authorization(source_result)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "source_calendar_date": source_calendar_date,
        "source_artifacts": source_hashes,
        "authorization_inputs": (
            "source mechanics gates and changed-root count only"
        ),
        "source_science_was_not_an_authorization_input": True,
        "structural_authorization": structural,
    }


def _write_source_stage_stop(
    *,
    output_dir: Path,
    run_id: str,
    balance_usd: float,
    source_result: dict[str, Any],
    source_hashes: dict[str, str],
    authorization: dict[str, Any],
) -> dict[str, Any]:
    with configured_daily_limits():
        composite = base._base_composite(
            run_id=run_id,
            starting_balance_usd=balance_usd,
            source_result=source_result,
            source_hashes=source_hashes,
        )
    structural = authorization["structural_authorization"]
    composite["interface_version"] = INTERFACE_VERSION
    composite["protocol"].update(
        {
            "daily_stages_amendment": True,
            "source_calendar_date": authorization[
                "source_calendar_date"
            ],
            "control_authorization": structural,
        }
    )
    composite["decision"] = structural["decision"]
    composite["status"] = (
        "mechanics_failed"
        if not structural["source_mechanics_passed"]
        else "opportunity_failed"
    )
    checkpoint(output_dir / "RESULT.json", composite)
    return composite


def run_source_stage(
    *,
    output_dir: Path,
    run_id: str,
    ledger: dict[str, Any],
    total_usage_usd: float,
    balance_usd: float,
    source_runner: Callable[..., dict[str, Any]] = base.run_fresh_source,
    now: datetime | None = None,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    base.validate_predecessors()
    budget = _check_stage_budget(
        ledger=ledger,
        projected_cost_usd=SOURCE_DAILY_CAP_USD,
        total_usage_usd=total_usage_usd,
        balance_usd=balance_usd,
        minimum_balance_usd=SOURCE_MIN_STARTING_BALANCE_USD,
        now=now,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    source_dir = output_dir / "source"
    with configured_daily_limits():
        source_result = source_runner(
            output_dir=source_dir,
            run_id=f"{run_id}-source",
        )
    source_result["protocol"].update(
        {
            "daily_stages_amendment": True,
            "source_calendar_date": ledger["date"],
            "source_daily_cap_usd": SOURCE_DAILY_CAP_USD,
        }
    )
    checkpoint(source_dir / "RESULT.json", source_result)
    source_hashes = base._source_artifact_hashes(source_dir)
    authorization = build_control_authorization(
        source_result=source_result,
        source_hashes=source_hashes,
        source_calendar_date=str(ledger["date"]),
    )
    authorization_path = output_dir / "CONTROL_AUTHORIZATION.json"
    checkpoint(authorization_path, authorization)
    structural = authorization["structural_authorization"]
    stage = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": (
            "control_authorized"
            if structural["control_authorized"]
            else "control_not_authorized"
        ),
        "decision": structural["decision"],
        "run_id": run_id,
        "source_calendar_date": ledger["date"],
        "source_daily_budget_at_start": budget,
        "source_starting_balance_usd": balance_usd,
        "source_usage": source_result["usage"],
        "source_artifacts": source_hashes,
        "control_authorization_sha256": base.sha256_file(
            authorization_path
        ),
        "structural_authorization": structural,
    }
    checkpoint(output_dir / "SOURCE_STAGE.json", stage)
    if not structural["control_authorized"]:
        stage["final_result"] = _write_source_stage_stop(
            output_dir=output_dir,
            run_id=run_id,
            balance_usd=balance_usd,
            source_result=source_result,
            source_hashes=source_hashes,
            authorization=authorization,
        )
    return stage


def validate_control_handoff(
    *,
    output_dir: Path,
    control_calendar_date: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    source_paths = _source_paths(output_dir)
    source_result = json.loads(
        source_paths["result"].read_text(encoding="utf-8")
    )
    source_hashes = base._source_artifact_hashes(source_paths["dir"])
    authorization_path = output_dir / "CONTROL_AUTHORIZATION.json"
    authorization = json.loads(
        authorization_path.read_text(encoding="utf-8")
    )
    source_stage = json.loads(
        (output_dir / "SOURCE_STAGE.json").read_text(encoding="utf-8")
    )
    if (
        base.sha256_file(authorization_path)
        != source_stage["control_authorization_sha256"]
    ):
        raise ValueError("control authorization artifact hash changed")
    if authorization["source_artifacts"] != source_hashes:
        raise ValueError("source artifact hashes changed after authorization")
    source_date = str(
        source_result["protocol"]["source_calendar_date"]
    )
    if source_date != authorization["source_calendar_date"]:
        raise ValueError("source calendar date binding changed")
    if control_calendar_date <= source_date:
        raise RuntimeError(
            "control stage requires a later Europe/London calendar date"
        )
    recomputed = base.source_control_authorization(source_result)
    if recomputed != authorization["structural_authorization"]:
        raise ValueError("structural control authorization changed")
    if not recomputed["control_authorized"]:
        raise RuntimeError("source structure does not authorize control calls")
    return source_result, authorization, source_hashes


def _control_mechanics_failure(
    *,
    output_dir: Path,
    composite: dict[str, Any],
    control_dir: Path,
) -> dict[str, Any]:
    failure_path = control_dir / "FAILURE.json"
    failure = json.loads(failure_path.read_text(encoding="utf-8"))
    composite["protocol"]["control_failure_sha256"] = base.sha256_file(
        failure_path
    )
    composite["control"] = {
        "status": "mechanics_failed",
        "usage": failure["usage"],
        "mechanics_gates": failure["mechanics_gates"],
    }
    composite["status"] = "mechanics_failed"
    composite["decision"] = "stop_at_control_mechanics"
    checkpoint(output_dir / "RESULT.json", composite)
    return composite


def run_control_stage(
    *,
    output_dir: Path,
    run_id: str,
    ledger: dict[str, Any],
    total_usage_usd: float,
    balance_usd: float,
    control_runner: Callable[..., dict[str, Any]] = base.run_fresh_control,
    control_adapter=None,
    bootstrap_samples: int = base.BOOTSTRAP_SAMPLES,
    now: datetime | None = None,
) -> dict[str, Any]:
    base.validate_predecessors()
    budget = _check_stage_budget(
        ledger=ledger,
        projected_cost_usd=CONTROL_DAILY_CAP_USD,
        total_usage_usd=total_usage_usd,
        balance_usd=balance_usd,
        minimum_balance_usd=CONTROL_MIN_STARTING_BALANCE_USD,
        now=now,
    )
    source_result, authorization, source_hashes = validate_control_handoff(
        output_dir=output_dir,
        control_calendar_date=str(ledger["date"]),
    )
    control_dir = output_dir / "control"
    if control_dir.exists() and any(control_dir.iterdir()):
        raise FileExistsError(f"control directory is not empty: {control_dir}")
    with configured_daily_limits():
        composite = base._base_composite(
            run_id=run_id,
            starting_balance_usd=float(
                json.loads(
                    (output_dir / "SOURCE_STAGE.json").read_text(
                        encoding="utf-8"
                    )
                )["source_starting_balance_usd"]
            ),
            source_result=source_result,
            source_hashes=source_hashes,
        )
    composite["interface_version"] = INTERFACE_VERSION
    composite["protocol"].update(
        {
            "daily_stages_amendment": True,
            "source_calendar_date": authorization[
                "source_calendar_date"
            ],
            "control_calendar_date": ledger["date"],
            "control_authorization": authorization[
                "structural_authorization"
            ],
            "control_authorization_sha256": base.sha256_file(
                output_dir / "CONTROL_AUTHORIZATION.json"
            ),
            "source_daily_cap_usd": SOURCE_DAILY_CAP_USD,
            "control_daily_cap_usd": CONTROL_DAILY_CAP_USD,
            "composite_daily_cap_usd": COMPOSITE_DAILY_CAP_USD,
            "source_science_was_not_an_authorization_input": True,
        }
    )
    try:
        control_result = control_runner(
            source_dir=output_dir / "source",
            output_dir=control_dir,
            run_id=f"{run_id}-control",
            adapter=control_adapter,
            remaining_credit=balance_usd,
            bootstrap_samples=bootstrap_samples,
        )
    except RuntimeError as exc:
        failure_path = control_dir / "FAILURE.json"
        if (
            str(exc) != "formal history-blind mechanics gates failed"
            or not failure_path.exists()
        ):
            raise
        return _control_mechanics_failure(
            output_dir=output_dir,
            composite=composite,
            control_dir=control_dir,
        )
    with configured_daily_limits():
        result = base.finalize_composite_with_control(
            output_dir=output_dir,
            composite=composite,
            source_result=source_result,
            control_result=control_result,
            control_dir=control_dir,
        )
    checkpoint(
        output_dir / "CONTROL_STAGE.json",
        {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": result["status"],
            "decision": result["decision"],
            "control_calendar_date": ledger["date"],
            "control_daily_budget_at_start": budget,
            "control_usage": control_result["usage"],
            "result_sha256": base.sha256_file(output_dir / "RESULT.json"),
        },
    )
    return result


def _load_ledger(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="stage", required=True)
    for stage in ("source", "control"):
        subparser = subparsers.add_parser(stage)
        subparser.add_argument("--output-dir", type=Path, required=True)
        subparser.add_argument("--run-id", required=True)
        subparser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    live = read_live_credits()
    ledger = _load_ledger(args.daily_ledger)
    try:
        if args.stage == "source":
            result = run_source_stage(
                output_dir=args.output_dir,
                run_id=args.run_id,
                ledger=ledger,
                total_usage_usd=live["total_usage_usd"],
                balance_usd=live["balance_usd"],
            )
        else:
            result = run_control_stage(
                output_dir=args.output_dir,
                run_id=args.run_id,
                ledger=ledger,
                total_usage_usd=live["total_usage_usd"],
                balance_usd=live["balance_usd"],
            )
    except Exception as exc:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        failure_path = args.output_dir / f"{args.stage.upper()}_RUNNER_FAILURE.json"
        if not failure_path.exists():
            checkpoint(
                failure_path,
                {
                    "schema_version": SCHEMA_VERSION,
                    "status": "failed_closed",
                    "stage": args.stage,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
