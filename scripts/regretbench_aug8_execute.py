#!/usr/bin/env python3
"""Run the frozen Aug 8 RegretBench dependency chain exactly once."""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

from scripts import bongard_openworld_luna_naive_first_link_daily_execute as baseline
from scripts import regretbench_deepseek_dynamic_depth2_policy_daily as policy
from scripts import regretbench_deepseek_support_recovery_daily as support
from scripts.openrouter_daily_budget import read_live_credits


DATE = "2026-08-08"
TIMEZONE = "Europe/London"
BASELINE_DAILY_SHA256 = (
    "db8d13229444abbf3f549967af4e70271103b6f5abd917fcce88573ec44b95c9"
)
SUPPORT_DAILY_SHA256 = (
    "d183e5b2785741d8d410c90d0b3f983da31db1a5b698a75fbd4b52b234c49399"
)
POLICY_DAILY_SHA256 = (
    "59fc575fb716b45ae76d0a1505b9293b5883c4cd99088f67e5e6159d64f0ebf7"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected JSON object: {path}")
    return value


def _validate_date(now: datetime | None = None) -> None:
    timezone = ZoneInfo(TIMEZONE)
    local = now.astimezone(timezone) if now else datetime.now(timezone)
    if local.date().isoformat() != DATE:
        raise RuntimeError(f"Aug 8 chain can run only on {DATE}")


def validate_bindings() -> None:
    bindings = (
        (baseline, BASELINE_DAILY_SHA256, "baseline daily"),
        (support, SUPPORT_DAILY_SHA256, "support daily"),
        (policy, POLICY_DAILY_SHA256, "policy daily"),
    )
    for module, expected, label in bindings:
        if _sha256(Path(module.__file__).resolve()) != expected:
            raise RuntimeError(f"{label} binding changed")


def _path_started(path: Path) -> bool:
    if not path.exists():
        return False
    return path.is_file() or any(path.iterdir())


def _refuse_downstream_before_baseline() -> None:
    paths = (
        support.SMOKE_DIR,
        support.DEVELOPMENT_DIR,
        support.LEDGER,
        support.ROOT / "DAILY_RESULT.json",
        policy.SMOKE_DIR,
        policy.NAIVE_SMOKE_DIR,
        policy.DEVELOPMENT_DIR,
        policy.LEDGER,
        policy.ROOT / "DAILY_RESULT.json",
    )
    if any(_path_started(path) for path in paths):
        raise RuntimeError("downstream artifact exists before baseline completion")


def _refuse_policy_before_support() -> None:
    paths = (
        policy.SMOKE_DIR,
        policy.NAIVE_SMOKE_DIR,
        policy.DEVELOPMENT_DIR,
        policy.LEDGER,
        policy.ROOT / "DAILY_RESULT.json",
    )
    if any(_path_started(path) for path in paths):
        raise RuntimeError("policy artifact exists before support completion")


def preflight(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader: Callable[[], dict[str, Any]] | None = None,
) -> dict[str, Any]:
    _validate_date(now)
    validate_bindings()
    catalog_kwargs = (
        {"catalog_reader": catalog_reader} if catalog_reader is not None else {}
    )

    if not baseline.SMOKE_RESULT.is_file():
        _refuse_downstream_before_baseline()
        ready = baseline.preflight_smoke(
            now=now, live_reader=live_reader, **catalog_kwargs
        )
        return {
            "status": "ready_without_paid_calls",
            "next_stage": "baseline_smoke",
            "stage_preflight": ready,
            "model_calls_made": 0,
            "files_written": 0,
        }

    baseline_result = _load(baseline.SMOKE_RESULT)
    if baseline_result.get("status") != "passed":
        return {
            "status": "terminal_baseline_result",
            "next_stage": None,
            "baseline_status": baseline_result.get("status"),
            "model_calls_made": 0,
            "files_written": 0,
        }

    support_daily_path = support.ROOT / "DAILY_RESULT.json"
    if not support_daily_path.is_file():
        _refuse_policy_before_support()
        ready = support.preflight(
            now=now, live_reader=live_reader, **catalog_kwargs
        )
        return {
            "status": "ready_without_paid_calls",
            "next_stage": "support_recovery",
            "stage_preflight": ready,
            "model_calls_made": 0,
            "files_written": 0,
        }

    support_daily = _load(support_daily_path)
    if support_daily.get("status") == "smoke_stopped":
        return {
            "status": "terminal_support_result",
            "next_stage": None,
            "support_development_status": None,
            "support_smoke_status": "stopped",
            "model_calls_made": 0,
            "files_written": 0,
        }
    if support_daily.get("status") != "complete_reconciled":
        raise RuntimeError("support daily artifact is partial or invalid")
    if support_daily.get("development_status") != "passed":
        return {
            "status": "terminal_support_result",
            "next_stage": None,
            "support_development_status": support_daily.get(
                "development_status"
            ),
            "model_calls_made": 0,
            "files_written": 0,
        }

    policy_daily_path = policy.ROOT / "DAILY_RESULT.json"
    if not policy_daily_path.is_file():
        ready = policy.preflight(
            now=now, live_reader=live_reader, **catalog_kwargs
        )
        return {
            "status": "ready_without_paid_calls",
            "next_stage": "dynamic_policy",
            "stage_preflight": ready,
            "model_calls_made": 0,
            "files_written": 0,
        }

    policy_daily = _load(policy_daily_path)
    if policy_daily.get("status") == "smoke_stopped":
        return {
            "status": "complete",
            "next_stage": None,
            "development_status": None,
            "policy_smoke_status": "stopped",
            "model_calls_made": 0,
            "files_written": 0,
        }
    if policy_daily.get("status") != "complete_reconciled":
        raise RuntimeError("policy daily artifact is partial or invalid")
    return {
        "status": "complete",
        "next_stage": None,
        "development_status": policy_daily.get("development_status"),
        "development_result_sha256": policy_daily.get(
            "development_result_sha256"
        ),
        "development_verification_sha256": policy_daily.get(
            "development_verification_sha256"
        ),
        "model_calls_made": 0,
        "files_written": 0,
    }


def execute(
    *,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    completed = []
    while True:
        ready = preflight(now=now, live_reader=live_reader)
        stage = ready["next_stage"]
        if stage is None:
            return {**ready, "stages_completed_by_invocation": completed}
        if stage == "baseline_smoke":
            result = baseline.execute_smoke(now=now, live_reader=live_reader)
            completed.append(stage)
            if result.get("status") != "passed":
                return {
                    "status": "terminal_baseline_result",
                    "next_stage": None,
                    "baseline_status": result.get("status"),
                    "stages_completed_by_invocation": completed,
                }
            continue
        if stage == "support_recovery":
            result = support.execute(now=now, live_reader=live_reader)
            completed.append(stage)
            if result.get("development_status") != "passed":
                return {
                    "status": "terminal_support_result",
                    "next_stage": None,
                    "support_development_status": result.get(
                        "development_status"
                    ),
                    "stages_completed_by_invocation": completed,
                }
            continue
        if stage == "dynamic_policy":
            result = policy.execute(now=now, live_reader=live_reader)
            completed.append(stage)
            return {
                "status": "complete",
                "next_stage": None,
                "development_status": result.get("development_status"),
                "development_result_sha256": result.get(
                    "development_result_sha256"
                ),
                "development_verification_sha256": result.get(
                    "development_verification_sha256"
                ),
                "stages_completed_by_invocation": completed,
            }
        raise RuntimeError(f"unknown Aug 8 chain stage: {stage}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    try:
        result = preflight() if args.preflight else execute()
    except Exception as exc:
        result = {
            "status": "failed_closed",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "model_calls_made": 0,
            "files_written": 0,
        }
        print(json.dumps(result, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
