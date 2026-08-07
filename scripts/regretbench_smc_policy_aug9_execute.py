#!/usr/bin/env python3
"""Run the single hash-bound Aug 9 RegretBench SMC policy command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_deepseek_smc_dynamic_depth2_daily as daily
from scripts import regretbench_deepseek_support_recovery as primary
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-smc-policy-aug9-execute-1"
BINDINGS = daily.ROOT / "EXECUTION_BINDINGS.json"
BINDINGS_SHA256 = (
    "8ae81b68ec666c3f64f583fd5dae4c95d4ef14a3e59933542a0850ccb2de973c"
)


def validate_bindings() -> dict[str, Any]:
    if primary.sha256_file(BINDINGS) != BINDINGS_SHA256:
        raise RuntimeError("SMC policy execution binding artifact changed")
    payload = json.loads(BINDINGS.read_text(encoding="utf-8"))
    expected = payload.get("files") or {}
    if (
        payload.get("status")
        != "prospectively_frozen_before_any_smc_policy_response"
        or payload.get("authorization")
        != "literal_verified_smc_support_pass_only"
        or float(payload.get("maximum_same_day_predecessor_spend_usd", -1.0))
        != 0.5
        or float(payload.get("maximum_policy_spend_usd", -1.0)) != 3.9
        or float(payload.get("maximum_combined_aug9_spend_usd", -1.0)) != 4.4
        or float(payload.get("daily_cap_usd", -1.0)) != 5.0
        or payload.get("paid_calls_made_while_freezing") != 0
        or float(payload.get("cost_usd_while_freezing", -1.0)) != 0.0
    ):
        raise RuntimeError("SMC policy execution binding metadata changed")
    for relative, digest in expected.items():
        path = REPO_ROOT / relative
        if not path.is_file() or primary.sha256_file(path) != digest:
            raise RuntimeError(f"SMC policy bound file changed: {relative}")
    return {
        "status": "verified_frozen_execution",
        "bindings_sha256": BINDINGS_SHA256,
        "bound_files": len(expected),
        "model_calls_made": 0,
        "cost_usd": 0.0,
    }


def preflight(
    *,
    now=None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    catalog_reader=None,
) -> dict[str, Any]:
    bindings = validate_bindings()
    kwargs = {"now": now, "live_reader": live_reader}
    if catalog_reader is not None:
        kwargs["catalog_reader"] = catalog_reader
    ready = daily.preflight(**kwargs)
    complete = ready["status"] == "already_complete_verified"
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": ready["status"],
        "next_stage": None if complete else "smc_dynamic_depth2_policy",
        "bindings": bindings,
        "stage_preflight": ready,
        "model_calls_made": 0,
        "files_written": 0,
    }


def execute(
    *,
    now=None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
) -> dict[str, Any]:
    validate_bindings()
    result = daily.execute(now=now, live_reader=live_reader)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "next_stage": None,
        "stage_result": result,
    }


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
