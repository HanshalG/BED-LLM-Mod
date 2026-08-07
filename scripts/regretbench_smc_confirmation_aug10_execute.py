#!/usr/bin/env python3
"""Run the single hash-bound Aug 10 RegretBench SMC confirmation command."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import regretbench_deepseek_smc_confirmation_daily as daily
from scripts import regretbench_deepseek_smc_confirmation_paper_fragment as paper_fragment
from scripts import regretbench_deepseek_smc_confirmation_report as frozen_report
from scripts import regretbench_deepseek_support_recovery as primary
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-smc-confirmation-aug10-execute-1"
BINDINGS = daily.ROOT / "EXECUTION_BINDINGS.json"
BINDINGS_SHA256 = (
    "f46b7e31ab419bd42c2fbc0f3c0994e242e7dbf974eff538b01e0b608350a805"
)


def validate_bindings() -> dict[str, Any]:
    if primary.sha256_file(BINDINGS) != BINDINGS_SHA256:
        raise RuntimeError("SMC confirmation binding artifact changed")
    payload = json.loads(BINDINGS.read_text(encoding="utf-8"))
    files = payload.get("files") or {}
    if (
        payload.get("status")
        != "prospectively_frozen_before_any_smc_policy_response"
        or payload.get("authorization")
        != "literal_verified_smc_development_pass_only"
        or payload.get("budget_branch")
        != "smc_confirmation_supersedes_bongard_aug10_only_after_authorization"
        or int(payload.get("parent_bank_requests", -1)) != 64
        or int(payload.get("maximum_policy_requests", -1)) != 8_768
        or int(payload.get("maximum_total_requests", -1)) != 8_832
        or float(payload.get("maximum_combined_aug10_spend_usd", -1.0))
        != 3.70
        or float(payload.get("daily_cap_usd", -1.0)) != 5.0
        or payload.get("paid_calls_made_while_freezing") != 0
        or float(payload.get("cost_usd_while_freezing", -1.0)) != 0.0
    ):
        raise RuntimeError("SMC confirmation binding metadata changed")
    for relative, digest in files.items():
        path = REPO_ROOT / relative
        if not path.is_file() or primary.sha256_file(path) != digest:
            raise RuntimeError(f"SMC confirmation bound file changed: {relative}")
    return {
        "status": "verified_frozen_execution",
        "bindings_sha256": BINDINGS_SHA256,
        "bound_files": len(files),
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
        "next_stage": None if complete else "smc_dynamic_depth2_confirmation",
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
    bindings = validate_bindings()
    result = daily.execute(now=now, live_reader=live_reader)
    reporting = None
    if result.get("status") == "complete_reconciled":
        report = frozen_report.write_report(
            daily.RUN_DIR, parent_dir=daily.PARENT_DIR
        )
        fragment = paper_fragment.write_fragment(
            daily.RUN_DIR, parent_dir=daily.PARENT_DIR
        )
        reporting = {
            "frozen_report": report,
            "paper_fragment": fragment,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "bindings": bindings,
        "daily_result": result,
        "reporting": reporting,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    result = preflight() if args.preflight else execute()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
