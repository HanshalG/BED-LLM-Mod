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
from scripts import regretbench_deepseek_smc_frozen_report as frozen_report
from scripts import regretbench_deepseek_smc_paper_fragment as paper_fragment
from scripts import regretbench_deepseek_support_recovery as primary
from scripts.openrouter_daily_budget import read_live_credits


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-smc-policy-aug9-execute-1"
BINDINGS = daily.ROOT / "EXECUTION_BINDINGS.json"
BINDINGS_SHA256 = (
    "62a7107be37a2e652357cc6bbb7453a59567fbfa2a94dc3c643cce0df71fc4f4"
)
REPORTING_BINDING = daily.ROOT / "REPORTING_BINDING.json"
REPORTING_BINDING_SHA256 = (
    "be10902ce1a703c295145808f7dfa26db8d399eaf90358f23c988e696d667574"
)
PAPER_FRAGMENT_BINDING = daily.ROOT / "PAPER_FRAGMENT_BINDING.json"
PAPER_FRAGMENT_BINDING_SHA256 = (
    "57dd8a53c10c3d9f1180625b8fb1ebf7e70232cc7c86ef7b9bbd603255a16371"
)


def _validate_derived_binding(path: Path, digest: str) -> int:
    if primary.sha256_file(path) != digest:
        raise RuntimeError(f"SMC derived binding changed: {path.name}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "prospectively_frozen_before_any_smc_policy_response":
        raise RuntimeError(f"SMC derived binding status changed: {path.name}")
    count = 0
    for row in payload.values():
        if not isinstance(row, dict) or "path" not in row:
            continue
        expected = row.get("sha256", row.get("preresult_sha256"))
        bound = REPO_ROOT / row["path"]
        if not bound.is_file() or primary.sha256_file(bound) != expected:
            raise RuntimeError(f"SMC derived bound file changed: {row['path']}")
        count += 1
    return count


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
    derived = _validate_derived_binding(
        REPORTING_BINDING, REPORTING_BINDING_SHA256
    ) + _validate_derived_binding(
        PAPER_FRAGMENT_BINDING, PAPER_FRAGMENT_BINDING_SHA256
    )
    return {
        "status": "verified_frozen_execution",
        "bindings_sha256": BINDINGS_SHA256,
        "bound_files": len(expected),
        "derived_bound_files": derived,
        "reporting_bindings_verified": True,
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
    reporting = None
    if result.get("status") == "complete_reconciled":
        report = frozen_report.write_report(
            daily.DEVELOPMENT_DIR,
            primary_dir=daily.PRIMARY_DEVELOPMENT_DIR,
        )
        fragment = paper_fragment.write_fragment(
            daily.DEVELOPMENT_DIR,
            primary_dir=daily.PRIMARY_DEVELOPMENT_DIR,
        )
        reporting = {
            "frozen_report": report,
            "paper_fragment": fragment,
        }
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "complete",
        "next_stage": None,
        "stage_result": result,
        "reporting": reporting,
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
