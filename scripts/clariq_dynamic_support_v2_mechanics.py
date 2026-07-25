#!/usr/bin/env python3
"""Run the full ClariQ dynamic-support V2 mechanics tree."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.clariq_dynamic_support_smoke import (
    DeterministicFixtureModel,
    SmokeExecutionError,
    _build_model,
    _checkpoint,
    run_smoke,
)
from scripts.clariq_dynamic_support_v2_serving import (
    MANIFEST_SHA256,
    load_manifest,
)


INTERFACE_VERSION = "clariq-dynamic-support-v2-mechanics-1"
EXPECTED_REQUESTS = 91
BRANCH_SUPPORT_CHANGE_MINIMUM = 80
BRANCH_PROFILE_DIVERSITY_MINIMUM = 75
POSITIVE_CONTINUATION_MINIMUM = 75
PROJECTED_COST_USD = 1.00
MAX_COST_USD = 1.25


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    task = load_manifest(args.manifest)
    if task["expected_model_requests"] != EXPECTED_REQUESTS:
        raise ValueError("ClariQ V2 mechanics request count changed")
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_REQUESTS - 1
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_smoke(
            config,
            source_root=args.source_root,
            manifest_path=args.manifest,
            raw_path=raw_path,
            model=model,
            task_override=task,
            interface_version=INTERFACE_VERSION,
            manifest_sha256=MANIFEST_SHA256,
            expected_requests=EXPECTED_REQUESTS,
            spaced_codes=True,
            branch_support_change_minimum=(
                BRANCH_SUPPORT_CHANGE_MINIMUM
            ),
            branch_profile_diversity_minimum=(
                BRANCH_PROFILE_DIVERSITY_MINIMUM
            ),
            positive_continuation_minimum=(
                POSITIVE_CONTINUATION_MINIMUM
            ),
            max_cost_usd=MAX_COST_USD,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        payload["protocol"]["serving_response_reused"] = False
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "MECHANICS_FAILURE.json", failure)
        raise
    _checkpoint(args.output_dir / "MECHANICS.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
