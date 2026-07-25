#!/usr/bin/env python3
"""Run manifest-bound ClariQ multisample likelihood development V2."""

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
from scripts.clariq_multisample_likelihood_development import (
    DevelopmentExecutionError,
    DeterministicFixtureModel,
    MAX_COST_USD,
    _build_model,
    _checkpoint,
    run_development,
)


INTERFACE_VERSION = "clariq-multisample-likelihood-development-2"
MANIFEST_SHA256 = (
    "97af5ebc228dfff94e5c270f1dd0f10b862a8d5627a26bcb36266ac3a8d1ce73"
)
SELECTED_TOPIC_IDS = ("136", "125", "149")
VALID_ROOT_COUNTS = (14, 13, 14)
EXPECTED_REQUESTS = 205


def load_manifest(path: Path) -> dict[str, dict[str, Any]]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("ClariQ V2 manifest hash does not match")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["status"] != "passed":
        raise ValueError("ClariQ V2 manifest did not pass")
    if tuple(payload["selected_topic_ids"]) != SELECTED_TOPIC_IDS:
        raise ValueError("ClariQ V2 manifest topic IDs changed")
    tasks = {
        task["topic_id"]: task for task in payload["selected_topics"]
    }
    if tuple(tasks) != SELECTED_TOPIC_IDS:
        raise ValueError("ClariQ V2 manifest task order changed")
    if tuple(len(tasks[topic_id]["questions"]) for topic_id in tasks) != (
        VALID_ROOT_COUNTS
    ):
        raise ValueError("ClariQ V2 valid root counts changed")
    if payload["expected_likelihood_requests"] != EXPECTED_REQUESTS:
        raise ValueError("ClariQ V2 request count changed")
    return tasks


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
    tasks = load_manifest(args.manifest)
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.20
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model = DeterministicFixtureModel() if args.dry_run else _build_model(config)
    try:
        payload = run_development(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            model=model,
            tasks_override=tasks,
            expected_requests=EXPECTED_REQUESTS,
            interface_version=INTERFACE_VERSION,
        )
        payload["protocol"]["manifest_sha256"] = MANIFEST_SHA256
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, DevelopmentExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "DEVELOPMENT_FAILURE.json", failure)
        raise
    output = args.output_dir / "DEVELOPMENT.json"
    _checkpoint(output, payload)
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
