#!/usr/bin/env python3
"""GPT-5.4 observational-equivalence gate for fixed-support Tau2 BED."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.tau2_account_fixed_support_v2 import run_gate as run_v2_gate


SCHEMA_VERSION = 3
ZERO_INFORMATION_ROOTS = (
    "customer_lookup",
    "status_bar",
    "speed_test",
    "payment_request",
    "sim_status",
)


def add_observational_equivalence_gates(
    payload: dict[str, Any],
) -> dict[str, Any]:
    records = payload["records"]
    zero_roots_exact = all(
        math.isclose(
            float(record["actions_by_id"][action]["predicted_d1_information"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for record in records
        for action in ZERO_INFORMATION_ROOTS
    )
    network_status_positive = all(
        float(
            record["actions_by_id"]["network_status"][
                "predicted_d1_information"
            ]
        )
        > 0.0
        for record in records
    )
    payload["summary"]["gates"].update(
        {
            "all_zero_information_roots_preserve_equivalence": (
                zero_roots_exact
            ),
            "network_status_predicted_informative": network_status_positive,
        }
    )
    payload["summary"]["gates"]["all_pass"] = all(
        payload["summary"]["gates"].values()
    )
    payload["schema_version"] = SCHEMA_VERSION
    payload["status"] = (
        "passed"
        if payload["summary"]["gates"]["all_pass"]
        else "gate_failed"
    )
    payload["protocol"].update(
        {
            "schema_version": SCHEMA_VERSION,
            "capability_test_model": "openai/gpt-5.4",
            "observational_equivalence_gated": True,
            "zero_information_roots": list(ZERO_INFORMATION_ROOTS),
        }
    )
    return payload


def run_gate(
    config: Config,
    *,
    t3_dir: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    return add_observational_equivalence_gates(
        run_v2_gate(
            config,
            t3_dir=t3_dir,
            stage=stage,
            raw_checkpoint_path=raw_checkpoint_path,
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--t3-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = 0.75
    else:
        config.openrouter_projected_cost_usd = 0.50
        config.openrouter_run_budget_usd = 2.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            t3_dir=args.t3_dir,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
