#!/usr/bin/env python3
"""Development gate for the four-world Tau2 account prerequisite."""

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
from scripts.tau2_account_prerequisite_ranking_gate import (
    FORMAL_PROMPTS,
    ROOT_ACTIONS,
)


SCHEMA_VERSION = 4
SELECTION_SEED = 24320
SMOKE_PROMPTS = (
    "The phone has no usable mobile data while airplane mode remains on.",
    "Carrier data service is unavailable during a fixed airplane-mode diagnostic.",
)
ACCOUNT_WORLD_IDS = (
    "available_roaming_enabled_device_off",
    "available_roaming_disabled_device_off",
    "exhausted_roaming_enabled_device_off",
    "exhausted_roaming_disabled_device_off",
)
ACCOUNT_HYPOTHESES = (
    {
        "id": "h1",
        "description": "Allowance available; account roaming enabled",
        "data_allowance": "available",
        "account_roaming": "enabled",
        "device_roaming": "off",
    },
    {
        "id": "h2",
        "description": "Allowance available; account roaming disabled",
        "data_allowance": "available",
        "account_roaming": "disabled",
        "device_roaming": "off",
    },
    {
        "id": "h3",
        "description": "Allowance exhausted; account roaming enabled",
        "data_allowance": "exhausted",
        "account_roaming": "enabled",
        "device_roaming": "off",
    },
    {
        "id": "h4",
        "description": "Allowance exhausted; account roaming disabled",
        "data_allowance": "exhausted",
        "account_roaming": "disabled",
        "device_roaming": "off",
    },
)


def stage_prompts(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_PROMPTS
    if stage == "formal":
        return FORMAL_PROMPTS
    raise ValueError("stage must be serving_smoke or formal")


def add_account_only_gates(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload["records"]
    all_roots_zero = all(
        math.isclose(
            float(action["predicted_d1_information"]),
            0.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for record in records
        for action in record["actions"]
    )
    d1_lookup = 0
    d2_lookup = 0
    for record in records:
        actions = record["actions"]
        d1 = max(
            actions,
            key=lambda row: (
                float(row["predicted_d1_information"]),
                row["action_id"],
            ),
        )
        d2 = max(
            actions,
            key=lambda row: (
                float(row["predicted_d2_information"]),
                row["action_id"],
            ),
        )
        d1_lookup += d1["action_id"] == "customer_lookup"
        d2_lookup += d2["action_id"] == "customer_lookup"

    gates = payload["summary"]["gates"]
    gates.pop("all_pass", None)
    gates.pop("mean_signature_coverage_at_least_5", None)
    gates["fixed_account_support_complete_4_of_4"] = all(
        record["official_signature_coverage"] == 4
        for record in records
    )
    gates["all_roots_preserve_zero_immediate_information"] = all_roots_zero
    if payload["protocol"]["stage"] == "serving_smoke":
        gates["d2_selects_lookup_both"] = d2_lookup == 2
        gates["d1_never_selects_lookup"] = d1_lookup == 0
    else:
        gates["d2_selects_lookup_at_least_10"] = d2_lookup >= 10
    gates["all_pass"] = all(gates.values())
    payload["summary"].update(
        {
            "d1_lookup_selected_count": d1_lookup,
            "d2_lookup_selected_count": d2_lookup,
        }
    )
    payload["schema_version"] = SCHEMA_VERSION
    payload["status"] = "passed" if gates["all_pass"] else "gate_failed"
    payload["protocol"].update(
        {
            "schema_version": SCHEMA_VERSION,
            "selection_seed": SELECTION_SEED,
            "development_only": True,
            "account_only_support": True,
            "device_roaming_fixed_off": True,
            "account_world_ids": list(ACCOUNT_WORLD_IDS),
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
    return add_account_only_gates(
        run_v2_gate(
            config,
            t3_dir=t3_dir,
            stage=stage,
            raw_checkpoint_path=raw_checkpoint_path,
            prompt_variants=stage_prompts(stage),
            fixed_hypotheses=ACCOUNT_HYPOTHESES,
            observation_world_ids=ACCOUNT_WORLD_IDS,
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
