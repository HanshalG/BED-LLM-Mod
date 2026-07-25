#!/usr/bin/env python3
"""Rescore frozen tau-Knowledge trees with thinking Gemma 4 26B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from model_factory import build_model_adapter
from scripts.tau_knowledge_cross_model_scorer import run_gate, sha256_file
from scripts.tau_knowledge_retrieval_opportunity import (
    GateExecutionError,
    SCHEMA_VERSION,
)


MODEL_ID = "google/gemma-4-26b-a4b-it"
INTERFACE_VERSION = "gemma26b-thinking-1"
EXPECTED_LOGICAL_REQUESTS = {
    "serving_smoke": 14,
    "confirmation": 140,
}
COST_CAP_USD = {
    "serving_smoke": 0.25,
    "confirmation": 2.00,
}


def apply_thinking_gates(
    payload: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected = EXPECTED_LOGICAL_REQUESTS[stage]
    usage = payload["usage"]
    generator = usage["generator"]
    physical = int(usage["physical_requests"])
    forced_exits = int(generator.get("forced_exits", 0))
    forced_requests = int(generator.get("forced_final_requests", 0))
    forced_successes = int(generator.get("forced_final_successes", 0))
    gates = dict(payload["summary"]["gates"])
    gates.pop("all_pass", None)
    gates.pop("exact_physical_request_count", None)
    gates.pop("zero_reasoning_tokens", None)
    gates.update(
        {
            "all_logical_responses_parsed": True,
            "physical_requests_between_logical_and_double": (
                expected <= physical <= 2 * expected
            ),
            "positive_reasoning_tokens": int(
                usage["reasoning_tokens"]
            )
            > 0,
            "all_forced_exits_finalized": (
                forced_exits == forced_requests == forced_successes
            ),
            "cost_within_stage_cap": (
                float(usage["adapter_cost_usd"]) <= COST_CAP_USD[stage]
            ),
        }
    )
    gates["all_pass"] = all(gates.values())
    payload["summary"]["gates"] = gates
    payload["status"] = "passed" if gates["all_pass"] else "gate_failed"
    protocol = payload["protocol"]
    protocol.update(
        {
            "expected_logical_requests": expected,
            "minimum_physical_requests": expected,
            "maximum_physical_requests": 2 * expected,
            "expected_physical_requests": None,
            "reasoning_requested": True,
            "thinking_max_new_tokens": 4096,
            "thinking_final_max_new_tokens": 1024,
            "forced_final_is_reasoning_disabled": True,
            "stage_cost_cap_usd": COST_CAP_USD[stage],
        }
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "confirmation"),
        required=True,
    )
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--nonsemantic-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.10 if args.stage == "serving_smoke" else 0.75
    )
    config.openrouter_run_budget_usd = COST_CAP_USD[args.stage]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "CONFIRMATION.json"
    )

    spec = config.model_pairs[0].questioner
    if spec.model != MODEL_ID or not spec.thinking:
        raise ValueError("Gemma thinking scorer config does not match")
    model = build_model_adapter(spec, config)
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            input_artifact=args.input_artifact,
            nonsemantic_analysis=args.nonsemantic_analysis,
            raw_checkpoint_path=raw_path,
            model_id=MODEL_ID,
            interface_version=INTERFACE_VERSION,
            model_adapter=model,
        )
        payload = apply_thinking_gates(payload, stage=args.stage)
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    output_path = args.output_dir / output_name
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    concise = {
        "status": payload["status"],
        "output": str(output_path),
        "focused_accuracy": summary["focused_pairwise_accuracy"],
        "usage": payload["usage"],
    }
    if args.stage == "confirmation":
        concise.update(
            {
                "root_accuracy": summary[
                    "nonmyopic_root_pairwise_accuracy"
                ],
                "root_accuracy_gain": summary[
                    "root_pairwise_accuracy_gain"
                ],
                "endpoint_total": summary["cross_model_endpoint_total"],
                "myopic_advantage": summary[
                    "end_to_end_total_advantage_over_myopic"
                ],
            }
        )
    print(json.dumps(concise, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
