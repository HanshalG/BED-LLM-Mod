#!/usr/bin/env python3
"""Run one GPT-5.4 scorer test-retest replicate on frozen tau trees."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts.tau_knowledge_cross_model_scorer import run_gate, sha256_file
from scripts.tau_knowledge_retrieval_opportunity import (
    GateExecutionError,
    SCHEMA_VERSION,
)


MODEL_ID = "openai/gpt-5.4"
INTERFACE_VERSION = "gpt54-retest-1"
REPLICATE_COUNT = 3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--input-artifact", type=Path, required=True)
    parser.add_argument("--nonsemantic-analysis", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--replicate-index",
        type=int,
        choices=range(1, REPLICATE_COUNT + 1),
        required=True,
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 1.40
    config.openrouter_run_budget_usd = 2.25
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_path = args.output_dir / f"REPLICATE_{args.replicate_index}.json"

    try:
        payload = run_gate(
            config,
            stage="confirmation",
            input_artifact=args.input_artifact,
            nonsemantic_analysis=args.nonsemantic_analysis,
            raw_checkpoint_path=raw_path,
            model_id=MODEL_ID,
            interface_version=INTERFACE_VERSION,
        )
        payload["protocol"]["replicate_index"] = args.replicate_index
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": "confirmation",
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "replicate_index": args.replicate_index,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        failure_path = (
            args.output_dir / f"REPLICATE_{args.replicate_index}_FAILURE.json"
        )
        failure_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    summary = payload["summary"]
    print(
        json.dumps(
            {
                "status": payload["status"],
                "replicate_index": args.replicate_index,
                "output": str(output_path),
                "root_accuracy": summary[
                    "nonmyopic_root_pairwise_accuracy"
                ],
                "root_accuracy_gain": summary[
                    "root_pairwise_accuracy_gain"
                ],
                "focused_accuracy": summary["focused_pairwise_accuracy"],
                "endpoint_total": summary["cross_model_endpoint_total"],
                "myopic_advantage": summary[
                    "end_to_end_total_advantage_over_myopic"
                ],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
