#!/usr/bin/env python3
"""Run tau receding V3 with a preregistered zero-padding parser amendment."""

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
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_receding_continuation_v2 import run_gate
from scripts.tau_knowledge_receding_continuation_v3 import (
    _score_schema,
    document_count_messages,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    GateExecutionError,
    SCHEMA_VERSION,
)


INTERFACE_VERSION = "3.1"


def parse_zero_padding_scores(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    if set(payload) != set(_score_schema()):
        raise ValueError("V3.1 response has unexpected keys")
    scores = []
    for index in range(1, 5):
        value = payload[f"followup_{index}_score"]
        if (
            not isinstance(value, str)
            or not value.isdigit()
            or not 1 <= len(value) <= 2
        ):
            raise ValueError("V3.1 score is not a one- or two-digit string")
        score = int(value)
        if not (
            0 <= score <= 9
            or 30 <= score <= 39
            or 60 <= score <= 69
            or 90 <= score <= 99
        ):
            raise ValueError("V3.1 score is outside a valid count band")
        scores.append(score)
    return {
        "scores": scores,
        "rationales": ["count-dominant document score"] * 4,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development", "confirmation"),
        required=True,
    )
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.08
        config.openrouter_run_budget_usd = 0.50
    elif args.stage == "development":
        config.openrouter_projected_cost_usd = 0.80
        config.openrouter_run_budget_usd = 3.00
    else:
        config.openrouter_projected_cost_usd = 3.20
        config.openrouter_run_budget_usd = 4.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = {
        "serving_smoke": "SERVING_SMOKE.json",
        "development": "DEVELOPMENT.json",
        "confirmation": "CONFIRMATION.json",
    }[args.stage]
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            tau_root=args.tau_root,
            input_artifact=args.input_artifact,
            raw_checkpoint_path=raw_path,
            message_builder=document_count_messages,
            response_parser=parse_zero_padding_scores,
            interface_version=INTERFACE_VERSION,
            protocol_extra={
                "count_dominant_document_scoring": True,
                "free_form_rationales_requested": False,
                "refreshed_beliefs_explicitly_fallible": True,
                "zero_padding_parser_amendment": True,
                "semantic_prompt_identical_to_v3": True,
                "openrouter_request_seed": config.mediq_seed,
            },
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
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
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
