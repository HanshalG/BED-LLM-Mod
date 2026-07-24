#!/usr/bin/env python3
"""Final robust-menu aligned CA-BED ranking gate for Animals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.animals_cabed_aligned_v12 import (
    AlignedSemanticAnimalsEnvironment,
    run_aligned_stage,
)
from scripts.animals_cabed_openrouter_v11 import (
    BatchedSemanticOpenRouterModel,
    FORMAL_RUN_CAP_USD,
    SMOKE_RUN_CAP_USD,
)


SCHEMA_VERSION = 13
CANDIDATE_OVERSAMPLE = 2


class OversampledAlignedAnimalsEnvironment(
    AlignedSemanticAnimalsEnvironment
):
    def generate_candidate_actions(
        self,
        belief_state: Any,
        history: Sequence[tuple[str, str]],
        model: Any,
        config: Config,
    ) -> Sequence[str]:
        requested = int(config.target_num_questions)
        config.target_num_questions = requested + CANDIDATE_OVERSAMPLE
        try:
            return super().generate_candidate_actions(
                belief_state,
                history,
                model,
                config,
            )
        finally:
            config.target_num_questions = requested


class RecordingBatchedSemanticModel(BatchedSemanticOpenRouterModel):
    def __init__(self, delegate: Any, config: Config) -> None:
        super().__init__(delegate, config)
        self.generation_records: list[dict[str, Any]] = []

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        num_responses: int = 1,
    ) -> list[str]:
        output = super().chat_complete(
            messages,
            temperature,
            num_responses=num_responses,
        )
        self.generation_records.append(
            {
                "messages": messages,
                "temperature": temperature,
                "outputs": output,
            }
        )
        return output


def run_v13_stage(
    config: Config,
    *,
    stage: str,
    model: RecordingBatchedSemanticModel,
) -> dict[str, Any]:
    payload = run_aligned_stage(
        config,
        stage=stage,
        model=model,
        environment_cls=OversampledAlignedAnimalsEnvironment,
    )
    payload["schema_version"] = SCHEMA_VERSION
    payload["protocol"].update(
        {
            "candidate_oversample": CANDIDATE_OVERSAMPLE,
            "requested_root_candidates": 6,
            "retained_root_candidates": 4,
            "requested_followup_candidates": 5,
            "retained_followup_candidates": 3,
            "no_menu_regeneration": True,
        }
    )
    payload["generation_records"] = model.generation_records
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = SMOKE_RUN_CAP_USD
    else:
        config.openrouter_projected_cost_usd = 1.00
        config.openrouter_run_budget_usd = FORMAL_RUN_CAP_USD
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
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

    delegate = build_model_adapter(config.model_pairs[0].questioner, config)
    model = RecordingBatchedSemanticModel(delegate, config)
    try:
        payload = run_v13_stage(
            config,
            stage=args.stage,
            model=model,
        )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": model.usage_snapshot(),
            "completed_semantic_batches": model.classification_records,
            "generation_records": model.generation_records,
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
