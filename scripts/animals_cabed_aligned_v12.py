#!/usr/bin/env python3
"""Aligned-response CA-BED shared-tree ranking gate for Animals."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.animals import AnimalsBEDEnvironment
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.animals_cabed_openrouter_v11 import (
    BatchedSemanticOpenRouterModel,
    FORMAL_RUN_CAP_USD,
    SMOKE_RUN_CAP_USD,
)
from scripts.animals_cabed_shared_tree_v10 import (
    FORMAL_TARGETS,
    SMOKE_TARGETS,
    run_state,
    summarize_records,
)


SCHEMA_VERSION = 12
SELECTION_SEED = 24310


class AlignedSemanticAnimalsEnvironment(AnimalsBEDEnvironment):
    """Realize the hidden target's label from the cached semantic table."""

    def observe(self, action: str, hidden_state: str, rng: Any) -> str:
        del rng
        key = (
            hidden_state.strip().casefold(),
            action.strip().casefold(),
        )
        if key not in self._likelihood_cache:
            raise ValueError(
                "aligned observation requires a precomputed target-blind "
                "semantic table"
            )
        yes_probability, no_probability = self._likelihood_cache[key]
        if math.isclose(
            yes_probability,
            no_probability,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        ):
            raise ValueError("aligned semantic label cannot be tied")
        return "Yes" if yes_probability > no_probability else "No"


def run_aligned_stage(
    config: Config,
    *,
    stage: str,
    model: BatchedSemanticOpenRouterModel,
    environment_cls: type[AlignedSemanticAnimalsEnvironment] = (
        AlignedSemanticAnimalsEnvironment
    ),
) -> dict[str, Any]:
    if stage not in {"serving_smoke", "formal"}:
        raise ValueError("stage must be serving_smoke or formal")
    targets = SMOKE_TARGETS if stage == "serving_smoke" else FORMAL_TARGETS
    support = tuple(config.animals[config.version])
    if len(support) != 64 or len(set(support)) != 64:
        raise ValueError("V12 requires the frozen 64-animal support")
    if not set(targets).issubset(support):
        raise ValueError("all V12 targets must be in the fixed support")

    env = environment_cls(
        config=config,
        answerer=model,
        target_animals=list(targets),
    )
    records = [
        run_state(
            env,
            model,
            config,
            state_index=state_index,
            target=target,
        )
        for state_index, target in enumerate(targets)
    ]
    summary = summarize_records(
        records,
        formal=stage == "formal",
    )
    likelihood_cache_valid = all(
        math.isfinite(yes_probability)
        and math.isfinite(no_probability)
        and 0.0 < yes_probability < 1.0
        and 0.0 < no_probability < 1.0
        and math.isclose(
            yes_probability + no_probability,
            1.0,
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
        for yes_probability, no_probability in env._likelihood_cache.values()
    )
    usage = model.usage_snapshot()
    summary["gates"][
        "all_cached_likelihoods_finite_and_complementary"
    ] = likelihood_cache_valid
    summary["gates"]["zero_reasoning_tokens"] = (
        int(usage["adapter_reasoning_tokens"]) == 0
    )
    summary["gates"]["all_pass"] = all(summary["gates"].values())
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "interface": "aligned_target_blind_semantic_table",
            "support_size": len(support),
            "targets": list(targets),
            "root_width": 4,
            "followup_width": 3,
            "likelihood_confidence": config.animals_likelihood_confidence,
            "semantic_raw_probabilities": [1.0, 0.0],
            "semantic_probability_after_confidence_smoothing": [0.85, 0.15],
            "semantic_batches_are_target_blind": True,
            "realized_answer_source": "cached_hidden_target_semantic_label",
            "independent_answer_llm_calls": False,
            "shared_tree_across_controls": True,
            "raw_reasoning_requested": False,
        },
        "summary": summary,
        "records": records,
        "semantic_classifications": model.classification_records,
        "likelihood_cache_entries": len(env._likelihood_cache),
        "usage": usage,
    }


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
    model = BatchedSemanticOpenRouterModel(delegate, config)
    try:
        payload = run_aligned_stage(
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
