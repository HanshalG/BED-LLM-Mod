#!/usr/bin/env python3
"""Verify provider-enforced executable Zendo outputs on already-open rules."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.zendo_final_readiness_belief_smoke import (
    MAX_NEW_TOKENS,
    _build_model,
    _checkpoint,
    _usage_snapshot,
    parse_particle_population,
)
from scripts.zendo_path_dependent_belief_gate import (
    MODEL_ID,
    RULE_ORDER,
    SOURCE_COMMIT,
    _canonical_json,
    _sha256,
    initial_messages,
    raw_official_scene,
    verify_source,
)
from scripts.zendo_structured_outputs import zendo_particle_response_format


INTERFACE_VERSION = "zendo-structured-output-serving-1"
TASKS = ("zeta", "phi", "mu")
EXPECTED_REQUESTS = len(TASKS)
COST_CAP_USD = 0.15
PROJECTED_COST_USD = 0.08


class ServingExecutionError(RuntimeError):
    """A failed-closed serving smoke with usage accounting."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def run_smoke(
    config: Config,
    *,
    source_dir: Path,
    raw_checkpoint_path: Path,
    model_adapter: Any | None = None,
    interface_version: str = INTERFACE_VERSION,
    use_responses_api: bool = False,
) -> dict[str, Any]:
    cases_path = verify_source(source_dir)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    case_by_name = dict(zip(RULE_ORDER, cases, strict=True))
    scenes = {
        name: raw_official_scene(case_by_name[name]["t"][0])
        for name in TASKS
    }
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "responses": {},
    }
    try:
        complete = (
            model.responses_complete_messages_batched_structured
            if use_responses_api
            else model.chat_complete_messages_batched_structured
        )
        responses = complete(
            [initial_messages(scenes[name]) for name in TASKS],
            temperature=0.0,
            block_size=config.batched_block_size,
            response_format=zendo_particle_response_format(),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        raw["responses"] = dict(zip(TASKS, responses, strict=True))
        _checkpoint(raw_checkpoint_path, raw)
        populations = {
            name: parse_particle_population(
                response, allow_duplicate_asts=True
            )
            for name, response in raw["responses"].items()
        }
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_checkpoint_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc
    unique_counts = {
        name: len(
            {
                _canonical_json(hypothesis["rule"])
                for hypothesis in population
            }
        )
        for name, population in populations.items()
    }
    generator = usage["generator"]
    gates = {
        "exact_3_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_3_http_attempts": (
            generator["http_attempts"] == EXPECTED_REQUESTS
        ),
        "zero_transport_retries": generator["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": generator["forced_exits"] == 0,
        "all_three_schema_responses_parse": len(populations) == len(TASKS),
        "all_populations_have_eight_unique_asts": all(
            count >= 8 for count in unique_counts.values()
        ),
        "cost_at_most_0_15": usage["adapter_cost_usd"] <= COST_CAP_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "tasks": list(TASKS),
            "model": MODEL_ID,
            "source_commit": SOURCE_COMMIT,
            "expected_physical_requests": EXPECTED_REQUESTS,
            "response_format": (
                "responses_text_strict_json_schema"
                if use_responses_api
                else "chat_strict_json_schema"
            ),
            "scientific_endpoints_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_retries": 0,
        },
        "summary": {
            "gates": gates,
            "unique_ast_counts": unique_counts,
        },
        "usage": usage,
    }


def run_cli(
    *,
    interface_version: str = INTERFACE_VERSION,
    use_responses_api: bool = False,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = COST_CAP_USD
    config.openrouter_concurrency = 3
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_smoke(
            config,
            source_dir=args.source_dir,
            raw_checkpoint_path=raw_path,
            interface_version=interface_version,
            use_responses_api=use_responses_api,
        )
        payload["protocol"]["private_raw_sha256"] = _sha256(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": interface_version,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256(raw_path)
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / "SMOKE.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    run_cli()
