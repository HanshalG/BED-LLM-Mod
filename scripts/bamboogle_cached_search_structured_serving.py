#!/usr/bin/env python3
"""Gate strict-schema serving before Bamboogle cached-search mechanics v2."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.bamboogle_cached_search_mechanics import (
    MAX_NEW_TOKENS,
    MECHANICS_IDS,
    TEMPERATURE,
    _build_model,
    _checkpoint,
    _usage_snapshot,
    initial_messages,
    initial_response_format,
    load_visible_tasks,
    parse_initial,
)
from scripts.bamboogle_semantic_bed_manifest import sha256_file


INTERFACE_VERSION = "bamboogle-cached-search-structured-serving-1"
EXPECTED_REQUESTS = len(MECHANICS_IDS)
MAX_COST_USD = 0.15


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def run_serving_gate(
    config: Config,
    *,
    data_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    tasks = load_visible_tasks(data_path)
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "task_ids": list(MECHANICS_IDS),
    }
    try:
        responses = model.chat_complete_messages_batched_structured(
            [initial_messages(task["question"]) for task in tasks],
            temperature=TEMPERATURE,
            block_size=len(tasks),
            response_format=initial_response_format(),
            max_new_tokens=MAX_NEW_TOKENS,
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        if len(responses) != EXPECTED_REQUESTS:
            raise ValueError("wrong structured serving response count")
        parsed = [parse_initial(response) for response in responses]
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    gates = {
        "exact_5_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_5_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_model_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_5_schema_responses_parse": len(parsed) == EXPECTED_REQUESTS,
        "all_beliefs_have_8_hypotheses": all(
            len(belief.hypotheses) == 8 for belief, _, _ in parsed
        ),
        "all_tasks_have_8_distinct_queries": all(
            len(roots) + len(fixed) == 8
            for _, roots, fixed in parsed
        ),
        "cost_at_most_0_15": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "task_ids": list(MECHANICS_IDS),
            "expected_requests": EXPECTED_REQUESTS,
            "response_format": "chat_strict_json_schema",
            "scientific_endpoints_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_retries": 0,
            "max_cost_usd": MAX_COST_USD,
        },
        "summary": {
            "gates": gates,
            "initial_entropy_nats": [
                belief.entropy_nats for belief, _, _ in parsed
            ],
        },
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.04
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_REQUESTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        result = run_serving_gate(
            config,
            data_path=args.data_path,
            raw_path=raw_path,
        )
        result["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        output = args.output_dir / "SERVING_FAILURE.json"
        output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output = args.output_dir / "SERVING.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
