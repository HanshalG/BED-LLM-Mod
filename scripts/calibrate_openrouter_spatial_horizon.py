"""Endpoint-free spatial horizon-counting calibration for model selection."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, ModelSpec
from openrouter_model import OpenRouterAdapter


CASES = (
    {
        "name": "east",
        "start": [0, 0],
        "fixed_root": "move-EAST",
        "position_after_root": [1, 0],
        "targets": [
            {"inspect_action": "inspect-A", "position": [3, 0]},
            {"inspect_action": "inspect-X", "position": [1, 3]},
        ],
        "expected": ["move-EAST", "move-EAST", "inspect-A"],
    },
    {
        "name": "south",
        "start": [5, 0],
        "fixed_root": "move-SOUTH",
        "position_after_root": [5, 1],
        "targets": [
            {"inspect_action": "inspect-B", "position": [5, 3]},
            {"inspect_action": "inspect-Y", "position": [2, 1]},
        ],
        "expected": ["move-SOUTH", "move-SOUTH", "inspect-B"],
    },
    {
        "name": "west",
        "start": [6, 6],
        "fixed_root": "move-WEST",
        "position_after_root": [5, 6],
        "targets": [
            {"inspect_action": "inspect-C", "position": [3, 6]},
            {"inspect_action": "inspect-Z", "position": [5, 3]},
        ],
        "expected": ["move-WEST", "move-WEST", "inspect-C"],
    },
)


MODEL_SPECS = {
    "gemma31b": ModelSpec(
        model="google/gemma-4-31b-it",
        backend="openrouter",
        thinking=True,
        thinking_max_new_tokens=4096,
        thinking_final_max_new_tokens=256,
        max_model_len=32768,
    ),
    "qwen32b": ModelSpec(
        model="qwen/qwen3-32b",
        backend="openrouter",
        thinking=True,
        thinking_max_new_tokens=4096,
        thinking_final_max_new_tokens=256,
        max_model_len=32768,
    ),
    "gpt54mini": ModelSpec(
        model="openai/gpt-5.4-mini",
        backend="openrouter",
        reasoning_effort="high",
        max_model_len=32768,
    ),
    "gpt54": ModelSpec(
        model="openai/gpt-5.4",
        backend="openrouter",
        reasoning_effort="high",
        max_model_len=32768,
    ),
}


def messages_for_case(case: dict[str, Any]) -> list[dict[str, str]]:
    context = {
        "grid": "x and y range from 0 through 6",
        "start": case["start"],
        "fixed_root": case["fixed_root"],
        "position_after_fixed_root": case["position_after_root"],
        "movement_effects": {
            "move-NORTH": "y decreases by 1",
            "move-EAST": "x increases by 1",
            "move-SOUTH": "y increases by 1",
            "move-WEST": "x decreases by 1",
        },
        "targets": case["targets"],
    }
    return [
        {
            "role": "system",
            "content": "Complete a short grid route. Return exact JSON only.",
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "The fixed root has already consumed action 1 of a four-action plan.",
                    "Return exactly three remaining actions.",
                    "An inspection is useful only when the rover is at that target's exact coordinate.",
                    "Choose a target that can be reached and inspected within the three remaining actions.",
                    'Return exactly {"tail":["action2","action3","action4"]}.',
                    "Do not include reasoning, labels, or extra fields.",
                    "CONTEXT=" + json.dumps(context, separators=(",", ":")),
                ]
            ),
        },
    ]


def parse_tail(response: str) -> list[str] | None:
    normalized = response.strip()
    try:
        payload, _end = json.JSONDecoder().raw_decode(normalized)
    except json.JSONDecodeError:
        return None
    if (
        not isinstance(payload, dict)
        or set(payload) != {"tail"}
        or not isinstance(payload["tail"], list)
        or len(payload["tail"]) != 3
        or not all(isinstance(action, str) for action in payload["tail"])
    ):
        return None
    return payload["tail"]


def run_model(
    model_key: str,
    *,
    output_dir: Path,
) -> dict[str, Any]:
    run_id = f"spatial-horizon-calibration-{model_key}-20260723"
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / f"{model_key}.log",
        openrouter_budget_usd=110.0,
        openrouter_run_budget_usd=1.0,
        openrouter_projected_cost_usd=0.10,
        openrouter_concurrency=3,
        openrouter_max_retries=5,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=120.0,
        openrouter_max_output_tokens=4352,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    adapter = OpenRouterAdapter(MODEL_SPECS[model_key], config)
    responses = adapter.chat_complete_messages_batched(
        [messages_for_case(case) for case in CASES],
        temperature=0.0,
        block_size=3,
        max_new_tokens=None,
    )
    records = []
    for case, response in zip(CASES, responses, strict=True):
        parsed = parse_tail(response)
        records.append(
            {
                "case": case["name"],
                "expected": case["expected"],
                "parsed": parsed,
                "exact": parsed == case["expected"],
                "raw_response": response,
            }
        )
    return {
        "model_key": model_key,
        "model": MODEL_SPECS[model_key].model,
        "endpoint_free": True,
        "passed": all(record["exact"] for record in records),
        "records": records,
        "usage": adapter.usage_snapshot(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=tuple(MODEL_SPECS),
        default=("gemma31b", "qwen32b"),
    )
    parser.add_argument("--model-concurrency", type=int, default=2)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.model_concurrency <= 0:
        parser.error("--model-concurrency must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=args.model_concurrency) as executor:
        results = list(
            executor.map(
                lambda key: run_model(key, output_dir=args.output_dir),
                args.models,
            )
        )
    payload = {
        "schema_version": 1,
        "stage": "endpoint_free_spatial_horizon_model_selection",
        "models": results,
    }
    (args.output_dir / "CALIBRATION.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                row["model_key"]: {
                    "passed": row["passed"],
                    "usage": row["usage"],
                }
                for row in results
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
