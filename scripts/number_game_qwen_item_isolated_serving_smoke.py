#!/usr/bin/env python3
"""Run the exact-10 Qwen item-isolated Number Game serving gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import (
    EXPECTED_REQUESTS,
    MAX_TOKENS,
    PROJECTED_COST_USD,
    RUN_BUDGET_USD,
    TEMPERATURE,
    serving_cases,
    sha256_file,
    summarize_usage,
)
from scripts.number_game_depth_three_development import (
    proposal_response_format,
)
from scripts.number_game_gptmini_planner_serving_smoke_v2 import (
    build_linked_supports,
    serving_gates,
)
from scripts.number_game_item_isolated_codec import (
    parse_proposals_item_isolated,
)
from scripts.number_game_predictive_risk_replication import _adapter


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-item-isolated-serving-smoke-1"
MODEL_ID = "qwen/qwen3.7-plus"
MODEL_SEED = 62_000
CONCURRENCY = 10


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    adapter = adapter or _adapter(
        model=MODEL_ID,
        run_id=run_id,
        output_dir=output_dir,
        request_seed=MODEL_SEED,
        concurrency=CONCURRENCY,
        projected_cost=PROJECTED_COST_USD,
        run_budget_usd=RUN_BUDGET_USD,
    )
    cases = serving_cases()
    responses = adapter.chat_complete_messages_batched_structured(
        [case["messages"] for case in cases],
        temperature=TEMPERATURE,
        block_size=EXPECTED_REQUESTS,
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    checkpoint(raw_path, {"responses": responses})

    generated = []
    diagnostics = []
    for case, response in zip(cases, responses, strict=True):
        support, diagnostic = parse_proposals_item_isolated(
            response,
            observations=case["observations"],
        )
        generated.append(support)
        diagnostics.append(diagnostic)
    first, second, retention = build_linked_supports(generated)
    usage = summarize_usage(adapter.usage_snapshot())
    gates = serving_gates(
        generated=generated,
        first=first,
        second=second,
        usage=usage,
    )
    gates["all_ten_live_responses_are_strict_json"] = all(
        diagnostic["codec_mode"] == "strict_json"
        for diagnostic in diagnostics
    )
    gates["all_pass"] = all(
        value for key, value in gates.items() if key != "all_pass"
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "model_seed": MODEL_SEED,
            "temperature": TEMPERATURE,
            "reasoning": False,
            "expected_requests": EXPECTED_REQUESTS,
            "prompt_mix": {
                "initial": 2,
                "linked_one_observation": 4,
                "linked_two_observations": 4,
            },
            "support_update": "retained_rejuvenation",
            "codec": "strict_json_with_complete_item_salvage",
            "semantic_repair": False,
            "efficacy_used_for_authorization": False,
            "run_budget_usd": RUN_BUDGET_USD,
        },
        "usage": usage,
        "gates": gates,
        "generated_valid_counts": [len(support) for support in generated],
        "merged_first_valid_counts": [len(support) for support in first],
        "merged_second_valid_counts": [len(support) for support in second],
        "diagnostics": diagnostics,
        "retention_diagnostics": retention,
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty: {args.output_dir}"
        )
    try:
        result = run_smoke(
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
    except Exception as exc:
        checkpoint(
            args.output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
        )
        raise
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
