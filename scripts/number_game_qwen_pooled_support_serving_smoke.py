#!/usr/bin/env python3
"""Run the exact-10 Qwen pooled-support serving gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import (
    MAX_TOKENS,
    PROJECTED_COST_USD,
    TEMPERATURE,
    serving_cases,
    sha256_file,
    summarize_usage,
)
from scripts.number_game_depth_three_development import (
    proposal_response_format,
    retain_parent_hypotheses,
)
from scripts.number_game_pooled_support import (
    PooledStructuredAdapter,
    parse_pooled_proposals,
)
from scripts.number_game_predictive_risk_replication import _adapter


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-qwen-pooled-support-serving-smoke-1"
MODEL_ID = "qwen/qwen3.7-plus"
MODEL_SEEDS = (63_000, 63_001)
HISTORY_INDICES = (0, 2, 3, 6, 7)
EXPECTED_HISTORIES = len(HISTORY_INDICES)
EXPECTED_REQUESTS = EXPECTED_HISTORIES * len(MODEL_SEEDS)
CONCURRENCY_PER_DRAW = EXPECTED_HISTORIES
RUN_BUDGET_USD = 0.12


def build_chain_supports(
    generated: Sequence[list[Any]],
) -> tuple[list[list[Any]], list[list[Any]], list[dict[str, Any]]]:
    if len(generated) != EXPECTED_HISTORIES:
        raise ValueError("pooled smoke requires exactly five histories")
    first = []
    retention = []
    for child_index, label in ((1, True), (2, False)):
        merged, diagnostic = retain_parent_hypotheses(
            parent_support=generated[0],
            generated_support=generated[child_index],
            query=10,
            label=label,
        )
        first.append(merged)
        retention.append(diagnostic)
    second = []
    for parent_index, child_index, label in (
        (0, 3, False),
        (1, 4, True),
    ):
        merged, diagnostic = retain_parent_hypotheses(
            parent_support=first[parent_index],
            generated_support=generated[child_index],
            query=20,
            label=label,
        )
        second.append(merged)
        retention.append(diagnostic)
    return first, second, retention


def serving_gates(
    *,
    generated,
    diagnostics,
    first,
    second,
    usage,
) -> dict[str, bool]:
    gates = {
        "exact_five_pooled_histories": (
            len(generated) == EXPECTED_HISTORIES
        ),
        "exact_ten_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_ten_http_attempts": (
            usage["http_attempts"] == EXPECTED_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": (
            usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_smoke_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
        "all_ten_draws_are_strict_json": all(
            draw["codec_mode"] == "strict_json"
            for diagnostic in diagnostics
            for draw in diagnostic["draw_diagnostics"]
        ),
        "each_second_draw_adds_at_least_two_extensions": all(
            diagnostic["draw_novel_contributions"][1] >= 2
            for diagnostic in diagnostics
        ),
        "pooled_initial_has_at_least_twenty_four": len(generated[0]) >= 24,
        "every_pooled_conditioned_support_has_at_least_eight": all(
            len(support) >= 8 for support in generated[1:]
        ),
        "both_merged_first_supports_have_at_least_twelve": all(
            len(support) >= 12 for support in first
        ),
        "both_merged_second_supports_have_at_least_eight": all(
            len(support) >= 8 for support in second
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    adapter: PooledStructuredAdapter | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = output_dir / "private"
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    if adapter is None:
        adapters = [
            _adapter(
                model=MODEL_ID,
                run_id=run_id,
                output_dir=output_dir,
                request_seed=seed,
                concurrency=CONCURRENCY_PER_DRAW,
                projected_cost=PROJECTED_COST_USD,
                run_budget_usd=RUN_BUDGET_USD,
            )
            for seed in MODEL_SEEDS
        ]
        adapter = PooledStructuredAdapter(adapters)
    all_cases = serving_cases()
    cases = [all_cases[index] for index in HISTORY_INDICES]
    responses = adapter.chat_complete_messages_batched_structured(
        [case["messages"] for case in cases],
        temperature=TEMPERATURE,
        block_size=EXPECTED_HISTORIES,
        response_format=proposal_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    checkpoint(raw_path, {"responses": responses})

    generated = []
    diagnostics = []
    for case, response in zip(cases, responses, strict=True):
        support, diagnostic = parse_pooled_proposals(
            response,
            observations=case["observations"],
        )
        generated.append(support)
        diagnostics.append(diagnostic)
    first, second, retention = build_chain_supports(generated)
    usage = summarize_usage(adapter.usage_snapshot())
    gates = serving_gates(
        generated=generated,
        diagnostics=diagnostics,
        first=first,
        second=second,
        usage=usage,
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gated_null",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "model_seeds": list(MODEL_SEEDS),
            "temperature": TEMPERATURE,
            "reasoning": False,
            "pool_size": 2,
            "expected_histories": EXPECTED_HISTORIES,
            "expected_requests": EXPECTED_REQUESTS,
            "support_update": "retained_rejuvenation",
            "semantic_repair": False,
            "efficacy_used_for_authorization": False,
            "run_budget_usd": RUN_BUDGET_USD,
        },
        "usage": usage,
        "gates": gates,
        "pooled_valid_counts": [len(support) for support in generated],
        "draw_valid_counts": [
            diagnostic["draw_valid_counts"] for diagnostic in diagnostics
        ],
        "draw_novel_contributions": [
            diagnostic["draw_novel_contributions"]
            for diagnostic in diagnostics
        ],
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
