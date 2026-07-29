#!/usr/bin/env python3
"""Run a linked exact-10 GPT-Mini retained-rejuvenation serving gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import (
    sha256_file,
    summarize_usage,
)
from scripts.number_game_depth_three_development import (
    MAX_TOKENS,
    TEMPERATURE,
    initial_messages,
    parse_proposals,
    proposal_response_format,
    retain_parent_hypotheses,
)
from scripts.number_game_generator_aware_bed import NUM_PROPOSALS, _GRAMMAR
from scripts.number_game_predictive_risk_replication import _adapter


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-gptmini-planner-serving-smoke-v2-1"
MODEL_ID = "openai/gpt-5.4-mini"
MODEL_SEED = 53_100
EXPECTED_REQUESTS = 10
CONCURRENCY = 10
RUN_BUDGET_USD = 0.10
PROJECTED_COST_USD = 0.05
MIN_INITIAL_VALID = 16
MIN_GENERATED_CONDITIONED_VALID = 4
MIN_MERGED_FIRST_VALID = 8
MIN_MERGED_SECOND_VALID = 4


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


def verified_history_messages(
    observations: Sequence[tuple[int, bool]],
    *,
    enforce_constraints: bool = True,
) -> list[dict[str, str]]:
    del enforce_constraints
    if not observations:
        return initial_messages()
    observation_lines = "\n".join(
        f"- Is {number} in the concept? {'YES' if label else 'NO'}."
        for number, label in observations
    )
    substitutions = "\n".join(
        f"- Substitute n={number}: the expression MUST evaluate to {label}."
        for number, label in observations
    )
    return [
        {
            "role": "system",
            "content": (
                "You rejuvenate executable hypotheses after an active Number "
                "Game query. Semantic constraint checking is mandatory. "
                "Return only the requested strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                f"The observations so far are:\n{observation_lines}\n\n"
                "Use this procedure privately for every candidate:\n"
                "1. Write one coherent general-rule expression.\n"
                "2. Literally substitute every observed integer into it.\n"
                "3. Reject the candidate if even one result differs from the "
                "required Boolean below.\n"
                "4. Continue until exactly "
                f"{NUM_PROPOSALS} distinct candidates have passed.\n\n"
                f"Required substitutions:\n{substitutions}\n\n"
                "Do not return a plausible near miss. In particular, a rule "
                "that includes a number labelled NO or excludes a number "
                "labelled YES must be replaced before emitting JSON. Generate "
                "coherent rules rather than adding observation-specific "
                f"exceptions.\n\n{_GRAMMAR}"
            ),
        },
    ]


def serving_cases() -> list[dict[str, Any]]:
    observations: tuple[tuple[tuple[int, bool], ...], ...] = (
        (),
        (),
        ((10, True),),
        ((10, False),),
        ((42, True),),
        ((42, False),),
        ((10, True), (20, False)),
        ((10, False), (20, True)),
        ((42, True), (75, True)),
        ((42, False), (75, False)),
    )
    return [
        {
            "case_index": index,
            "observations": history,
            "messages": verified_history_messages(history),
        }
        for index, history in enumerate(observations)
    ]


def build_linked_supports(
    generated: Sequence[list[Any]],
) -> tuple[list[list[Any]], list[list[Any]], list[dict[str, Any]]]:
    if len(generated) != EXPECTED_REQUESTS:
        raise ValueError("linked smoke requires exactly ten generated supports")
    first_specs = (
        (0, 2, 10, True),
        (0, 3, 10, False),
        (1, 4, 42, True),
        (1, 5, 42, False),
    )
    first = []
    retention = []
    for parent_index, child_index, query, label in first_specs:
        merged, diagnostic = retain_parent_hypotheses(
            parent_support=generated[parent_index],
            generated_support=generated[child_index],
            query=query,
            label=label,
        )
        first.append(merged)
        retention.append(diagnostic)
    second_specs = (
        (0, 6, 20, False),
        (1, 7, 20, True),
        (2, 8, 75, True),
        (3, 9, 75, False),
    )
    second = []
    for parent_index, child_index, query, label in second_specs:
        merged, diagnostic = retain_parent_hypotheses(
            parent_support=first[parent_index],
            generated_support=generated[child_index],
            query=query,
            label=label,
        )
        second.append(merged)
        retention.append(diagnostic)
    return first, second, retention


def serving_gates(
    *,
    generated: Sequence[list[Any]],
    first: Sequence[list[Any]],
    second: Sequence[list[Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    gates = {
        "exact_ten_parsed_responses": len(generated) == EXPECTED_REQUESTS,
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
        "both_initial_supports_have_at_least_sixteen_valid": (
            all(len(support) >= MIN_INITIAL_VALID for support in generated[:2])
        ),
        "every_conditioned_draw_adds_at_least_four_valid": all(
            len(support) >= MIN_GENERATED_CONDITIONED_VALID
            for support in generated[2:]
        ),
        "all_merged_first_supports_have_at_least_eight": all(
            len(support) >= MIN_MERGED_FIRST_VALID for support in first
        ),
        "all_merged_second_supports_have_at_least_four": all(
            len(support) >= MIN_MERGED_SECOND_VALID for support in second
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


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
        support, diagnostic = parse_proposals(
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
