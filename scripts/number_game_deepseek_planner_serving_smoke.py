#!/usr/bin/env python3
"""Run the exact-10 DeepSeek Number Game planner serving gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_depth_three_development import (
    MAX_TOKENS,
    TEMPERATURE,
    history_messages,
    initial_messages,
    parse_proposals,
    proposal_response_format,
)
from scripts.number_game_predictive_risk_replication import _adapter


SCHEMA_VERSION = 1
INTERFACE_VERSION = "number-game-deepseek-planner-serving-smoke-1"
MODEL_ID = "deepseek/deepseek-v4-pro"
MODEL_SEED = 42_000
EXPECTED_REQUESTS = 10
CONCURRENCY = 10
RUN_BUDGET_USD = 0.10
PROJECTED_COST_USD = 0.05
MIN_INITIAL_VALID = 16
MIN_CONDITIONED_VALID = 8


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


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
            "messages": (
                initial_messages()
                if not history
                else history_messages(history, enforce_constraints=True)
            ),
        }
        for index, history in enumerate(observations)
    ]


def summarize_usage(snapshot: dict[str, Any]) -> dict[str, Any]:
    return {
        "adapter_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "provider_error_retries": int(
            snapshot.get("provider_error_retries", 0)
        ),
        "adapter_reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0)
        ),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "run_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "prompt_tokens": int(snapshot.get("adapter_prompt_tokens", 0)),
        "completion_tokens": int(
            snapshot.get("adapter_completion_tokens", 0)
        ),
    }


def serving_gates(
    *,
    diagnostics: Sequence[dict[str, Any]],
    cases: Sequence[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, bool]:
    initial = [
        diagnostic
        for diagnostic, case in zip(diagnostics, cases, strict=True)
        if not case["observations"]
    ]
    conditioned = [
        diagnostic
        for diagnostic, case in zip(diagnostics, cases, strict=True)
        if case["observations"]
    ]
    gates = {
        "exact_ten_cases": len(cases) == EXPECTED_REQUESTS,
        "exact_ten_parsed_responses": len(diagnostics) == EXPECTED_REQUESTS,
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
            len(initial) == 2
            and all(
                item["valid_unique_count"] >= MIN_INITIAL_VALID
                for item in initial
            )
        ),
        "all_conditioned_supports_have_at_least_eight_valid": (
            len(conditioned) == 8
            and all(
                item["valid_unique_count"] >= MIN_CONDITIONED_VALID
                for item in conditioned
            )
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
    diagnostics = []
    public_cases = []
    for case, response in zip(cases, responses, strict=True):
        hypotheses, diagnostic = parse_proposals(
            response,
            observations=case["observations"],
        )
        diagnostics.append(diagnostic)
        public_cases.append(
            {
                "case_index": case["case_index"],
                "observations": [
                    [number, label]
                    for number, label in case["observations"]
                ],
                "valid_unique_count": len(hypotheses),
                "diagnostics": diagnostic,
                "extension_sha256s": [
                    hypothesis.public_dict()["extension_sha256"]
                    for hypothesis in hypotheses
                ],
            }
        )
    usage = summarize_usage(adapter.usage_snapshot())
    gates = serving_gates(
        diagnostics=diagnostics,
        cases=cases,
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
                "one_observation": 4,
                "two_observations": 4,
            },
            "efficacy_used_for_authorization": False,
            "run_budget_usd": RUN_BUDGET_USD,
        },
        "usage": usage,
        "gates": gates,
        "cases": public_cases,
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
