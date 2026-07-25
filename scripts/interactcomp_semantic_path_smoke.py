#!/usr/bin/env python3
"""Smoke InteractComp auxiliary generation, validation, and classification."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.interactcomp_first_link_opportunity import (
    ChatModel,
    GENERATOR_MODEL_ID,
    Hypothesis,
    PARTICLE_COUNT,
    QUESTION_COUNT,
    _checkpoint,
    _classification_messages,
    parse_classification,
    parse_hypothesis,
    verify_source,
)
from scripts.interactcomp_model_criticism_validation import (
    _auxiliary_messages,
    _semantic_validation_messages,
    parse_semantic_distinctness,
)


INTERFACE_VERSION = "interactcomp-semantic-path-smoke-1"
OPEN_TASK_INDEX = 141
OPEN_TASK_ID = 142
PRIOR_PRIVATE_SHA256 = (
    "691130ba00f29936ba1bb250b84b6e7a0e5692dd62e48c94ec80c5cf3c696c06"
)
PROPOSAL_COUNT = 4
EXPECTED_REQUESTS = 12
MAX_COST_USD = 0.15


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def parse_classification_ascii_whitespace(text: str) -> str:
    compact = text.translate(
        {ord(character): None for character in " \t\r\n"}
    )
    return parse_classification(compact)


def _usage(model: ChatModel) -> dict[str, Any]:
    return model.usage_snapshot()


def _load_open_inputs(
    source_root: Path,
    prior_private_path: Path,
) -> tuple[str, list[Hypothesis], list[str]]:
    _path, rows = verify_source(source_root)
    if int(rows[OPEN_TASK_INDEX]["id"]) != OPEN_TASK_ID:
        raise ValueError("open smoke task changed")
    if hashlib.sha256(prior_private_path.read_bytes()).hexdigest() != (
        PRIOR_PRIVATE_SHA256
    ):
        raise ValueError("prior private artifact hash changed")
    raw = json.loads(prior_private_path.read_text(encoding="utf-8"))
    screen_indices = [int(index) for index in raw["screen_indices"]]
    enrolled_indices = [int(index) for index in raw["enrolled_indices"]]
    screen_offset = screen_indices.index(OPEN_TASK_INDEX)
    enrolled_offset = enrolled_indices.index(OPEN_TASK_INDEX)
    initial = [
        parse_hypothesis(response)
        for response in raw["initial"][
            screen_offset
            * PARTICLE_COUNT : (screen_offset + 1)
            * PARTICLE_COUNT
        ]
    ]
    questions = raw["questions"][
        enrolled_offset
        * QUESTION_COUNT : (enrolled_offset + 1)
        * QUESTION_COUNT
    ]
    # The encrypted source question is needed for proposal generation, not a
    # context or target. Recover it from the prior initial request data is not
    # possible, so decrypt only that source field here.
    from scripts.interactcomp_first_link_opportunity import _decrypted_fields

    question = _decrypted_fields(rows, ("question",))[OPEN_TASK_INDEX][
        "question"
    ]
    return question, initial, questions


def run_smoke(
    config: Config,
    *,
    source_root: Path,
    prior_private_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    question, initial, questions = _load_open_inputs(
        source_root,
        prior_private_path,
    )
    raw: dict[str, Any] = {
        "open_task_index": OPEN_TASK_INDEX,
        "open_task_id": OPEN_TASK_ID,
    }
    try:
        proposal_messages = [
            _auxiliary_messages(
                task_id=OPEN_TASK_ID,
                question=question,
                initial=initial,
                sample_index=sample_index,
            )
            for sample_index in range(PROPOSAL_COUNT)
        ]
        proposal_raw = model.chat_complete_messages_batched(
            proposal_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["proposals"] = proposal_raw
        proposals = [
            parse_hypothesis(response) for response in proposal_raw
        ]

        validation_messages = [
            _semantic_validation_messages(
                current=initial,
                candidate=candidate,
            )
            for candidate in proposals
        ]
        validation_raw = model.chat_complete_messages_batched(
            validation_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=8,
        )
        raw["semantic_validations"] = validation_raw
        decisions = [
            parse_semantic_distinctness(response)
            for response in validation_raw
        ]

        classification_messages = [
            _classification_messages(
                hypothesis=candidate,
                questions=questions,
            )
            for candidate in proposals
        ]
        classification_raw = model.chat_complete_messages_batched(
            classification_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["classifications"] = classification_raw
        classifications = [
            parse_classification_ascii_whitespace(response)
            for response in classification_raw
        ]
        raw["target_free_smoke_complete"] = True
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc
    gates = {
        "exact_request_count": int(usage.get("adapter_requests", 0))
        == EXPECTED_REQUESTS,
        "exact_http_attempt_count": int(usage.get("http_attempts", 0))
        == EXPECTED_REQUESTS,
        "zero_transport_retries": int(usage.get("retry_count", 0)) == 0,
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "four_valid_proposals": len(proposals) == PROPOSAL_COUNT,
        "at_least_two_semantically_distinct": sum(decisions) >= 2,
        "four_valid_compact_classifications": len(classifications)
        == PROPOSAL_COUNT,
        "cost_at_most_0_15": float(usage.get("adapter_cost_usd", 0.0))
        <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "open_task_index": OPEN_TASK_INDEX,
            "open_task_id": OPEN_TASK_ID,
            "prior_private_sha256": PRIOR_PRIVATE_SHA256,
            "model": GENERATOR_MODEL_ID,
            "reasoning_requested": False,
            "expected_requests": EXPECTED_REQUESTS,
            "ascii_whitespace_compaction_only": True,
            "hidden_context_loaded": False,
            "target_answer_loaded": False,
        },
        "metrics": {
            "proposal_count": len(proposals),
            "semantically_distinct_count": sum(decisions),
            "classification_count": len(classifications),
        },
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            if "semantic_candidate" in request:
                responses.append("D")
            elif "questions" in request:
                responses.append("Y N U Y")
            else:
                sample = int(request["outside_sample"])
                responses.append(
                    f"ENTITY: Fixture Alternative {sample}\n"
                    f"PROFILE: Alternative profile {sample}."
                )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != GENERATOR_MODEL_ID:
        raise ValueError("semantic smoke config selects wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--prior-private-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.03
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 12
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_smoke(
            config,
            source_root=args.source_root,
            prior_private_path=args.prior_private_path,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SMOKE_FAILURE.json", failure)
        raise
    output = args.output_dir / "SMOKE.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
