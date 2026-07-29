#!/usr/bin/env python3
"""Run the preregistered ICAE-Bench exact-10 semantic serving smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Protocol

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.icae_bench_source_opportunity_audit import (
    content_tokens,
    sha256_bytes,
)
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "icae-bench-semantic-serving-smoke-1"
EXPECTED_MANIFEST_SHA256 = (
    "47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f"
)
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "f63a313adb8bc3bd1090fe4542d1b38b4c2e63abca30c559447a70a6e6348552"
)
SELECTION_SEED = 50_100
PLANNER_SEED = 50_200
ORACLE_SEED = 50_300
PLANNER_MODEL_ID = "openai/gpt-5.4"
ORACLE_MODEL_ID = "google/gemini-3.1-flash-lite"
NUM_TASKS = 2
NUM_HYPOTHESES = 12
NUM_QUESTIONS = 6
EXPECTED_REQUESTS = 10
PLANNER_MAX_TOKENS = 3000
ORACLE_MAX_TOKENS = 1200
CONCURRENCY = 8
PROJECTED_COST_USD = 0.10
RUN_BUDGET_USD = 0.25
MIN_CHANGED_FOLLOWUPS = 4
MIN_ANSWER_TERMS_IN_FOLLOWUPS = 1
GENERIC_UNMATCHED_QUESTION = (
    "What catering menu should the team order for next Thursday?"
)


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


class ServingModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def canonical_text(value: str) -> str:
    return " ".join(value.casefold().split())


def planner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_semantic_support",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "questions"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": NUM_HYPOTHESES,
                        "maxItems": NUM_HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["requirement", "weight"],
                            "properties": {
                                "requirement": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 500,
                                },
                                "weight": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                            },
                        },
                    },
                    "questions": {
                        "type": "array",
                        "minItems": NUM_QUESTIONS,
                        "maxItems": NUM_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["question", "rationale"],
                            "properties": {
                                "question": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 500,
                                },
                                "rationale": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 500,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def parse_support(response: str, *, label: str) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != {"hypotheses", "questions"}:
        raise ValueError(f"{label} has unexpected fields")
    hypotheses = value["hypotheses"]
    questions = value["questions"]
    if not isinstance(hypotheses, list) or len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError(f"{label} must contain 12 hypotheses")
    if not isinstance(questions, list) or len(questions) != NUM_QUESTIONS:
        raise ValueError(f"{label} must contain 6 questions")

    parsed_hypotheses = []
    for index, row in enumerate(hypotheses):
        if not isinstance(row, dict) or set(row) != {"requirement", "weight"}:
            raise ValueError(f"{label}.hypotheses[{index}] has wrong fields")
        requirement = " ".join(str(row["requirement"]).split())
        weight = row["weight"]
        if not 10 <= len(requirement) <= 500:
            raise ValueError(f"{label}.hypotheses[{index}] has invalid text")
        if not isinstance(weight, int) or isinstance(weight, bool) or not 1 <= weight <= 100:
            raise ValueError(f"{label}.hypotheses[{index}] has invalid weight")
        parsed_hypotheses.append(
            {"requirement": requirement, "weight": weight}
        )
    if len(
        {canonical_text(row["requirement"]) for row in parsed_hypotheses}
    ) != NUM_HYPOTHESES:
        raise ValueError(f"{label} contains duplicate hypotheses")

    parsed_questions = []
    for index, row in enumerate(questions):
        if not isinstance(row, dict) or set(row) != {"question", "rationale"}:
            raise ValueError(f"{label}.questions[{index}] has wrong fields")
        question = " ".join(str(row["question"]).split())
        rationale = " ".join(str(row["rationale"]).split())
        if not 10 <= len(question) <= 500 or not question.endswith("?"):
            raise ValueError(f"{label}.questions[{index}] is not a question")
        if not 10 <= len(rationale) <= 500:
            raise ValueError(f"{label}.questions[{index}] has invalid rationale")
        parsed_questions.append(
            {"question": question, "rationale": rationale}
        )
    if len(
        {canonical_text(row["question"]) for row in parsed_questions}
    ) != NUM_QUESTIONS:
        raise ValueError(f"{label} contains duplicate questions")
    return {
        "hypotheses": parsed_hypotheses,
        "questions": parsed_questions,
    }


def parse_oracle(response: str, *, label: str) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != {"_internal_log", "reply"}:
        raise ValueError(f"{label} has unexpected fields")
    if not isinstance(value["reply"], str) or not value["reply"].strip():
        raise ValueError(f"{label}.reply is empty")
    internal = value["_internal_log"]
    if not isinstance(internal, dict):
        raise ValueError(f"{label}._internal_log is not an object")
    triggers = internal.get("triggers_hit")
    if not isinstance(triggers, list) or not all(
        isinstance(item, str) for item in triggers
    ):
        raise ValueError(f"{label}.triggers_hit is invalid")
    for key in (
        "api_alignment_triggered",
        "fallback_triggered",
        "cheating_attempt_detected",
    ):
        if not isinstance(internal.get(key), bool):
            raise ValueError(f"{label}.{key} is invalid")
    score = internal.get("score_adjustment")
    if not isinstance(score, int) or isinstance(score, bool):
        raise ValueError(f"{label}.score_adjustment is invalid")
    return value


def select_mechanics_tasks(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    mechanics = manifest["partitions"]["mechanics"]
    return sorted(
        mechanics,
        key=lambda row: (
            hashlib.sha256(
                f"{SELECTION_SEED}:{row['alias']}".encode("utf-8")
            ).hexdigest(),
            row["alias"],
        ),
    )[:NUM_TASKS]


def initial_messages(fuzzy_prd: str) -> list[dict[str, str]]:
    request = {
        "task": (
            "Construct a semantic belief support over requirements omitted "
            "from a deliberately fuzzy software PRD, then propose clarification "
            "questions that could distinguish consequential alternatives."
        ),
        "fuzzy_prd": fuzzy_prd,
        "requirements": [
            "Return exactly 12 distinct plausible hidden requirements.",
            "Assign each a positive integer relative weight from 1 to 100.",
            "Return exactly 6 distinct, direct clarification questions.",
            "Questions must ask about task-specific behavior, interfaces, edge cases, or output contracts.",
            "Do not ask for source code, test files, a complete checklist, or all hidden requirements.",
            "Make each question answerable by a strict product owner in one short reply.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You design information-gathering experiments for ambiguous "
                "software requirements. Follow the supplied schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def followup_messages(
    fuzzy_prd: str,
    initial_support: dict[str, Any],
    question: str,
    answer: str,
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Regenerate the semantic requirement support and candidate "
            "clarification questions after one real product-owner answer."
        ),
        "fuzzy_prd": fuzzy_prd,
        "previous_support": initial_support,
        "history": [{"question": question, "answer": answer}],
        "requirements": [
            "Return exactly 12 distinct plausible hidden requirements.",
            "Return exactly 6 distinct direct clarification questions.",
            "Use concrete concepts introduced by the answer when they expose useful unresolved details.",
            "Do not repeat already answered questions.",
            "Do not ask for source code, test files, a complete checklist, or all hidden requirements.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You update experimental-design beliefs after observations. "
                "Follow the supplied schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def oracle_system_prompt(repo: Path, record: dict[str, Any]) -> str:
    init_prompt = (repo / "user_agent/init.md").read_text(encoding="utf-8")
    injected = json.dumps(record["oracle_data"], ensure_ascii=False, indent=2)
    return (
        f"{init_prompt}\n\n"
        "# Injected oracle_data (your single source of truth)\n"
        f"```json\n{injected}\n```\n"
    )


def oracle_messages(
    repo: Path,
    record: dict[str, Any],
    question: str,
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": oracle_system_prompt(repo, record)},
        {"role": "user", "content": question},
    ]


def usage_summary(models: list[ServingModel]) -> dict[str, Any]:
    snapshots = [model.usage_snapshot() for model in models]
    keys = (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "forced_final_requests",
        "forced_final_successes",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
        "adapter_cost_usd",
    )
    result = {
        key: sum(float(snapshot.get(key, 0) or 0) for snapshot in snapshots)
        for key in keys
    }
    for key in keys:
        if key != "adapter_cost_usd":
            result[key] = int(result[key])
    result["run_cost_usd"] = result.pop("adapter_cost_usd")
    return result


def changed_question_count(
    initial: dict[str, Any],
    followup: dict[str, Any],
) -> int:
    initial_questions = {
        canonical_text(row["question"]) for row in initial["questions"]
    }
    return sum(
        canonical_text(row["question"]) not in initial_questions
        for row in followup["questions"]
    )


def answer_term_count(
    *,
    fuzzy_prd: str,
    initial: dict[str, Any],
    answer: str,
    followup: dict[str, Any],
) -> int:
    initially_visible = content_tokens(fuzzy_prd)
    initially_visible |= content_tokens(
        " ".join(
            row["question"] + " " + row["rationale"]
            for row in initial["questions"]
        )
    )
    introduced = content_tokens(answer) - initially_visible
    followup_tokens = content_tokens(
        " ".join(
            row["question"] + " " + row["rationale"]
            for row in followup["questions"]
        )
    )
    return len(introduced & followup_tokens)


def _adapter(
    *,
    model: str,
    run_id: str,
    output_dir: Path,
    request_seed: int,
    max_tokens: int,
    projected_cost: float,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=300.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=projected_cost,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=max_tokens,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=131072),
        config,
        request_seed=request_seed,
    )


def run_smoke(
    *,
    repo: Path,
    manifest_path: Path,
    source_audit_path: Path,
    output_dir: Path,
    run_id: str,
    planner_model: ServingModel,
    oracle_model: ServingModel,
) -> dict[str, Any]:
    if sha256_bytes(manifest_path.read_bytes()) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen ICAE manifest hash mismatch")
    if sha256_bytes(source_audit_path.read_bytes()) != EXPECTED_SOURCE_AUDIT_SHA256:
        raise ValueError("Frozen ICAE source-audit hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_mechanics_tasks(manifest)
    records = [
        json.loads((repo / row["oracle_record"]).read_text(encoding="utf-8"))
        for row in selected
    ]

    initial_raw = planner_model.chat_complete_messages_batched_structured(
        [initial_messages(record["fuzzy_prd"]) for record in records],
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )
    initial = [
        parse_support(raw, label=f"initial[{index}]")
        for index, raw in enumerate(initial_raw)
    ]
    roots = [support["questions"][0]["question"] for support in initial]

    targeted_messages = [
        oracle_messages(repo, record, root)
        for record, root in zip(records, roots, strict=True)
    ]
    targeted_raw = oracle_model.chat_complete_messages_batched(
        targeted_messages,
        temperature=0.0,
        block_size=NUM_TASKS,
        max_new_tokens=ORACLE_MAX_TOKENS,
    )
    targeted = [
        parse_oracle(raw, label=f"targeted[{index}]")
        for index, raw in enumerate(targeted_raw)
    ]

    repeated_raw = oracle_model.chat_complete_messages_batched(
        targeted_messages,
        temperature=0.0,
        block_size=NUM_TASKS,
        max_new_tokens=ORACLE_MAX_TOKENS,
    )
    repeated = [
        parse_oracle(raw, label=f"repeated[{index}]")
        for index, raw in enumerate(repeated_raw)
    ]

    generic_raw = oracle_model.chat_complete_messages_batched(
        [
            oracle_messages(repo, record, GENERIC_UNMATCHED_QUESTION)
            for record in records
        ],
        temperature=0.0,
        block_size=NUM_TASKS,
        max_new_tokens=ORACLE_MAX_TOKENS,
    )
    generic = [
        parse_oracle(raw, label=f"generic[{index}]")
        for index, raw in enumerate(generic_raw)
    ]

    followup_raw = planner_model.chat_complete_messages_batched_structured(
        [
            followup_messages(
                record["fuzzy_prd"],
                support,
                root,
                oracle["reply"],
            )
            for record, support, root, oracle in zip(
                records, initial, roots, targeted, strict=True
            )
        ],
        temperature=0.0,
        block_size=NUM_TASKS,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )
    followup = [
        parse_support(raw, label=f"followup[{index}]")
        for index, raw in enumerate(followup_raw)
    ]

    task_rows = []
    for row, record, first, target, repeat, generic_reply, refreshed in zip(
        selected,
        records,
        initial,
        targeted,
        repeated,
        generic,
        followup,
        strict=True,
    ):
        changed = changed_question_count(first, refreshed)
        answer_terms = answer_term_count(
            fuzzy_prd=record["fuzzy_prd"],
            initial=first,
            answer=target["reply"],
            followup=refreshed,
        )
        task_rows.append(
            {
                "alias": row["alias"],
                "language": row["language"],
                "initial_hypothesis_count": len(first["hypotheses"]),
                "initial_question_count": len(first["questions"]),
                "targeted_trigger_count": len(
                    target["_internal_log"]["triggers_hit"]
                ),
                "targeted_fallback": target["_internal_log"][
                    "fallback_triggered"
                ],
                "fresh_session_exact_replay": target == repeat,
                "generic_exact_fallback": (
                    generic_reply["reply"]
                    == record["oracle_data"]["fallback_response"]
                    and generic_reply["_internal_log"]["fallback_triggered"]
                    and not generic_reply["_internal_log"]["triggers_hit"]
                ),
                "followup_hypothesis_count": len(refreshed["hypotheses"]),
                "followup_question_count": len(refreshed["questions"]),
                "changed_followup_questions": changed,
                "new_answer_terms_used": answer_terms,
            }
        )

    usage = usage_summary([planner_model, oracle_model])
    gates = {
        "exact_request_count": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": (
            usage["retry_count"] == 0
            and usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_and_forced_exits": (
            usage["adapter_reasoning_tokens"] == 0
            and usage["forced_exits"] == 0
            and usage["forced_final_requests"] == 0
        ),
        "targeted_questions_match_hidden_requirements": all(
            row["targeted_trigger_count"] >= 1
            and not row["targeted_fallback"]
            for row in task_rows
        ),
        "fresh_session_oracle_replay_is_exact": all(
            row["fresh_session_exact_replay"] for row in task_rows
        ),
        "generic_queries_make_no_progress": all(
            row["generic_exact_fallback"] for row in task_rows
        ),
        "history_changes_question_support": all(
            row["changed_followup_questions"] >= MIN_CHANGED_FOLLOWUPS
            for row in task_rows
        ),
        "followups_use_answer_introduced_terms": all(
            row["new_answer_terms_used"] >= MIN_ANSWER_TERMS_IN_FOLLOWUPS
            for row in task_rows
        ),
        "cost_within_cap": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    all_pass = all(gates.values())
    private = {
        "selected": selected,
        "initial_raw": initial_raw,
        "targeted_raw": targeted_raw,
        "repeated_raw": repeated_raw,
        "generic_raw": generic_raw,
        "followup_raw": followup_raw,
    }
    private_path = output_dir / "private" / "RAW_RESPONSES.json"
    checkpoint(private_path, private)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if all_pass else "gated_null",
        "run_id": run_id,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_audit_sha256": EXPECTED_SOURCE_AUDIT_SHA256,
        "models": {
            "planner": PLANNER_MODEL_ID,
            "oracle": ORACLE_MODEL_ID,
        },
        "seeds": {
            "selection": SELECTION_SEED,
            "planner": PLANNER_SEED,
            "oracle": ORACLE_SEED,
        },
        "selected_tasks": [
            {"alias": row["alias"], "language": row["language"]}
            for row in selected
        ],
        "usage": usage,
        "tasks": task_rows,
        "gates": gates,
        "all_pass": all_pass,
        "decision": (
            "authorize_separately_frozen_mechanics_first_link_test"
            if all_pass
            else "close_exact_interface"
        ),
        "private_raw_sha256": sha256_bytes(private_path.read_bytes()),
        "endpoint_accessed": False,
        "development_or_later_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    planner = _adapter(
        model=PLANNER_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=PLANNER_SEED,
        max_tokens=PLANNER_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.8,
    )
    oracle = _adapter(
        model=ORACLE_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=ORACLE_SEED,
        max_tokens=ORACLE_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.2,
    )
    result_path = output_dir / "SERVING.json"
    failure_path = output_dir / "FAILURE.json"
    try:
        payload = run_smoke(
            repo=args.repo.resolve(),
            manifest_path=args.manifest.resolve(),
            source_audit_path=args.source_audit.resolve(),
            output_dir=output_dir,
            run_id=args.run_id,
            planner_model=planner,
            oracle_model=oracle,
        )
        checkpoint(result_path, payload)
    except Exception as exc:
        usage = usage_summary([planner, oracle])
        payload = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "run_id": args.run_id,
            "error": str(exc),
            "usage": usage,
            "endpoint_accessed": False,
            "development_or_later_opened": False,
        }
        checkpoint(failure_path, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
