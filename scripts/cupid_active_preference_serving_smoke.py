#!/usr/bin/env python3
"""Run the frozen CUPID active-preference planner/target serving smoke."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import sys
from typing import Any, Protocol, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import cupid_active_preference_source_audit as source
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_predictive_risk_replication import (
    SeededStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "cupid-active-preference-serving-smoke-1"
PLANNER_MODEL_ID = "openai/gpt-5.4-mini"
TARGET_MODEL_ID = "google/gemini-2.5-flash"
PLANNER_SEED = 37_500
TARGET_SEED = 37_600
TEMPERATURE = 0.7
NUM_CASES = 5
NUM_QUESTIONS = 6
NUM_HYPOTHESES = 12
EXPECTED_REQUESTS = NUM_CASES * 2
CONCURRENCY = 5
PLANNER_MAX_TOKENS = 6_000
TARGET_MAX_TOKENS = 128
PROJECTED_COST_USD = 0.10
RUN_BUDGET_USD = 0.25
MIN_UNIQUE_HYPOTHESES = 10
MIN_UNIQUE_SIGNATURES = 8
MIN_PARTITION_MINORITY = 2
MIN_MEAN_PARTITION_ENTROPY = 0.45
MIN_EXACT_TARGET_COVERAGE_CASES = 4
MAX_NEAREST_TARGET_HAMMING = 1
MIN_UNIQUE_TARGET_SIGNATURES = 3
BITSTRING_RE = re.compile(r"[01]{6}")


class StructuredModel(Protocol):
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


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def planner_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cupid_preference_interview",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["questions", "hypotheses"],
                "properties": {
                    "questions": {
                        "type": "array",
                        "minItems": NUM_QUESTIONS,
                        "maxItems": NUM_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["id", "question"],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 3,
                                },
                                "question": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 360,
                                },
                            },
                        },
                    },
                    "hypotheses": {
                        "type": "array",
                        "minItems": NUM_HYPOTHESES,
                        "maxItems": NUM_HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "id",
                                "preference",
                                "answer_signature",
                            ],
                            "properties": {
                                "id": {
                                    "type": "string",
                                    "minLength": 2,
                                    "maxLength": 3,
                                },
                                "preference": {
                                    "type": "string",
                                    "minLength": 5,
                                    "maxLength": 600,
                                },
                                "answer_signature": {
                                    "type": "string",
                                    "minLength": NUM_QUESTIONS,
                                    "maxLength": NUM_QUESTIONS,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def target_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "cupid_preference_answers",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["answer_signature"],
                "properties": {
                    "answer_signature": {
                        "type": "string",
                        "minLength": NUM_QUESTIONS,
                        "maxLength": NUM_QUESTIONS,
                    }
                },
            },
        },
    }


def canonical_text(value: str) -> str:
    return " ".join(value.casefold().split())


def strict_json_object(response: str, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(response)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label} is not exact JSON: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} is not a JSON object")
    return value


def parse_planner_response(response: str) -> dict[str, Any]:
    value = strict_json_object(response, label="planner response")
    if set(value) != {"questions", "hypotheses"}:
        raise ValueError("planner response has the wrong top-level fields")

    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != NUM_QUESTIONS:
        raise ValueError(f"planner must return exactly {NUM_QUESTIONS} questions")
    parsed_questions = []
    for index, item in enumerate(questions, start=1):
        if not isinstance(item, dict) or set(item) != {"id", "question"}:
            raise ValueError(f"questions[{index - 1}] has the wrong fields")
        if item["id"] != f"Q{index}":
            raise ValueError("question IDs must be Q1 through Q6 in order")
        question = item["question"]
        if (
            not isinstance(question, str)
            or not 10 <= len(question.strip()) <= 360
            or not question.strip().endswith("?")
        ):
            raise ValueError(f"questions[{index - 1}] is not a valid question")
        parsed_questions.append(question.strip())
    if len({canonical_text(item) for item in parsed_questions}) != NUM_QUESTIONS:
        raise ValueError("planner questions are not unique")

    hypotheses = value["hypotheses"]
    if not isinstance(hypotheses, list) or len(hypotheses) != NUM_HYPOTHESES:
        raise ValueError(
            f"planner must return exactly {NUM_HYPOTHESES} hypotheses"
        )
    parsed_hypotheses = []
    for index, item in enumerate(hypotheses, start=1):
        if not isinstance(item, dict) or set(item) != {
            "id",
            "preference",
            "answer_signature",
        }:
            raise ValueError(f"hypotheses[{index - 1}] has the wrong fields")
        if item["id"] != f"H{index}":
            raise ValueError("hypothesis IDs must be H1 through H12 in order")
        preference = item["preference"]
        signature = item["answer_signature"]
        if (
            not isinstance(preference, str)
            or not 5 <= len(preference.strip()) <= 600
        ):
            raise ValueError(f"hypotheses[{index - 1}] has invalid preference")
        if not isinstance(signature, str) or BITSTRING_RE.fullmatch(signature) is None:
            raise ValueError(f"hypotheses[{index - 1}] has invalid signature")
        parsed_hypotheses.append(
            {
                "id": item["id"],
                "preference": preference.strip(),
                "answer_signature": signature,
            }
        )
    return {
        "questions": parsed_questions,
        "hypotheses": parsed_hypotheses,
    }


def parse_target_response(response: str) -> str:
    value = strict_json_object(response, label="target response")
    if set(value) != {"answer_signature"}:
        raise ValueError("target response has the wrong fields")
    signature = value["answer_signature"]
    if not isinstance(signature, str) or BITSTRING_RE.fullmatch(signature) is None:
        raise ValueError("target answer_signature must be exactly six bits")
    return signature


def planner_messages(row: dict[str, Any]) -> list[dict[str, str]]:
    request = {
        "case_id": source.row_id(row),
        "observation": source.candidate_payload(row),
    }
    return [
        {
            "role": "system",
            "content": (
                "You design an active interview to infer an unobserved contextual "
                "user preference. The hidden preference and evaluation checklist are "
                "not available. From only the current request/context and supplied "
                "background dialogues, produce twelve plausible, mutually contrasting "
                "open-text preference hypotheses for the current context and six "
                "semantically distinct yes/no clarification questions. Each question "
                "must be answerable by the user from their preference, ask one clear "
                "thing, and must not ask the user to select a hypothesis or option. "
                "For every hypothesis, predict its six answers in Q1..Q6 order, using "
                "1 for yes and 0 for no. Design the questions jointly so each splits "
                "the hypotheses and the signatures distinguish competing preferences. "
                "Do not quote or invent an evaluation checklist."
            ),
        },
        {
            "role": "user",
            "content": source.canonical_json(request),
        },
    ]


def target_messages(
    row: dict[str, Any],
    questions: Sequence[str],
) -> list[dict[str, str]]:
    request = {
        "case_id": source.row_id(row),
        "current_request": row["current_request"],
        "current_context_factor": row["current_context_factor"],
        "hidden_contextual_preference": row["current_contextual_preference"],
        "questions": [
            {"id": f"Q{index}", "question": question}
            for index, question in enumerate(questions, start=1)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Simulate the user whose hidden contextual preference is supplied. "
                "Answer each clarification from that preference, not from generic "
                "taste. Use 1 when the preference endorses the behavior in the "
                "question and 0 when it does not. When wording is imperfect, choose "
                "the answer most consistent with the supplied preference; do not "
                "hedge. Return the six bits in Q1..Q6 order."
            ),
        },
        {
            "role": "user",
            "content": source.canonical_json(request),
        },
    ]


def binary_entropy(ones: int, total: int) -> float:
    if ones in {0, total}:
        return 0.0
    probability = ones / total
    return -(
        probability * math.log(probability)
        + (1.0 - probability) * math.log(1.0 - probability)
    )


def hamming(left: str, right: str) -> int:
    if len(left) != len(right):
        raise ValueError("cannot compare signatures with different lengths")
    return sum(a != b for a, b in zip(left, right, strict=True))


def case_metrics(
    *,
    row: dict[str, Any],
    planner: dict[str, Any],
    target_signature: str,
) -> dict[str, Any]:
    signatures = [
        item["answer_signature"] for item in planner["hypotheses"]
    ]
    preferences = [
        canonical_text(item["preference"]) for item in planner["hypotheses"]
    ]
    ones = [
        sum(signature[index] == "1" for signature in signatures)
        for index in range(NUM_QUESTIONS)
    ]
    minorities = [min(count, NUM_HYPOTHESES - count) for count in ones]
    entropies = [
        binary_entropy(count, NUM_HYPOTHESES) for count in ones
    ]
    nearest = min(hamming(target_signature, item) for item in signatures)
    return {
        "id": source.row_id(row),
        "instance_type": row["instance_type"],
        "question_count": len(planner["questions"]),
        "unique_question_count": len(
            {canonical_text(item) for item in planner["questions"]}
        ),
        "hypothesis_count": len(preferences),
        "unique_hypothesis_count": len(set(preferences)),
        "unique_hypothesis_signature_count": len(set(signatures)),
        "partition_minority_counts": minorities,
        "partition_entropies_nats": entropies,
        "mean_partition_entropy_nats": statistics.fmean(entropies),
        "target_signature_exactly_covered": target_signature in signatures,
        "target_signature_nearest_hamming": nearest,
    }


def aggregate_gates(
    *,
    cases: Sequence[dict[str, Any]],
    target_signatures: Sequence[str],
    usage: dict[str, Any],
) -> dict[str, bool]:
    exact_covered = sum(
        case["target_signature_exactly_covered"] for case in cases
    )
    gates = {
        "exact_10_accepted_requests": (
            usage["adapter_requests"] == EXPECTED_REQUESTS
        ),
        "exact_10_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_five_planner_and_target_schemas_parse": len(cases) == NUM_CASES,
        "all_cases_have_six_unique_questions": all(
            case["unique_question_count"] == NUM_QUESTIONS for case in cases
        ),
        "all_cases_have_at_least_ten_unique_hypotheses": all(
            case["unique_hypothesis_count"] >= MIN_UNIQUE_HYPOTHESES
            for case in cases
        ),
        "all_cases_have_at_least_eight_unique_signatures": all(
            case["unique_hypothesis_signature_count"]
            >= MIN_UNIQUE_SIGNATURES
            for case in cases
        ),
        "every_question_has_minority_at_least_two": all(
            min(case["partition_minority_counts"]) >= MIN_PARTITION_MINORITY
            for case in cases
        ),
        "every_case_mean_partition_entropy_at_least_0_45_nats": all(
            case["mean_partition_entropy_nats"]
            >= MIN_MEAN_PARTITION_ENTROPY
            for case in cases
        ),
        "exact_target_signature_covered_on_at_least_four_cases": (
            exact_covered >= MIN_EXACT_TARGET_COVERAGE_CASES
        ),
        "target_signature_within_one_bit_on_all_cases": all(
            case["target_signature_nearest_hamming"]
            <= MAX_NEAREST_TARGET_HAMMING
            for case in cases
        ),
        "at_least_three_unique_target_signatures": (
            len(set(target_signatures)) >= MIN_UNIQUE_TARGET_SIGNATURES
        ),
        "cost_at_most_0_25": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _usage(models: Sequence[StructuredModel]) -> dict[str, Any]:
    snapshots = [model.usage_snapshot() for model in models]
    totals: dict[str, Any] = {}
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
        "adapter_cost_usd",
    ):
        totals[key] = sum(float(item.get(key, 0) or 0) for item in snapshots)
    for key in (
        "adapter_requests",
        "http_attempts",
        "retry_count",
        "provider_error_retries",
        "adapter_reasoning_tokens",
        "forced_exits",
        "adapter_prompt_tokens",
        "adapter_completion_tokens",
    ):
        totals[key] = int(totals[key])
    totals["run_cost_usd"] = totals.pop("adapter_cost_usd")
    totals["models"] = snapshots
    return totals


def _checkpoint_private(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    *,
    rows: Sequence[dict[str, Any]],
    planner_model: StructuredModel,
    target_model: StructuredModel,
    raw_path: Path,
) -> dict[str, Any]:
    if len(rows) != NUM_CASES:
        raise ValueError(f"serving smoke requires exactly {NUM_CASES} rows")
    raw: dict[str, Any] = {
        "source_row_ids": [source.row_id(row) for row in rows],
        "planner_responses": [],
        "target_responses": [],
        "hidden_target_accessed": False,
        "checklist_accessed": False,
    }
    try:
        planner_responses = planner_model.chat_complete_messages_batched_structured(
            [planner_messages(row) for row in rows],
            temperature=TEMPERATURE,
            block_size=CONCURRENCY,
            response_format=planner_response_format(),
            max_new_tokens=PLANNER_MAX_TOKENS,
        )
        raw["planner_responses"] = list(planner_responses)
        _checkpoint_private(raw_path, raw)
        if len(planner_responses) != NUM_CASES:
            raise ValueError("planner response count changed")
        planners = [
            parse_planner_response(response) for response in planner_responses
        ]

        raw["hidden_target_accessed"] = True
        target_responses = target_model.chat_complete_messages_batched_structured(
            [
                target_messages(row, planner["questions"])
                for row, planner in zip(rows, planners, strict=True)
            ],
            temperature=0.0,
            block_size=CONCURRENCY,
            response_format=target_response_format(),
            max_new_tokens=TARGET_MAX_TOKENS,
        )
        raw["target_responses"] = list(target_responses)
        _checkpoint_private(raw_path, raw)
        if len(target_responses) != NUM_CASES:
            raise ValueError("target response count changed")
        target_signatures = [
            parse_target_response(response) for response in target_responses
        ]
        cases = [
            case_metrics(
                row=row,
                planner=planner,
                target_signature=target_signature,
            )
            for row, planner, target_signature in zip(
                rows,
                planners,
                target_signatures,
                strict=True,
            )
        ]
        usage = _usage((planner_model, target_model))
    except Exception as exc:
        _checkpoint_private(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage((planner_model, target_model)),
        ) from exc

    gates = aggregate_gates(
        cases=cases,
        target_signatures=target_signatures,
        usage=usage,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_manifest_sha256": (
                "2f742ddbacd64eade99fa0148b3c5a11a8676f5cfceb5e666f0966fa6785f790"
            ),
            "planner_model": PLANNER_MODEL_ID,
            "target_model": TARGET_MODEL_ID,
            "planner_seed": PLANNER_SEED,
            "target_seed": TARGET_SEED,
            "temperature": TEMPERATURE,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "checklist_accessed": False,
            "policy_endpoint_accessed": False,
            "holdout_accessed": False,
        },
        "metrics": {
            "case_count": len(cases),
            "exact_target_coverage_cases": sum(
                case["target_signature_exactly_covered"] for case in cases
            ),
            "unique_target_signature_count": len(set(target_signatures)),
            "mean_nearest_target_hamming": statistics.fmean(
                case["target_signature_nearest_hamming"] for case in cases
            ),
            "mean_partition_entropy_nats": statistics.fmean(
                case["mean_partition_entropy_nats"] for case in cases
            ),
        },
        "cases": cases,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, role: str) -> None:
        self.role = role
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, response_format, max_new_tokens
        patterns = (
            "010101",
            "101010",
            "001111",
            "110000",
            "011001",
            "100110",
            "000111",
            "111000",
            "010110",
            "101001",
            "011010",
            "100101",
        )
        target_patterns = (
            "010101",
            "001111",
            "011001",
            "000111",
            "010110",
        )
        responses = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            case_index = self.requests + len(responses)
            if self.role == "planner":
                case_id = request["case_id"]
                responses.append(
                    json.dumps(
                        {
                            "questions": [
                                {
                                    "id": f"Q{index}",
                                    "question": (
                                        f"For {case_id}, would preference feature "
                                        f"{index} be desirable?"
                                    ),
                                }
                                for index in range(1, NUM_QUESTIONS + 1)
                            ],
                            "hypotheses": [
                                {
                                    "id": f"H{index}",
                                    "preference": (
                                        f"Distinct fixture preference {index} "
                                        f"for {case_id}"
                                    ),
                                    "answer_signature": patterns[index - 1],
                                }
                                for index in range(1, NUM_HYPOTHESES + 1)
                            ],
                        }
                    )
                )
            else:
                responses.append(
                    json.dumps(
                        {"answer_signature": target_patterns[case_index]}
                    )
                )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "provider_error_retries": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def _adapter(
    *,
    model: str,
    seed: int,
    run_id: str,
    output_dir: Path,
) -> SeededStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_output_tokens=PLANNER_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return SeededStructuredAdapter(
        ModelSpec(model=model, backend="openrouter", max_model_len=131_072),
        config,
        request_seed=seed,
    )


def serving_rows() -> list[dict[str, Any]]:
    rows = source.load_rows()
    serving, _, _, _ = source.split_rows(rows)
    expected_ids = [
        item["id"]
        for item in json.loads(
            (
                REPO_ROOT
                / "results/nonmyopic/cupid_active_preference_source_audit/"
                "cupid-active-preference-source-audit-20260729/MANIFEST.json"
            ).read_text(encoding="utf-8")
        )["splits"]["serving_smoke"]
    ]
    observed_ids = [source.row_id(row) for row in serving]
    if observed_ids != expected_ids:
        raise ValueError("CUPID serving split differs from the bound manifest")
    return serving


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"

    if args.dry_run:
        planner: StructuredModel = DeterministicFixtureModel("planner")
        target: StructuredModel = DeterministicFixtureModel("target")
    else:
        planner = _adapter(
            model=PLANNER_MODEL_ID,
            seed=PLANNER_SEED,
            run_id=f"{args.run_id}-planner",
            output_dir=args.output_dir,
        )
        target = _adapter(
            model=TARGET_MODEL_ID,
            seed=TARGET_SEED,
            run_id=f"{args.run_id}-target",
            output_dir=args.output_dir,
        )
    try:
        payload = run_smoke(
            rows=serving_rows(),
            planner_model=planner,
            target_model=target,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
            "holdout_accessed": False,
            "policy_endpoint_accessed": False,
        }
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    checkpoint(args.output_dir / "SERVING.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
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
