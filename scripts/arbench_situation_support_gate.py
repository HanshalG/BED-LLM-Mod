#!/usr/bin/env python3
"""Test path-dependent semantic support recovery on AR-Bench Situation Puzzles."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


AR_BENCH_COMMIT = "9971322fe9e4d77cb4d303b7e279ab1d1cb5dba1"
AR_BENCH_TEST_SHA256 = (
    "e3c34d58b8bad06d0152fee771f2ee19a01c17dd5ba65c3a6dfb69c02b52f485"
)
SELECTION_SEED = 24301
FORMAL_POSITIONS = (10, 12, 19, 20, 28, 34, 47, 60, 62, 67, 91, 96)
SMOKE_POSITIONS = FORMAL_POSITIONS[:2]
NUM_INITIAL_EXPLANATIONS = 8
NUM_CANDIDATE_QUESTIONS = 4
NUM_REFRESHED_EXPLANATIONS = 8
SEMANTIC_COVERAGE_THRESHOLD = 0.80
FORMAL_EXPECTED_REQUESTS = 132
SMOKE_EXPECTED_REQUESTS = 10


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        stripped = fenced.group(1)
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        payload = json.loads(stripped[start : end + 1])
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return payload


def _dedupe_text(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        clean = " ".join(value.strip().split())
        key = clean.casefold()
        if clean and key not in seen:
            result.append(clean)
            seen.add(key)
    return result


def parse_text_list(text: str, key: str, count: int) -> list[str]:
    raw = _parse_json_object(text).get(key)
    if not isinstance(raw, list):
        raise ValueError(f"JSON field {key!r} must be a list")
    if any(not isinstance(value, str) for value in raw):
        raise ValueError(f"JSON field {key!r} must contain strings only")
    values = _dedupe_text(raw)
    if len(values) != count:
        raise ValueError(
            f"JSON field {key!r} must contain exactly {count} unique non-empty "
            f"strings; got {len(values)}"
        )
    return values


def parse_answer(text: str) -> str:
    answer = _parse_json_object(text).get("answer")
    if answer not in {"Yes", "No", "Unknown"}:
        raise ValueError("answer must be exactly Yes, No, or Unknown")
    return str(answer)


def parse_coverage_response(
    text: str,
    support_ids: Sequence[str],
    support_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    raw = _parse_json_object(text).get("supports")
    if not isinstance(raw, list) or len(raw) != len(support_ids):
        raise ValueError("coverage response must contain one row per support")
    parsed: list[dict[str, Any]] = []
    for expected_id, support_size, row in zip(
        support_ids, support_sizes, raw, strict=True
    ):
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("coverage response changed support IDs or order")
        score = row.get("best_match_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("best_match_score must be numeric")
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("best_match_score must be in [0, 1]")
        index = row.get("best_explanation_index")
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < support_size
        ):
            raise ValueError("best_explanation_index is outside its support")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("coverage response requires a reason")
        parsed.append(
            {
                "id": expected_id,
                "best_match_score": score,
                "best_explanation_index": index,
                "reason": reason.strip(),
                "covered": score >= SEMANTIC_COVERAGE_THRESHOLD,
            }
        )
    return parsed


def load_arbench_situation_puzzles(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != AR_BENCH_TEST_SHA256:
        raise ValueError(
            "AR-Bench Situation Puzzle test hash mismatch: "
            f"expected {AR_BENCH_TEST_SHA256}, got {digest}"
        )
    rows = json.loads(raw)
    if not isinstance(rows, list) or len(rows) != 100:
        raise ValueError("AR-Bench Situation Puzzle test file must contain 100 rows")
    required = {"surface", "bottom", "key_question", "index"}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError(f"AR-Bench row {position} has an invalid schema")
        if (
            not isinstance(row["surface"], str)
            or not row["surface"].strip()
            or not isinstance(row["bottom"], str)
            or not row["bottom"].strip()
        ):
            raise ValueError(f"AR-Bench row {position} has empty story text")
    return rows


def initial_explanation_messages(
    surface: str,
    count: int = NUM_INITIAL_EXPLANATIONS,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Generate diverse causal hypotheses for a situation puzzle. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Public puzzle:\n{surface}\n\n"
                f"Generate exactly {count} mutually distinct, complete explanations. "
                "Each explanation must be a self-contained causal story that could "
                "resolve every public fact. Cover genuinely different mechanisms, not "
                "minor wording variants. You do not know the hidden answer. Return "
                + json.dumps({"explanations": ["..."] * count})
                + "."
            ),
        },
    ]


def candidate_question_messages(
    surface: str,
    explanations: Sequence[str],
    count: int = NUM_CANDIDATE_QUESTIONS,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Design discriminating questions for a situation puzzle. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Public puzzle:\n{surface}\n\n"
                "Current possible explanations:\n- "
                + "\n- ".join(explanations)
                + f"\n\nGenerate exactly {count} distinct atomic yes-or-no questions. "
                "Each must test one factual proposition, help distinguish the current "
                "explanations, and be answerable as Yes, No, or Unknown from the hidden "
                "story. Do not ask for the complete solution. Return "
                + json.dumps({"questions": ["..."] * count})
                + "."
            ),
        },
    ]


def answer_messages(
    surface: str,
    hidden_story: str,
    question: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are the truth oracle for a situation puzzle. Use the hidden "
                "story exactly and return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Public puzzle:\n{surface}\n\n"
                f"Hidden story:\n{hidden_story}\n\n"
                f"Question:\n{question}\n\n"
                "Return Yes if the proposition is supported, No if contradicted, and "
                "Unknown if neither follows from the public puzzle and hidden story. "
                'Return exactly {"answer":"Yes"}, {"answer":"No"}, or '
                '{"answer":"Unknown"}.'
            ),
        },
    ]


def refreshed_explanation_messages(
    surface: str,
    initial_explanations: Sequence[str],
    question: str,
    answer: str,
    count: int = NUM_REFRESHED_EXPLANATIONS,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Revise a situation-puzzle hypothesis space after new evidence. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Public puzzle:\n{surface}\n\n"
                "Previous possible explanations:\n- "
                + "\n- ".join(initial_explanations)
                + f"\n\nObserved evidence:\nQuestion: {question}\nAnswer: {answer}\n\n"
                f"Generate exactly {count} mutually distinct, complete explanations "
                "that are consistent with all public and observed evidence. Revise "
                "the hypothesis space substantively: include newly suggested causal "
                "mechanisms instead of merely paraphrasing the previous list. Do not "
                "invent additional observations. Return "
                + json.dumps({"explanations": ["..."] * count})
                + "."
            ),
        },
    ]


def semantic_coverage_messages(
    surface: str,
    hidden_story: str,
    supports: Sequence[tuple[str, Sequence[str]]],
) -> list[dict[str, str]]:
    payload = {
        "public_puzzle": surface,
        "hidden_story": hidden_story,
        "supports": [
            {"id": support_id, "explanations": list(explanations)}
            for support_id, explanations in supports
        ],
    }
    schema = {
        "supports": [
            {
                "id": support_id,
                "best_match_score": 0.0,
                "best_explanation_index": 0,
                "reason": "brief",
            }
            for support_id, _explanations in supports
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict semantic coverage evaluator. The hidden story was "
                "withheld from every explanation and question generator. Return strict "
                "JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each support, score its single best explanation against the "
                "hidden story. A score of 1 requires the same central mechanism and "
                "all causal links needed to resolve the public puzzle. A score of at "
                "least 0.8 requires the same central mechanism and enough correct "
                "causal links to explain every public fact; incidental hidden-story "
                "details may be omitted. Plausibility, a shared theme, or one matching "
                "fact is insufficient. Preserve support IDs and order. Return "
                + json.dumps(schema, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def _build_models(config: Config, judge_model: str) -> tuple[Any, Any]:
    if len(config.model_pairs) != 1:
        raise ValueError("support gate requires exactly one model pair")
    generator_spec = config.model_pairs[0].questioner
    generator = build_model_adapter(generator_spec, config)
    evaluator_spec = replace(
        generator_spec,
        model=judge_model,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    evaluator = build_model_adapter(evaluator_spec, config)
    return generator, evaluator


def _write_raw_checkpoint(
    path: Path | None,
    *,
    stage: str,
    positions: Sequence[int],
    raw: dict[str, Any],
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_stage": stage,
                "positions": list(positions),
                "responses": raw,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _usage(generator: Any, evaluator: Any) -> dict[str, Any]:
    snapshots = {
        "generator": generator.usage_snapshot(),
        "evaluator": evaluator.usage_snapshot(),
    }
    return {
        "physical_requests": sum(
            int(snapshot["adapter_requests"]) for snapshot in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(snapshot["adapter_reasoning_tokens"])
            for snapshot in snapshots.values()
        ),
        "adapter_cost_usd": sum(
            float(snapshot["adapter_cost_usd"]) for snapshot in snapshots.values()
        ),
        "by_role": snapshots,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    initial_omitted = sum(not record["initial"]["covered"] for record in records)
    recovered = sum(
        not record["initial"]["covered"]
        and any(branch["coverage"]["covered"] for branch in record["branches"])
        for record in records
    )
    branch_count = sum(len(record["branches"]) for record in records)
    spreads = [
        max(branch["coverage"]["best_match_score"] for branch in record["branches"])
        - min(branch["coverage"]["best_match_score"] for branch in record["branches"])
        for record in records
    ]
    oracle_gains = [
        max(branch["coverage"]["best_match_score"] for branch in record["branches"])
        - record["initial"]["best_match_score"]
        for record in records
    ]
    expected_tasks = 2 if stage == "serving_smoke" else 12
    expected_branches = 2 if stage == "serving_smoke" else 48
    expected_requests = (
        SMOKE_EXPECTED_REQUESTS
        if stage == "serving_smoke"
        else FORMAL_EXPECTED_REQUESTS
    )
    summary = {
        "num_tasks": len(records),
        "num_branches": branch_count,
        "initial_covered": len(records) - initial_omitted,
        "initial_omitted": initial_omitted,
        "initially_omitted_recovered_by_any_branch": recovered,
        "recovery_fraction_among_initial_omissions": (
            recovered / initial_omitted if initial_omitted else 0.0
        ),
        "mean_oracle_best_match_gain": (
            sum(oracle_gains) / len(oracle_gains) if oracle_gains else 0.0
        ),
        "tasks_with_branch_score_spread_at_least_0_15": sum(
            spread >= 0.15 for spread in spreads
        ),
        "mean_branch_score_spread": (
            sum(spreads) / len(spreads) if spreads else 0.0
        ),
    }
    base_gates = {
        "all_tasks_completed": len(records) == expected_tasks,
        "all_branches_completed": branch_count == expected_branches,
        "exact_physical_request_count": (
            int(usage["physical_requests"]) == expected_requests
        ),
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_answers_valid": all(
            branch["answer"] in {"Yes", "No", "Unknown"}
            for record in records
            for branch in record["branches"]
        ),
    }
    if stage == "serving_smoke":
        gates = base_gates
    else:
        gates = {
            **base_gates,
            "initial_support_non_saturated": initial_omitted >= 6,
            "at_least_three_omitted_truths_recovered": recovered >= 3,
            "mean_oracle_best_match_gain_at_least_0_10": (
                summary["mean_oracle_best_match_gain"] >= 0.10
            ),
            "at_least_four_tasks_have_branch_spread": (
                summary["tasks_with_branch_score_spread_at_least_0_15"] >= 4
            ),
        }
    gates["all_pass"] = all(gates.values())
    return {**summary, "gates": gates}


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    judge_model: str,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"serving_smoke", "formal"}:
        raise ValueError("stage must be serving_smoke or formal")
    rows = load_arbench_situation_puzzles(data_path)
    positions = SMOKE_POSITIONS if stage == "serving_smoke" else FORMAL_POSITIONS
    selected = [rows[position] for position in positions]
    generator, evaluator = _build_models(config, judge_model)
    raw: dict[str, Any] = {}

    initial_prompts = [
        initial_explanation_messages(row["surface"]) for row in selected
    ]
    initial_raw = generator.chat_complete_messages_batched(
        initial_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial_explanations"] = initial_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, positions=positions, raw=raw
    )
    initial_supports = [
        parse_text_list(response, "explanations", NUM_INITIAL_EXPLANATIONS)
        for response in initial_raw
    ]

    question_prompts = [
        candidate_question_messages(row["surface"], support)
        for row, support in zip(selected, initial_supports, strict=True)
    ]
    question_raw = generator.chat_complete_messages_batched(
        question_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["candidate_questions"] = question_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, positions=positions, raw=raw
    )
    question_counts = (
        [1] * len(selected)
        if stage == "serving_smoke"
        else [NUM_CANDIDATE_QUESTIONS] * len(selected)
    )
    questions_many = [
        parse_text_list(response, "questions", NUM_CANDIDATE_QUESTIONS)[:count]
        for response, count in zip(question_raw, question_counts, strict=True)
    ]

    flat_rows = [
        row
        for row, questions in zip(selected, questions_many, strict=True)
        for _question in questions
    ]
    flat_initial = [
        support
        for support, questions in zip(initial_supports, questions_many, strict=True)
        for _question in questions
    ]
    flat_questions = [
        question for questions in questions_many for question in questions
    ]
    answer_prompts = [
        answer_messages(row["surface"], row["bottom"], question)
        for row, question in zip(flat_rows, flat_questions, strict=True)
    ]
    answer_raw = evaluator.chat_complete_messages_batched(
        answer_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=128,
    )
    raw["oracle_answers"] = answer_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, positions=positions, raw=raw
    )
    answers = [parse_answer(response) for response in answer_raw]

    refresh_prompts = [
        refreshed_explanation_messages(
            row["surface"],
            support,
            question,
            answer,
        )
        for row, support, question, answer in zip(
            flat_rows, flat_initial, flat_questions, answers, strict=True
        )
    ]
    refresh_raw = generator.chat_complete_messages_batched(
        refresh_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["refreshed_explanations"] = refresh_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, positions=positions, raw=raw
    )
    refreshed = [
        parse_text_list(response, "explanations", NUM_REFRESHED_EXPLANATIONS)
        for response in refresh_raw
    ]

    coverage_prompts: list[list[dict[str, str]]] = []
    metadata: list[tuple[list[str], list[int]]] = []
    offset = 0
    for row, support, questions in zip(
        selected, initial_supports, questions_many, strict=True
    ):
        branch_supports = refreshed[offset : offset + len(questions)]
        support_ids = [
            "initial",
            *[f"candidate_{index}" for index in range(len(questions))],
        ]
        supports = [support, *branch_supports]
        coverage_prompts.append(
            semantic_coverage_messages(
                row["surface"],
                row["bottom"],
                list(zip(support_ids, supports, strict=True)),
            )
        )
        metadata.append((support_ids, [len(values) for values in supports]))
        offset += len(questions)
    coverage_raw = evaluator.chat_complete_messages_batched(
        coverage_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["semantic_coverage"] = coverage_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, positions=positions, raw=raw
    )
    coverage_many = [
        parse_coverage_response(response, support_ids, support_sizes)
        for response, (support_ids, support_sizes) in zip(
            coverage_raw, metadata, strict=True
        )
    ]

    records: list[dict[str, Any]] = []
    offset = 0
    for position, row, initial, questions, coverage in zip(
        positions,
        selected,
        initial_supports,
        questions_many,
        coverage_many,
        strict=True,
    ):
        branches = []
        for branch_index, question in enumerate(questions):
            branches.append(
                {
                    "branch_index": branch_index,
                    "question": question,
                    "answer": answers[offset + branch_index],
                    "refreshed_support": refreshed[offset + branch_index],
                    "coverage": coverage[branch_index + 1],
                }
            )
        records.append(
            {
                "position": position,
                "dataset_index": row["index"],
                "surface": row["surface"],
                "initial_support": initial,
                "initial": coverage[0],
                "branches": branches,
            }
        )
        offset += len(questions)

    usage = _usage(generator, evaluator)
    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "ar_bench_commit": AR_BENCH_COMMIT,
            "test_sha256": AR_BENCH_TEST_SHA256,
            "selection_seed": SELECTION_SEED,
            "positions": list(positions),
            "dataset_indices": [row["index"] for row in selected],
            "num_initial_explanations": NUM_INITIAL_EXPLANATIONS,
            "num_candidate_questions": (
                1 if stage == "serving_smoke" else NUM_CANDIDATE_QUESTIONS
            ),
            "num_refreshed_explanations": NUM_REFRESHED_EXPLANATIONS,
            "semantic_coverage_threshold": SEMANTIC_COVERAGE_THRESHOLD,
            "target_hidden_from_generation": True,
            "target_used_only_for_oracle_and_measurement": True,
            "key_questions_unused": True,
            "judge_model": judge_model,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        default="formal",
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = args.output_dir / "RAW_RESPONSES.json"
    output_name = "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            data_path=args.data_path,
            judge_model=args.judge_model,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
