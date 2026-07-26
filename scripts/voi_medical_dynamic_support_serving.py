#!/usr/bin/env python3
"""Target-blind serving gate for free-form MedDG hypotheses and questions."""

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
from scripts.voi_medical_future_tree_mechanics import (
    MODEL_ID,
    MechanicsExecutionError,
    _checkpoint,
    _usage_snapshot,
    normalize_question,
    parse_questions,
    sha256_file,
)


INTERFACE_VERSION = "voi-medical-dynamic-support-serving-1"
SOURCE_SHA256 = "e851864a9cb53c36304245bc3213a8a894cf7b86f8945d60978923e1f1ef0169"
MANIFEST_SHA256 = "70a943c1c943e7d53944cb88a4e3aff809a09219dd07da39fc38bc04a07d9c37"
MECHANICS_IDS = (474, 67, 151, 50, 284)
HYPOTHESIS_COUNT = 6
QUESTION_COUNT = 4
EXPECTED_REQUESTS = 10
PROJECTED_COST_USD = 0.03
MAX_COST_USD = 0.10


def normalize_hypothesis(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def parse_hypotheses(text: str) -> list[dict[str, str]]:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError(f"expected exactly {HYPOTHESIS_COUNT} hypothesis lines")
    result = []
    for index, line in enumerate(lines, start=1):
        fields = [field.strip() for field in line.split("|")]
        if len(fields) != 3 or fields[0] != f"H{index}":
            raise ValueError("hypothesis line has invalid fields or index")
        diagnosis, rationale = fields[1:]
        if not 2 <= len(diagnosis) <= 100:
            raise ValueError("hypothesis diagnosis length is invalid")
        if not 10 <= len(rationale) <= 280:
            raise ValueError("hypothesis rationale length is invalid")
        result.append({"diagnosis": diagnosis, "rationale": rationale})
    if len({normalize_hypothesis(row["diagnosis"]) for row in result}) != len(
        result
    ):
        raise ValueError("hypothesis diagnoses must be distinct")
    return result


def hypothesis_messages(self_report: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are constructing a differential diagnosis in an open semantic "
                "space. Generate exactly six distinct plausible clinical hypotheses "
                "from the patient self-report. Use specific diagnosis names, not "
                "symptom restatements or broad organ categories. Do not assume a "
                "closed candidate list. Return exactly six lines and no other text: "
                "H1|<diagnosis>|<one-sentence evidence rationale> through H6."
            ),
        },
        {"role": "user", "content": "Patient self-report:\n" + self_report},
    ]


def question_messages(
    self_report: str,
    hypotheses: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    support = "\n".join(
        f"- {row['diagnosis']}: {row['rationale']}" for row in hypotheses
    )
    return [
        {
            "role": "system",
            "content": (
                "Design four distinct atomic yes/no questions that would "
                "discriminate among an open differential diagnosis. Questions must "
                "ask about patient-observable symptoms, timing, triggers, or history. "
                "Do not name or directly guess a diagnosis, combine multiple "
                "features, or repeat facts already explicit in the self-report. "
                "Return exactly four lines and no other text: Q1|<question> through "
                "Q4|<question>."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Patient self-report:\n{self_report}\n\n"
                f"Current generated differential:\n{support}"
            ),
        },
    ]


def target_is_covered(target: str, hypotheses: Sequence[dict[str, str]]) -> bool:
    target_normalized = normalize_hypothesis(target)
    return any(
        target_normalized in normalize_hypothesis(row["diagnosis"])
        or normalize_hypothesis(row["diagnosis"]) in target_normalized
        for row in hypotheses
    )


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("dynamic-support serving config selects the wrong model")
    return build_model_adapter(spec, config)


def run_serving(
    config: Config,
    *,
    source_path: Path,
    manifest_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    if sha256_file(source_path) != SOURCE_SHA256:
        raise ValueError("MedDG source hash mismatch")
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("dynamic-support manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if tuple(manifest["splits"]["mechanics"]) != MECHANICS_IDS:
        raise ValueError("mechanics IDs do not match the frozen manifest")
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    visible = [{"self_repo": rows[index]["self_repo"]} for index in MECHANICS_IDS]
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        support_responses = model.chat_complete_messages_batched(
            [hypothesis_messages(row["self_repo"]) for row in visible],
            temperature=0.7,
            block_size=len(visible),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["hypotheses"] = support_responses
        _checkpoint(raw_path, raw)
        supports = [parse_hypotheses(response) for response in support_responses]

        question_responses = model.chat_complete_messages_batched(
            [
                question_messages(row["self_repo"], support)
                for row, support in zip(visible, supports, strict=True)
            ],
            temperature=0.7,
            block_size=len(visible),
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["questions"] = question_responses
        _checkpoint(raw_path, raw)
        questions = [
            parse_questions(response, QUESTION_COUNT)
            for response in question_responses
        ]
        for support, task_questions in zip(supports, questions, strict=True):
            diagnoses = {
                normalize_hypothesis(row["diagnosis"]) for row in support
            }
            if any(
                any(diagnosis in normalize_question(question) for diagnosis in diagnoses)
                for question in task_questions
            ):
                raise ValueError("question directly names a generated diagnosis")
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    # Targets become visible only after all target-blind outputs parse and freeze.
    targets = [rows[index]["target"] for index in MECHANICS_IDS]
    leaked = [
        normalize_hypothesis(target)
        in normalize_hypothesis(row["self_repo"])
        for target, row in zip(targets, visible, strict=True)
    ]
    coverage = [
        target_is_covered(target, support)
        for target, support in zip(targets, supports, strict=True)
    ]
    eligible_coverage = [
        covered for covered, is_leaked in zip(coverage, leaked, strict=True)
        if not is_leaked
    ]
    unique_hypotheses = len(
        {
            normalize_hypothesis(row["diagnosis"])
            for support in supports
            for row in support
        }
    )
    unique_questions = len(
        {
            normalize_question(question)
            for task_questions in questions
            for question in task_questions
        }
    )
    generator = usage["generator"]
    gates = {
        "exact_source_and_manifest": True,
        "exact_10_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_10_http_attempts_zero_retries": (
            usage["http_attempts"] == EXPECTED_REQUESTS
            and usage["retry_count"] == 0
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits_or_finalization": (
            usage["forced_exits"] == 0
            and int(generator.get("forced_final_requests", 0)) == 0
        ),
        "five_complete_six_hypothesis_sets": (
            len(supports) == len(MECHANICS_IDS)
            and all(len(support) == HYPOTHESIS_COUNT for support in supports)
        ),
        "five_complete_four_question_sets": (
            len(questions) == len(MECHANICS_IDS)
            and all(len(values) == QUESTION_COUNT for values in questions)
        ),
        "at_least_20_unique_hypotheses": unique_hypotheses >= 20,
        "at_least_15_unique_questions": unique_questions >= 15,
        "exactly_one_frozen_leak_exclusion": sum(leaked) == 1,
        "eligible_initial_target_coverage_at_least_half": (
            sum(eligible_coverage) >= 2
        ),
        "cost_at_most_0_10": usage["adapter_cost_usd"] <= MAX_COST_USD,
        "adapter_reports_nonreasoning_model": (
            generator.get("model") == MODEL_ID
            and generator.get("reasoning_enabled") is False
            and int(generator.get("reasoning_tokens", 0)) == 0
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "source_sha256": SOURCE_SHA256,
            "manifest_sha256": MANIFEST_SHA256,
            "mechanics_ids": list(MECHANICS_IDS),
            "target_exposed_to_generation": False,
            "candidate_list_exposed": False,
            "reasoning_requested": False,
            "scientific_retries_or_repairs": 0,
        },
        "hypotheses": supports,
        "questions": questions,
        "metrics": {
            "unique_hypotheses": unique_hypotheses,
            "unique_questions": unique_questions,
            "target_leakage_by_task": leaked,
            "initial_target_coverage_by_task": coverage,
            "eligible_initial_target_coverage": sum(eligible_coverage),
            "eligible_tasks": len(eligible_coverage),
        },
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 10
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_serving(
            config,
            source_path=args.source,
            manifest_path=args.manifest,
            raw_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, MechanicsExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        _checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    output = args.output_dir / "SERVING.json"
    _checkpoint(output, payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
