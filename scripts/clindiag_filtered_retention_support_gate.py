#!/usr/bin/env python3
"""Gate BED-LLM-style filtered retention on deterministic ClinDiag evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable, Sequence, TypeVar

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mediq.parsing import parse_json_object
from helpers import Config, load_config
from scripts.clindiag_fixed_slot_audit import fixed_evidence_slots
from scripts.clindiag_fixed_slot_support_gate import (
    source_evidence_contains_target,
)
from scripts.clindiag_mc_joint_model_gate import _build_role
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import _dedupe, parse_diagnoses


SELECTION_SEED = 24299
SMOKE_IDS = ("11388546", "rare203")
ACTION_ID = "lab_1"
FILTER_THRESHOLD = 0.20
EXPECTED_REQUESTS = 14
SUPPORT_IDS = ("initial", "filtered_retention", "filtered_retention_duplicate")

T = TypeVar("T")


class StructuredStageError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        stage: str,
        row: int,
        raw_responses: Sequence[str],
    ) -> None:
        super().__init__(message)
        self.stage = stage
        self.row = row
        self.raw_responses = list(raw_responses)


def generated_candidate_messages(
    case: ClinDiagCase,
    acquired_observations: Sequence[tuple[str, Any]],
    count: int = DIAGNOSIS_COUNT,
) -> list[dict[str, str]]:
    evidence = [
        {"action_id": action_id, "stored_observation": observation}
        for action_id, observation in acquired_observations
    ]
    return [
        {
            "role": "system",
            "content": (
                "Generate diverse candidate diagnoses from the complete visible "
                "patient history. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation:\n{case.initial_information}\n\n"
                "Acquired evidence in order:\n"
                + json.dumps(evidence, ensure_ascii=True, sort_keys=True)
                + f"\n\nGenerate exactly {count} distinct, specific, unifying "
                "candidate diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Generate jointly for '
                "diversity. Each item must be one diagnosis-name string, not an "
                "explanation or object. Include rare conditions when warranted. "
                "No previous differential, answer options, case title, final "
                "diagnosis, or hidden benchmark label is provided."
            ),
        },
    ]


def compatibility_messages(
    case: ClinDiagCase,
    diagnoses: Sequence[str],
    evidence_to_check: Sequence[tuple[str, Any]],
) -> list[dict[str, str]]:
    evidence = [
        {"id": evidence_id, "stored_observation": observation}
        for evidence_id, observation in evidence_to_check
    ]
    candidates = [
        {"id": f"d{index:02d}", "diagnosis": diagnosis}
        for index, diagnosis in enumerate(diagnoses)
    ]
    evidence_ids = [row["id"] for row in evidence]
    likelihood_template = ",".join(
        f'"{evidence_id}":0.0' for evidence_id in evidence_ids
    )
    return [
        {
            "role": "system",
            "content": (
                "Estimate clinical observation likelihoods for filtering a "
                "hypothesis set. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation context:\n{case.initial_information}\n\n"
                "Evidence to check:\n"
                + json.dumps(evidence, ensure_ascii=True, sort_keys=True)
                + "\n\nCandidate diagnoses:\n"
                + json.dumps(candidates, ensure_ascii=True, sort_keys=True)
                + "\n\nFor every diagnosis and every evidence item, estimate in "
                "[0,1] the probability of observing that exact evidence if the "
                "diagnosis were true. This is p(evidence|diagnosis), not the "
                "posterior probability of the diagnosis. Use low values for "
                "contradictory or very surprising evidence. Preserve IDs and order. "
                "Return exactly "
                '{"diagnoses":[{"id":"d00","likelihoods":{'
                + likelihood_template
                + '},"reason":"brief"}, ...]}.'
            ),
        },
    ]


def parse_compatibility(
    text: str,
    *,
    num_diagnoses: int,
    evidence_ids: Sequence[str],
) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("diagnoses")
    if not isinstance(raw, list) or len(raw) != num_diagnoses:
        raise ValueError("compatibility response must contain one row per diagnosis")
    expected_evidence = tuple(evidence_ids)
    parsed = []
    for index, row in enumerate(raw):
        expected_id = f"d{index:02d}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("compatibility response changed diagnosis IDs or order")
        likelihoods = row.get("likelihoods")
        if not isinstance(likelihoods, dict) or set(likelihoods) != set(
            expected_evidence
        ):
            raise ValueError("likelihood keys must exactly match evidence IDs")
        values = []
        for evidence_id in expected_evidence:
            value = likelihoods[evidence_id]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError("each compatibility likelihood must be numeric")
            value = float(value)
            if not 0.0 <= value <= 1.0:
                raise ValueError("each compatibility likelihood must be in [0,1]")
            values.append(value)
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("each compatibility row requires a reason")
        parsed.append(
            {
                "id": expected_id,
                "likelihoods": {
                    evidence_id: value
                    for evidence_id, value in zip(
                        expected_evidence, values, strict=True
                    )
                },
                "minimum_likelihood": min(values),
                "reason": reason.strip(),
            }
        )
    return parsed


def retain_by_threshold(
    diagnoses: Sequence[str],
    compatibility: Sequence[dict[str, Any]],
    threshold: float = FILTER_THRESHOLD,
) -> list[str]:
    if len(diagnoses) != len(compatibility):
        raise ValueError("diagnoses and compatibility rows must have equal length")
    return [
        diagnosis
        for diagnosis, row in zip(diagnoses, compatibility, strict=True)
        if float(row["minimum_likelihood"]) >= threshold
    ]


def merge_support(
    retained_old: Sequence[str],
    filtered_new: Sequence[str],
    count: int = DIAGNOSIS_COUNT,
) -> list[str]:
    return _dedupe([*retained_old, *filtered_new])[:count]


def semantic_audit_messages(
    true_diagnosis: str,
    supports: Sequence[tuple[str, Sequence[str]]],
) -> list[dict[str, str]]:
    payload = {
        "true_diagnosis": true_diagnosis,
        "supports": [
            {"id": support_id, "diagnoses": list(diagnoses)}
            for support_id, diagnoses in supports
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict diagnosis-equivalence and set-overlap evaluator. "
                "The truth was hidden from generation and filtering. Return strict "
                "JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each support, score its closest diagnosis to the truth in [0,1]. "
                "Use at least 0.8 only for the same diagnosis or a standard clinical "
                "synonym; broad parents, related diseases, and alternative subtypes "
                "score below 0.8. Compare filtered_retention with its independent "
                "exact-generation-prompt replay semantically. In each direction, "
                "report the fraction having a same-disease or standard-synonym match "
                "in the other list. Return exactly "
                '{"supports":[{"id":"initial","best_match_score":0.0,'
                '"reason":"brief"},{"id":"filtered_retention",'
                '"best_match_score":0.0,"reason":"brief"},'
                '{"id":"filtered_retention_duplicate","best_match_score":0.0,'
                '"reason":"brief"}],"final_to_duplicate_overlap":0.0,'
                '"duplicate_to_final_overlap":0.0,"overlap_reason":"brief"}; '
                "preserve support IDs and order.\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_semantic_audit(text: str) -> dict[str, Any]:
    raw = parse_json_object(text)
    rows = raw.get("supports")
    if not isinstance(rows, list) or len(rows) != len(SUPPORT_IDS):
        raise ValueError("audit must contain one row per support")
    supports = []
    for support_id, row in zip(SUPPORT_IDS, rows, strict=True):
        if not isinstance(row, dict) or row.get("id") != support_id:
            raise ValueError("audit changed support IDs or order")
        score = row.get("best_match_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("best_match_score must be numeric")
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("best_match_score must be in [0,1]")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("each support score requires a reason")
        supports.append(
            {
                "id": support_id,
                "best_match_score": score,
                "reason": reason.strip(),
            }
        )
    overlaps = []
    for field in ("final_to_duplicate_overlap", "duplicate_to_final_overlap"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0,1]")
        overlaps.append(value)
    reason = raw.get("overlap_reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("overlap_reason must be nonempty")
    return {
        "supports": supports,
        "final_to_duplicate_overlap": overlaps[0],
        "duplicate_to_final_overlap": overlaps[1],
        "duplicate_semantic_overlap": min(overlaps),
        "overlap_reason": reason.strip(),
    }


def _complete_stage(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    *,
    config: Config,
    temperature: float,
    parser: Callable[[str], T],
    stage: str,
) -> tuple[list[T], list[str]]:
    responses = model.chat_complete_messages_batched(
        messages,
        temperature=temperature,
        block_size=256,
        max_new_tokens=int(config.openrouter_max_output_tokens),
    )
    parsed = []
    for row, response in enumerate(responses):
        try:
            parsed.append(parser(response))
        except ValueError as exc:
            raise StructuredStageError(
                str(exc),
                stage=stage,
                row=row,
                raw_responses=responses,
            ) from exc
    return parsed, list(responses)


def _usage(models: dict[str, Any]) -> dict[str, Any]:
    snapshots = {name: model.usage_snapshot() for name, model in models.items()}
    return {
        "snapshots": snapshots,
        "physical_requests": sum(
            int(item["adapter_requests"]) for item in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(item["adapter_reasoning_tokens"]) for item in snapshots.values()
        ),
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    candidate_generation_prompts_exact: bool,
) -> dict[str, Any]:
    overlaps = [float(row["duplicate_semantic_overlap"]) for row in records]
    truth_gaps = [float(row["duplicate_truth_score_gap"]) for row in records]
    gates = {
        "exactly_14_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "candidate_generation_prompts_exact": candidate_generation_prompts_exact,
        "all_final_supports_size_twelve": all(
            len(row["supports"][support_id]) == DIAGNOSIS_COUNT
            for row in records
            for support_id in ("filtered_retention", "filtered_retention_duplicate")
        ),
        "both_cases_prune_old_hypotheses": all(
            row["num_old_pruned"] >= 1 for row in records
        ),
        "both_cases_introduce_filtered_new_hypotheses": all(
            row["num_new_introduced"] >= 1
            and row["num_duplicate_new_introduced"] >= 1
            for row in records
        ),
        "both_duplicate_semantic_overlaps_at_least_0_80": (
            len(overlaps) == 2 and all(value >= 0.80 for value in overlaps)
        ),
        "both_duplicate_truth_score_gaps_at_most_0_05": (
            len(truth_gaps) == 2 and all(value <= 0.05 for value in truth_gaps)
        ),
        "no_hidden_target_in_source_evidence": all(
            not row["source_target_leak"] for row in records
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "duplicate_semantic_overlaps": overlaps,
        "duplicate_truth_score_gaps": truth_gaps,
        "old_pruned": [row["num_old_pruned"] for row in records],
        "new_introduced": [row["num_new_introduced"] for row in records],
        "duplicate_new_introduced": [
            row["num_duplicate_new_introduced"] for row in records
        ],
        "gates": gates,
    }


def run_gate(
    config: Config,
    data_zip: Path,
    *,
    generator_model: str,
    filter_model: str,
    judge_model: str,
) -> dict[str, Any]:
    models = {
        "initial": _build_role(config, generator_model),
        "old_filter": _build_role(config, filter_model),
        "candidate": _build_role(config, generator_model),
        "candidate_duplicate": _build_role(config, generator_model),
        "new_filter": _build_role(config, filter_model),
        "new_filter_duplicate": _build_role(config, filter_model),
        "judge": _build_role(config, judge_model),
    }
    cases = load_selected_cases(data_zip, SMOKE_IDS)
    slots = [fixed_evidence_slots(case) for case in cases]
    raw_by_stage: dict[str, list[str]] = {}

    initial, raw_by_stage["initial"] = _complete_stage(
        models["initial"],
        [initial_differential_messages(case) for case in cases],
        config=config,
        temperature=0.0,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="initial",
    )
    old_filter_messages = [
        compatibility_messages(
            case,
            support,
            [(ACTION_ID, case_slots[ACTION_ID])],
        )
        for case, case_slots, support in zip(cases, slots, initial, strict=True)
    ]
    old_compatibility, raw_by_stage["old_filter"] = _complete_stage(
        models["old_filter"],
        old_filter_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_compatibility(
            text,
            num_diagnoses=DIAGNOSIS_COUNT,
            evidence_ids=(ACTION_ID,),
        ),
        stage="old_filter",
    )
    retained_old = [
        retain_by_threshold(support, rows)
        for support, rows in zip(initial, old_compatibility, strict=True)
    ]

    candidate_messages = [
        generated_candidate_messages(
            case,
            [(ACTION_ID, case_slots[ACTION_ID])],
        )
        for case, case_slots in zip(cases, slots, strict=True)
    ]
    duplicate_candidate_messages = [
        generated_candidate_messages(
            case,
            [(ACTION_ID, case_slots[ACTION_ID])],
        )
        for case, case_slots in zip(cases, slots, strict=True)
    ]
    candidates, raw_by_stage["candidate"] = _complete_stage(
        models["candidate"],
        candidate_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="candidate",
    )
    duplicate_candidates, raw_by_stage["candidate_duplicate"] = _complete_stage(
        models["candidate_duplicate"],
        duplicate_candidate_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="candidate_duplicate",
    )

    full_evidence = [
        (
            ("initial", case.initial_information),
            (ACTION_ID, case_slots[ACTION_ID]),
        )
        for case, case_slots in zip(cases, slots, strict=True)
    ]
    new_filter_messages = [
        compatibility_messages(case, support, evidence)
        for case, support, evidence in zip(
            cases, candidates, full_evidence, strict=True
        )
    ]
    duplicate_filter_messages = [
        compatibility_messages(case, support, evidence)
        for case, support, evidence in zip(
            cases, duplicate_candidates, full_evidence, strict=True
        )
    ]
    new_compatibility, raw_by_stage["new_filter"] = _complete_stage(
        models["new_filter"],
        new_filter_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_compatibility(
            text,
            num_diagnoses=DIAGNOSIS_COUNT,
            evidence_ids=("initial", ACTION_ID),
        ),
        stage="new_filter",
    )
    duplicate_compatibility, raw_by_stage["new_filter_duplicate"] = _complete_stage(
        models["new_filter_duplicate"],
        duplicate_filter_messages,
        config=config,
        temperature=0.0,
        parser=lambda text: parse_compatibility(
            text,
            num_diagnoses=DIAGNOSIS_COUNT,
            evidence_ids=("initial", ACTION_ID),
        ),
        stage="new_filter_duplicate",
    )
    filtered_new = [
        retain_by_threshold(support, rows)
        for support, rows in zip(candidates, new_compatibility, strict=True)
    ]
    duplicate_filtered_new = [
        retain_by_threshold(support, rows)
        for support, rows in zip(
            duplicate_candidates, duplicate_compatibility, strict=True
        )
    ]
    merged = [
        merge_support(old, new)
        for old, new in zip(retained_old, filtered_new, strict=True)
    ]
    duplicate_merged = [
        merge_support(old, new)
        for old, new in zip(retained_old, duplicate_filtered_new, strict=True)
    ]
    support_sets = [
        list(
            zip(
                SUPPORT_IDS,
                (initial_support, final_support, duplicate_support),
                strict=True,
            )
        )
        for initial_support, final_support, duplicate_support in zip(
            initial, merged, duplicate_merged, strict=True
        )
    ]
    audits, raw_by_stage["semantic_audit"] = _complete_stage(
        models["judge"],
        [
            semantic_audit_messages(case.final_diagnosis, supports)
            for case, supports in zip(cases, support_sets, strict=True)
        ],
        config=config,
        temperature=0.0,
        parser=parse_semantic_audit,
        stage="semantic_audit",
    )

    records = []
    for index, (
        case,
        case_slots,
        initial_support,
        old_rows,
        retained,
        generated,
        duplicate_generated,
        new_rows,
        duplicate_rows,
        valid_new,
        duplicate_valid_new,
        final_support,
        duplicate_support,
        audit,
    ) in enumerate(
        zip(
            cases,
            slots,
            initial,
            old_compatibility,
            retained_old,
            candidates,
            duplicate_candidates,
            new_compatibility,
            duplicate_compatibility,
            filtered_new,
            duplicate_filtered_new,
            merged,
            duplicate_merged,
            audits,
            strict=True,
        )
    ):
        scores = {
            row["id"]: float(row["best_match_score"])
            for row in audit["supports"]
        }
        final_keys = {value.casefold() for value in retained}
        duplicate_keys = set(final_keys)
        records.append(
            {
                "row": index,
                "source_id": case.source_id,
                "subset": case.subset,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "action_id": ACTION_ID,
                "stored_observation": case_slots[ACTION_ID],
                "source_target_leak": source_evidence_contains_target(
                    case, case_slots
                ),
                "supports": {
                    "initial": initial_support,
                    "retained_old": retained,
                    "generated_new": generated,
                    "filtered_new": valid_new,
                    "generated_new_duplicate": duplicate_generated,
                    "filtered_new_duplicate": duplicate_valid_new,
                    "filtered_retention": final_support,
                    "filtered_retention_duplicate": duplicate_support,
                },
                "old_compatibility": old_rows,
                "new_compatibility": new_rows,
                "duplicate_new_compatibility": duplicate_rows,
                "num_old_pruned": len(initial_support) - len(retained),
                "num_new_introduced": sum(
                    value.casefold() not in final_keys for value in final_support
                ),
                "num_duplicate_new_introduced": sum(
                    value.casefold() not in duplicate_keys
                    for value in duplicate_support
                ),
                "semantic_audit": audit,
                "duplicate_semantic_overlap": audit[
                    "duplicate_semantic_overlap"
                ],
                "duplicate_truth_score_gap": abs(
                    scores["filtered_retention"]
                    - scores["filtered_retention_duplicate"]
                ),
            }
        )

    usage = _usage(models)
    prompts_exact = candidate_messages == duplicate_candidate_messages
    summary = summarize(
        records,
        usage,
        candidate_generation_prompts_exact=prompts_exact,
    )
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "smoke_ids": list(SMOKE_IDS),
            "action_id": ACTION_ID,
            "filter_threshold": FILTER_THRESHOLD,
            "support_size": DIAGNOSIS_COUNT,
            "generator_model": generator_model,
            "filter_model": filter_model,
            "judge_model": judge_model,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_disabled": True,
            "temperatures": {
                "initial": 0.0,
                "candidate": 0.0,
                "filter": 0.0,
                "audit": 0.0,
            },
            "retries": 0,
            "truth_hidden_from_generation_and_filtering": True,
            "truth_enters_only_semantic_audit": True,
            "candidate_generation_prompts_exact": prompts_exact,
        },
        "summary": summary,
        "records": records,
        "raw_responses": raw_by_stage,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--filter-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "SERVING_GATE.json"
    try:
        payload = run_gate(
            config,
            args.data_zip,
            generator_model=args.generator_model,
            filter_model=args.filter_model,
            judge_model=args.judge_model,
        )
    except StructuredStageError as exc:
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "runtime_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "stage": exc.stage,
                    "row": exc.row,
                    "raw_responses": exc.raw_responses,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    except Exception as exc:
        output_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "status": "runtime_failure",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(output_path)
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    if payload["status"] != "passed":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
