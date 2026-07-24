#!/usr/bin/env python3
"""Qualify path-dependent support refreshes over deterministic ClinDiag slots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.clindiag_fixed_slot_audit import fixed_evidence_slots
from scripts.clindiag_mc_joint_model_gate import _build_role
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    _normalized_text,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    complete_parsed_many,
    parse_diagnoses,
)


SELECTION_SEED = 24294
SMOKE_IDS = ("25992750", "rare167")
REFRESH_ACTION_IDS = ("present_illness", "lab_1")
SUPPORT_IDS = (
    "initial",
    "after_present_illness",
    "after_lab_1",
    "after_lab_1_duplicate",
)
EXPECTED_REQUESTS = 10


def refresh_differential_messages(
    case: ClinDiagCase,
    acquired_observations: Sequence[tuple[str, Any]],
    previous_support: Sequence[str],
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
                "Regenerate a broad open-world clinical differential from only the "
                "visible patient evidence. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation:\n{case.initial_information}\n\n"
                "Acquired evidence in order:\n"
                + json.dumps(evidence, ensure_ascii=True, sort_keys=True)
                + "\n\nPrevious generated differential:\n- "
                + "\n- ".join(previous_support)
                + f"\n\nReturn exactly {count} updated, distinct, specific, "
                "unifying diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an explanation or object. Use all "
                "visible evidence, retain plausible earlier diagnoses, and include "
                "rare conditions when warranted. No answer options, case title, "
                "final diagnosis, or hidden benchmark label is provided."
            ),
        },
    ]


def stability_audit_messages(
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
                "The true diagnosis was hidden from every differential generator. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each support, score its closest diagnosis to the truth in [0,1]. "
                "Use at least 0.8 only for the same diagnosis or a standard clinical "
                "synonym; broad parents, related diseases, and alternative subtypes "
                "score below 0.8. Then compare after_lab_1 with its exact-prompt "
                "duplicate semantically. For each direction, report the fraction of "
                "diagnoses having a same-disease or standard-synonym match in the "
                "other list; related diagnoses do not count. Return exactly "
                '{"supports":['
                '{"id":"initial","best_match_score":0.0,"reason":"brief"},'
                '{"id":"after_present_illness","best_match_score":0.0,'
                '"reason":"brief"},'
                '{"id":"after_lab_1","best_match_score":0.0,"reason":"brief"},'
                '{"id":"after_lab_1_duplicate","best_match_score":0.0,'
                '"reason":"brief"}],'
                '"lab_to_duplicate_overlap":0.0,'
                '"duplicate_to_lab_overlap":0.0,"overlap_reason":"brief"}; '
                "preserve support IDs and order.\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_stability_audit(
    text: str,
    support_ids: Sequence[str] = SUPPORT_IDS,
) -> dict[str, Any]:
    from environments.mediq.parsing import parse_json_object

    raw = parse_json_object(text)
    rows = raw.get("supports")
    if not isinstance(rows, list) or len(rows) != len(support_ids):
        raise ValueError("audit must contain one row per support")
    supports = []
    for support_id, row in zip(support_ids, rows, strict=True):
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
    directional = []
    for field in ("lab_to_duplicate_overlap", "duplicate_to_lab_overlap"):
        value = raw.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{field} must be numeric")
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{field} must be in [0,1]")
        directional.append(value)
    overlap_reason = raw.get("overlap_reason")
    if not isinstance(overlap_reason, str) or not overlap_reason.strip():
        raise ValueError("overlap_reason must be nonempty")
    return {
        "supports": supports,
        "lab_to_duplicate_overlap": directional[0],
        "duplicate_to_lab_overlap": directional[1],
        "duplicate_semantic_overlap": min(directional),
        "overlap_reason": overlap_reason.strip(),
    }


def source_evidence_contains_target(
    case: ClinDiagCase,
    slots: dict[str, Any],
) -> bool:
    target = _normalized_text(case.final_diagnosis)
    visible_source = (
        _normalized_text(case.initial_information),
        *(_normalized_text(slots[action_id]) for action_id in REFRESH_ACTION_IDS),
    )
    return bool(target) and any(target in value for value in visible_source)


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
    all_supports_size_twelve: bool,
    duplicate_prompts_exact: bool,
    source_target_leaks: int,
) -> dict[str, Any]:
    overlaps = [float(record["duplicate_semantic_overlap"]) for record in records]
    score_gaps = [float(record["duplicate_truth_score_gap"]) for record in records]
    gates = {
        "exactly_10_requests": usage["physical_requests"] == EXPECTED_REQUESTS,
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "all_supports_size_twelve": all_supports_size_twelve,
        "duplicate_prompts_exact": duplicate_prompts_exact,
        "no_hidden_target_in_source_evidence": source_target_leaks == 0,
        "both_duplicate_semantic_overlaps_at_least_0_80": (
            len(overlaps) == 2 and all(value >= 0.80 for value in overlaps)
        ),
        "both_duplicate_truth_score_gaps_at_most_0_05": (
            len(score_gaps) == 2 and all(value <= 0.05 for value in score_gaps)
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "duplicate_semantic_overlaps": overlaps,
        "duplicate_truth_score_gaps": score_gaps,
        "source_target_leaks": source_target_leaks,
        "gates": gates,
    }


def _complete_supports(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    config: Config,
    stage: str,
) -> list[list[str]]:
    return complete_parsed_many(
        model,
        messages,
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage=stage,
        retries=0,
    )


def run_smoke(
    config: Config,
    data_zip: Path,
    *,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    models = {
        "initial": _build_role(config, generator_model),
        "present": _build_role(config, generator_model),
        "lab": _build_role(config, generator_model),
        "duplicate": _build_role(config, generator_model),
        "judge": _build_role(config, judge_model),
    }
    cases = load_selected_cases(data_zip, SMOKE_IDS)
    slots = [fixed_evidence_slots(case) for case in cases]

    initial = _complete_supports(
        models["initial"],
        [initial_differential_messages(case) for case in cases],
        config,
        "initial_support",
    )
    present_messages = [
        refresh_differential_messages(
            case,
            [("present_illness", case_slots["present_illness"])],
            support,
        )
        for case, case_slots, support in zip(cases, slots, initial, strict=True)
    ]
    after_present = _complete_supports(
        models["present"],
        present_messages,
        config,
        "present_illness_refresh",
    )
    lab_messages = [
        refresh_differential_messages(
            case,
            [
                ("present_illness", case_slots["present_illness"]),
                ("lab_1", case_slots["lab_1"]),
            ],
            support,
        )
        for case, case_slots, support in zip(
            cases, slots, after_present, strict=True
        )
    ]
    duplicate_messages = [
        refresh_differential_messages(
            case,
            [
                ("present_illness", case_slots["present_illness"]),
                ("lab_1", case_slots["lab_1"]),
            ],
            support,
        )
        for case, case_slots, support in zip(
            cases, slots, after_present, strict=True
        )
    ]
    after_lab = _complete_supports(
        models["lab"],
        lab_messages,
        config,
        "lab_1_refresh",
    )
    after_lab_duplicate = _complete_supports(
        models["duplicate"],
        duplicate_messages,
        config,
        "lab_1_refresh_duplicate",
    )
    support_sets = [
        list(zip(SUPPORT_IDS, values, strict=True))
        for values in zip(
            initial,
            after_present,
            after_lab,
            after_lab_duplicate,
            strict=True,
        )
    ]
    raw_audits = models["judge"].chat_complete_messages_batched(
        [
            stability_audit_messages(case.final_diagnosis, case_supports)
            for case, case_supports in zip(cases, support_sets, strict=True)
        ],
        temperature=0.0,
        block_size=256,
        max_new_tokens=int(config.openrouter_max_output_tokens),
    )
    audits: list[dict[str, Any] | None] = []
    audit_errors = []
    for index, response in enumerate(raw_audits):
        try:
            audits.append(parse_stability_audit(response))
        except ValueError as exc:
            audits.append(None)
            audit_errors.append(
                {
                    "row": index,
                    "error": str(exc),
                    "raw_response": response,
                }
            )

    usage = _usage(models)
    if audit_errors:
        return {
            "schema_version": 1,
            "status": "runtime_failure",
            "error_type": "StructuredAuditError",
            "error": audit_errors[0]["error"],
            "protocol": {
                "selection_seed": SELECTION_SEED,
                "smoke_ids": list(SMOKE_IDS),
                "refresh_action_ids": list(REFRESH_ACTION_IDS),
                "generator_model": generator_model,
                "judge_model": judge_model,
                "expected_requests": EXPECTED_REQUESTS,
                "support_size": DIAGNOSIS_COUNT,
                "reasoning_disabled": True,
                "retries": 0,
                "truth_hidden_from_generators": True,
                "truth_enters_only_semantic_audit": True,
            },
            "summary": {
                "physical_requests": usage["physical_requests"],
                "reasoning_tokens": usage["reasoning_tokens"],
                "structured_audit_errors": len(audit_errors),
            },
            "partial_records": [
                {
                    "source_id": case.source_id,
                    "subset": case.subset,
                    "supports": {
                        support_id: support
                        for support_id, support in case_supports
                    },
                }
                for case, case_supports in zip(cases, support_sets, strict=True)
            ],
            "raw_audit_responses": raw_audits,
            "audit_errors": audit_errors,
            "usage": usage,
        }

    records = []
    for case, case_slots, case_supports, audit in zip(
        cases, slots, support_sets, audits, strict=True
    ):
        assert audit is not None
        scores = {
            row["id"]: row["best_match_score"] for row in audit["supports"]
        }
        records.append(
            {
                "source_id": case.source_id,
                "subset": case.subset,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "stored_observations": {
                    action_id: case_slots[action_id]
                    for action_id in REFRESH_ACTION_IDS
                },
                "supports": {
                    support_id: support
                    for support_id, support in case_supports
                },
                "semantic_audit": audit,
                "duplicate_semantic_overlap": audit[
                    "duplicate_semantic_overlap"
                ],
                "duplicate_truth_score_gap": abs(
                    scores["after_lab_1"]
                    - scores["after_lab_1_duplicate"]
                ),
            }
        )

    summary = summarize(
        records,
        usage,
        all_supports_size_twelve=all(
            len(support) == DIAGNOSIS_COUNT
            for case_supports in support_sets
            for _, support in case_supports
        ),
        duplicate_prompts_exact=lab_messages == duplicate_messages,
        source_target_leaks=sum(
            source_evidence_contains_target(case, case_slots)
            for case, case_slots in zip(cases, slots, strict=True)
        ),
    )
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "smoke_ids": list(SMOKE_IDS),
            "refresh_action_ids": list(REFRESH_ACTION_IDS),
            "generator_model": generator_model,
            "judge_model": judge_model,
            "expected_requests": EXPECTED_REQUESTS,
            "support_size": DIAGNOSIS_COUNT,
            "reasoning_disabled": True,
            "retries": 0,
            "truth_hidden_from_generators": True,
            "truth_enters_only_semantic_audit": True,
        },
        "summary": summary,
        "records": records,
        "raw_audit_responses": raw_audits,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "SERVING_SMOKE.json"
    try:
        payload = run_smoke(
            config,
            args.data_zip,
            generator_model=args.generator_model,
            judge_model=args.judge_model,
        )
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
