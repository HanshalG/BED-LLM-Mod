#!/usr/bin/env python3
"""Audit a fixed generic evidence-slot construction for ClinDiag."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import zipfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clindiag_binary_joint_model_gate import INTERFACE_IDS, JOINT_IDS
from scripts.clindiag_branch_opportunity_gate import (
    OPPORTUNITY_IDS,
    SMOKE_IDS as OPPORTUNITY_SMOKE_IDS,
)
from scripts.clindiag_joint_response_function_gate import (
    FIDELITY_IDS,
    QUALIFICATION_IDS,
)
from scripts.clindiag_mc_joint_model_gate import SMOKE_IDS as MC_SMOKE_IDS
from scripts.clindiag_staged_generator_gate import (
    CLINDIAG_ZIP_SHA256,
    DEVELOPMENT_IDS,
    HOLDOUT_IDS,
    SMOKE_IDS as GENERATOR_SMOKE_IDS,
    ClinDiagCase,
    _assert_no_lexical_target_leak,
    _normalized_text,
    _parse_case,
)


ACTION_IDS = (
    "present_illness",
    "prior_history",
    "family_social",
    "exam_1",
    "lab_1",
    "lab_2",
    "imaging_1",
    "other_1",
)
BLOCKED_EVIDENCE_PATTERN = re.compile(
    r"\b(?:"
    r"biops(?:y|ies)|histopath\w*|patholog\w*|histolog\w*|genetic\w*|"
    r"genomic\w*|gene|genes|mutation\w*|sequenc\w*|molecular\w*|"
    r"immunohistochem\w*|cytogenetic\w*|karyotyp\w*|autops\w*|"
    r"surgery|surgical|resection|implantation|transfus\w*|treatment|therapy|"
    r"postoperative|transplant\w*"
    r")\b",
    re.IGNORECASE,
)
RESERVED_IDS = frozenset(
    DEVELOPMENT_IDS
    + HOLDOUT_IDS
    + GENERATOR_SMOKE_IDS
    + OPPORTUNITY_IDS
    + OPPORTUNITY_SMOKE_IDS
    + MC_SMOKE_IDS
    + INTERFACE_IDS
    + JOINT_IDS
    + QUALIFICATION_IDS
    + FIDELITY_IDS
)


def _eligible_entries(values: Any, count: int) -> list[dict[str, Any]]:
    result = []
    for value in values or []:
        if not isinstance(value, dict):
            continue
        if BLOCKED_EVIDENCE_PATTERN.search(json.dumps(value, sort_keys=True)):
            continue
        result.append(value)
        if len(result) == count:
            break
    return result


def fixed_evidence_slots(case: ClinDiagCase) -> dict[str, Any]:
    history = case.medical_history.get("medical_history") or {}
    physical = _eligible_entries(
        case.physical_examination.get("physical_examinations"),
        1,
    )
    laboratory = _eligible_entries(
        case.diagnostic_test.get("laboratory_examinations"),
        2,
    )
    imaging = _eligible_entries(
        case.diagnostic_test.get("radiographic_examinations"),
        1,
    )
    other = _eligible_entries(
        case.diagnostic_test.get("other_examinations"),
        1,
    )
    return {
        "present_illness": history.get("history_of_present_illness"),
        "prior_history": history.get("past_medical_history"),
        "family_social": {
            "family_history": history.get("family_history"),
            "social_history": history.get("social_history"),
        },
        "exam_1": physical[0] if physical else None,
        "lab_1": laboratory[0] if len(laboratory) >= 1 else None,
        "lab_2": laboratory[1] if len(laboratory) >= 2 else None,
        "imaging_1": imaging[0] if imaging else None,
        "other_1": other[0] if other else None,
    }


def missing_slot_ids(slots: dict[str, Any]) -> list[str]:
    return [
        action_id
        for action_id in ACTION_IDS
        if _normalized_text(slots.get(action_id)) in {"", "none", "not specified"}
    ]


def audit_archive(data_zip: Path, *, verify_hash: bool = True) -> dict[str, Any]:
    if verify_hash:
        digest = hashlib.sha256(data_zip.read_bytes()).hexdigest()
        if digest != CLINDIAG_ZIP_SHA256:
            raise ValueError(
                f"ClinDiag archive hash mismatch: expected {CLINDIAG_ZIP_SHA256}, "
                f"got {digest}"
            )
    eligible = []
    missing = Counter()
    case_exclusions = Counter()
    with zipfile.ZipFile(data_zip) as archive:
        source_ids = sorted(
            {
                name.split("/", 1)[0]
                for name in archive.namelist()
                if "/" in name and name.split("/", 1)[0]
            }
        )
        for source_id in source_ids:
            if source_id in RESERVED_IDS:
                continue
            try:
                case = _parse_case(archive, source_id)
                _assert_no_lexical_target_leak(case)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                case_exclusions[type(exc).__name__] += 1
                continue
            slots = fixed_evidence_slots(case)
            missing_ids = missing_slot_ids(slots)
            if missing_ids:
                missing.update(missing_ids)
                continue
            eligible.append(
                {
                    "source_id": source_id,
                    "subset": case.subset,
                    "diagnosis": case.final_diagnosis,
                }
            )
    by_subset = Counter(row["subset"] for row in eligible)
    return {
        "schema_version": 1,
        "archive_sha256": CLINDIAG_ZIP_SHA256,
        "num_archive_cases": len(source_ids),
        "num_reserved_cases": len(RESERVED_IDS),
        "action_ids": list(ACTION_IDS),
        "num_eligible_cases": len(eligible),
        "eligible_by_subset": dict(sorted(by_subset.items())),
        "case_exclusions": dict(sorted(case_exclusions.items())),
        "missing_slot_exclusions": dict(sorted(missing.items())),
        "construction": {
            "same_action_labels_for_every_case": True,
            "all_selected_slots_nonempty": True,
            "confirmatory_and_intervention_entries_filtered": True,
            "observations_are_stored_not_generated": True,
            "procedure_names_hidden_until_slot_selected": True,
        },
        "eligible_cases": eligible,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-zip",
        type=Path,
        default=Path("external/ClinDiag/Clindiag_Benchmark.zip"),
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--skip-hash-check", action="store_true")
    args = parser.parse_args(argv)
    payload = audit_archive(
        args.data_zip,
        verify_hash=not args.skip_hash_check,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(args.output)
    print(
        json.dumps(
            {
                "num_eligible_cases": payload["num_eligible_cases"],
                "eligible_by_subset": payload["eligible_by_subset"],
                "missing_slot_exclusions": payload["missing_slot_exclusions"],
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
