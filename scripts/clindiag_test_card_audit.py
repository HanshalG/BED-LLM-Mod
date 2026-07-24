#!/usr/bin/env python3
"""Audit whether ClinDiag procedure records define a valid BED action space."""

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

from scripts.clindiag_branch_opportunity_gate import (
    OPPORTUNITY_IDS,
    SMOKE_IDS as OPPORTUNITY_SMOKE_IDS,
)
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


TEST_CATEGORIES = (
    "laboratory_examinations",
    "radiographic_examinations",
    "other_examinations",
)
CONFIRMATORY_PATTERN = re.compile(
    r"\b(?:"
    r"biops(?:y|ies)|histopath\w*|patholog\w*|genetic\w*|genomic\w*|"
    r"gene|genes|mutation\w*|sequenc\w*|molecular\w*|immunohistochem\w*|"
    r"cytogenetic\w*|karyotyp\w*|fish|autops\w*|postmortem"
    r")\b",
    re.IGNORECASE,
)
INTERVENTION_PATTERN = re.compile(
    r"\b(?:"
    r"treatment|therapy|transfus\w*|surgery|surgical|resection|implantation|"
    r"pacing|angioplasty|cricothyrotomy|transplant\w*|bursectomy|"
    r"administration|replacement"
    r")\b",
    re.IGNORECASE,
)
RESERVED_IDS = frozenset(
    DEVELOPMENT_IDS
    + HOLDOUT_IDS
    + GENERATOR_SMOKE_IDS
    + OPPORTUNITY_IDS
    + OPPORTUNITY_SMOKE_IDS
)


def normalize_action_name(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def eligible_test_cards(
    case: ClinDiagCase,
) -> tuple[list[dict[str, Any]], Counter[str]]:
    """Return grouped diagnostic cards after fixed target-blind filters."""

    grouped: dict[str, dict[str, Any]] = {}
    excluded: Counter[str] = Counter()
    for category in TEST_CATEGORIES:
        for item in case.diagnostic_test.get(category) or []:
            if not isinstance(item, dict):
                excluded["non_object"] += 1
                continue
            procedure_name = str(item.get("procedure_name") or "").strip()
            if not procedure_name or procedure_name.casefold() == "not specified":
                excluded["missing_name"] += 1
                continue
            if CONFIRMATORY_PATTERN.search(procedure_name):
                excluded["confirmatory"] += 1
                continue
            if INTERVENTION_PATTERN.search(procedure_name):
                excluded["intervention"] += 1
                continue
            key = normalize_action_name(procedure_name)
            card = grouped.setdefault(
                key,
                {
                    "procedure_name": procedure_name,
                    "category": category,
                    "observations": [],
                },
            )
            card["observations"].append(item)
    return list(grouped.values()), excluded


def _archive_case_ids(archive: zipfile.ZipFile) -> list[str]:
    return sorted(
        {
            name.split("/", 1)[0]
            for name in archive.namelist()
            if "/" in name and name.split("/", 1)[0]
        }
    )


def audit_archive(
    data_zip: Path,
    *,
    verify_hash: bool = True,
    include_cases: bool = False,
) -> dict[str, Any]:
    if verify_hash:
        digest = hashlib.sha256(data_zip.read_bytes()).hexdigest()
        if digest != CLINDIAG_ZIP_SHA256:
            raise ValueError(
                f"ClinDiag archive hash mismatch: expected {CLINDIAG_ZIP_SHA256}, "
                f"got {digest}"
            )

    rows: list[dict[str, Any]] = []
    parse_exclusions: Counter[str] = Counter()
    card_exclusions: Counter[str] = Counter()
    lexical_action_leaks: list[dict[str, Any]] = []
    with zipfile.ZipFile(data_zip) as archive:
        all_ids = _archive_case_ids(archive)
        for source_id in all_ids:
            if source_id in RESERVED_IDS:
                continue
            try:
                case = _parse_case(archive, source_id)
                _assert_no_lexical_target_leak(case)
            except (ValueError, KeyError, json.JSONDecodeError) as exc:
                parse_exclusions[type(exc).__name__] += 1
                continue

            cards, exclusions = eligible_test_cards(case)
            card_exclusions.update(exclusions)
            target = _normalized_text(case.final_diagnosis)
            leaks = [
                card["procedure_name"]
                for card in cards
                if target and target in _normalized_text(card["procedure_name"])
            ]
            if leaks:
                lexical_action_leaks.append(
                    {
                        "source_id": source_id,
                        "diagnosis": case.final_diagnosis,
                        "procedure_names": leaks,
                    }
                )
                continue
            rows.append(
                {
                    "source_id": source_id,
                    "subset": case.subset,
                    "diagnosis": case.final_diagnosis,
                    "num_test_cards": len(cards),
                    "test_cards": [
                        {
                            "procedure_name": card["procedure_name"],
                            "category": card["category"],
                            "num_grouped_observations": len(card["observations"]),
                        }
                        for card in cards
                    ],
                }
            )

    thresholds = {}
    for threshold in (4, 5, 6, 7, 8, 10):
        selected = [row for row in rows if row["num_test_cards"] >= threshold]
        by_subset = Counter(row["subset"] for row in selected)
        thresholds[str(threshold)] = {
            "total": len(selected),
            "challenging": by_subset["challenging"],
            "rare": by_subset["rare"],
        }

    card_count_distribution = Counter(row["num_test_cards"] for row in rows)
    result = {
        "schema_version": 1,
        "archive_sha256": CLINDIAG_ZIP_SHA256,
        "num_archive_cases": len(all_ids),
        "num_reserved_cases": len(RESERVED_IDS),
        "num_fresh_statically_eligible_cases": len(rows),
        "parse_or_case_exclusions": dict(sorted(parse_exclusions.items())),
        "test_entry_exclusions": dict(sorted(card_exclusions.items())),
        "lexical_action_leaks": lexical_action_leaks,
        "test_card_count_distribution": {
            str(count): frequency
            for count, frequency in sorted(card_count_distribution.items())
        },
        "minimum_card_thresholds": thresholds,
        "bed_validity": {
            "valid_test_card_environment": False,
            "reasons": [
                (
                    "The recorded procedure set is selected retrospectively under the "
                    "true case, so action availability is not target independent."
                ),
                (
                    "Unperformed tests have no observed outcome, so the archive does "
                    "not identify counterfactual branches p(y|theta,x)."
                ),
                (
                    "Disease-specific procedure names can reveal the source clinicians' "
                    "differential even when the diagnosis string does not match lexically."
                ),
            ],
        },
    }
    if include_cases:
        result["cases"] = rows
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-zip",
        type=Path,
        default=Path("external/ClinDiag/Clindiag_Benchmark.zip"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--skip-hash-check", action="store_true")
    parser.add_argument("--include-cases", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    result = audit_archive(
        args.data_zip,
        verify_hash=not args.skip_hash_check,
        include_cases=args.include_cases,
    )
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(rendered, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
        print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
