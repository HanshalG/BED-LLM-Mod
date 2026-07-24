#!/usr/bin/env python3
"""Qualify open-world diagnosis generation on staged ClinDiag evidence."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import zipfile

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.icraft_staged_diagnosis_unlock import (
    _dedupe,
    complete_parsed_many,
    parse_diagnoses,
    parse_semantic_diagnosis,
    semantic_diagnosis_messages,
)


CLINDIAG_SOURCE_COMMIT = "f9b5c181e8d120d6a244accba9422ffb89d1f319"
CLINDIAG_ZIP_SHA256 = (
    "a9ea339fc3a6ded91f5c1769589c5c374dd32c266072511260d4102e544539c1"
)
SELECTION_SEED = 24289
DIAGNOSIS_COUNT = 12
COVERAGE_THRESHOLD = 0.80

SMOKE_IDS = (
    "16837683",
    "rare297",
    "21506744",
    "rare28",
)
DEVELOPMENT_IDS = (
    "rare27",
    "11232006",
    "rare26",
    "rare197",
    "rare238",
    "rare145",
    "rare93",
    "14715828",
    "23802518",
    "rare164",
    "30044928",
    "20068123",
    "rare245",
    "19213685",
    "rare151",
    "17845596",
    "23071170",
    "24104373",
    "25210096",
    "rare42",
)
HOLDOUT_IDS = (
    "rare103",
    "15705922",
    "23866852",
    "11096017",
    "rare117",
    "16798335",
    "rare111",
    "rare227",
    "11772594",
    "rare274",
    "rare224",
    "30231229",
    "12894069",
    "12350194",
    "11396442",
    "21776542",
    "24491977",
    "rare291",
    "15583598",
    "rare244",
    "17484731",
    "rare239",
    "29878124",
    "24423348",
    "rare163",
    "21289269",
    "rare96",
    "rare67",
    "rare31",
    "rare40",
    "rare251",
    "rare182",
    "rare191",
    "20494731",
    "11900499",
    "25555030",
    "20554986",
    "rare240",
    "rare253",
    "10987731",
    "15010448",
    "374752",
    "655289",
    "rare60",
    "rare218",
    "rare127",
    "rare210",
    "11423847",
    "17035606",
    "rare90",
    "rare12",
    "12444199",
    "rare91",
    "31216403",
    "rare80",
    "28240659",
    "rare75",
    "rare299",
    "rare114",
    "10698822",
)


@dataclass(frozen=True)
class ClinDiagCase:
    source_id: str
    subset: str
    initial_information: str
    medical_history: dict[str, Any]
    physical_examination: dict[str, Any]
    diagnostic_test: dict[str, Any]
    final_diagnosis: str

    def full_evidence(self) -> str:
        sections = (
            ("Medical history", self.medical_history),
            ("Physical examination", self.physical_examination),
            ("Diagnostic tests", self.diagnostic_test),
        )
        return "\n\n".join(
            f"{label}:\n{json.dumps(value, ensure_ascii=True, sort_keys=True)}"
            for label, value in sections
        )


def _normalized_text(value: Any) -> str:
    if isinstance(value, dict):
        return " ".join(_normalized_text(item) for item in value.values())
    if isinstance(value, list):
        return " ".join(_normalized_text(item) for item in value)
    return " ".join(re.findall(r"[a-z0-9]+", str(value).casefold()))


def _read_json(archive: zipfile.ZipFile, source_id: str, filename: str) -> Any:
    with archive.open(f"{source_id}/{filename}") as handle:
        return json.load(handle)


def _parse_case(archive: zipfile.ZipFile, source_id: str) -> ClinDiagCase:
    initial = _read_json(archive, source_id, "initial_information.json")
    history = _read_json(archive, source_id, "medical_history.json")
    physical = _read_json(archive, source_id, "physical_examination.json")
    tests = _read_json(archive, source_id, "diagnostic_test.json")
    diagnosis = _read_json(archive, source_id, "diagnosis.json")
    initial_text = initial.get("initial_information")
    final_diagnosis = (diagnosis.get("diagnosis") or {}).get("final_diagnosis")
    if not isinstance(initial_text, str) or not initial_text.strip():
        raise ValueError(f"{source_id}: missing initial information")
    if not isinstance(final_diagnosis, str) or not final_diagnosis.strip():
        raise ValueError(f"{source_id}: missing final diagnosis")
    if not all(isinstance(item, dict) for item in (history, physical, tests)):
        raise ValueError(f"{source_id}: evidence stages must be JSON objects")
    return ClinDiagCase(
        source_id=source_id,
        subset="rare" if source_id.startswith("rare") else "challenging",
        initial_information=initial_text.strip(),
        medical_history=history,
        physical_examination=physical,
        diagnostic_test=tests,
        final_diagnosis=final_diagnosis.strip(),
    )


def _assert_no_lexical_target_leak(case: ClinDiagCase) -> None:
    target = _normalized_text(case.final_diagnosis)
    tokens = target.split()
    if not target or len(tokens) > 12:
        raise ValueError(f"{case.source_id}: ineligible diagnosis length")
    evidence = (
        _normalized_text(case.initial_information),
        _normalized_text(case.medical_history),
        _normalized_text(case.physical_examination),
        _normalized_text(case.diagnostic_test),
    )
    if any(target in item for item in evidence):
        raise ValueError(f"{case.source_id}: full diagnosis leaks lexically")
    first_four = " ".join(tokens[: min(4, len(tokens))])
    if len(first_four) >= 12 and any(first_four in item for item in evidence):
        raise ValueError(f"{case.source_id}: diagnosis prefix leaks lexically")


def load_selected_cases(
    data_zip: Path,
    source_ids: Sequence[str],
    *,
    verify_hash: bool = True,
) -> list[ClinDiagCase]:
    if verify_hash:
        digest = hashlib.sha256(data_zip.read_bytes()).hexdigest()
        if digest != CLINDIAG_ZIP_SHA256:
            raise ValueError(
                f"ClinDiag archive hash mismatch: expected {CLINDIAG_ZIP_SHA256}, "
                f"got {digest}"
            )
    with zipfile.ZipFile(data_zip) as archive:
        cases = [_parse_case(archive, source_id) for source_id in source_ids]
    for case in cases:
        _assert_no_lexical_target_leak(case)
    return cases


def initial_differential_messages(
    case: ClinDiagCase,
    count: int = DIAGNOSIS_COUNT,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Construct a broad open-world clinical differential using only "
                "the evidence shown. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation:\n{case.initial_information}\n\n"
                f"Return exactly {count} distinct, specific, unifying diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an explanation or object. Include rare "
                "conditions when plausible. No answer options, case title, final "
                "diagnosis, or hidden benchmark label is provided."
            ),
        },
    ]


def full_differential_messages(
    case: ClinDiagCase,
    initial_diagnoses: Sequence[str],
    count: int = DIAGNOSIS_COUNT,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Regenerate a precise open-world differential after receiving the "
                "complete clinical workup. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Initial presentation:\n{case.initial_information}\n\n"
                f"Complete workup evidence:\n{case.full_evidence()}\n\n"
                "Earlier differential:\n- "
                + "\n- ".join(initial_diagnoses)
                + f"\n\nReturn exactly {count} updated, distinct, specific, "
                "unifying diagnoses as "
                '{"diagnoses":["diagnosis name", ...]}. Each item must be one '
                "diagnosis-name string, not an explanation or object. Infer the "
                "diagnosis from all findings, including rare diseases, genetics, "
                "pathology, and imaging when present. No answer options, case title, "
                "final diagnosis, or hidden benchmark label is provided."
            ),
        },
    ]


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    initial_covered = sum(
        record["initial_measurement"]["covered"] for record in records
    )
    full_covered = sum(record["full_measurement"]["covered"] for record in records)
    omissions = len(records) - initial_covered
    recovered = sum(
        not record["initial_measurement"]["covered"]
        and record["full_measurement"]["covered"]
        for record in records
    )
    gains = [
        record["full_measurement"]["best_match_score"]
        - record["initial_measurement"]["best_match_score"]
        for record in records
    ]
    subset_coverage = {
        subset: sum(
            record["full_measurement"]["covered"]
            for record in records
            if record["subset"] == subset
        )
        for subset in ("challenging", "rare")
    }
    summary = {
        "num_tasks": len(records),
        "initial_covered": initial_covered,
        "initial_omitted": omissions,
        "full_generated_covered": full_covered,
        "initially_omitted_recovered_by_full_evidence": recovered,
        "recovery_fraction_among_initial_omissions": (
            recovered / omissions if omissions else 0.0
        ),
        "mean_full_best_match_gain": float(np.mean(gains)),
        "full_coverage_by_subset": subset_coverage,
    }
    gates = {
        "all_twenty_tasks_completed": len(records) == 20,
        "initial_support_not_saturated": initial_covered <= 8,
        "full_support_covers_at_least_sixteen": full_covered >= 16,
        "at_least_ten_initial_omissions_recovered": recovered >= 10,
        "recovery_fraction_at_least_0_70": (
            summary["recovery_fraction_among_initial_omissions"] >= 0.70
        ),
        "mean_match_gain_at_least_0_30": (
            summary["mean_full_best_match_gain"] >= 0.30
        ),
        "challenging_coverage_at_least_eight": (
            subset_coverage["challenging"] >= 8
        ),
        "rare_coverage_at_least_seven": subset_coverage["rare"] >= 7,
    }
    gates["all_pass"] = all(gates.values())
    return {**summary, "gates": gates}


def _build_models(
    config: Config,
    generator_model: str,
    judge_model: str,
) -> tuple[Any, Any]:
    base = config.model_pairs[0].questioner
    common = {
        "thinking": None,
        "reasoning_effort": "none",
        "reasoning_max_tokens": None,
        "thinking_max_new_tokens": None,
        "thinking_final_max_new_tokens": None,
    }
    generator = build_model_adapter(
        replace(base, model=generator_model, **common),
        config,
    )
    judge = build_model_adapter(
        replace(base, model=judge_model, **common),
        config,
    )
    return generator, judge


def _run_cases(
    config: Config,
    data_zip: Path,
    source_ids: Sequence[str],
    *,
    generator_model: str,
    judge_model: str,
    retries: int,
    semantic_limit: int | None = None,
) -> tuple[
    list[dict[str, Any]],
    dict[str, Any],
    list[list[str]],
    list[list[str]],
]:
    generator, judge = _build_models(config, generator_model, judge_model)
    cases = load_selected_cases(data_zip, source_ids)
    parser = lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT)
    initial = complete_parsed_many(
        generator,
        [initial_differential_messages(case) for case in cases],
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=parser,
        stage="initial_differential",
        retries=retries,
    )
    full = complete_parsed_many(
        generator,
        [
            full_differential_messages(case, diagnoses)
            for case, diagnoses in zip(cases, initial, strict=True)
        ],
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=parser,
        stage="full_differential",
        retries=retries,
    )
    measured_cases = cases[:semantic_limit] if semantic_limit is not None else cases
    support_ids = ("initial", "full_generated", "merged")
    semantic = complete_parsed_many(
        judge,
        [
            semantic_diagnosis_messages(
                case.final_diagnosis,
                (
                    ("initial", initial[index]),
                    ("full_generated", full[index]),
                    ("merged", _dedupe([*initial[index], *full[index]])),
                ),
            )
            for index, case in enumerate(measured_cases)
        ],
        temperature=0.0,
        max_new_tokens=int(config.openrouter_max_output_tokens),
        parser=lambda text: parse_semantic_diagnosis(text, support_ids),
        stage="semantic_measurement",
        retries=retries,
    )
    records = []
    for index, (case, measurements) in enumerate(
        zip(measured_cases, semantic, strict=True)
    ):
        records.append(
            {
                "source_id": case.source_id,
                "subset": case.subset,
                "initial_information": case.initial_information,
                "full_evidence": case.full_evidence(),
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "initial_diagnoses": initial[index],
                "full_generated_diagnoses": full[index],
                "initial_measurement": measurements[0],
                "full_measurement": measurements[1],
                "merged_measurement": measurements[2],
            }
        )
    return (
        records,
        {
            "generator": generator.usage_snapshot(),
            "judge": judge.usage_snapshot(),
        },
        initial,
        full,
    )


def run_development(
    config: Config,
    data_zip: Path,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    records, usage, _, _ = _run_cases(
        config,
        data_zip,
        DEVELOPMENT_IDS,
        generator_model=generator_model,
        judge_model=judge_model,
        retries=int(config.mediq_structured_max_retries),
    )
    summary = summarize(records)
    return {
        "schema_version": 1,
        "status": (
            "passed" if summary["gates"]["all_pass"] else "development_gate_failed"
        ),
        "protocol": {
            "dataset": "ClinDiag-Benchmark",
            "source_commit": CLINDIAG_SOURCE_COMMIT,
            "archive_sha256": CLINDIAG_ZIP_SHA256,
            "selection_seed": SELECTION_SEED,
            "smoke_ids": list(SMOKE_IDS),
            "development_ids": list(DEVELOPMENT_IDS),
            "holdout_ids": list(HOLDOUT_IDS),
            "diagnosis_count": DIAGNOSIS_COUNT,
            "coverage_threshold": COVERAGE_THRESHOLD,
            "lexical_target_leaks_excluded": True,
            "truth_used_for_measurement_only": True,
            "generator_model": generator_model,
            "judge_model": judge_model,
            "reasoning_disabled": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def run_serving_smoke(
    config: Config,
    data_zip: Path,
    generator_model: str,
    judge_model: str,
) -> dict[str, Any]:
    records, usage, initial, full = _run_cases(
        config,
        data_zip,
        SMOKE_IDS,
        generator_model=generator_model,
        judge_model=judge_model,
        retries=0,
        semantic_limit=2,
    )
    requests = sum(int(item["adapter_requests"]) for item in usage.values())
    reasoning = sum(int(item["adapter_reasoning_tokens"]) for item in usage.values())
    support_sizes_valid = (
        len(initial) == 4
        and len(full) == 4
        and all(len(values) == DIAGNOSIS_COUNT for values in [*initial, *full])
    )
    return {
        "schema_version": 1,
        "status": (
            "passed"
            if requests == 10 and reasoning == 0 and support_sizes_valid
            else "failed"
        ),
        "physical_requests": requests,
        "expected_physical_requests": 10,
        "reasoning_tokens": reasoning,
        "stage_counts": {"initial": 4, "full": 4, "semantic": 2},
        "parsed_initial_sizes": [len(values) for values in initial],
        "parsed_full_sizes": [len(values) for values in full],
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development"),
        required=True,
    )
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = (
            run_serving_smoke(
                config,
                args.data_zip,
                args.generator_model,
                args.judge_model,
            )
            if args.stage == "serving_smoke"
            else run_development(
                config,
                args.data_zip,
                args.generator_model,
                args.judge_model,
            )
        )
        filename = (
            "SERVING_SMOKE.json"
            if args.stage == "serving_smoke"
            else "DEVELOPMENT.json"
        )
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "failing_stage": getattr(exc, "stage", None),
            "failing_row": getattr(exc, "row", None),
            "raw_failing_response": getattr(exc, "response", None),
        }
        filename = (
            "SERVING_SMOKE_FAILURE.json"
            if args.stage == "serving_smoke"
            else "DEVELOPMENT_FAILURE.json"
        )
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = (
        {"status": payload["status"], **payload["summary"]}
        if "summary" in payload
        else {
            "status": payload["status"],
            "physical_requests": payload["physical_requests"],
            "reasoning_tokens": payload["reasoning_tokens"],
        }
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
