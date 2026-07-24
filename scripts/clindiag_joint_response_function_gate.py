#!/usr/bin/env python3
"""Qualify a jointly generated hidden-patient response function."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mediq.parsing import parse_json_object
from helpers import Config, load_config
from scripts.clindiag_binary_joint_model_gate import (
    QUERY_IDS,
    binary_likelihood_messages,
    binary_query_messages,
    parse_binary_likelihoods,
    parse_binary_queries,
)
from scripts.clindiag_mc_joint_model_gate import (
    LIKELIHOOD_SUPPORT_SIZE,
    _build_role,
    _normalized,
)
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    complete_parsed_many,
    parse_diagnoses,
)


SELECTION_SEED = 24293
QUALIFICATION_IDS = ("24450895", "rare214")
FIDELITY_IDS = ("25688156", "12830418", "rare43", "rare88")
EXPECTED_QUALIFICATION_REQUESTS = 10
EXPECTED_FIDELITY_REQUESTS = 24
JOINT_MISSINGNESS_PATTERN = re.compile(
    r"\b(?:"
    r"not available|unavailable|not measured|not recorded|not reported|"
    r"not specified|no documentation|not documented|absent from (?:the )?"
    r"(?:record|chart)|record does not|chart does not|insufficient sample|"
    r"assay failure|unknown"
    r")\b",
    re.IGNORECASE,
)


def joint_response_messages(
    case: ClinDiagCase,
    queries: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "hidden_case": {
            "initial_information": case.initial_information,
            "medical_history": case.medical_history,
            "physical_examination": case.physical_examination,
            "diagnostic_tests": case.diagnostic_test,
            "true_diagnosis_for_consistency_only": case.final_diagnosis,
        },
        "queries": list(queries),
    }
    return [
        {
            "role": "system",
            "content": (
                "Instantiate one complete, internally consistent hidden patient "
                "response function. Answer all six binary propositions jointly. Use "
                "recorded facts where available; for every absent fact synthesize the "
                "most plausible yes/no patient fact under the complete hidden case and "
                "true condition. Never expose chart missingness, diagnose, interpret, "
                "or hint. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return q1 through q6 in order as "
                '{"answers":[{"query_id":"q1","answer":"yes",'
                '"finding":"objective patient finding","source":"recorded"}]}. '
                "answer is yes or no; source is recorded or synthetic. Every finding "
                "must assert a patient fact. Never say not reported, not recorded, "
                "unknown, unavailable, or otherwise describe source-data absence.\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_joint_response(text: str) -> list[dict[str, str]]:
    raw = parse_json_object(text).get("answers")
    if not isinstance(raw, list) or len(raw) != len(QUERY_IDS):
        raise ValueError("response function must contain six answers")
    result = []
    for query_id, row in zip(QUERY_IDS, raw, strict=True):
        if not isinstance(row, dict) or row.get("query_id") != query_id:
            raise ValueError("response function query IDs or order changed")
        answer = row.get("answer")
        if answer not in {"yes", "no"}:
            raise ValueError("answer must be yes or no")
        finding = row.get("finding")
        if not isinstance(finding, str) or not finding.strip():
            raise ValueError("finding must be nonempty")
        if JOINT_MISSINGNESS_PATTERN.search(finding):
            raise ValueError("finding exposes chart missingness")
        source = row.get("source")
        if source not in {"recorded", "synthetic"}:
            raise ValueError("source must be recorded or synthetic")
        result.append(
            {
                "query_id": query_id,
                "answer": answer,
                "finding": finding.strip(),
                "source": source,
            }
        )
    return result


AUDIT_FIELDS = (
    "original_relevant",
    "original_objective",
    "original_no_target_leak",
    "original_case_consistent",
    "duplicate_relevant",
    "duplicate_objective",
    "duplicate_no_target_leak",
    "duplicate_case_consistent",
    "duplicate_semantically_consistent",
    "provenance_consistent",
)


def joint_audit_messages(
    case: ClinDiagCase,
    queries: Sequence[dict[str, str]],
    original: Sequence[dict[str, str]],
    duplicate: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "hidden_case": {
            "initial_information": case.initial_information,
            "medical_history": case.medical_history,
            "physical_examination": case.physical_examination,
            "diagnostic_tests": case.diagnostic_test,
            "true_diagnosis": case.final_diagnosis,
        },
        "queries": list(queries),
        "original_response_function": list(original),
        "exact_prompt_duplicate_response_function": list(duplicate),
    }
    return [
        {
            "role": "system",
            "content": (
                "Independently audit two hidden-patient response functions. Judge "
                "environment behavior, not diagnostic quality. Return strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "For q1-q6 return all ten booleans: original_relevant, "
                "original_objective, original_no_target_leak, "
                "original_case_consistent, duplicate_relevant, "
                "duplicate_objective, duplicate_no_target_leak, "
                "duplicate_case_consistent, duplicate_semantically_consistent, and "
                "provenance_consistent. Synthetic facts must be clinically plausible "
                "under the complete hidden case. Provenance is consistent only when "
                "the two source labels agree. Return "
                '{"queries":[{"id":"q1","original_relevant":true,'
                '"original_objective":true,"original_no_target_leak":true,'
                '"original_case_consistent":true,"duplicate_relevant":true,'
                '"duplicate_objective":true,"duplicate_no_target_leak":true,'
                '"duplicate_case_consistent":true,'
                '"duplicate_semantically_consistent":true,'
                '"provenance_consistent":true}]}.\n'
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_joint_audit(text: str) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("queries")
    if not isinstance(raw, list) or len(raw) != len(QUERY_IDS):
        raise ValueError("audit must contain six query rows")
    result = []
    for query_id, row in zip(QUERY_IDS, raw, strict=True):
        if not isinstance(row, dict) or row.get("id") != query_id:
            raise ValueError("audit query IDs or order changed")
        if any(not isinstance(row.get(field), bool) for field in AUDIT_FIELDS):
            raise ValueError("all joint audit fields must be booleans")
        result.append({"id": query_id, **{field: row[field] for field in AUDIT_FIELDS}})
    return result


def _raw_many(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    max_tokens: int,
) -> list[str]:
    return model.chat_complete_messages_batched(
        list(messages),
        temperature=0.0,
        block_size=256,
        max_new_tokens=max_tokens,
    )


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


def _literal_leaks(
    case: ClinDiagCase,
    response: Sequence[dict[str, str]],
) -> list[str]:
    target = _normalized(case.final_diagnosis)
    return [
        row["query_id"]
        for row in response
        if target and target in _normalized(row["finding"])
    ]


def _duplicate_label_matches(
    original: Sequence[dict[str, str]],
    duplicate: Sequence[dict[str, str]],
) -> int:
    return sum(
        left["answer"] == right["answer"]
        for left, right in zip(original, duplicate, strict=True)
    )


def _build_common(
    config: Config,
    data_zip: Path,
    source_ids: Sequence[str],
    model_name: str,
    *,
    include_likelihood: bool,
) -> tuple[
    list[ClinDiagCase],
    list[list[str]],
    list[list[dict[str, str]]],
    list[list[dict[str, str]]],
    list[list[dict[str, str]]],
    list[list[dict[str, Any]]],
    list[tuple[list[dict[str, str]], list[dict[str, str]]]],
    list[list[dict[str, Any]]],
    dict[str, Any],
]:
    role_names = ["support", "query", "environment", "duplicate", "judge"]
    if include_likelihood:
        role_names.append("likelihood")
    models = {name: _build_role(config, model_name) for name in role_names}
    cases = load_selected_cases(data_zip, source_ids)
    max_tokens = int(config.openrouter_max_output_tokens)
    supports = complete_parsed_many(
        models["support"],
        [initial_differential_messages(case) for case in cases],
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=max_tokens,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="initial_support",
        retries=0,
    )
    queries = complete_parsed_many(
        models["query"],
        [
            binary_query_messages(case, support)
            for case, support in zip(cases, supports, strict=True)
        ],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=parse_binary_queries,
        stage="binary_queries",
        retries=0,
    )
    response_messages = [
        joint_response_messages(case, case_queries)
        for case, case_queries in zip(cases, queries, strict=True)
    ]
    originals = [
        parse_joint_response(text)
        for text in _raw_many(models["environment"], response_messages, max_tokens)
    ]
    duplicates = [
        parse_joint_response(text)
        for text in _raw_many(models["duplicate"], response_messages, max_tokens)
    ]
    audits = complete_parsed_many(
        models["judge"],
        [
            joint_audit_messages(case, case_queries, original, duplicate)
            for case, case_queries, original, duplicate in zip(
                cases, queries, originals, duplicates, strict=True
            )
        ],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=parse_joint_audit,
        stage="joint_audit",
        retries=0,
    )
    likelihood_payloads: list[
        tuple[list[dict[str, str]], list[dict[str, str]]]
    ] = []
    likelihoods: list[list[dict[str, Any]]] = []
    if include_likelihood:
        likelihood_payloads = [
            binary_likelihood_messages(case, support, case_queries)
            for case, support, case_queries in zip(
                cases, supports, queries, strict=True
            )
        ]
        likelihoods = [
            parse_binary_likelihoods(text, payload[1])
            for text, payload in zip(
                _raw_many(
                    models["likelihood"],
                    [payload[0] for payload in likelihood_payloads],
                    max_tokens,
                ),
                likelihood_payloads,
                strict=True,
            )
        ]
    return (
        cases,
        supports,
        queries,
        originals,
        duplicates,
        audits,
        likelihood_payloads,
        likelihoods,
        _usage(models),
    )


def run_qualification(
    config: Config,
    data_zip: Path,
    model_name: str,
) -> dict[str, Any]:
    (
        cases,
        supports,
        queries,
        originals,
        duplicates,
        audits,
        _,
        _,
        usage,
    ) = _build_common(
        config,
        data_zip,
        QUALIFICATION_IDS,
        model_name,
        include_likelihood=False,
    )
    records = []
    leaks = 0
    label_matches = 0
    audit_passes = 0
    for case, support, case_queries, original, duplicate, audit in zip(
        cases, supports, queries, originals, duplicates, audits, strict=True
    ):
        case_leaks = sorted(
            set(_literal_leaks(case, original) + _literal_leaks(case, duplicate))
        )
        leaks += len(case_leaks)
        case_matches = _duplicate_label_matches(original, duplicate)
        label_matches += case_matches
        case_audit_passes = sum(all(row[field] for field in AUDIT_FIELDS) for row in audit)
        audit_passes += case_audit_passes
        records.append(
            {
                "source_id": case.source_id,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "support": support,
                "queries": case_queries,
                "original_response_function": original,
                "duplicate_response_function": duplicate,
                "audit": audit,
                "literal_leak_query_ids": case_leaks,
                "duplicate_label_matches": case_matches,
                "fully_valid_audit_rows": case_audit_passes,
            }
        )
    gates = {
        "exactly_10_requests": (
            usage["physical_requests"] == EXPECTED_QUALIFICATION_REQUESTS
        ),
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "all_supports_size_twelve": all(len(value) == 12 for value in supports),
        "all_query_sets_size_six": all(len(value) == 6 for value in queries),
        "no_literal_target_leaks": leaks == 0,
        "all_12_duplicate_labels_match": label_matches == 12,
        "all_12_audit_rows_pass": audit_passes == 12,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "qualification_ids": list(QUALIFICATION_IDS),
            "model": model_name,
            "expected_requests": EXPECTED_QUALIFICATION_REQUESTS,
            "reasoning_disabled": True,
            "retries": 0,
        },
        "summary": {
            "literal_target_leaks": leaks,
            "duplicate_label_matches": label_matches,
            "fully_valid_audit_rows": audit_passes,
            "gates": gates,
        },
        "records": records,
        "usage": usage,
    }


def _fidelity_metrics(
    response: Sequence[dict[str, str]],
    likelihoods: Sequence[dict[str, Any]],
) -> tuple[list[float], list[float]]:
    answers = {row["query_id"]: row["answer"] for row in response}
    p_yes = {
        (row["hypothesis_id"], row["query_id"]): row["p_yes"]
        for row in likelihoods
    }
    true_values = []
    margins = []
    for query_id in QUERY_IDS:
        realized_yes = answers[query_id] == "yes"
        truth = p_yes[("hT", query_id)]
        truth = truth if realized_yes else 1.0 - truth
        generated = []
        for index in range(LIKELIHOOD_SUPPORT_SIZE):
            value = p_yes[(f"h{index + 1}", query_id)]
            generated.append(value if realized_yes else 1.0 - value)
        true_values.append(truth)
        margins.append(truth - float(np.mean(generated)))
    return true_values, margins


def run_fidelity(
    config: Config,
    data_zip: Path,
    model_name: str,
) -> dict[str, Any]:
    (
        cases,
        supports,
        queries,
        originals,
        duplicates,
        audits,
        likelihood_payloads,
        likelihoods,
        usage,
    ) = _build_common(
        config,
        data_zip,
        FIDELITY_IDS,
        model_name,
        include_likelihood=True,
    )
    records = []
    all_true = []
    all_margins = []
    audit_passes = 0
    label_matches = 0
    leaks = 0
    for (
        case,
        support,
        case_queries,
        original,
        duplicate,
        audit,
        payload,
        case_likelihoods,
    ) in zip(
        cases,
        supports,
        queries,
        originals,
        duplicates,
        audits,
        likelihood_payloads,
        likelihoods,
        strict=True,
    ):
        true_values, margins = _fidelity_metrics(original, case_likelihoods)
        all_true.extend(true_values)
        all_margins.extend(margins)
        case_audits = sum(all(row[field] for field in AUDIT_FIELDS) for row in audit)
        case_labels = _duplicate_label_matches(original, duplicate)
        case_leaks = sorted(
            set(_literal_leaks(case, original) + _literal_leaks(case, duplicate))
        )
        audit_passes += case_audits
        label_matches += case_labels
        leaks += len(case_leaks)
        records.append(
            {
                "source_id": case.source_id,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "support": support,
                "queries": case_queries,
                "original_response_function": original,
                "duplicate_response_function": duplicate,
                "audit": audit,
                "likelihood_hypotheses": payload[1],
                "likelihoods": case_likelihoods,
                "true_realized_probabilities": true_values,
                "margins": margins,
                "fully_valid_audit_rows": case_audits,
                "duplicate_label_matches": case_labels,
                "literal_leak_query_ids": case_leaks,
            }
        )
    summary = {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "fully_valid_audit_rows": audit_passes,
        "duplicate_label_matches": label_matches,
        "literal_target_leaks": leaks,
        "mean_true_realized_probability": float(np.mean(all_true)),
        "positive_margin_queries": sum(value > 0.0 for value in all_margins),
        "mean_true_minus_generated_margin": float(np.mean(all_margins)),
    }
    gates = {
        "exactly_24_requests": (
            usage["physical_requests"] == EXPECTED_FIDELITY_REQUESTS
        ),
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "all_24_audit_rows_pass": audit_passes == 24,
        "all_24_duplicate_labels_match": label_matches == 24,
        "no_literal_target_leaks": leaks == 0,
        "mean_true_probability_at_least_0_65": (
            summary["mean_true_realized_probability"] >= 0.65
        ),
        "positive_margin_on_at_least_12_queries": (
            summary["positive_margin_queries"] >= 12
        ),
        "mean_margin_at_least_0_10": (
            summary["mean_true_minus_generated_margin"] >= 0.10
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "fidelity_ids": list(FIDELITY_IDS),
            "model": model_name,
            "expected_requests": EXPECTED_FIDELITY_REQUESTS,
            "reasoning_disabled": True,
            "retries": 0,
            "truth_hidden_until_queries_and_response_functions_frozen": True,
        },
        "summary": summary,
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
        choices=("qualification", "fidelity"),
        required=True,
    )
    parser.add_argument("--model", default="openai/gpt-5.4")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        "QUALIFICATION.json"
        if args.stage == "qualification"
        else "FIDELITY.json"
    )
    try:
        payload = (
            run_qualification(config, args.data_zip, args.model)
            if args.stage == "qualification"
            else run_fidelity(config, args.data_zip, args.model)
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
