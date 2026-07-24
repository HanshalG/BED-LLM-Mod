#!/usr/bin/env python3
"""Qualify binary clinical queries for semantic sequential BED."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mediq.parsing import parse_json_object
from helpers import Config, load_config
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


SELECTION_SEED = 24292
INTERFACE_IDS = ("17983370", "rare231")
JOINT_IDS = ("15492352", "24867998", "rare264", "rare173")
QUERY_IDS = ("q1", "q2", "q3", "q4", "q5", "q6")
EXPECTED_INTERFACE_REQUESTS = 10
EXPECTED_JOINT_REQUESTS = 44
BLOCKED_ACTION_PATTERN = re.compile(
    r"\b(?:"
    r"biops(?:y|ies)|histopath\w*|patholog\w*|histolog\w*|genetic\w*|"
    r"genomic\w*|gene|genes|mutation\w*|sequenc\w*|molecular\w*|"
    r"immunohistochem\w*|cytogenetic\w*|karyotyp\w*|autops\w*|"
    r"surgery|surgical|resection|implantation|transfus\w*|treatment|therapy"
    r")\b",
    re.IGNORECASE,
)
MISSING_DATA_PATTERN = re.compile(
    r"\b(?:not available|unavailable|not measured|not recorded|not specified|"
    r"insufficient sample|assay failure|unknown)\b",
    re.IGNORECASE,
)


def parse_binary_queries(text: str) -> list[dict[str, str]]:
    raw = parse_json_object(text).get("queries")
    if not isinstance(raw, list) or len(raw) != len(QUERY_IDS):
        raise ValueError("queries must contain exactly six rows")
    result = []
    requests = set()
    for query_id, row in zip(QUERY_IDS, raw, strict=True):
        if not isinstance(row, dict) or row.get("id") != query_id:
            raise ValueError("query IDs or order changed")
        kind = row.get("kind")
        if kind not in {"history", "examination", "test"}:
            raise ValueError("query kind must be history, examination, or test")
        request = row.get("request")
        positive = row.get("positive_finding")
        negative = row.get("negative_finding")
        if not all(
            isinstance(value, str) and value.strip()
            for value in (request, positive, negative)
        ):
            raise ValueError("query text fields must be nonempty strings")
        request = request.strip()
        if request.casefold() in requests:
            raise ValueError("query requests must be unique")
        requests.add(request.casefold())
        if BLOCKED_ACTION_PATTERN.search(request):
            raise ValueError("query uses a blocked confirmatory or intervention action")
        if positive.strip().casefold() == negative.strip().casefold():
            raise ValueError("positive and negative findings must differ")
        result.append(
            {
                "id": query_id,
                "kind": kind,
                "request": request,
                "positive_finding": positive.strip(),
                "negative_finding": negative.strip(),
            }
        )
    return result


def binary_query_messages(
    case: ClinDiagCase,
    diagnoses: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "initial_presentation": case.initial_information,
        "current_differential": list(diagnoses),
    }
    return [
        {
            "role": "system",
            "content": (
                "Propose safe, specific binary clinical information-gathering "
                "actions that discriminate the current differential. You do not know "
                "the true diagnosis or hidden record. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly six unique queries with IDs q1 through q6. Each must "
                "have kind history, examination, or test; one request answerable yes "
                "or no; and objective positive_finding and negative_finding text. "
                "Ask one observable fact per query. Do not diagnose, interpret, ask "
                "for treatment, bundle tests, or request biopsy, pathology, histology, "
                "genetic/genomic/molecular testing, sequencing, surgery, resection, "
                "implantation, or transfusion. Never use missing, unavailable, or "
                "unknown as an outcome. Return "
                '{"queries":[{"id":"q1","kind":"history","request":"...",'
                '"positive_finding":"...","negative_finding":"..."}]}.\n'
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_binary_answer(text: str) -> dict[str, str]:
    raw = parse_json_object(text)
    answer = raw.get("answer")
    if answer not in {"yes", "no"}:
        raise ValueError("answer must be yes or no")
    finding = raw.get("finding")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("finding must be nonempty")
    if MISSING_DATA_PATTERN.search(finding):
        raise ValueError("finding reports source-data missingness")
    source = raw.get("source")
    if source not in {"recorded", "synthetic"}:
        raise ValueError("source must be recorded or synthetic")
    return {"answer": answer, "finding": finding.strip(), "source": source}


def binary_gatekeeper_messages(
    case: ClinDiagCase,
    query: dict[str, str],
) -> list[dict[str, str]]:
    hidden_case = {
        "initial_information": case.initial_information,
        "medical_history": case.medical_history,
        "physical_examination": case.physical_examination,
        "diagnostic_tests": case.diagnostic_test,
        "true_diagnosis_for_consistency_only": case.final_diagnosis,
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a hidden clinical environment, not a diagnostician. Answer "
                "the single binary proposition. Use the record when it answers the "
                "request; otherwise synthesize the most plausible patient-specific "
                "yes/no finding consistent with the complete hidden case and true "
                "condition. Never expose missingness, diagnose, interpret, or hint. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly "
                '{"answer":"yes","finding":"objective finding",'
                '"source":"recorded"}; source is recorded or synthetic.\n'
                + json.dumps(
                    {"hidden_case": hidden_case, "query": query},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def _hypotheses(
    diagnoses: Sequence[str],
    true_diagnosis: str,
) -> list[dict[str, str]]:
    return [
        *[
            {"id": f"h{index + 1}", "name": diagnosis}
            for index, diagnosis in enumerate(
                diagnoses[:LIKELIHOOD_SUPPORT_SIZE]
            )
        ],
        {"id": "hT", "name": true_diagnosis},
    ]


def binary_likelihood_messages(
    case: ClinDiagCase,
    diagnoses: Sequence[str],
    queries: Sequence[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hypotheses = _hypotheses(diagnoses, case.final_diagnosis)
    payload = {
        "observed_history": {"initial_presentation": case.initial_information},
        "hypotheses": hypotheses,
        "queries": list(queries),
    }
    return (
        [
            {
                "role": "system",
                "content": (
                    "Estimate calibrated clinical answer likelihoods conditional on "
                    "the observed history, one hypothesis, and one binary query. "
                    "Return strict JSON only. Do not use any hidden patient record."
                ),
            },
            {
                "role": "user",
                "content": (
                    "Return every hypothesis-query pair in hypothesis order then "
                    "query order as "
                    '{"rows":[{"hypothesis_id":"h1","query_id":"q1",'
                    '"p_yes":0.5}]}. Preserve IDs. p_yes must be a finite number in '
                    "[0,1].\n"
                    + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                ),
            },
        ],
        hypotheses,
    )


def parse_binary_likelihoods(
    text: str,
    hypotheses: Sequence[dict[str, str]],
) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("rows")
    expected = [
        (hypothesis["id"], query_id)
        for hypothesis in hypotheses
        for query_id in QUERY_IDS
    ]
    if not isinstance(raw, list) or len(raw) != len(expected):
        raise ValueError("likelihood response has the wrong number of rows")
    result = []
    for (hypothesis_id, query_id), row in zip(expected, raw, strict=True):
        if (
            not isinstance(row, dict)
            or row.get("hypothesis_id") != hypothesis_id
            or row.get("query_id") != query_id
        ):
            raise ValueError("likelihood row IDs or order changed")
        value = row.get("p_yes")
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
        ):
            raise ValueError("p_yes must be finite and in [0,1]")
        result.append(
            {
                "hypothesis_id": hypothesis_id,
                "query_id": query_id,
                "p_yes": float(value),
            }
        )
    return result


def binary_audit_messages(
    case: ClinDiagCase,
    queries: Sequence[dict[str, str]],
    answers: Sequence[dict[str, str]],
    duplicate: dict[str, str],
) -> list[dict[str, str]]:
    payload = {
        "hidden_case": {
            "initial_information": case.initial_information,
            "medical_history": case.medical_history,
            "physical_examination": case.physical_examination,
            "diagnostic_tests": case.diagnostic_test,
            "true_diagnosis": case.final_diagnosis,
        },
        "queries_and_answers": [
            {"query": query, "answer": answer}
            for query, answer in zip(queries, answers, strict=True)
        ],
        "exact_prompt_duplicate_for_q1": duplicate,
    }
    return [
        {
            "role": "system",
            "content": (
                "Audit a hidden clinical environment. Judge response behavior, not "
                "diagnostic quality. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For q1-q6 report booleans relevant, objective, no_target_leak, and "
                "case_consistent. For q1 also judge duplicate_semantically_consistent; "
                "set it true for q2-q6. A synthetic answer is case-consistent only if "
                "it is clinically plausible under the whole hidden case. Return "
                '{"queries":[{"id":"q1","relevant":true,"objective":true,'
                '"no_target_leak":true,"case_consistent":true,'
                '"duplicate_semantically_consistent":true}]}.\n'
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_binary_audit(text: str) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("queries")
    if not isinstance(raw, list) or len(raw) != len(QUERY_IDS):
        raise ValueError("audit must contain six query rows")
    fields = (
        "relevant",
        "objective",
        "no_target_leak",
        "case_consistent",
        "duplicate_semantically_consistent",
    )
    result = []
    for query_id, row in zip(QUERY_IDS, raw, strict=True):
        if not isinstance(row, dict) or row.get("id") != query_id:
            raise ValueError("audit query IDs or order changed")
        if any(not isinstance(row.get(field), bool) for field in fields):
            raise ValueError("all audit fields must be booleans")
        result.append({"id": query_id, **{field: row[field] for field in fields}})
    return result


def _raw_many(
    model: Any,
    messages: Sequence[list[dict[str, str]]],
    *,
    temperature: float,
    max_tokens: int,
) -> list[str]:
    return model.chat_complete_messages_batched(
        list(messages),
        temperature=temperature,
        block_size=256,
        max_new_tokens=max_tokens,
    )


def _usage_summary(models: dict[str, Any]) -> dict[str, Any]:
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


def run_interface_smoke(
    config: Config,
    data_zip: Path,
    model_name: str,
) -> dict[str, Any]:
    models = {
        "support": _build_role(config, model_name),
        "query": _build_role(config, model_name),
        "gatekeeper": _build_role(config, model_name),
    }
    cases = load_selected_cases(data_zip, INTERFACE_IDS)
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
    original_specs = [
        (case, query)
        for case, case_queries in zip(cases, queries, strict=True)
        for query in case_queries[:2]
    ]
    original_raw = _raw_many(
        models["gatekeeper"],
        [binary_gatekeeper_messages(case, query) for case, query in original_specs],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    originals = [parse_binary_answer(text) for text in original_raw]
    duplicate_raw = _raw_many(
        models["gatekeeper"],
        [
            binary_gatekeeper_messages(case, case_queries[0])
            for case, case_queries in zip(cases, queries, strict=True)
        ],
        temperature=0.0,
        max_tokens=max_tokens,
    )
    duplicates = [parse_binary_answer(text) for text in duplicate_raw]
    usage = _usage_summary(models)
    records = []
    literal_leaks = 0
    duplicate_matches = 0
    for index, (case, support, case_queries, duplicate) in enumerate(
        zip(cases, supports, queries, duplicates, strict=True)
    ):
        case_answers = originals[index * 2 : index * 2 + 2]
        target = _normalized(case.final_diagnosis)
        leaks = [
            query["id"]
            for query, answer in zip(case_queries[:2], case_answers, strict=True)
            if target and target in _normalized(answer["finding"])
        ]
        literal_leaks += len(leaks)
        match = case_answers[0]["answer"] == duplicate["answer"]
        duplicate_matches += match
        records.append(
            {
                "source_id": case.source_id,
                "true_diagnosis_measurement_only": case.final_diagnosis,
                "support": support,
                "queries": case_queries,
                "answers": case_answers,
                "duplicate": duplicate,
                "literal_leak_query_ids": leaks,
                "duplicate_answer_match": match,
            }
        )
    gates = {
        "exactly_10_requests": (
            usage["physical_requests"] == EXPECTED_INTERFACE_REQUESTS
        ),
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "supports_size_twelve": all(len(value) == 12 for value in supports),
        "six_queries_per_case": all(len(value) == 6 for value in queries),
        "no_literal_target_leaks": literal_leaks == 0,
        "both_duplicate_answers_match": duplicate_matches == 2,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "interface_ids": list(INTERFACE_IDS),
            "expected_requests": EXPECTED_INTERFACE_REQUESTS,
            "model": model_name,
            "reasoning_disabled": True,
            "retries": 0,
        },
        "summary": {
            "literal_target_leaks": literal_leaks,
            "duplicate_answer_matches": duplicate_matches,
            "gates": gates,
        },
        "records": records,
        "usage": usage,
    }


def _joint_case_metrics(record: dict[str, Any]) -> dict[str, Any]:
    answers = {item["query_id"]: item for item in record["answers"]}
    probabilities = {
        (item["hypothesis_id"], item["query_id"]): item["p_yes"]
        for item in record["likelihoods"]
    }
    true_probs = []
    margins = []
    for query_id in QUERY_IDS:
        realized_yes = answers[query_id]["answer"] == "yes"
        true_p_yes = probabilities[("hT", query_id)]
        true_realized = true_p_yes if realized_yes else 1.0 - true_p_yes
        generated = []
        for index in range(LIKELIHOOD_SUPPORT_SIZE):
            p_yes = probabilities[(f"h{index + 1}", query_id)]
            generated.append(p_yes if realized_yes else 1.0 - p_yes)
        true_probs.append(true_realized)
        margins.append(true_realized - float(np.mean(generated)))
    valid_audits = sum(
        all(
            row[field]
            for field in ("relevant", "objective", "no_target_leak", "case_consistent")
        )
        for row in record["audit"]
    )
    return {
        "true_realized_probabilities": true_probs,
        "margins": margins,
        "mean_true_realized_probability": float(np.mean(true_probs)),
        "mean_margin": float(np.mean(margins)),
        "positive_margin_queries": sum(value > 0.0 for value in margins),
        "valid_audits": valid_audits,
        "duplicate_answer_match": (
            record["answers"][0]["answer"] == record["duplicate"]["answer"]
        ),
        "duplicate_semantically_consistent": record["audit"][0][
            "duplicate_semantically_consistent"
        ],
    }


def run_joint_smoke(
    config: Config,
    data_zip: Path,
    model_name: str,
) -> dict[str, Any]:
    models = {
        name: _build_role(config, model_name)
        for name in ("support", "query", "gatekeeper", "likelihood", "judge")
    }
    cases = load_selected_cases(data_zip, JOINT_IDS)
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
    flat_specs = [
        (case, query)
        for case, case_queries in zip(cases, queries, strict=True)
        for query in case_queries
    ]
    flat_answers = [
        parse_binary_answer(text)
        for text in _raw_many(
            models["gatekeeper"],
            [binary_gatekeeper_messages(case, query) for case, query in flat_specs],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    ]
    answers = [
        [
            {"query_id": query["id"], **flat_answers[i * 6 + j]}
            for j, query in enumerate(case_queries)
        ]
        for i, case_queries in enumerate(queries)
    ]
    duplicates = [
        parse_binary_answer(text)
        for text in _raw_many(
            models["gatekeeper"],
            [
                binary_gatekeeper_messages(case, case_queries[0])
                for case, case_queries in zip(cases, queries, strict=True)
            ],
            temperature=0.0,
            max_tokens=max_tokens,
        )
    ]
    likelihood_payloads = [
        binary_likelihood_messages(case, support, case_queries)
        for case, support, case_queries in zip(cases, supports, queries, strict=True)
    ]
    likelihoods = [
        parse_binary_likelihoods(text, payload[1])
        for text, payload in zip(
            _raw_many(
                models["likelihood"],
                [payload[0] for payload in likelihood_payloads],
                temperature=0.0,
                max_tokens=max_tokens,
            ),
            likelihood_payloads,
            strict=True,
        )
    ]
    audits = complete_parsed_many(
        models["judge"],
        [
            binary_audit_messages(case, case_queries, case_answers, duplicate)
            for case, case_queries, case_answers, duplicate in zip(
                cases, queries, answers, duplicates, strict=True
            )
        ],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=parse_binary_audit,
        stage="behavior_audit",
        retries=0,
    )
    records = [
        {
            "source_id": case.source_id,
            "true_diagnosis_measurement_only": case.final_diagnosis,
            "support": support,
            "queries": case_queries,
            "answers": case_answers,
            "duplicate": duplicate,
            "likelihood_hypotheses": payload[1],
            "likelihoods": case_likelihoods,
            "audit": audit,
        }
        for (
            case,
            support,
            case_queries,
            case_answers,
            duplicate,
            payload,
            case_likelihoods,
            audit,
        ) in zip(
            cases,
            supports,
            queries,
            answers,
            duplicates,
            likelihood_payloads,
            likelihoods,
            audits,
            strict=True,
        )
    ]
    usage = _usage_summary(models)
    metrics = [_joint_case_metrics(record) for record in records]
    true_probs = [value for item in metrics for value in item["true_realized_probabilities"]]
    margins = [value for item in metrics for value in item["margins"]]
    valid_audits = sum(item["valid_audits"] for item in metrics)
    duplicate_matches = sum(item["duplicate_answer_match"] for item in metrics)
    duplicate_semantic = sum(
        item["duplicate_semantically_consistent"] for item in metrics
    )
    summary = {
        "physical_requests": usage["physical_requests"],
        "reasoning_tokens": usage["reasoning_tokens"],
        "valid_response_audits": valid_audits,
        "duplicate_answer_matches": duplicate_matches,
        "duplicate_semantic_matches": duplicate_semantic,
        "mean_true_realized_probability": float(np.mean(true_probs)),
        "positive_margin_queries": sum(value > 0.0 for value in margins),
        "mean_true_minus_generated_margin": float(np.mean(margins)),
        "case_metrics": [
            {"source_id": record["source_id"], **item}
            for record, item in zip(records, metrics, strict=True)
        ],
    }
    gates = {
        "exactly_44_requests": usage["physical_requests"] == EXPECTED_JOINT_REQUESTS,
        "zero_reasoning": usage["reasoning_tokens"] == 0,
        "all_24_responses_pass_audit": valid_audits == 24,
        "all_four_duplicate_answers_match": duplicate_matches == 4,
        "all_four_duplicates_semantically_consistent": duplicate_semantic == 4,
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
            "joint_ids": list(JOINT_IDS),
            "expected_requests": EXPECTED_JOINT_REQUESTS,
            "model": model_name,
            "reasoning_disabled": True,
            "retries": 0,
            "truth_hidden_until_queries_and_answers_frozen": True,
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
        choices=("interface_smoke", "joint_smoke"),
        required=True,
    )
    parser.add_argument("--model", default="openai/gpt-5.4")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / (
        "INTERFACE_SMOKE.json"
        if args.stage == "interface_smoke"
        else "JOINT_SMOKE.json"
    )
    try:
        payload = (
            run_interface_smoke(config, args.data_zip, args.model)
            if args.stage == "interface_smoke"
            else run_joint_smoke(config, args.data_zip, args.model)
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
