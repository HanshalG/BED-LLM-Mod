#!/usr/bin/env python3
"""Test a multiple-choice ClinDiag gatekeeper and semantic likelihood model."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from environments.mediq.parsing import parse_json_object
from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.clindiag_staged_generator_gate import (
    DIAGNOSIS_COUNT,
    ClinDiagCase,
    initial_differential_messages,
    load_selected_cases,
)
from scripts.icraft_staged_diagnosis_unlock import (
    StructuredBatchError,
    complete_parsed_many,
    parse_diagnoses,
)


SELECTION_SEED = 24291
SMOKE_IDS = (
    "22704279",
    "13424741",
    "rare126",
    "rare82",
)
QUERY_IDS = ("q1", "q2", "q3", "q4")
OUTCOME_IDS = ("A", "B", "C", "D")
LIKELIHOOD_SUPPORT_SIZE = 6
EXPECTED_REQUESTS = 36


def parse_candidate_queries(text: str) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("queries")
    if not isinstance(raw, list) or len(raw) != len(QUERY_IDS):
        raise ValueError("queries must contain exactly four rows")
    parsed = []
    requests = set()
    for query_id, row in zip(QUERY_IDS, raw, strict=True):
        if not isinstance(row, dict) or row.get("id") != query_id:
            raise ValueError("query IDs or order changed")
        kind = row.get("kind")
        if kind not in {"history", "examination", "test"}:
            raise ValueError("query kind must be history, examination, or test")
        request = row.get("request")
        if not isinstance(request, str) or not request.strip():
            raise ValueError("query request must be nonempty")
        request = request.strip()
        request_key = request.casefold()
        if request_key in requests:
            raise ValueError("query requests must be unique")
        requests.add(request_key)
        outcomes = row.get("outcomes")
        if not isinstance(outcomes, list) or len(outcomes) != len(OUTCOME_IDS):
            raise ValueError("each query must have exactly four outcomes")
        parsed_outcomes = []
        labels = set()
        for outcome_id, outcome in zip(OUTCOME_IDS, outcomes, strict=True):
            if not isinstance(outcome, dict) or outcome.get("id") != outcome_id:
                raise ValueError("outcome IDs or order changed")
            label = outcome.get("label")
            if not isinstance(label, str) or not label.strip():
                raise ValueError("outcome label must be nonempty")
            label = label.strip()
            if label.casefold() in labels:
                raise ValueError("outcome labels must be unique")
            labels.add(label.casefold())
            parsed_outcomes.append({"id": outcome_id, "label": label})
        parsed.append(
            {
                "id": query_id,
                "kind": kind,
                "request": request,
                "outcomes": parsed_outcomes,
            }
        )
    return parsed


def candidate_query_messages(
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
                "Propose specific clinical information-gathering actions that "
                "discriminate the current differential. You do not know the true "
                "diagnosis or hidden record. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly four unique queries with IDs q1,q2,q3,q4. Each query "
                "must have kind history, examination, or test; one specific request; "
                "and exactly four mutually exclusive, collectively useful outcomes "
                "with IDs A,B,C,D. Outcome labels must be objective findings, not "
                "diagnoses or interpretations. Avoid vague requests, treatment, and "
                "bundled panels. Use schema "
                '{"queries":[{"id":"q1","kind":"test","request":"...",'
                '"outcomes":[{"id":"A","label":"..."},'
                '{"id":"B","label":"..."},{"id":"C","label":"..."},'
                '{"id":"D","label":"..."}]}]}.\n'
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_gatekeeper_response(
    text: str,
    query: dict[str, Any],
) -> dict[str, str]:
    raw = parse_json_object(text)
    outcome_id = raw.get("outcome_id")
    if outcome_id not in OUTCOME_IDS:
        raise ValueError("outcome_id must be A, B, C, or D")
    finding = raw.get("finding")
    if not isinstance(finding, str) or not finding.strip():
        raise ValueError("finding must be nonempty")
    source = raw.get("source")
    if source not in {"recorded", "synthetic"}:
        raise ValueError("source must be recorded or synthetic")
    valid_ids = {item["id"] for item in query["outcomes"]}
    if outcome_id not in valid_ids:
        raise ValueError("outcome_id is not present in the query")
    return {
        "outcome_id": outcome_id,
        "finding": finding.strip(),
        "source": source,
    }


def gatekeeper_messages(
    case: ClinDiagCase,
    query: dict[str, Any],
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
                "only the requested question or test. Use a recorded finding when "
                "available; otherwise synthesize a plausible patient-specific finding "
                "consistent with the hidden case. Select the closest offered outcome. "
                "Never state, imply, or interpret the diagnosis. Return strict JSON."
            ),
        },
        {
            "role": "user",
            "content": (
                "Return exactly "
                '{"outcome_id":"A","finding":"objective finding",'
                '"source":"recorded"}; source is recorded or synthetic.\n'
                + json.dumps(
                    {"hidden_case": hidden_case, "query": query},
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
            ),
        },
    ]


def _likelihood_hypotheses(
    diagnoses: Sequence[str],
    true_diagnosis: str,
) -> list[dict[str, str]]:
    values = [
        {"id": f"h{index + 1}", "name": diagnosis}
        for index, diagnosis in enumerate(diagnoses[:LIKELIHOOD_SUPPORT_SIZE])
    ]
    values.append({"id": "hT", "name": true_diagnosis})
    return values


def likelihood_messages(
    case: ClinDiagCase,
    diagnoses: Sequence[str],
    queries: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    hypotheses = _likelihood_hypotheses(diagnoses, case.final_diagnosis)
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
                    "Estimate clinical outcome likelihoods conditional on the observed "
                    "history, one hypothesis, and one query. Return calibrated "
                    "probabilities and strict JSON only. Do not use any hidden case."
                ),
            },
            {
                "role": "user",
                "content": (
                    "For every hypothesis-query pair, return one row in hypothesis "
                    "order then query order. Preserve IDs. Probabilities for A,B,C,D "
                    "must be finite, nonnegative, and sum to 1. Use schema "
                    '{"rows":[{"hypothesis_id":"h1","query_id":"q1",'
                    '"probabilities":{"A":0.25,"B":0.25,"C":0.25,"D":0.25}}]}.\n'
                    + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
                ),
            },
        ],
        hypotheses,
    )


def parse_likelihood_matrix(
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
    parsed = []
    for (hypothesis_id, query_id), row in zip(expected, raw, strict=True):
        if not isinstance(row, dict):
            raise ValueError("likelihood row must be an object")
        if (
            row.get("hypothesis_id") != hypothesis_id
            or row.get("query_id") != query_id
        ):
            raise ValueError("likelihood row IDs or order changed")
        probabilities = row.get("probabilities")
        if not isinstance(probabilities, dict) or set(probabilities) != set(OUTCOME_IDS):
            raise ValueError("likelihood row must contain exactly A,B,C,D")
        values = []
        for outcome_id in OUTCOME_IDS:
            value = probabilities[outcome_id]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or float(value) < 0.0
            ):
                raise ValueError("likelihood probabilities must be finite and nonnegative")
            values.append(float(value))
        total = sum(values)
        if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-4):
            raise ValueError("likelihood probabilities must sum to one")
        parsed.append(
            {
                "hypothesis_id": hypothesis_id,
                "query_id": query_id,
                "probabilities": dict(zip(OUTCOME_IDS, values, strict=True)),
            }
        )
    return parsed


def response_audit_messages(
    case: ClinDiagCase,
    queries: Sequence[dict[str, Any]],
    responses: Sequence[dict[str, str]],
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
        "queries_and_responses": [
            {"query": query, "response": response}
            for query, response in zip(queries, responses, strict=True)
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
                "For q1-q4, report booleans: relevant (answers the exact request), "
                "objective (does not interpret), no_target_leak (does not state or "
                "hint the true diagnosis), and case_consistent (recorded facts match "
                "and synthetic facts are clinically plausible). For q1 also report "
                "duplicate_semantically_consistent. Preserve order and return "
                '{"queries":[{"id":"q1","relevant":true,"objective":true,'
                '"no_target_leak":true,"case_consistent":true,'
                '"duplicate_semantically_consistent":true},...]}. For q2-q4 set '
                "duplicate_semantically_consistent to true because no duplicate was "
                "requested.\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_response_audit(text: str) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("queries")
    if not isinstance(raw, list) or len(raw) != len(QUERY_IDS):
        raise ValueError("audit must contain four query rows")
    fields = (
        "relevant",
        "objective",
        "no_target_leak",
        "case_consistent",
        "duplicate_semantically_consistent",
    )
    parsed = []
    for query_id, row in zip(QUERY_IDS, raw, strict=True):
        if not isinstance(row, dict) or row.get("id") != query_id:
            raise ValueError("audit query IDs or order changed")
        if any(not isinstance(row.get(field), bool) for field in fields):
            raise ValueError("all audit judgments must be booleans")
        parsed.append({"id": query_id, **{field: row[field] for field in fields}})
    return parsed


def _build_role(config: Config, model_name: str) -> Any:
    base = config.model_pairs[0].questioner
    return build_model_adapter(
        replace(
            base,
            model=model_name,
            thinking=None,
            reasoning_effort="none",
            reasoning_max_tokens=None,
            thinking_max_new_tokens=None,
            thinking_final_max_new_tokens=None,
        ),
        config,
    )


def _normalized(value: str) -> str:
    return " ".join(
        token
        for token in "".join(
            character if character.isalnum() else " "
            for character in value.casefold()
        ).split()
        if token
    )


def _case_metrics(record: dict[str, Any]) -> dict[str, Any]:
    response_by_query = {
        response["query_id"]: response for response in record["responses"]
    }
    likelihood_by_key = {
        (row["hypothesis_id"], row["query_id"]): row["probabilities"]
        for row in record["likelihood_rows"]
    }
    true_probabilities = []
    distractor_probabilities = []
    margins = []
    for query_id in QUERY_IDS:
        outcome_id = response_by_query[query_id]["outcome_id"]
        true_probability = likelihood_by_key[("hT", query_id)][outcome_id]
        distractor_probability = float(
            np.mean(
                [
                    likelihood_by_key[(f"h{index + 1}", query_id)][outcome_id]
                    for index in range(LIKELIHOOD_SUPPORT_SIZE)
                ]
            )
        )
        true_probabilities.append(true_probability)
        distractor_probabilities.append(distractor_probability)
        margins.append(true_probability - distractor_probability)
    target = _normalized(record["true_diagnosis_measurement_only"])
    lexical_leaks = [
        response["query_id"]
        for response in record["responses"]
        if target and target in _normalized(response["finding"])
    ]
    audit_valid = [
        all(
            row[field]
            for field in ("relevant", "objective", "no_target_leak", "case_consistent")
        )
        for row in record["response_audit"]
    ]
    return {
        "mean_true_realized_probability": float(np.mean(true_probabilities)),
        "mean_distractor_realized_probability": float(
            np.mean(distractor_probabilities)
        ),
        "mean_true_minus_distractor_margin": float(np.mean(margins)),
        "positive_margin_queries": sum(margin > 0.0 for margin in margins),
        "true_realized_probabilities": true_probabilities,
        "distractor_realized_probabilities": distractor_probabilities,
        "margins": margins,
        "lexical_target_leak_query_ids": lexical_leaks,
        "audit_valid_queries": sum(audit_valid),
        "duplicate_outcome_match": (
            record["responses"][0]["outcome_id"]
            == record["duplicate_response"]["outcome_id"]
        ),
        "duplicate_semantically_consistent": record["response_audit"][0][
            "duplicate_semantically_consistent"
        ],
    }


def summarize(records: Sequence[dict[str, Any]], usage: dict[str, Any]) -> dict[str, Any]:
    case_metrics = [_case_metrics(record) for record in records]
    true_probabilities = [
        value
        for metrics in case_metrics
        for value in metrics["true_realized_probabilities"]
    ]
    margins = [
        value for metrics in case_metrics for value in metrics["margins"]
    ]
    requests = sum(int(item["adapter_requests"]) for item in usage.values())
    reasoning_tokens = sum(
        int(item["adapter_reasoning_tokens"]) for item in usage.values()
    )
    valid_audits = sum(item["audit_valid_queries"] for item in case_metrics)
    duplicate_outcome_matches = sum(
        item["duplicate_outcome_match"] for item in case_metrics
    )
    duplicate_semantic_matches = sum(
        item["duplicate_semantically_consistent"] for item in case_metrics
    )
    lexical_leaks = sum(
        len(item["lexical_target_leak_query_ids"]) for item in case_metrics
    )
    summary = {
        "num_cases": len(records),
        "physical_requests": requests,
        "reasoning_tokens": reasoning_tokens,
        "valid_response_audits": valid_audits,
        "duplicate_outcome_matches": duplicate_outcome_matches,
        "duplicate_semantic_matches": duplicate_semantic_matches,
        "lexical_target_leaks": lexical_leaks,
        "mean_true_realized_outcome_probability": float(
            np.mean(true_probabilities)
        ),
        "positive_true_minus_generated_margin_queries": sum(
            margin > 0.0 for margin in margins
        ),
        "mean_true_minus_generated_margin": float(np.mean(margins)),
        "case_metrics": [
            {"source_id": record["source_id"], **metrics}
            for record, metrics in zip(records, case_metrics, strict=True)
        ],
    }
    gates = {
        "exactly_four_cases": len(records) == 4,
        "exactly_36_physical_requests": requests == EXPECTED_REQUESTS,
        "zero_reasoning_tokens": reasoning_tokens == 0,
        "all_supports_size_twelve": all(
            len(record["initial_diagnoses"]) == DIAGNOSIS_COUNT for record in records
        ),
        "all_queries_and_options_valid": all(
            len(record["queries"]) == 4
            and all(len(query["outcomes"]) == 4 for query in record["queries"])
            for record in records
        ),
        "all_sixteen_responses_pass_audit": valid_audits == 16,
        "no_lexical_target_leaks": lexical_leaks == 0,
        "all_four_duplicate_outcomes_match": duplicate_outcome_matches == 4,
        "all_four_duplicates_semantically_consistent": (
            duplicate_semantic_matches == 4
        ),
        "mean_true_probability_at_least_0_35": (
            summary["mean_true_realized_outcome_probability"] >= 0.35
        ),
        "positive_margin_on_at_least_eight_queries": (
            summary["positive_true_minus_generated_margin_queries"] >= 8
        ),
        "mean_margin_at_least_0_05": (
            summary["mean_true_minus_generated_margin"] >= 0.05
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {**summary, "gates": gates}


def run_smoke(
    config: Config,
    data_zip: Path,
    *,
    generator_model: str,
    environment_model: str,
    likelihood_model: str,
    judge_model: str,
) -> dict[str, Any]:
    generator = _build_role(config, generator_model)
    query_generator = _build_role(config, generator_model)
    gatekeeper = _build_role(config, environment_model)
    likelihood = _build_role(config, likelihood_model)
    judge = _build_role(config, judge_model)
    cases = load_selected_cases(data_zip, SMOKE_IDS)
    max_tokens = int(config.openrouter_max_output_tokens)

    initial = complete_parsed_many(
        generator,
        [initial_differential_messages(case) for case in cases],
        temperature=float(config.generation_temperature_diverse),
        max_new_tokens=max_tokens,
        parser=lambda text: parse_diagnoses(text, DIAGNOSIS_COUNT),
        stage="initial_support",
        retries=0,
    )
    queries = complete_parsed_many(
        query_generator,
        [
            candidate_query_messages(case, diagnoses)
            for case, diagnoses in zip(cases, initial, strict=True)
        ],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=parse_candidate_queries,
        stage="candidate_queries",
        retries=0,
    )

    flat_gatekeeper_messages = [
        gatekeeper_messages(case, query)
        for case, case_queries in zip(cases, queries, strict=True)
        for query in case_queries
    ]
    flat_queries = [query for case_queries in queries for query in case_queries]
    flat_responses = complete_parsed_many(
        gatekeeper,
        flat_gatekeeper_messages,
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=lambda text_query: text_query,
        stage="gatekeeper_raw",
        retries=0,
    )
    parsed_flat_responses = [
        parse_gatekeeper_response(text, query)
        for text, query in zip(flat_responses, flat_queries, strict=True)
    ]
    responses = [
        [
            {
                "query_id": query["id"],
                **parsed_flat_responses[case_index * len(QUERY_IDS) + query_index],
            }
            for query_index, query in enumerate(case_queries)
        ]
        for case_index, case_queries in enumerate(queries)
    ]

    duplicate_raw = complete_parsed_many(
        gatekeeper,
        [gatekeeper_messages(case, case_queries[0]) for case, case_queries in zip(cases, queries, strict=True)],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=lambda text_query: text_query,
        stage="gatekeeper_duplicate_raw",
        retries=0,
    )
    duplicates = [
        parse_gatekeeper_response(text, case_queries[0])
        for text, case_queries in zip(duplicate_raw, queries, strict=True)
    ]

    likelihood_payloads = [
        likelihood_messages(case, diagnoses, case_queries)
        for case, diagnoses, case_queries in zip(cases, initial, queries, strict=True)
    ]
    likelihood_rows = complete_parsed_many(
        likelihood,
        [payload[0] for payload in likelihood_payloads],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=lambda text_rows: text_rows,
        stage="likelihood_raw",
        retries=0,
    )
    parsed_likelihoods = [
        parse_likelihood_matrix(text, payload[1])
        for text, payload in zip(likelihood_rows, likelihood_payloads, strict=True)
    ]

    audits = complete_parsed_many(
        judge,
        [
            response_audit_messages(case, case_queries, case_responses, duplicate)
            for case, case_queries, case_responses, duplicate in zip(
                cases, queries, responses, duplicates, strict=True
            )
        ],
        temperature=0.0,
        max_new_tokens=max_tokens,
        parser=parse_response_audit,
        stage="response_audit",
        retries=0,
    )

    records = [
        {
            "source_id": case.source_id,
            "subset": case.subset,
            "initial_information": case.initial_information,
            "true_diagnosis_measurement_only": case.final_diagnosis,
            "initial_diagnoses": diagnoses,
            "queries": case_queries,
            "responses": case_responses,
            "duplicate_response": duplicate,
            "likelihood_hypotheses": likelihood_payload[1],
            "likelihood_rows": case_likelihoods,
            "response_audit": audit,
        }
        for (
            case,
            diagnoses,
            case_queries,
            case_responses,
            duplicate,
            likelihood_payload,
            case_likelihoods,
            audit,
        ) in zip(
            cases,
            initial,
            queries,
            responses,
            duplicates,
            likelihood_payloads,
            parsed_likelihoods,
            audits,
            strict=True,
        )
    ]
    usage = {
        "support_generator": generator.usage_snapshot(),
        "query_generator": query_generator.usage_snapshot(),
        "gatekeeper": gatekeeper.usage_snapshot(),
        "likelihood": likelihood.usage_snapshot(),
        "judge": judge.usage_snapshot(),
    }
    summary = summarize(records, usage)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "failed",
        "protocol": {
            "selection_seed": SELECTION_SEED,
            "smoke_ids": list(SMOKE_IDS),
            "query_ids": list(QUERY_IDS),
            "outcome_ids": list(OUTCOME_IDS),
            "likelihood_support_size": LIKELIHOOD_SUPPORT_SIZE,
            "expected_requests": EXPECTED_REQUESTS,
            "generator_model": generator_model,
            "environment_model": environment_model,
            "likelihood_model": likelihood_model,
            "judge_model": judge_model,
            "reasoning_disabled": True,
            "retries": 0,
            "truth_hidden_until_queries_and_outcomes_frozen": True,
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
    parser.add_argument("--generator-model", default="openai/gpt-5.4")
    parser.add_argument("--environment-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--likelihood-model", default="openai/gpt-5.4-mini")
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
            environment_model=args.environment_model,
            likelihood_model=args.likelihood_model,
            judge_model=args.judge_model,
        )
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "status": "runtime_failure",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
        if isinstance(exc, StructuredBatchError):
            failure.update(
                {
                    "stage": exc.stage,
                    "row": exc.row,
                    "response": exc.response,
                }
            )
        output_path.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
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
