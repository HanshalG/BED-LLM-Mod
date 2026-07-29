#!/usr/bin/env python3
"""Smoke the complete ICAE model-aware first-link measurement path."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.icae_bench_exact_response_controller_smoke import (
    EXCLUDED_OPENED_ALIASES,
    exact_controller_reply,
    matcher_messages,
    matcher_response_format,
    parse_matcher,
    trigger_catalog,
)
from scripts.icae_bench_semantic_serving_smoke import (
    NUM_HYPOTHESES,
    NUM_QUESTIONS,
    ServingModel,
    _adapter,
    canonical_text,
    changed_question_count,
    followup_messages,
    initial_messages,
    parse_support,
    planner_response_format,
    strict_json_object,
    usage_summary,
)
from scripts.icae_bench_source_opportunity_audit import (
    substantive_constraints,
    sha256_bytes,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "icae-first-link-instrument-smoke-1"
EXPECTED_MANIFEST_SHA256 = (
    "47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f"
)
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "f63a313adb8bc3bd1090fe4542d1b38b4c2e63abca30c559447a70a6e6348552"
)
ADDITIONALLY_OPENED_ALIASES = {"realcode@235", "realcode@185"}
SELECTION_SEED = 50_700
PLANNER_SEED = 50_800
EVALUATOR_SEED = 50_900
PLANNER_MODEL_ID = "openai/gpt-5.4"
EVALUATOR_MODEL_ID = "openai/gpt-5.4-mini"
EXPECTED_REQUESTS = 10
PLANNER_MAX_TOKENS = 3200
EVALUATOR_MAX_TOKENS = 3500
PROJECTED_COST_USD = 0.18
RUN_BUDGET_USD = 0.40
SIMULATED_FALLBACK = "No additional requirement is specified for that topic."
MIN_DISTINCT_LIKELIHOODS = 3
MIN_PATH_COVERAGE_DIFFERENCE = 1
MIN_ENDPOINT_COVERAGE = 0.20
MAX_ENDPOINT_COVERAGE = 0.90


def select_instrument_task(manifest: dict[str, Any]) -> dict[str, Any]:
    excluded = EXCLUDED_OPENED_ALIASES | ADDITIONALLY_OPENED_ALIASES
    eligible = [
        row
        for row in manifest["partitions"]["mechanics"]
        if row["alias"] not in excluded
    ]
    return min(
        eligible,
        key=lambda row: (
            hashlib.sha256(
                f"{SELECTION_SEED}:{row['alias']}".encode("utf-8")
            ).hexdigest(),
            row["alias"],
        ),
    )


def branch_answers_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_hypothetical_positive_answers",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["branches"],
                "properties": {
                    "branches": {
                        "type": "array",
                        "minItems": NUM_QUESTIONS,
                        "maxItems": NUM_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["question_index", "positive_answer"],
                            "properties": {
                                "question_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": NUM_QUESTIONS - 1,
                                },
                                "positive_answer": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 800,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def parse_branch_answers(response: str) -> list[str]:
    value = strict_json_object(response, label="branch answers")
    if set(value) != {"branches"}:
        raise ValueError("branch answers have unexpected fields")
    rows = value["branches"]
    if not isinstance(rows, list) or len(rows) != NUM_QUESTIONS:
        raise ValueError("branch answers must contain six rows")
    by_index: dict[int, str] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "question_index",
            "positive_answer",
        }:
            raise ValueError("branch answer row has unexpected fields")
        index = row["question_index"]
        answer = " ".join(str(row["positive_answer"]).split())
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < NUM_QUESTIONS
            or not 10 <= len(answer) <= 800
        ):
            raise ValueError("branch answer row is invalid")
        if index in by_index:
            raise ValueError("branch answer index is duplicated")
        by_index[index] = answer
    if set(by_index) != set(range(NUM_QUESTIONS)):
        raise ValueError("branch answers do not cover every question")
    return [by_index[index] for index in range(NUM_QUESTIONS)]


def branch_answer_messages(
    fuzzy_prd: str,
    support: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Construct one plausible informative positive product-owner "
            "answer for each candidate clarification question."
        ),
        "fuzzy_prd": fuzzy_prd,
        "hypotheses": support["hypotheses"],
        "questions": [
            row["question"] for row in support["questions"]
        ],
        "rules": [
            "Answer under a plausible world represented by the hypotheses.",
            "State one concrete requirement or contract.",
            "Do not mention hidden data, tests, trigger IDs, or uncertainty.",
            "Do not use a fallback or say that no requirement exists.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate counterfactual observations for semantic "
                "experimental design. Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def likelihood_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_hypothesis_question_likelihoods",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cells"],
                "properties": {
                    "cells": {
                        "type": "array",
                        "minItems": NUM_HYPOTHESES * NUM_QUESTIONS,
                        "maxItems": NUM_HYPOTHESES * NUM_QUESTIONS,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "hypothesis_index",
                                "question_index",
                                "positive_probability",
                            ],
                            "properties": {
                                "hypothesis_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": NUM_HYPOTHESES - 1,
                                },
                                "question_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": NUM_QUESTIONS - 1,
                                },
                                "positive_probability": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def parse_likelihoods(response: str) -> list[list[float]]:
    value = strict_json_object(response, label="likelihoods")
    if set(value) != {"cells"}:
        raise ValueError("likelihood response has unexpected fields")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != (
        NUM_HYPOTHESES * NUM_QUESTIONS
    ):
        raise ValueError("likelihood response has wrong cardinality")
    matrix = [[-1.0] * NUM_QUESTIONS for _ in range(NUM_HYPOTHESES)]
    seen = set()
    for cell in cells:
        if not isinstance(cell, dict) or set(cell) != {
            "hypothesis_index",
            "question_index",
            "positive_probability",
        }:
            raise ValueError("likelihood cell has unexpected fields")
        h = cell["hypothesis_index"]
        q = cell["question_index"]
        probability = cell["positive_probability"]
        if (
            not isinstance(h, int)
            or isinstance(h, bool)
            or not isinstance(q, int)
            or isinstance(q, bool)
            or not isinstance(probability, int)
            or isinstance(probability, bool)
            or not 0 <= h < NUM_HYPOTHESES
            or not 0 <= q < NUM_QUESTIONS
            or not 0 <= probability <= 100
            or (h, q) in seen
        ):
            raise ValueError("likelihood cell is invalid")
        seen.add((h, q))
        matrix[h][q] = probability / 100.0
    if len(seen) != NUM_HYPOTHESES * NUM_QUESTIONS:
        raise ValueError("likelihood matrix is incomplete")
    return matrix


def likelihood_messages(
    support: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Estimate semantic observation likelihoods. For every hypothesis "
            "and question, estimate the probability that a strict product "
            "owner in that hypothesis world gives a concrete informative "
            "positive answer rather than the fixed fallback."
        ),
        "hypotheses": [
            row["requirement"] for row in support["hypotheses"]
        ],
        "questions": [
            row["question"] for row in support["questions"]
        ],
        "rules": [
            "Return every hypothesis-question pair exactly once.",
            "Use integer probabilities from 0 to 100.",
            "Judge semantic relevance, not lexical overlap.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You estimate semantic likelihoods for Bayesian experimental "
                "design. Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def coverage_response_format(
    *,
    row_count: int,
    name: str,
    id_key: str,
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rows"],
                "properties": {
                    "rows": {
                        "type": "array",
                        "minItems": row_count,
                        "maxItems": row_count,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [id_key, "covered"],
                            "properties": {
                                id_key: {"type": "string"},
                                "covered": {"type": "boolean"},
                            },
                        },
                    }
                },
            },
        },
    }


def parse_coverage(
    response: str,
    *,
    expected_ids: list[str],
    id_key: str,
    label: str,
) -> dict[str, bool]:
    value = strict_json_object(response, label=label)
    if set(value) != {"rows"}:
        raise ValueError(f"{label} has unexpected fields")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != len(expected_ids):
        raise ValueError(f"{label} has wrong cardinality")
    result: dict[str, bool] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {id_key, "covered"}:
            raise ValueError(f"{label} row has unexpected fields")
        identifier = row[id_key]
        covered = row["covered"]
        if (
            identifier not in expected_ids
            or identifier in result
            or not isinstance(covered, bool)
        ):
            raise ValueError(f"{label} row is invalid")
        result[identifier] = covered
    if list(result) != expected_ids:
        raise ValueError(f"{label} rows are not in frozen order")
    return result


def proxy_coverage_messages(
    initial: dict[str, Any],
    positive: dict[str, Any],
    negative: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "For each initial hypothesis, judge whether each refreshed "
            "support still contains a semantically equivalent requirement."
        ),
        "targets": [
            {"id": f"H{index:02d}", "text": row["requirement"]}
            for index, row in enumerate(initial["hypotheses"])
        ],
        "positive_support": [
            row["requirement"] for row in positive["hypotheses"]
        ],
        "negative_support": [
            row["requirement"] for row in negative["hypotheses"]
        ],
        "rules": [
            "Semantic equivalence or a more specific statement counts as covered.",
            "Topical relatedness without the same requirement does not count.",
            "Return positive rows H00..H11 followed by negative rows H00..H11.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict semantic coverage evaluator. Follow the "
                "schema and preserve row order."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def proxy_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_branch_retention",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rows"],
                "properties": {
                    "rows": {
                        "type": "array",
                        "minItems": 2 * NUM_HYPOTHESES,
                        "maxItems": 2 * NUM_HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["branch", "hypothesis_id", "covered"],
                            "properties": {
                                "branch": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                },
                                "hypothesis_id": {"type": "string"},
                                "covered": {"type": "boolean"},
                            },
                        },
                    }
                },
            },
        },
    }


def parse_proxy_coverage(response: str) -> dict[str, dict[str, bool]]:
    value = strict_json_object(response, label="proxy coverage")
    if set(value) != {"rows"}:
        raise ValueError("proxy coverage has unexpected fields")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != 2 * NUM_HYPOTHESES:
        raise ValueError("proxy coverage has wrong cardinality")
    expected = [
        (branch, f"H{index:02d}")
        for branch in ("positive", "negative")
        for index in range(NUM_HYPOTHESES)
    ]
    parsed = {"positive": {}, "negative": {}}
    observed = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "branch",
            "hypothesis_id",
            "covered",
        }:
            raise ValueError("proxy coverage row has unexpected fields")
        branch = row["branch"]
        identifier = row["hypothesis_id"]
        covered = row["covered"]
        if (
            branch not in parsed
            or identifier in parsed[branch]
            or not isinstance(covered, bool)
        ):
            raise ValueError("proxy coverage row is invalid")
        parsed[branch][identifier] = covered
        observed.append((branch, identifier))
    if observed != expected:
        raise ValueError("proxy coverage rows are not in frozen order")
    return parsed


def endpoint_messages(
    constraints: list[tuple[str, dict[str, Any]]],
    support: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Judge whether the generated belief support covers each released "
            "hidden requirement."
        ),
        "hidden_requirements": [
            {"id": identifier, "text": row["oracle_response"]}
            for identifier, row in constraints
        ],
        "generated_support": [
            row["requirement"] for row in support["hypotheses"]
        ],
        "rules": [
            "A semantically equivalent or more specific generated hypothesis counts.",
            "Topical relatedness without the same behavioral contract does not count.",
            "Return every hidden requirement exactly once in supplied order.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an independent strict semantic endpoint evaluator. "
                "Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def run_instrument_smoke(
    *,
    repo: Path,
    manifest_path: Path,
    source_audit_path: Path,
    output_dir: Path,
    run_id: str,
    planner_model: ServingModel,
    evaluator_model: ServingModel,
) -> dict[str, Any]:
    if sha256_bytes(manifest_path.read_bytes()) != EXPECTED_MANIFEST_SHA256:
        raise ValueError("Frozen ICAE manifest hash mismatch")
    if sha256_bytes(source_audit_path.read_bytes()) != EXPECTED_SOURCE_AUDIT_SHA256:
        raise ValueError("Frozen ICAE source-audit hash mismatch")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = select_instrument_task(manifest)
    record = json.loads(
        (repo / selected["oracle_record"]).read_text(encoding="utf-8")
    )

    initial_raw = planner_model.chat_complete_messages_batched_structured(
        [initial_messages(record["fuzzy_prd"])],
        temperature=0.0,
        block_size=1,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    initial = parse_support(initial_raw, label="initial")

    branch_answers_raw = planner_model.chat_complete_messages_batched_structured(
        [branch_answer_messages(record["fuzzy_prd"], initial)],
        temperature=0.0,
        block_size=1,
        response_format=branch_answers_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    branch_answers = parse_branch_answers(branch_answers_raw)

    likelihood_raw = evaluator_model.chat_complete_messages_batched_structured(
        [likelihood_messages(initial)],
        temperature=0.0,
        block_size=1,
        response_format=likelihood_response_format(),
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    likelihoods = parse_likelihoods(likelihood_raw)

    root_question = initial["questions"][0]["question"]
    positive_raw = planner_model.chat_complete_messages_batched_structured(
        [
            followup_messages(
                record["fuzzy_prd"],
                initial,
                root_question,
                branch_answers[0],
            )
        ],
        temperature=0.0,
        block_size=1,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    negative_raw = planner_model.chat_complete_messages_batched_structured(
        [
            followup_messages(
                record["fuzzy_prd"],
                initial,
                root_question,
                SIMULATED_FALLBACK,
            )
        ],
        temperature=0.0,
        block_size=1,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    positive = parse_support(positive_raw, label="positive refresh")
    negative = parse_support(negative_raw, label="negative refresh")

    proxy_raw = evaluator_model.chat_complete_messages_batched_structured(
        [proxy_coverage_messages(initial, positive, negative)],
        temperature=0.0,
        block_size=1,
        response_format=proxy_response_format(),
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    proxy = parse_proxy_coverage(proxy_raw)

    valid_ids = [row["id"] for row in trigger_catalog(record)]
    actual_match_raw = evaluator_model.chat_complete_messages_batched_structured(
        [matcher_messages(record, root_question)],
        temperature=0.0,
        block_size=1,
        response_format=matcher_response_format(),
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    actual_match = parse_matcher(
        actual_match_raw,
        valid_ids=valid_ids,
        label="actual match",
    )
    actual_reply = exact_controller_reply(
        record, actual_match["matched_ids"]
    )

    actual_refresh_raw = planner_model.chat_complete_messages_batched_structured(
        [
            followup_messages(
                record["fuzzy_prd"],
                initial,
                root_question,
                actual_reply,
            )
        ],
        temperature=0.0,
        block_size=1,
        response_format=planner_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    actual_refresh = parse_support(actual_refresh_raw, label="actual refresh")

    constraints = substantive_constraints(
        record["oracle_data"]["hidden_constraints"]
    )
    endpoint_ids = [identifier for identifier, _ in constraints]
    endpoint_format = coverage_response_format(
        row_count=len(endpoint_ids),
        name="icae_hidden_requirement_coverage",
        id_key="constraint_id",
    )
    endpoint_prompt = endpoint_messages(constraints, actual_refresh)
    endpoint_raw = evaluator_model.chat_complete_messages_batched_structured(
        [endpoint_prompt],
        temperature=0.0,
        block_size=1,
        response_format=endpoint_format,
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    endpoint_repeat_raw = evaluator_model.chat_complete_messages_batched_structured(
        [endpoint_prompt],
        temperature=0.0,
        block_size=1,
        response_format=endpoint_format,
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    endpoint = parse_coverage(
        endpoint_raw,
        expected_ids=endpoint_ids,
        id_key="constraint_id",
        label="endpoint",
    )
    endpoint_repeat = parse_coverage(
        endpoint_repeat_raw,
        expected_ids=endpoint_ids,
        id_key="constraint_id",
        label="endpoint repeat",
    )

    distinct_likelihoods = len(
        {round(value, 2) for row in likelihoods for value in row}
    )
    positive_covered = sum(proxy["positive"].values())
    negative_covered = sum(proxy["negative"].values())
    endpoint_covered = sum(endpoint.values())
    endpoint_rate = endpoint_covered / len(endpoint)
    usage = usage_summary([planner_model, evaluator_model])
    gates = {
        "exact_request_count": usage["adapter_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": (
            usage["retry_count"] == 0
            and usage["provider_error_retries"] == 0
        ),
        "zero_reasoning_and_forced_exits": (
            usage["adapter_reasoning_tokens"] == 0
            and usage["forced_exits"] == 0
            and usage["forced_final_requests"] == 0
        ),
        "likelihoods_are_nonconstant": (
            distinct_likelihoods >= MIN_DISTINCT_LIKELIHOODS
        ),
        "simulated_branches_change_support": (
            changed_question_count(initial, positive) >= 4
            and changed_question_count(initial, negative) >= 4
            and {
                canonical_text(row["requirement"])
                for row in positive["hypotheses"]
            }
            != {
                canonical_text(row["requirement"])
                for row in negative["hypotheses"]
            }
        ),
        "proxy_detects_path_dependent_retention": (
            abs(positive_covered - negative_covered)
            >= MIN_PATH_COVERAGE_DIFFERENCE
        ),
        "realized_root_matches_controller": bool(
            actual_match["matched_ids"]
        )
        and not actual_match["fallback"],
        "actual_history_changes_support": (
            changed_question_count(initial, actual_refresh) >= 4
        ),
        "endpoint_replays_exactly": endpoint == endpoint_repeat,
        "endpoint_is_unsaturated": (
            MIN_ENDPOINT_COVERAGE <= endpoint_rate <= MAX_ENDPOINT_COVERAGE
        ),
        "cost_within_cap": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    all_pass = all(gates.values())
    private = {
        "selected": selected,
        "initial_raw": initial_raw,
        "branch_answers_raw": branch_answers_raw,
        "likelihood_raw": likelihood_raw,
        "positive_raw": positive_raw,
        "negative_raw": negative_raw,
        "proxy_raw": proxy_raw,
        "actual_match_raw": actual_match_raw,
        "actual_refresh_raw": actual_refresh_raw,
        "endpoint_raw": endpoint_raw,
        "endpoint_repeat_raw": endpoint_repeat_raw,
    }
    private_path = output_dir / "private" / "RAW_RESPONSES.json"
    checkpoint(private_path, private)
    return {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if all_pass else "gated_null",
        "run_id": run_id,
        "manifest_sha256": EXPECTED_MANIFEST_SHA256,
        "source_audit_sha256": EXPECTED_SOURCE_AUDIT_SHA256,
        "selected_task": {
            "alias": selected["alias"],
            "language": selected["language"],
        },
        "models": {
            "planner": PLANNER_MODEL_ID,
            "evaluator_matcher": EVALUATOR_MODEL_ID,
        },
        "seeds": {
            "selection": SELECTION_SEED,
            "planner": PLANNER_SEED,
            "evaluator": EVALUATOR_SEED,
        },
        "usage": usage,
        "diagnostics": {
            "distinct_likelihood_values": distinct_likelihoods,
            "positive_proxy_covered": positive_covered,
            "negative_proxy_covered": negative_covered,
            "actual_match_count": len(actual_match["matched_ids"]),
            "endpoint_constraint_count": len(endpoint),
            "endpoint_covered_count": endpoint_covered,
            "endpoint_coverage_rate": endpoint_rate,
        },
        "gates": gates,
        "all_pass": all_pass,
        "decision": (
            "authorize_separately_frozen_paired_mechanics_measurement"
            if all_pass
            else "close_exact_first_link_instrument"
        ),
        "private_raw_sha256": sha256_bytes(private_path.read_bytes()),
        "development_or_later_opened": False,
        "executable_endpoint_opened": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    planner = _adapter(
        model=PLANNER_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=PLANNER_SEED,
        max_tokens=PLANNER_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.65,
    )
    evaluator = _adapter(
        model=EVALUATOR_MODEL_ID,
        run_id=args.run_id,
        output_dir=output_dir,
        request_seed=EVALUATOR_SEED,
        max_tokens=EVALUATOR_MAX_TOKENS,
        projected_cost=PROJECTED_COST_USD * 0.35,
    )
    try:
        payload = run_instrument_smoke(
            repo=args.repo.resolve(),
            manifest_path=args.manifest.resolve(),
            source_audit_path=args.source_audit.resolve(),
            output_dir=output_dir,
            run_id=args.run_id,
            planner_model=planner,
            evaluator_model=evaluator,
        )
        checkpoint(output_dir / "SERVING.json", payload)
    except Exception as exc:
        payload = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "run_id": args.run_id,
            "error": str(exc),
            "usage": usage_summary([planner, evaluator]),
            "development_or_later_opened": False,
            "executable_endpoint_opened": False,
        }
        checkpoint(output_dir / "FAILURE.json", payload)
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
