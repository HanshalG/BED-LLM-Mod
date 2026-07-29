#!/usr/bin/env python3
"""Qualify selective full-world beliefs for ICAE sequential BED."""

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
    exact_controller_reply,
    matcher_response_format,
    trigger_catalog,
)
from scripts.icae_bench_first_link_instrument_smoke import (
    branch_answers_response_format,
    parse_branch_answers,
)
from scripts.icae_bench_first_link_instrument_smoke_v2 import (
    matcher_messages_set,
    parse_matcher_set,
)
from scripts.icae_bench_semantic_serving_smoke import (
    ServingModel,
    _adapter,
    canonical_text,
    strict_json_object,
    usage_summary,
)
from scripts.icae_bench_source_opportunity_audit import (
    sha256_bytes,
    substantive_constraints,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "icae-full-world-instrument-smoke-1"
EXPECTED_MANIFEST_SHA256 = (
    "47ab7f2fcf985433389f53873fa7fe8ccbebd88dd8390d654de93083bad84f5f"
)
EXPECTED_SOURCE_AUDIT_SHA256 = (
    "f63a313adb8bc3bd1090fe4542d1b38b4c2e63abca30c559447a70a6e6348552"
)
SELECTION_SEED = 51_300
PLANNER_SEED = 51_400
EVALUATOR_SEED = 51_500
PLANNER_MODEL_ID = "openai/gpt-5.4"
EVALUATOR_MODEL_ID = "openai/gpt-5.4-mini"
WORLD_COUNT = 8
CLAUSES_PER_WORLD = 6
QUESTION_COUNT = 6
EXPECTED_REQUESTS = 10
PLANNER_MAX_TOKENS = 5200
EVALUATOR_MAX_TOKENS = 5200
PROJECTED_COST_USD = 0.25
RUN_BUDGET_USD = 0.50
SIMULATED_FALLBACK = "No additional requirement is specified for that topic."
MIN_CHANGED_QUESTIONS = 4
MIN_RETENTION_DIFFERENCE = 1
MIN_DISTINCT_LIKELIHOODS = 3
MIN_ENDPOINT_COVERAGE = 0.20
MAX_ENDPOINT_COVERAGE = 0.90
MIN_ENDPOINT_RANGE = 0.10
MIN_EFFECTIVE_WORLDS = 2.0


def select_development_task(
    manifest: dict[str, Any],
    source_audit: dict[str, Any],
) -> dict[str, Any]:
    eligible_aliases = {
        row["alias"]
        for row in source_audit["tasks"]
        if row["substantive_constraint_count"] >= 10
        and row["lexical_unlock_target_count"] >= 5
    }
    eligible = [
        row
        for row in manifest["partitions"]["mechanics"]
        if row["alias"] in eligible_aliases
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


def world_support_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_latent_contract_worlds",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["worlds", "questions"],
                "properties": {
                    "worlds": {
                        "type": "array",
                        "minItems": WORLD_COUNT,
                        "maxItems": WORLD_COUNT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "world_index",
                                "probability_percent",
                                "clauses",
                            ],
                            "properties": {
                                "world_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": WORLD_COUNT - 1,
                                },
                                "probability_percent": {
                                    "type": "integer",
                                    "minimum": 1,
                                    "maximum": 100,
                                },
                                "clauses": {
                                    "type": "array",
                                    "minItems": CLAUSES_PER_WORLD,
                                    "maxItems": CLAUSES_PER_WORLD,
                                    "items": {
                                        "type": "string",
                                        "minLength": 10,
                                        "maxLength": 500,
                                    },
                                },
                            },
                        },
                    },
                    "questions": {
                        "type": "array",
                        "minItems": QUESTION_COUNT,
                        "maxItems": QUESTION_COUNT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["question_index", "question"],
                            "properties": {
                                "question_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": QUESTION_COUNT - 1,
                                },
                                "question": {
                                    "type": "string",
                                    "minLength": 10,
                                    "maxLength": 500,
                                },
                            },
                        },
                    },
                },
            },
        },
    }


def _normalized_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    normalized = " ".join(value.split())
    if not 10 <= len(normalized) <= 500:
        raise ValueError(f"{label} has invalid length")
    return normalized


def parse_world_support(response: str, *, label: str) -> dict[str, Any]:
    value = strict_json_object(response, label=label)
    if set(value) != {"worlds", "questions"}:
        raise ValueError(f"{label} has unexpected fields")
    worlds_raw = value["worlds"]
    questions_raw = value["questions"]
    if not isinstance(worlds_raw, list) or len(worlds_raw) != WORLD_COUNT:
        raise ValueError(f"{label} must contain eight worlds")
    if (
        not isinstance(questions_raw, list)
        or len(questions_raw) != QUESTION_COUNT
    ):
        raise ValueError(f"{label} must contain six questions")

    worlds = []
    world_keys = set()
    total_probability = 0
    for expected_index, row in enumerate(worlds_raw):
        if not isinstance(row, dict) or set(row) != {
            "world_index",
            "probability_percent",
            "clauses",
        }:
            raise ValueError(f"{label} world has unexpected fields")
        if row["world_index"] != expected_index:
            raise ValueError(f"{label} worlds are not in frozen order")
        probability = row["probability_percent"]
        if (
            not isinstance(probability, int)
            or isinstance(probability, bool)
            or not 1 <= probability <= 100
        ):
            raise ValueError(f"{label} world probability is invalid")
        clauses_raw = row["clauses"]
        if (
            not isinstance(clauses_raw, list)
            or len(clauses_raw) != CLAUSES_PER_WORLD
        ):
            raise ValueError(f"{label} world must contain six clauses")
        clauses = [
            _normalized_text(
                clause,
                label=f"{label}.world[{expected_index}].clause",
            )
            for clause in clauses_raw
        ]
        clause_keys = [canonical_text(clause) for clause in clauses]
        if len(set(clause_keys)) != CLAUSES_PER_WORLD:
            raise ValueError(f"{label} world contains duplicate clauses")
        world_key = tuple(sorted(clause_keys))
        if world_key in world_keys:
            raise ValueError(f"{label} contains duplicate worlds")
        world_keys.add(world_key)
        total_probability += probability
        worlds.append(
            {
                "world_index": expected_index,
                "probability_percent": probability,
                "probability": probability / 100.0,
                "clauses": clauses,
            }
        )
    if total_probability != 100:
        raise ValueError(f"{label} world probabilities do not sum to 100")

    questions = []
    seen_questions = set()
    for expected_index, row in enumerate(questions_raw):
        if not isinstance(row, dict) or set(row) != {
            "question_index",
            "question",
        }:
            raise ValueError(f"{label} question has unexpected fields")
        if row["question_index"] != expected_index:
            raise ValueError(f"{label} questions are not in frozen order")
        question = _normalized_text(
            row["question"],
            label=f"{label}.question[{expected_index}]",
        )
        key = canonical_text(question)
        if key in seen_questions:
            raise ValueError(f"{label} contains duplicate questions")
        seen_questions.add(key)
        questions.append(question)
    return {"worlds": worlds, "questions": questions}


def initial_world_messages(fuzzy_prd: str) -> list[dict[str, str]]:
    request = {
        "task": (
            "Infer a probability distribution over competing latent "
            "implementation-contract worlds and propose discriminating "
            "clarification questions."
        ),
        "fuzzy_prd": fuzzy_prd,
        "rules": [
            "Return exactly eight mutually competing worlds.",
            "Each world is one coherent conjunction of exactly six unresolved atomic behavioral clauses.",
            "A clause must be concrete, falsifiable, and express one behavior; no vague umbrella, list, or disjunction.",
            "Shared requirements already explicit in the PRD may be omitted.",
            "Worlds must differ on meaningful behavioral alternatives, not wording.",
            "Assign positive integer probabilities summing exactly to 100.",
            "Return six specific questions chosen to separate these worlds.",
            "Do not ask for everything, use a generic anything-else question, or mention hidden data.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You construct finite semantic belief states for Bayesian "
                "experimental design. Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


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
        "worlds": [
            {
                "probability_percent": world["probability_percent"],
                "clauses": world["clauses"],
            }
            for world in support["worlds"]
        ],
        "questions": support["questions"],
        "rules": [
            "Each answer must be concrete and consistent with at least one supplied world.",
            "State one behavioral contract rather than a fallback.",
            "Do not mention hidden data, trigger IDs, or uncertainty.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You generate counterfactual semantic observations for "
                "experimental design. Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def likelihood_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_world_question_likelihoods",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["cells"],
                "properties": {
                    "cells": {
                        "type": "array",
                        "minItems": WORLD_COUNT * QUESTION_COUNT,
                        "maxItems": WORLD_COUNT * QUESTION_COUNT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "world_index",
                                "question_index",
                                "positive_probability",
                            ],
                            "properties": {
                                "world_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": WORLD_COUNT - 1,
                                },
                                "question_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": QUESTION_COUNT - 1,
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


def likelihood_messages(
    support: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Estimate the semantic observation likelihood for every "
            "world-question pair."
        ),
        "worlds": [
            {"world_index": index, "clauses": world["clauses"]}
            for index, world in enumerate(support["worlds"])
        ],
        "questions": [
            {"question_index": index, "question": question}
            for index, question in enumerate(support["questions"])
        ],
        "rules": [
            "positive_probability is the chance that a strict owner in that complete world gives a concrete informative answer rather than the fixed fallback.",
            "Use integer probabilities from 0 to 100.",
            "Return every world-question pair exactly once in world-major then question-major order.",
            "Judge semantic entailment, not lexical overlap.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You estimate semantic likelihoods for Bayesian experimental "
                "design. Follow the schema and frozen row order exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def parse_likelihoods(response: str) -> list[list[float]]:
    value = strict_json_object(response, label="likelihoods")
    if set(value) != {"cells"}:
        raise ValueError("likelihoods have unexpected fields")
    cells = value["cells"]
    if not isinstance(cells, list) or len(cells) != (
        WORLD_COUNT * QUESTION_COUNT
    ):
        raise ValueError("likelihoods have wrong cardinality")
    matrix = [[0.0] * QUESTION_COUNT for _ in range(WORLD_COUNT)]
    for offset, cell in enumerate(cells):
        expected_world = offset // QUESTION_COUNT
        expected_question = offset % QUESTION_COUNT
        if not isinstance(cell, dict) or set(cell) != {
            "world_index",
            "question_index",
            "positive_probability",
        }:
            raise ValueError("likelihood cell has unexpected fields")
        probability = cell["positive_probability"]
        if (
            cell["world_index"] != expected_world
            or cell["question_index"] != expected_question
            or not isinstance(probability, int)
            or isinstance(probability, bool)
            or not 0 <= probability <= 100
        ):
            raise ValueError("likelihood cells are not valid and ordered")
        matrix[expected_world][expected_question] = probability / 100.0
    return matrix


def refresh_messages(
    fuzzy_prd: str,
    initial: dict[str, Any],
    question: str,
    answer: str,
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Regenerate the latent implementation-contract belief after one "
            "observed clarification answer."
        ),
        "fuzzy_prd": fuzzy_prd,
        "prior_worlds": [
            {
                "probability_percent": world["probability_percent"],
                "clauses": world["clauses"],
            }
            for world in initial["worlds"]
        ],
        "history": [{"question": question, "answer": answer}],
        "rules": [
            "Discard and regenerate the latent support from the complete visible history.",
            "Return exactly eight competing worlds of six unresolved atomic clauses each.",
            "Treat a fallback as evidence that the queried requirement is absent; do not repeat that topic.",
            "Do not repeat a positively resolved clause as unresolved.",
            "Clauses must be concrete, falsifiable, single behaviors without vague umbrellas or disjunctions.",
            "Assign positive integer probabilities summing exactly to 100.",
            "Return six new specific questions that discriminate the refreshed worlds.",
            "Do not ask for everything or use a generic anything-else question.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You update a finite semantic belief state from observations. "
                "Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def retention_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_world_branch_retention",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["rows"],
                "properties": {
                    "rows": {
                        "type": "array",
                        "minItems": 2 * WORLD_COUNT,
                        "maxItems": 2 * WORLD_COUNT,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "branch",
                                "initial_world_index",
                                "represented",
                            ],
                            "properties": {
                                "branch": {
                                    "type": "string",
                                    "enum": ["positive", "negative"],
                                },
                                "initial_world_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": WORLD_COUNT - 1,
                                },
                                "represented": {"type": "boolean"},
                            },
                        },
                    }
                },
            },
        },
    }


def retention_messages(
    initial: dict[str, Any],
    positive: dict[str, Any],
    negative: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Judge whether each initial complete world remains represented "
            "in each refreshed support."
        ),
        "initial_worlds": [
            {"world_index": index, "clauses": world["clauses"]}
            for index, world in enumerate(initial["worlds"])
        ],
        "positive_worlds": [
            world["clauses"] for world in positive["worlds"]
        ],
        "negative_worlds": [
            world["clauses"] for world in negative["worlds"]
        ],
        "rules": [
            "represented=true only if one refreshed world preserves at least four of the initial world's six behavioral clauses semantically.",
            "Paraphrases and more specific forms count; broad topical overlap does not.",
            "Return positive rows 0..7 followed by negative rows 0..7.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict semantic world-retention evaluator. "
                "Follow the schema and row order exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def parse_retention(response: str) -> dict[str, list[bool]]:
    value = strict_json_object(response, label="retention")
    if set(value) != {"rows"}:
        raise ValueError("retention has unexpected fields")
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != 2 * WORLD_COUNT:
        raise ValueError("retention has wrong cardinality")
    parsed = {"positive": [], "negative": []}
    for offset, row in enumerate(rows):
        branch = "positive" if offset < WORLD_COUNT else "negative"
        expected_index = offset % WORLD_COUNT
        if not isinstance(row, dict) or set(row) != {
            "branch",
            "initial_world_index",
            "represented",
        }:
            raise ValueError("retention row has unexpected fields")
        if (
            row["branch"] != branch
            or row["initial_world_index"] != expected_index
            or not isinstance(row["represented"], bool)
        ):
            raise ValueError("retention rows are not valid and ordered")
        parsed[branch].append(row["represented"])
    return parsed


def endpoint_response_format(row_count: int) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "icae_world_hidden_coverage",
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
                            "required": [
                                "world_index",
                                "constraint_id",
                                "covered",
                            ],
                            "properties": {
                                "world_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": WORLD_COUNT - 1,
                                },
                                "constraint_id": {"type": "string"},
                                "covered": {"type": "boolean"},
                            },
                        },
                    }
                },
            },
        },
    }


def endpoint_messages(
    constraints: list[tuple[str, dict[str, Any]]],
    support: dict[str, Any],
) -> list[dict[str, str]]:
    request = {
        "task": (
            "Judge hidden-requirement coverage separately for every "
            "competing generated world."
        ),
        "hidden_requirements": [
            {"constraint_id": identifier, "text": row["oracle_response"]}
            for identifier, row in constraints
        ],
        "worlds": [
            {"world_index": index, "clauses": world["clauses"]}
            for index, world in enumerate(support["worlds"])
        ],
        "rules": [
            "For each world independently, covered=true only when one of its clauses is semantically equivalent to or more specific than the hidden behavioral contract.",
            "Broad topical overlap, a disjunction, or combining evidence across different worlds does not count.",
            "Return every world-constraint pair in world-major then supplied-constraint order.",
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are an independent strict semantic endpoint evaluator. "
                "Never union evidence across worlds. Follow the schema exactly."
            ),
        },
        {"role": "user", "content": json.dumps(request, sort_keys=True)},
    ]


def parse_endpoint(
    response: str,
    *,
    constraint_ids: list[str],
) -> list[list[bool]]:
    value = strict_json_object(response, label="endpoint")
    if set(value) != {"rows"}:
        raise ValueError("endpoint has unexpected fields")
    expected_count = WORLD_COUNT * len(constraint_ids)
    rows = value["rows"]
    if not isinstance(rows, list) or len(rows) != expected_count:
        raise ValueError("endpoint has wrong cardinality")
    matrix = [
        [False] * len(constraint_ids) for _ in range(WORLD_COUNT)
    ]
    for offset, row in enumerate(rows):
        world_index = offset // len(constraint_ids)
        constraint_offset = offset % len(constraint_ids)
        constraint_id = constraint_ids[constraint_offset]
        if not isinstance(row, dict) or set(row) != {
            "world_index",
            "constraint_id",
            "covered",
        }:
            raise ValueError("endpoint row has unexpected fields")
        if (
            row["world_index"] != world_index
            or row["constraint_id"] != constraint_id
            or not isinstance(row["covered"], bool)
        ):
            raise ValueError("endpoint rows are not valid and ordered")
        matrix[world_index][constraint_offset] = row["covered"]
    return matrix


def changed_question_count(
    first: dict[str, Any],
    second: dict[str, Any],
) -> int:
    first_keys = {canonical_text(question) for question in first["questions"]}
    return sum(
        canonical_text(question) not in first_keys
        for question in second["questions"]
    )


def world_fingerprint(support: dict[str, Any]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted(
            tuple(sorted(canonical_text(clause) for clause in world["clauses"]))
            for world in support["worlds"]
        )
    )


def effective_world_count(support: dict[str, Any]) -> float:
    return 1.0 / sum(
        world["probability"] ** 2 for world in support["worlds"]
    )


def endpoint_statistics(
    support: dict[str, Any],
    coverage: list[list[bool]],
) -> tuple[float, list[float]]:
    rates = [
        sum(row) / len(row)
        for row in coverage
    ]
    expected = sum(
        world["probability"] * rate
        for world, rate in zip(support["worlds"], rates, strict=True)
    )
    return expected, rates


def run_full_world_smoke(
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
    source_audit = json.loads(source_audit_path.read_text(encoding="utf-8"))
    selected = select_development_task(manifest, source_audit)
    record_path = repo / selected["oracle_record"]
    if sha256_bytes(record_path.read_bytes()) != selected["oracle_record_sha256"]:
        raise ValueError("Selected ICAE Oracle record hash mismatch")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    private_path = output_dir / "private" / "RAW_RESPONSES.json"
    private: dict[str, Any] = {"selected": selected}

    def save_raw(name: str, value: Any) -> None:
        private[name] = value
        checkpoint(private_path, private)

    initial_raw = planner_model.chat_complete_messages_batched_structured(
        [initial_world_messages(record["fuzzy_prd"])],
        temperature=0.0,
        block_size=1,
        response_format=world_support_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    save_raw("initial_raw", initial_raw)
    initial = parse_world_support(initial_raw, label="initial")

    branch_answers_raw = planner_model.chat_complete_messages_batched_structured(
        [branch_answer_messages(record["fuzzy_prd"], initial)],
        temperature=0.0,
        block_size=1,
        response_format=branch_answers_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    save_raw("branch_answers_raw", branch_answers_raw)
    branch_answers = parse_branch_answers(branch_answers_raw)

    likelihood_raw = evaluator_model.chat_complete_messages_batched_structured(
        [likelihood_messages(initial)],
        temperature=0.0,
        block_size=1,
        response_format=likelihood_response_format(),
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    save_raw("likelihood_raw", likelihood_raw)
    likelihoods = parse_likelihoods(likelihood_raw)

    root_question = initial["questions"][0]
    positive_raw = planner_model.chat_complete_messages_batched_structured(
        [
            refresh_messages(
                record["fuzzy_prd"],
                initial,
                root_question,
                branch_answers[0],
            )
        ],
        temperature=0.0,
        block_size=1,
        response_format=world_support_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    save_raw("positive_raw", positive_raw)
    positive = parse_world_support(positive_raw, label="positive")

    negative_raw = planner_model.chat_complete_messages_batched_structured(
        [
            refresh_messages(
                record["fuzzy_prd"],
                initial,
                root_question,
                SIMULATED_FALLBACK,
            )
        ],
        temperature=0.0,
        block_size=1,
        response_format=world_support_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    save_raw("negative_raw", negative_raw)
    negative = parse_world_support(negative_raw, label="negative")

    retention_raw = evaluator_model.chat_complete_messages_batched_structured(
        [retention_messages(initial, positive, negative)],
        temperature=0.0,
        block_size=1,
        response_format=retention_response_format(),
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    save_raw("retention_raw", retention_raw)
    retention = parse_retention(retention_raw)

    valid_ids = [row["id"] for row in trigger_catalog(record)]
    match_raw = evaluator_model.chat_complete_messages_batched_structured(
        [matcher_messages_set(record, root_question)],
        temperature=0.0,
        block_size=1,
        response_format=matcher_response_format(),
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    save_raw("match_raw", match_raw)
    match = parse_matcher_set(
        match_raw,
        valid_ids=valid_ids,
        label="actual match",
    )
    actual_reply = exact_controller_reply(record, match["matched_ids"])

    actual_raw = planner_model.chat_complete_messages_batched_structured(
        [
            refresh_messages(
                record["fuzzy_prd"],
                initial,
                root_question,
                actual_reply,
            )
        ],
        temperature=0.0,
        block_size=1,
        response_format=world_support_response_format(),
        max_new_tokens=PLANNER_MAX_TOKENS,
    )[0]
    save_raw("actual_raw", actual_raw)
    actual = parse_world_support(actual_raw, label="actual")

    constraints = substantive_constraints(
        record["oracle_data"]["hidden_constraints"]
    )
    constraint_ids = [identifier for identifier, _ in constraints]
    endpoint_prompt = endpoint_messages(constraints, actual)
    endpoint_format = endpoint_response_format(
        WORLD_COUNT * len(constraint_ids)
    )
    endpoint_raw = evaluator_model.chat_complete_messages_batched_structured(
        [endpoint_prompt],
        temperature=0.0,
        block_size=1,
        response_format=endpoint_format,
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    save_raw("endpoint_raw", endpoint_raw)
    endpoint = parse_endpoint(
        endpoint_raw,
        constraint_ids=constraint_ids,
    )
    endpoint_repeat_raw = evaluator_model.chat_complete_messages_batched_structured(
        [endpoint_prompt],
        temperature=0.0,
        block_size=1,
        response_format=endpoint_format,
        max_new_tokens=EVALUATOR_MAX_TOKENS,
    )[0]
    save_raw("endpoint_repeat_raw", endpoint_repeat_raw)
    endpoint_repeat = parse_endpoint(
        endpoint_repeat_raw,
        constraint_ids=constraint_ids,
    )

    distinct_likelihoods = len(
        {round(value, 2) for row in likelihoods for value in row}
    )
    positive_retained = sum(retention["positive"])
    negative_retained = sum(retention["negative"])
    endpoint_expected, endpoint_rates = endpoint_statistics(actual, endpoint)
    endpoint_range = max(endpoint_rates) - min(endpoint_rates)
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
            changed_question_count(initial, positive)
            >= MIN_CHANGED_QUESTIONS
            and changed_question_count(initial, negative)
            >= MIN_CHANGED_QUESTIONS
            and world_fingerprint(positive) != world_fingerprint(negative)
        ),
        "retention_is_path_dependent": (
            abs(positive_retained - negative_retained)
            >= MIN_RETENTION_DIFFERENCE
        ),
        "realized_root_matches_controller": (
            bool(match["matched_ids"]) and not match["fallback"]
        ),
        "actual_history_changes_support": (
            changed_question_count(initial, actual)
            >= MIN_CHANGED_QUESTIONS
            and world_fingerprint(initial) != world_fingerprint(actual)
        ),
        "beliefs_remain_noncollapsed": all(
            effective_world_count(support) >= MIN_EFFECTIVE_WORLDS
            for support in (initial, positive, negative, actual)
        ),
        "endpoint_replays_exactly": endpoint == endpoint_repeat,
        "endpoint_is_unsaturated": (
            MIN_ENDPOINT_COVERAGE
            <= endpoint_expected
            <= MAX_ENDPOINT_COVERAGE
        ),
        "endpoint_distinguishes_worlds": (
            endpoint_range >= MIN_ENDPOINT_RANGE
        ),
        "cost_within_cap": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    all_pass = all(gates.values())
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
            "positive_changed_questions": changed_question_count(
                initial, positive
            ),
            "negative_changed_questions": changed_question_count(
                initial, negative
            ),
            "actual_changed_questions": changed_question_count(initial, actual),
            "positive_retained_worlds": positive_retained,
            "negative_retained_worlds": negative_retained,
            "effective_world_counts": {
                name: effective_world_count(support)
                for name, support in (
                    ("initial", initial),
                    ("positive", positive),
                    ("negative", negative),
                    ("actual", actual),
                )
            },
            "actual_match_count": len(match["matched_ids"]),
            "endpoint_constraint_count": len(constraint_ids),
            "endpoint_expected_coverage": endpoint_expected,
            "endpoint_world_coverage_rates": endpoint_rates,
            "endpoint_world_coverage_range": endpoint_range,
        },
        "gates": gates,
        "all_pass": all_pass,
        "decision": (
            "authorize_separately_frozen_full_world_paired_measurement"
            if all_pass
            else "close_exact_full_world_instrument"
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
        payload = run_full_world_smoke(
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
