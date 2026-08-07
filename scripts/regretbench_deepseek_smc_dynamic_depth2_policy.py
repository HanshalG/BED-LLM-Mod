#!/usr/bin/env python3
"""Zero-call core for the sealed RegretBench SMC depth-two policy."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts import regretbench_deepseek_smc_support_recovery as smc_support
from scripts import regretbench_deepseek_support_recovery as primary


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-dynamic-depth2-core-1"
MODEL_ID = smc_support.MODEL_ID
TEMPERATURE = 0.7
MAX_TOKENS = 2_400
PARTICLES = 8
QUESTIONS = 4
MIN_RETAINED = 2
MAX_RETAINED = 6
PROBABILITY_FLOOR = 1e-12
PROTOCOL = primary.REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_SMC_DYNAMIC_DEPTH2_POLICY_PREREGISTRATION_20260807.md"
)
PROTOCOL_SHA256 = (
    "97e9e7a582d38ae989042352755ee500ae4a42726dd6a431309a8e66102dd3cd"
)


ANNOTATION_SYSTEM_PROMPT = """Predict how a user would answer each supplied clarification question under every supplied semantic particle. The particles and questions are fixed model-generated objects. Return exactly one record for every parent_index 0 through 7 and exactly four nonempty replies aligned to the four supplied questions. Do not revise, merge, remove, reweight, or add particles or questions. Use only the supplied prompt and model-generated objects. Return only the required JSON object."""


TRANSITION_SYSTEM_PROMPT = """Update a finite semantic particle population after the visible clarification dialogue. Return exactly one child for every parent_index 0 through 7. Keep useful semantic particles exactly and revise incompatible or weak particles using only the supplied prompt, dialogue, and model-generated parent population. Mark an unchanged interpretation and final answer as retained and a changed one as revised. Retain between two and six children inclusive. Return eight unique children, four ranked distinct single-dimension questions, and four predicted user replies per child aligned to those questions. Do not ask for the entity name, final factual answer, or an omnibus list. Return only the required JSON object."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def validate_protocol_binding() -> None:
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("SMC dynamic policy protocol changed")


def validate_smc_support_predecessor(
    *,
    result_path: Path,
    verification_path: Path,
    daily_result_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    daily = json.loads(daily_result_path.read_text(encoding="utf-8"))
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    result_hash = sha256_file(result_path)
    verification_hash = sha256_file(verification_path)
    if (
        result.get("interface_version")
        != "regretbench-deepseek-smc-support-recovery-daily-1"
        or result.get("status") != "passed"
        or result.get("authorizes")
        != "separately_preregistered_smc_policy_only"
        or result.get("protocol", {}).get("protocol_sha256")
        != smc_support.PROTOCOL_SHA256
        or result.get("protocol", {}).get("smc_policy_endpoint_opened") is not False
        or result.get("protocol", {}).get("primary_policy_endpoint_opened")
        is not False
        or result.get("protocol", {}).get("primary_confirmation_opened")
        is not False
        or result.get("mechanics_gates", {}).get("all_pass") is not True
        or result.get("science", {}).get("gates", {}).get("all_pass") is not True
        or verification.get("status") != "verified"
        or verification.get("result_status") != "passed"
        or verification.get("mismatches") != []
        or verification.get("model_calls") != 0
        or float(verification.get("cost_usd", math.inf)) != 0.0
        or verification.get("artifact_sha256", {}).get("RESULT.json")
        != result_hash
        or daily.get("status") != "complete_reconciled"
        or daily.get("development_status") != "passed"
        or daily.get("independent_replay_passed") is not True
        or daily.get("smc_policy_endpoint_opened") is not False
        or daily.get("primary_policy_endpoint_opened") is not False
        or daily.get("primary_confirmation_opened") is not False
        or daily.get("authorizes")
        != "separately_preregistered_smc_policy_only"
        or daily.get("result_sha256") != result_hash
        or daily.get("verification_sha256") != verification_hash
        or daily.get("ledger_sha256") != sha256_file(ledger_path)
        or ledger.get("date") != "2026-08-09"
        or ledger.get("timezone") != "Europe/London"
        or float(ledger.get("daily_cap_usd", 0.0)) != 5.0
        or ledger.get("account_wide_usage_counts_against_cap") is not True
        or ledger.get("stage", {}).get("status") != "passed"
    ):
        raise ValueError("SMC support result does not authorize the policy")
    return {
        "status": "authorized_smc_policy_development_only",
        "result_sha256": result_hash,
        "verification_sha256": verification_hash,
        "daily_result_sha256": sha256_file(daily_result_path),
        "ledger_sha256": sha256_file(ledger_path),
    }


def annotation_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_smc_parent_annotation",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["particles"],
                "properties": {
                    "particles": {
                        "type": "array",
                        "minItems": PARTICLES,
                        "maxItems": PARTICLES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["parent_index", "predicted_replies"],
                            "properties": {
                                "parent_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": PARTICLES - 1,
                                },
                                "predicted_replies": {
                                    "type": "array",
                                    "minItems": QUESTIONS,
                                    "maxItems": QUESTIONS,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 240,
                                    },
                                },
                            },
                        },
                    }
                },
            },
        },
    }


def transition_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_smc_enriched_transition",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "questions"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": PARTICLES,
                        "maxItems": PARTICLES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "parent_index",
                                "revision_type",
                                "interpretation",
                                "final_answer",
                                "prior_weight",
                                "predicted_replies",
                            ],
                            "properties": {
                                "parent_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": PARTICLES - 1,
                                },
                                "revision_type": {
                                    "type": "string",
                                    "enum": ["retained", "revised"],
                                },
                                "interpretation": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 320,
                                },
                                "final_answer": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 200,
                                },
                                "prior_weight": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                                "predicted_replies": {
                                    "type": "array",
                                    "minItems": QUESTIONS,
                                    "maxItems": QUESTIONS,
                                    "items": {
                                        "type": "string",
                                        "minLength": 1,
                                        "maxLength": 240,
                                    },
                                },
                            },
                        },
                    },
                    "questions": {
                        "type": "array",
                        "minItems": QUESTIONS,
                        "maxItems": QUESTIONS,
                        "items": {
                            "type": "string",
                            "minLength": 2,
                            "maxLength": 240,
                        },
                    },
                },
            },
        },
    }


def _validate_questions(questions: Sequence[Any]) -> list[str]:
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        raise ValueError("support must contain exactly four questions")
    cleaned = []
    seen = set()
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError(f"question {index} is not a question")
        text = question.strip()
        key = primary.normalize_text(text)
        if not key or key in seen:
            raise ValueError("support contains duplicate questions")
        seen.add(key)
        cleaned.append(text)
    return cleaned


def _validate_replies(replies: Any) -> list[str]:
    if not isinstance(replies, list) or len(replies) != QUESTIONS:
        raise ValueError("particle must contain four predicted replies")
    output = []
    for reply in replies:
        if not isinstance(reply, str) or not reply.strip():
            raise ValueError("particle contains an empty predicted reply")
        output.append(reply.strip())
    return output


def support_sha256(support: Mapping[str, Any]) -> str:
    payload = {
        "hypotheses": support["hypotheses"],
        "questions": support["questions"],
    }
    return hashlib.sha256(canonical_json(payload).encode()).hexdigest()


def _public_parent_particles(support: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = support.get("hypotheses") or support.get("particles")
    if not isinstance(rows, list) or len(rows) != PARTICLES:
        raise ValueError("parent support must contain eight indexed particles")
    output = []
    for index, row in enumerate(rows):
        item = {
            "parent_index": index,
            "interpretation": row["interpretation"],
            "final_answer": row["final_answer"],
            "probability": float(row["probability"]),
        }
        if "predicted_replies" in row:
            item["predicted_replies"] = list(row["predicted_replies"])
        output.append(item)
    total = sum(row["probability"] for row in output)
    if total <= 0 or not all(
        math.isfinite(row["probability"]) and row["probability"] >= 0
        for row in output
    ):
        raise ValueError("parent probabilities are invalid")
    for row in output:
        row["probability"] /= total
    return output


def annotation_messages_for(
    cig: Any,
    parent_population: Mapping[str, Any],
    questions: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    fixed_questions = _validate_questions(list(questions))
    particles = _public_parent_particles(parent_population)
    parent_hash = str(parent_population["raw_parent_sha256"])
    base = primary.public_payload(cig, [])
    base_audit = primary.privacy_audit(cig, base)
    payload = {
        **base,
        "parent_particles": particles,
        "questions": fixed_questions,
        "parent_population_sha256": parent_hash,
        "parent_source": "verified_primary_raw_root",
    }
    audit = {
        "passed": base_audit["passed"] is True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "parent_population_sha256": parent_hash,
        "parent_source": payload["parent_source"],
    }
    return [
        {"role": "system", "content": ANNOTATION_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_parent_annotation(
    raw: str,
    parent_population: Mapping[str, Any],
    questions: Sequence[str],
) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("annotation response has wrong top-level fields")
    annotations = value["particles"]
    if not isinstance(annotations, list) or len(annotations) != PARTICLES:
        raise ValueError("annotation must contain eight particles")
    by_index = {}
    for item in annotations:
        if not isinstance(item, dict) or set(item) != {
            "parent_index",
            "predicted_replies",
        }:
            raise ValueError("annotation particle has wrong fields")
        index = item["parent_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(PARTICLES)
            or index in by_index
        ):
            raise ValueError("annotation parent indexes are not an exact permutation")
        by_index[index] = _validate_replies(item["predicted_replies"])
    if sorted(by_index) != list(range(PARTICLES)):
        raise ValueError("annotation parent indexes are not an exact permutation")
    parents = _public_parent_particles(parent_population)
    hypotheses = [
        {**parent, "predicted_replies": by_index[index]}
        for index, parent in enumerate(parents)
    ]
    support = {
        "hypotheses": hypotheses,
        "questions": _validate_questions(list(questions)),
        "diagnostic": {
            "codec_mode": "strict_json",
            "parent_index_permutation_exact": True,
            "particle_count": PARTICLES,
            "question_count": QUESTIONS,
            "parent_population_sha256": parent_population["raw_parent_sha256"],
            "initial_hypotheses_regenerated": False,
            "initial_questions_regenerated": False,
        },
    }
    support["support_sha256"] = support_sha256(support)
    return support


def transition_messages_for(
    cig: Any,
    dialogue: Sequence[Mapping[str, str]],
    parent_support: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    base = primary.public_payload(cig, dialogue)
    base_audit = primary.privacy_audit(cig, base)
    particles = _public_parent_particles(parent_support)
    questions = _validate_questions(list(parent_support["questions"]))
    parent_hash = support_sha256(parent_support)
    payload = {
        **base,
        "parent_particles": particles,
        "parent_questions": questions,
        "parent_support_sha256": parent_hash,
        "parent_source": "model_generated_semantic_particles",
    }
    audit = {
        "passed": base_audit["passed"] is True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(canonical_json(payload).encode()).hexdigest(),
        "parent_support_sha256": parent_hash,
        "parent_source": payload["parent_source"],
    }
    return [
        {"role": "system", "content": TRANSITION_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_enriched_transition(
    raw: str, parent_support: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("transition response has wrong top-level fields")
    raw_hypotheses = value["hypotheses"]
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != PARTICLES:
        raise ValueError("transition must contain eight particles")
    parents = _public_parent_particles(parent_support)
    children = []
    indexes = []
    seen = set()
    retained = 0
    for position, item in enumerate(raw_hypotheses):
        if not isinstance(item, dict) or set(item) != {
            "parent_index",
            "revision_type",
            "interpretation",
            "final_answer",
            "prior_weight",
            "predicted_replies",
        }:
            raise ValueError(f"transition particle {position} has wrong fields")
        index = item["parent_index"]
        revision_type = item["revision_type"]
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(PARTICLES)
            or revision_type not in {"retained", "revised"}
            or not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
        ):
            raise ValueError(f"transition particle {position} has invalid values")
        unchanged = (
            primary.normalize_text(interpretation)
            == primary.normalize_text(parents[index]["interpretation"])
            and primary.normalize_text(answer)
            == primary.normalize_text(parents[index]["final_answer"])
        )
        if (revision_type == "retained") != unchanged:
            raise ValueError("transition revision label disagrees with lineage")
        retained += revision_type == "retained"
        key = (
            primary.normalize_text(interpretation),
            primary.normalize_text(answer),
        )
        if key in seen:
            raise ValueError("transition contains duplicate children")
        seen.add(key)
        indexes.append(index)
        children.append(
            {
                "parent_index": index,
                "revision_type": revision_type,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "probability": float(weight),
                "predicted_replies": _validate_replies(item["predicted_replies"]),
            }
        )
    if sorted(indexes) != list(range(PARTICLES)):
        raise ValueError("transition parent indexes are not the exact permutation")
    if not MIN_RETAINED <= retained <= MAX_RETAINED:
        raise ValueError("transition must retain between two and six parents")
    total = sum(row["probability"] for row in children)
    if total <= 0:
        raise ValueError("transition weights sum to zero")
    for row in children:
        row["probability"] /= total
    support = {
        "hypotheses": children,
        "questions": _validate_questions(value["questions"]),
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": PARTICLES,
            "parent_index_permutation_exact": True,
            "retained_count": retained,
            "revised_count": PARTICLES - retained,
            "question_count": QUESTIONS,
            "parent_support_sha256": support_sha256(parent_support),
        },
    }
    support["support_sha256"] = support_sha256(support)
    return support


def posterior_parent_after_reply(
    support: Mapping[str, Any], question_index: int, reply: str
) -> tuple[dict[str, Any], bool]:
    if question_index not in range(QUESTIONS):
        raise ValueError("question index is outside the support")
    normalized = primary.normalize_text(reply)
    if not normalized:
        raise ValueError("realized reply is empty")
    rows = _public_parent_particles(support)
    matched = [
        primary.normalize_text(row["predicted_replies"][question_index])
        == normalized
        for row in rows
    ]
    mass = sum(row["probability"] for row, keep in zip(rows, matched) if keep)
    hypotheses = []
    for row, keep in zip(rows, matched):
        probability = row["probability"] / mass if mass > 0 and keep else 0.0
        if mass <= 0:
            probability = row["probability"]
        hypotheses.append({**row, "probability": probability})
    updated = {
        "hypotheses": hypotheses,
        "questions": list(support["questions"]),
        "diagnostic": {
            **dict(support.get("diagnostic") or {}),
            "realized_reply_represented": mass > 0,
            "posterior_update_applied": mass > 0,
            "unmodelled_reply_preserves_parent_weights": mass <= 0,
        },
    }
    updated["support_sha256"] = support_sha256(updated)
    return updated, mass > 0
