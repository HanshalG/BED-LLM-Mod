#!/usr/bin/env python3
"""History-free likelihood annotation for RegretBench child particles."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping, Sequence

from scripts import regretbench_deepseek_smc_dynamic_depth2_policy as core
from scripts import regretbench_deepseek_support_recovery as primary


INTERFACE_VERSION = "regretbench-static-child-likelihood-core-1"
PARTICLES = core.PARTICLES
QUESTIONS = core.QUESTIONS

SYSTEM_PROMPT = """Predict how a user with each supplied semantic interpretation and final answer would answer each supplied clarification question. Treat each particle independently. The particles and questions are fixed model-generated objects. Return exactly one record for every particle_index 0 through 7 and exactly four nonempty replies aligned to the four supplied questions. Do not revise, merge, remove, reweight, or add particles or questions. No interaction history, particle weight, lineage label, or previous reply prediction is provided; do not invent or condition on one. Use only the supplied original prompt, particle meaning, final answer, and question. Return only the required JSON object."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=True, separators=(",", ":"), sort_keys=True
    )


def structural_child_sha256(
    particles: Sequence[Mapping[str, Any]], questions: Sequence[str]
) -> str:
    value = {"particles": list(particles), "questions": list(questions)}
    return hashlib.sha256(_canonical(value).encode()).hexdigest()


def response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_static_child_likelihood",
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
                            "required": ["particle_index", "predicted_replies"],
                            "properties": {
                                "particle_index": {
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


def _public_child_particles(
    child_support: Mapping[str, Any],
) -> list[dict[str, Any]]:
    rows = child_support.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != PARTICLES:
        raise ValueError("child support must contain exactly eight hypotheses")
    output = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping):
            raise ValueError("child hypothesis is not an object")
        interpretation = row.get("interpretation")
        final_answer = row.get("final_answer")
        if not isinstance(interpretation, str) or not interpretation.strip():
            raise ValueError("child interpretation is empty")
        if not isinstance(final_answer, str) or not final_answer.strip():
            raise ValueError("child final answer is empty")
        output.append(
            {
                "particle_index": index,
                "interpretation": interpretation.strip(),
                "final_answer": final_answer.strip(),
            }
        )
    return output


def messages_for(
    cig: Any, child_support: Mapping[str, Any]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    questions = core._validate_questions(list(child_support.get("questions") or []))
    particles = _public_child_particles(child_support)
    base = primary.public_payload(cig, [])
    base_audit = primary.privacy_audit(cig, base)
    structural_hash = structural_child_sha256(particles, questions)
    payload = {
        **base,
        "child_particles": particles,
        "questions": questions,
        "structural_child_sha256": structural_hash,
        "likelihood_factorization": "static_particle_question_only",
        "particle_source": "history_conditioned_children_with_history_removed",
    }
    serialized = _canonical(payload)
    replies_excluded = all(
        set(row) == {"particle_index", "interpretation", "final_answer"}
        for row in particles
    )
    audit = {
        "passed": base_audit["passed"] is True and replies_excluded,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        "structural_child_sha256": structural_hash,
        "dialogue_excluded": True,
        "particle_probabilities_excluded": True,
        "lineage_metadata_excluded": True,
        "updated_predicted_replies_excluded": replies_excluded,
        "hidden_truth_exposed": False,
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": serialized},
    ], audit


def parse_annotation(
    raw: str, child_support: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("static annotation has wrong top-level fields")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != PARTICLES:
        raise ValueError("static annotation must contain eight particles")
    by_index: dict[int, list[str]] = {}
    for row in rows:
        if not isinstance(row, dict) or set(row) != {
            "particle_index",
            "predicted_replies",
        }:
            raise ValueError("static annotation particle has wrong fields")
        index = row["particle_index"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(PARTICLES)
            or index in by_index
        ):
            raise ValueError("static annotation indexes are not an exact permutation")
        by_index[index] = core._validate_replies(row["predicted_replies"])
    if sorted(by_index) != list(range(PARTICLES)):
        raise ValueError("static annotation indexes are not an exact permutation")

    source_particles = _public_child_particles(child_support)
    questions = core._validate_questions(list(child_support.get("questions") or []))
    source_hash = core.support_sha256(child_support)
    hypotheses = [
        {**row, "predicted_replies": by_index[index]}
        for index, row in enumerate(child_support["hypotheses"])
    ]
    diagnostic = dict(child_support.get("diagnostic") or {})
    diagnostic.update(
        {
            "likelihood_factorization": "static_particle_question_only",
            "history_conditioned_child_particles_preserved": True,
            "updated_likelihood_replies_discarded": True,
            "static_annotation_index_permutation_exact": True,
            "source_child_support_sha256": source_hash,
            "structural_child_sha256": structural_child_sha256(
                source_particles, questions
            ),
        }
    )
    output = {
        "hypotheses": hypotheses,
        "questions": questions,
        "diagnostic": diagnostic,
    }
    output["support_sha256"] = core.support_sha256(output)
    return output
