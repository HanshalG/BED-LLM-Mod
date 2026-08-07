#!/usr/bin/env python3
"""Core instrument for the sealed RegretBench SMC support contingency.

This module makes no model calls. It defines the prospectively frozen child
particle contract, validates its primary-null predecessor, and computes the
registered support-recovery endpoints. A dated executor may import this core
only after the authorization boundary is satisfied.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from scripts import regretbench_deepseek_support_recovery as primary


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-smc-support-recovery-core-1"
MODEL_ID = primary.MODEL_ID
TEMPERATURE = 0.7
MAX_TOKENS = 2_200
PARENT_PARTICLES = 8
CHILD_PARTICLES = 8
QUESTIONS = 4
MIN_RETAINED = 2
MAX_RETAINED = 6
BRANCH_SEED_START = 202608270000
BOOTSTRAP_SEED = 202608280000
BOOTSTRAP_SAMPLES = 20_000
RUN_BUDGET_USD = 0.50
PROJECTED_COST_USD = 0.30
PROTOCOL = primary.REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_SMC_SUPPORT_RECOVERY_CONTINGENCY_20260807.md"
)
PROTOCOL_SHA256 = (
    "b8eafe438c21e4793a59bd24a0991d20ecf4f53efa5a19fe21dc4eed7c22f807"
)


SYSTEM_PROMPT = """You update a finite semantic particle population after a clarification.
The eight supplied parent particles are model-generated beliefs, not ground truth. Return exactly one child for every parent_index 0 through 7, with no repeated index. Keep useful parents exactly and revise incompatible or weak parents using only the supplied prompt and dialogue. Mark an unchanged child as retained and a changed child as revised. Retain between two and six children inclusive. Return eight unique children and four ranked, distinct, single-dimension clarification questions. Do not ask for the entity name, final factual answer, or an omnibus list. Return only the required JSON object."""


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_protocol_binding() -> None:
    if not PROTOCOL.is_file() or sha256_file(PROTOCOL) != PROTOCOL_SHA256:
        raise ValueError("SMC support contingency protocol changed")


def validate_primary_null_predecessor(
    result_path: Path, verification_path: Path
) -> dict[str, Any]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    verification = json.loads(verification_path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    mechanics = result.get("mechanics_gates") or {}
    if (
        result.get("interface_version") != primary.INTERFACE_VERSION
        or result.get("status") != "gated_null"
        or result.get("authorizes") != "nothing"
        or protocol.get("stage") != "development"
        or protocol.get("model") != MODEL_ID
        or protocol.get("support_recovery_endpoint_accessed") is not True
        or protocol.get("policy_endpoint_opened") is not False
        or mechanics.get("all_pass") is not True
        or verification.get("status") != "verified"
        or verification.get("result_status") != "gated_null"
        or verification.get("mismatches") != []
        or verification.get("model_calls") != 0
        or float(verification.get("cost_usd", math.inf)) != 0.0
        or verification.get("artifact_sha256", {}).get("RESULT.json")
        != sha256_file(result_path)
    ):
        raise ValueError("primary support result does not authorize SMC contingency")
    return {
        "result_path": str(result_path),
        "result_sha256": sha256_file(result_path),
        "verification_path": str(verification_path),
        "verification_sha256": sha256_file(verification_path),
        "status": "authorized_primary_scientific_null",
    }


def child_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_smc_child_support",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "questions"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": CHILD_PARTICLES,
                        "maxItems": CHILD_PARTICLES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "parent_index",
                                "revision_type",
                                "interpretation",
                                "final_answer",
                                "prior_weight",
                            ],
                            "properties": {
                                "parent_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": PARENT_PARTICLES - 1,
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


def parse_parent_population(raw_root: str) -> dict[str, Any]:
    value = json.loads(raw_root)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("parent root response has wrong top-level fields")
    raw = value["hypotheses"]
    if not isinstance(raw, list) or len(raw) != PARENT_PARTICLES:
        raise ValueError("parent root response must contain eight particle slots")
    particles = []
    for index, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {
            "interpretation",
            "final_answer",
            "prior_weight",
        }:
            raise ValueError(f"parent particle {index} has wrong fields")
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError(f"parent particle {index} has invalid values")
        particles.append(
            {
                "parent_index": index,
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "prior_weight": float(weight),
            }
        )
    total = sum(item["prior_weight"] for item in particles)
    if total <= 0.0:
        raise ValueError("parent particle weights sum to zero")
    public = [
        {
            "parent_index": item["parent_index"],
            "interpretation": item["interpretation"],
            "final_answer": item["final_answer"],
            "probability": item["prior_weight"] / total,
        }
        for item in particles
    ]
    return {
        "particles": public,
        "raw_parent_sha256": hashlib.sha256(raw_root.encode()).hexdigest(),
    }


def public_payload(
    cig: Any,
    dialogue: Sequence[Mapping[str, str]],
    parent_population: Mapping[str, Any],
) -> dict[str, Any]:
    base = primary.public_payload(cig, dialogue)
    particles = parent_population.get("particles")
    if not isinstance(particles, list) or len(particles) != PARENT_PARTICLES:
        raise ValueError("parent population is incomplete")
    return {
        **base,
        "parent_particles": particles,
        "parent_population_sha256": parent_population["raw_parent_sha256"],
        "parent_source": "verified_primary_raw_root",
    }


def messages_for(
    cig: Any,
    dialogue: Sequence[Mapping[str, str]],
    parent_population: Mapping[str, Any],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    base = primary.public_payload(cig, dialogue)
    base_audit = primary.privacy_audit(cig, base)
    payload = public_payload(cig, dialogue, parent_population)
    expected_hash = parent_population["raw_parent_sha256"]
    if payload["parent_population_sha256"] != expected_hash:
        raise ValueError("parent population provenance changed")
    audit = {
        "passed": base_audit["passed"] is True,
        "payload_keys": sorted(payload),
        "payload_sha256": hashlib.sha256(
            primary.canonical_json(payload).encode()
        ).hexdigest(),
        "parent_population_sha256": expected_hash,
        "parent_source": payload["parent_source"],
    }
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": primary.canonical_json(payload)},
    ], audit


def _same_particle(child: Mapping[str, Any], parent: Mapping[str, Any]) -> bool:
    return (
        primary.normalize_text(child["interpretation"])
        == primary.normalize_text(parent["interpretation"])
        and primary.normalize_text(child["final_answer"])
        == primary.normalize_text(parent["final_answer"])
    )


def parse_child_support(
    raw_child: str, parent_population: Mapping[str, Any]
) -> dict[str, Any]:
    value = json.loads(raw_child)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("child response has wrong top-level fields")
    raw = value["hypotheses"]
    if not isinstance(raw, list) or len(raw) != CHILD_PARTICLES:
        raise ValueError("child response must contain eight particles")
    parents = parent_population["particles"]
    children = []
    indexes = []
    seen = set()
    retained = 0
    for position, item in enumerate(raw):
        if not isinstance(item, dict) or set(item) != {
            "parent_index",
            "revision_type",
            "interpretation",
            "final_answer",
            "prior_weight",
        }:
            raise ValueError(f"child particle {position} has wrong fields")
        parent_index = item["parent_index"]
        revision_type = item["revision_type"]
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        if (
            isinstance(parent_index, bool)
            or not isinstance(parent_index, int)
            or parent_index not in range(PARENT_PARTICLES)
            or revision_type not in {"retained", "revised"}
            or not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0.0
        ):
            raise ValueError(f"child particle {position} has invalid values")
        child = {
            "parent_index": parent_index,
            "revision_type": revision_type,
            "interpretation": interpretation.strip(),
            "final_answer": answer.strip(),
            "probability": float(weight),
        }
        unchanged = _same_particle(child, parents[parent_index])
        if revision_type == "retained" and not unchanged:
            raise ValueError("retained child differs from its parent")
        if revision_type == "revised" and unchanged:
            raise ValueError("revised child is unchanged from its parent")
        retained += revision_type == "retained"
        key = (
            primary.normalize_text(child["interpretation"]),
            primary.normalize_text(child["final_answer"]),
        )
        if key in seen:
            raise ValueError("child response contains duplicate hypotheses")
        seen.add(key)
        indexes.append(parent_index)
        children.append(child)
    if sorted(indexes) != list(range(PARENT_PARTICLES)):
        raise ValueError("child parent indexes are not the exact permutation")
    if not MIN_RETAINED <= retained <= MAX_RETAINED:
        raise ValueError("child response must retain between two and six parents")
    total = sum(item["probability"] for item in children)
    if total <= 0.0:
        raise ValueError("child particle weights sum to zero")
    for item in children:
        item["probability"] /= total

    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        raise ValueError("child response must contain four questions")
    cleaned = []
    question_keys = set()
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError(f"child question {index} is not a question")
        text = question.strip()
        key = primary.normalize_text(text)
        if not key or key in question_keys:
            raise ValueError("child response contains duplicate questions")
        question_keys.add(key)
        cleaned.append(text)
    return {
        "hypotheses": children,
        "questions": cleaned,
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": CHILD_PARTICLES,
            "parent_index_permutation_exact": True,
            "retained_count": retained,
            "revised_count": CHILD_PARTICLES - retained,
            "question_count": QUESTIONS,
            "parent_population_sha256": parent_population[
                "raw_parent_sha256"
            ],
        },
    }


def truth_covered(child_support: Mapping[str, Any], aliases: str) -> bool:
    return primary.truth_covered(child_support, aliases)


def branch_seed(task_index: int) -> int:
    if task_index not in range(64):
        raise ValueError("SMC task index is outside the frozen cohort")
    return BRANCH_SEED_START + task_index


def mechanics_gates(
    *,
    rows: Sequence[Mapping[str, Any]],
    privacy: Sequence[Mapping[str, Any]],
    usage: Mapping[str, Any],
) -> dict[str, bool]:
    supports = [
        support
        for row in rows
        for support in (row["conditioned_support"], row["blind_support"])
    ]
    exact_lineage = all(
        support["diagnostic"]["codec_mode"] == "strict_json"
        and support["diagnostic"]["valid_unique_count"] == CHILD_PARTICLES
        and support["diagnostic"]["parent_index_permutation_exact"] is True
        and MIN_RETAINED
        <= support["diagnostic"]["retained_count"]
        <= MAX_RETAINED
        and support["diagnostic"]["revised_count"]
        == CHILD_PARTICLES - support["diagnostic"]["retained_count"]
        and support["diagnostic"]["question_count"] == QUESTIONS
        for support in supports
    )
    paired_provenance = all(
        row["parent_population_sha256"]
        == row["conditioned_support"]["diagnostic"][
            "parent_population_sha256"
        ]
        == row["blind_support"]["diagnostic"]["parent_population_sha256"]
        for row in rows
    )
    exact_schedule = len(rows) == 64 and all(
        row["task_index"] == index
        and row["refresh_seed"] == branch_seed(index)
        and row["conditioned_dispatch_index"] == 2 * index
        and row["blind_dispatch_index"] == 2 * index + 1
        for index, row in enumerate(rows)
    )
    gates = {
        "exact_64_tasks": len(rows) == 64,
        "unique_task_ids": len({row["task_id"] for row in rows}) == len(rows),
        "exact_128_child_responses": len(supports) == 128,
        "exact_128_accepted_requests": usage["adapter_requests"] == 128,
        "exact_128_http_attempts": usage["http_attempts"] == 128,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_child_lineages_strict_and_exact": exact_lineage,
        "conditioned_blind_parent_provenance_exact": paired_provenance,
        "conditioned_blind_seed_formula_and_adjacency_exact": exact_schedule,
        "at_least_48_supported_roots": sum(row["supported"] for row in rows)
        >= 48,
        "all_128_privacy_and_provenance_audits_pass": len(privacy) == 128
        and all(item["passed"] for item in privacy),
        "within_run_budget": float(usage["run_cost_usd"])
        <= RUN_BUDGET_USD + 1e-12,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _bootstrap(values: Sequence[float], samples: int) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return {
            "mean": None,
            "ci95": [None, None],
            "probability_positive": None,
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indexes = rng.integers(0, array.size, size=(samples, array.size))
    means = array[indexes].mean(axis=1)
    return {
        "mean": float(np.mean(array)),
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_positive": float(np.mean(means > 0.0)),
        "samples": samples,
        "seed": BOOTSTRAP_SEED,
    }


def scientific_summary(
    rows: Sequence[Mapping[str, Any]], samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    supported = [row for row in rows if row["supported"]]
    root_missing = [row for row in supported if not row["root_covered"]]
    root_covered = [row for row in supported if row["root_covered"]]
    differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in supported
    ]
    missing_differences = [
        float(row["conditioned_covered"]) - float(row["blind_covered"])
        for row in root_missing
    ]
    recoveries = sum(value > 0 for value in differences)
    losses = sum(value < 0 for value in differences)
    conditioned_root_losses = sum(
        not row["conditioned_covered"] for row in root_covered
    )
    blind_root_losses = sum(not row["blind_covered"] for row in root_covered)
    overall = _bootstrap(differences, samples)
    missing = _bootstrap(missing_differences, samples)
    root_retention = (
        float(np.mean([row["conditioned_covered"] for row in root_covered]))
        if root_covered
        else None
    )
    gates = {
        "at_least_48_supported_tasks": len(supported) >= 48,
        "at_least_16_root_missing_supported_tasks": len(root_missing) >= 16,
        "conditioned_minus_blind_coverage_at_least_005": (
            overall["mean"] is not None and overall["mean"] >= 0.05
        ),
        "bootstrap_probability_positive_at_least_080": (
            overall["probability_positive"] is not None
            and overall["probability_positive"] >= 0.80
        ),
        "conditioned_recoveries_exceed_losses": recoveries > losses,
        "root_missing_recovery_difference_at_least_010": (
            missing["mean"] is not None and missing["mean"] >= 0.10
        ),
        "conditioned_root_covered_retention_at_least_090": (
            root_retention is not None and root_retention >= 0.90
        ),
        "conditioned_root_covered_losses_no_more_than_blind": (
            conditioned_root_losses <= blind_root_losses
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "population": {
            "all_tasks": len(rows),
            "supported_tasks": len(supported),
            "root_missing_supported_tasks": len(root_missing),
            "root_covered_supported_tasks": len(root_covered),
        },
        "coverage": {
            "conditioned_minus_history_blind": overall,
            "root_missing_conditioned_minus_history_blind": missing,
            "conditioned_root_covered_retention": root_retention,
        },
        "paired_outcomes": {
            "conditioned_recoveries": recoveries,
            "conditioned_losses": losses,
            "ties": len(differences) - recoveries - losses,
            "conditioned_root_covered_losses": conditioned_root_losses,
            "blind_root_covered_losses": blind_root_losses,
        },
        "gates": gates,
    }
