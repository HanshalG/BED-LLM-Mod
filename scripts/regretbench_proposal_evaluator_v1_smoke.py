#!/usr/bin/env python3
"""Run the RegretBench proposal-evaluator v1 exact-20 mechanics smoke."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Protocol, Sequence


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_ROOT = REPO_ROOT / "external/RegretBench"
REGRETBENCH_SRC = REGRETBENCH_ROOT / "src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import Config, ModelSpec
from regretbench.mapping.semantic_action_mapper import SemanticActionMapper
from regretbench.schemas.cig import CIG, load_cig
from scripts import regretbench_deepseek_support_recovery as source_tools
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-proposal-evaluator-v1-exact20-smoke-1"
MODEL_ID = "deepseek/deepseek-v4-flash-0731"
TEMPERATURE = 0.7
PROPOSAL_MAX_TOKENS = 2_000
EVALUATOR_MAX_TOKENS = 1_800
EXPECTED_REQUESTS = 20
MAX_RETRIES = 4
CONCURRENCY = 20
RUN_BUDGET_USD = 0.20
PROJECTED_COST_USD = 0.04
HYPOTHESES = 8
QUESTIONS = 4
BRANCHES = 2
ROOT_PROPOSAL_SEED_START = 202608530000
ROOT_EVALUATOR_SEED_START = 202608531000
BRANCH_PROPOSAL_SEED_START = 202608532000
BRANCH_EVALUATOR_SEED_START = 202608533000
MIN_BRANCH_MASS = 0.10
MIN_OWN_MINUS_OPPOSITE_MEAN = 0.10
MIN_OWN_MINUS_BLIND_MEAN = 0.05
MIN_POSITIVE_SIGNAL_COUNT = 3

SOURCE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_PROPOSAL_EVALUATOR_V1_SOURCE_PROTOCOL_20260811.md"
)
SOURCE_PROTOCOL_SHA256 = (
    "9edb0a7828dfa74b660bea02fc80c50db05be4b6df571efe0911693ee41c190e"
)
SMOKE_PROTOCOL = REPO_ROOT / (
    "results/nonmyopic/"
    "REGRETBENCH_PROPOSAL_EVALUATOR_V1_EXACT20_SMOKE_PROTOCOL_20260811.md"
)
SMOKE_PROTOCOL_SHA256 = (
    "c56939f2f45c1da2a5dd2dc4e6ab8fd7921cf3033febe085a4d3a0933cd5583b"
)
SOURCE_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/regretbench_proposal_evaluator_v1_source_audit/"
    "SOURCE_TASK_MANIFEST.json"
)
SOURCE_MANIFEST_SHA256 = (
    "d3c92a4edecc39b6524790f0e46b813ae8c04ad8507ac67535a95680b9afe7ad"
)
SOURCE_RESULT = REPO_ROOT / (
    "results/nonmyopic/regretbench_proposal_evaluator_v1_source_audit/RESULT.json"
)
SOURCE_RESULT_SHA256 = (
    "e3e627a69578f19671e57d9e0a731c8c1eb30d53984c5f238c6a83f91bc93be9"
)


class StructuredAdapter(Protocol):
    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, str]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=True, separators=(",", ":"), sort_keys=True)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def structural_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def validate_bindings() -> dict[str, str]:
    expected = {
        "source_protocol": (SOURCE_PROTOCOL, SOURCE_PROTOCOL_SHA256),
        "smoke_protocol": (SMOKE_PROTOCOL, SMOKE_PROTOCOL_SHA256),
        "source_manifest": (SOURCE_MANIFEST, SOURCE_MANIFEST_SHA256),
        "source_result": (SOURCE_RESULT, SOURCE_RESULT_SHA256),
    }
    for name, (path, digest) in expected.items():
        if not path.is_file() or sha256_file(path) != digest:
            raise ValueError(f"proposal-evaluator {name} changed")
    source = load_object(SOURCE_RESULT)
    manifest = load_object(SOURCE_MANIFEST)
    if (
        source.get("status") != "source_protocol_pass"
        or source.get("authorizes")
        != "proposal_evaluator_v1_exact20_smoke_only"
        or source.get("gates", {}).get("all_pass") is not True
        or len(manifest.get("successor_smoke_tasks", [])) != 2
        or manifest.get("development_or_confirmation_tasks_loaded") is not False
    ):
        raise ValueError("proposal-evaluator source does not authorize smoke")
    return {name: digest for name, (_, digest) in expected.items()}


def load_smoke_cigs() -> list[CIG]:
    validate_bindings()
    rows = load_object(SOURCE_MANIFEST)["successor_smoke_tasks"]
    ids = [row["task_id"] for row in rows]
    cigs = [
        load_cig(REGRETBENCH_ROOT / f"data/OpenDomainQA/test/{cig_id}.json")
        for cig_id in ids
    ]
    if [cig.cig_id for cig in cigs] != ids:
        raise ValueError("proposal-evaluator smoke task order changed")
    for cig, row in zip(cigs, rows, strict=True):
        path = REGRETBENCH_ROOT / f"data/OpenDomainQA/test/{cig.cig_id}.json"
        if sha256_file(path) != row["task_file_sha256"]:
            raise ValueError("proposal-evaluator smoke task file changed")
    return cigs


def proposal_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_proposal_evaluator_v1_proposal",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["hypotheses", "questions"],
                "properties": {
                    "hypotheses": {
                        "type": "array",
                        "minItems": HYPOTHESES,
                        "maxItems": HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": ["interpretation", "final_answer"],
                            "properties": {
                                "interpretation": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 800,
                                },
                                "final_answer": {
                                    "type": "string",
                                    "minLength": 1,
                                    "maxLength": 240,
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


def evaluator_response_format(question_count: int) -> dict[str, Any]:
    if question_count not in {4, 5}:
        raise ValueError("evaluator question count must be four or five")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": f"regretbench_proposal_evaluator_v1_evaluator_{question_count}",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["particles"],
                "properties": {
                    "particles": {
                        "type": "array",
                        "minItems": HYPOTHESES,
                        "maxItems": HYPOTHESES,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "particle_index",
                                "prior_weight",
                                "predicted_replies",
                            ],
                            "properties": {
                                "particle_index": {
                                    "type": "integer",
                                    "minimum": 0,
                                    "maximum": HYPOTHESES - 1,
                                },
                                "prior_weight": {
                                    "type": "number",
                                    "minimum": 0,
                                    "maximum": 100,
                                },
                                "predicted_replies": {
                                    "type": "array",
                                    "minItems": question_count,
                                    "maxItems": question_count,
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


PROPOSAL_SYSTEM_PROMPT = """Generate a semantic proposal population for an ambiguous factual question. Return exactly eight plausible, meaningfully distinct interpretations with one concise factual final answer each, plus exactly four ranked, distinct, single-facet clarification questions. Questions must be answerable by the user, must not repeat an answered question, and must not ask for the entity name, final factual answer, or an omnibus list. Do not assign probabilities, predict replies, cite parent particles, or describe retention or revision. Use only the public prompt and supplied dialogue. Return only the required JSON object."""

EVALUATOR_SYSTEM_PROMPT = """Evaluate a fixed semantic proposal without changing it. For every particle_index, assign a nonnegative prior plausibility weight using only the public ambiguous prompt and fixed particle text, then predict one concise cooperative-user reply to each fixed question in order. Do not add, remove, merge, revise, quote, or reorder particles or questions. No observed answer or dialogue is available; do not infer that one was supplied. Return only the required JSON object."""


def proposal_messages_for(
    cig: CIG, dialogue: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = source_tools.public_payload(cig, dialogue)
    audit = source_tools.privacy_audit(cig, payload)
    audit.update(
        {
            "interface_role": "conditioned_proposal" if dialogue else "answer_free_proposal",
            "dialogue_count": len(dialogue),
            "contains_probability_or_likelihood_fields": False,
            "contains_lineage_fields": False,
        }
    )
    return [
        {"role": "system", "content": PROPOSAL_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_proposal(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("proposal response has wrong top-level fields")
    hypotheses = []
    seen = set()
    rows = value["hypotheses"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("proposal must contain eight hypotheses")
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {"interpretation", "final_answer"}:
            raise ValueError(f"proposal hypothesis {position} has wrong fields")
        interpretation = row["interpretation"]
        answer = row["final_answer"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError(f"proposal hypothesis {position} has invalid values")
        key = (
            source_tools.normalize_text(interpretation),
            source_tools.normalize_text(answer),
        )
        if key in seen:
            raise ValueError("proposal contains duplicate hypotheses")
        seen.add(key)
        hypotheses.append(
            {"interpretation": interpretation.strip(), "final_answer": answer.strip()}
        )
    questions = []
    seen_questions = set()
    rows = value["questions"]
    if not isinstance(rows, list) or len(rows) != QUESTIONS:
        raise ValueError("proposal must contain four questions")
    for position, question in enumerate(rows):
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError(f"proposal question {position} is not interrogative")
        cleaned = question.strip()
        key = source_tools.normalize_text(cleaned)
        if not key or key in seen_questions:
            raise ValueError("proposal contains duplicate questions")
        seen_questions.add(key)
        questions.append(cleaned)
    public = {"hypotheses": hypotheses, "questions": questions}
    return {
        **public,
        "diagnostic": {
            "codec_mode": "strict_json",
            "valid_unique_count": len(hypotheses),
            "question_count": len(questions),
            "structural_sha256": structural_hash(public),
            "contains_weights_or_likelihoods": False,
            "contains_lineage": False,
        },
    }


def _public_particles(proposal: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = proposal.get("hypotheses")
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("evaluator proposal must contain eight hypotheses")
    return [
        {
            "particle_index": index,
            "interpretation": row["interpretation"],
            "final_answer": row["final_answer"],
        }
        for index, row in enumerate(rows)
    ]


def evaluator_messages_for(
    cig: CIG,
    proposal: Mapping[str, Any],
    questions: Sequence[str],
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    if len(questions) not in {4, 5}:
        raise ValueError("evaluator requires four or five questions")
    particles = _public_particles(proposal)
    fixed_questions = [str(question).strip() for question in questions]
    if any(not question.endswith("?") for question in fixed_questions):
        raise ValueError("evaluator question is not interrogative")
    payload = {
        "task_id": cig.cig_id,
        "prompt": cig.prompt,
        "particles": particles,
        "questions": fixed_questions,
        "proposal_sha256": structural_hash(
            {"hypotheses": proposal["hypotheses"], "questions": proposal["questions"]}
        ),
        "source": "model_generated_semantic_proposal",
    }
    expected_keys = {
        "task_id",
        "prompt",
        "particles",
        "questions",
        "proposal_sha256",
        "source",
    }
    if set(payload) != expected_keys:
        raise ValueError("evaluator payload fields changed")
    audit = {
        "passed": True,
        "interface_role": "history_free_evaluator",
        "payload_keys": sorted(payload),
        "payload_sha256": structural_hash(payload),
        "proposal_sha256": payload["proposal_sha256"],
        "question_count": len(fixed_questions),
        "dialogue_present": False,
        "dedicated_observed_or_simulated_reply_field_present": False,
        "probability_or_likelihood_input_present": False,
        "lineage_input_present": False,
        "old_prediction_input_present": False,
        "truth_or_mapping_input_present": False,
    }
    return [
        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
        {"role": "user", "content": canonical_json(payload)},
    ], audit


def parse_evaluation(
    raw: str,
    proposal: Mapping[str, Any],
    questions: Sequence[str],
) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"particles"}:
        raise ValueError("evaluator response has wrong top-level fields")
    rows = value["particles"]
    if not isinstance(rows, list) or len(rows) != HYPOTHESES:
        raise ValueError("evaluator must contain eight particles")
    by_index = {}
    for position, row in enumerate(rows):
        if not isinstance(row, dict) or set(row) != {
            "particle_index",
            "prior_weight",
            "predicted_replies",
        }:
            raise ValueError(f"evaluator particle {position} has wrong fields")
        index = row["particle_index"]
        weight = row["prior_weight"]
        replies = row["predicted_replies"]
        if (
            isinstance(index, bool)
            or not isinstance(index, int)
            or index not in range(HYPOTHESES)
            or index in by_index
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            or not isinstance(replies, list)
            or len(replies) != len(questions)
            or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
        ):
            raise ValueError(f"evaluator particle {position} has invalid values")
        by_index[index] = {
            "prior_weight": float(weight),
            "predicted_replies": [reply.strip() for reply in replies],
        }
    if set(by_index) != set(range(HYPOTHESES)):
        raise ValueError("evaluator indexes are not an exact permutation")
    total = sum(row["prior_weight"] for row in by_index.values())
    if total <= 0:
        raise ValueError("evaluator prior weights have zero mass")
    hypotheses = []
    for index, original in enumerate(proposal["hypotheses"]):
        evaluated = by_index[index]
        hypotheses.append(
            {
                "interpretation": original["interpretation"],
                "final_answer": original["final_answer"],
                "probability": evaluated["prior_weight"] / total,
                "predicted_replies": evaluated["predicted_replies"],
            }
        )
    return {
        "hypotheses": hypotheses,
        "questions": list(questions),
        "diagnostic": {
            "codec_mode": "strict_json",
            "particle_index_permutation_exact": True,
            "question_count": len(questions),
            "prior_sum_before_normalization": total,
            "normalized_probability_sum": sum(
                row["probability"] for row in hypotheses
            ),
            "proposal_sha256": proposal["diagnostic"]["structural_sha256"],
        },
    }


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def question_eig(support: Mapping[str, Any], question_index: int) -> float:
    masses: dict[str, float] = {}
    for row in support["hypotheses"]:
        key = source_tools.normalize_text(row["predicted_replies"][question_index])
        masses[key] = masses.get(key, 0.0) + float(row["probability"])
    return entropy(list(masses.values()))


def mapped_action(cig: CIG, question: str) -> dict[str, Any]:
    parsed = SemanticActionMapper().map_question(cig, question)
    supported = parsed.facet is not None and parsed.semantic_action != "UNSUPPORTED"
    return {
        "supported": supported,
        "facet": parsed.facet if supported else None,
        "confidence": float(parsed.confidence),
        "method": parsed.method,
    }


def select_root_question(cig: CIG, support: Mapping[str, Any]) -> dict[str, Any]:
    actions = [mapped_action(cig, question) for question in support["questions"]]
    candidates = [index for index, action in enumerate(actions) if action["supported"]]
    if not candidates:
        raise ValueError("root proposal has no mapper-supported question")
    index = min(candidates, key=lambda item: (-question_eig(support, item), item))
    return {
        "index": index,
        "question": support["questions"][index],
        "eig": question_eig(support, index),
        "action": actions[index],
        "supported_question_count": len(candidates),
    }


def reply_groups(support: Mapping[str, Any], question_index: int) -> list[dict[str, Any]]:
    groups: dict[str, dict[str, Any]] = {}
    for row in support["hypotheses"]:
        reply = row["predicted_replies"][question_index]
        key = source_tools.normalize_text(reply)
        if key not in groups:
            groups[key] = {"key": key, "reply": reply, "mass": 0.0}
        groups[key]["mass"] += float(row["probability"])
    return sorted(groups.values(), key=lambda row: (-row["mass"], row["key"]))


def source_reply_is_real(cig: CIG, facet: str, reply: str) -> bool:
    normalized = source_tools.normalize_text(reply)
    values = {
        source_tools.normalize_text(str((intent.slots or {}).get(facet, "")))
        for intent in cig.intents
    }
    values.discard("")
    return normalized in values


def reply_mass(support: Mapping[str, Any], question_index: int, reply: str) -> float:
    target = source_tools.normalize_text(reply)
    return sum(
        float(row["probability"])
        for row in support["hypotheses"]
        if source_tools.normalize_text(row["predicted_replies"][question_index])
        == target
    )


def condition_on_reply(
    support: Mapping[str, Any], question_index: int, reply: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    target = source_tools.normalize_text(reply)
    matches = [
        index
        for index, row in enumerate(support["hypotheses"])
        if source_tools.normalize_text(row["predicted_replies"][question_index])
        == target
    ]
    mass = sum(float(support["hypotheses"][index]["probability"]) for index in matches)
    if not matches or mass <= 0:
        raise ValueError("branch reply has zero evaluator likelihood")
    updated = copy.deepcopy(support)
    for index, row in enumerate(updated["hypotheses"]):
        row["probability"] = (
            float(row["probability"]) / mass if index in matches else 0.0
        )
    posterior_mass = reply_mass(updated, question_index, reply)
    diagnostic = {
        "matching_particle_count": len(matches),
        "prior_predictive_mass": mass,
        "posterior_probability_sum": sum(
            float(row["probability"]) for row in updated["hypotheses"]
        ),
        "posterior_predictive_matched_reply": posterior_mass,
    }
    return updated, diagnostic


def select_child_question(
    cig: CIG, support: Mapping[str, Any], root_facet: str
) -> dict[str, Any]:
    actions = [mapped_action(cig, question) for question in support["questions"]]
    candidates = [
        index
        for index, action in enumerate(actions)
        if index > 0
        and action["supported"]
        and action["facet"] is not None
        and action["facet"] != root_facet
    ]
    if not candidates:
        raise ValueError("child support has no novel mapper-supported question")
    index = min(candidates, key=lambda item: (-question_eig(support, item), item))
    return {
        "index": index,
        "question": support["questions"][index],
        "eig": question_eig(support, index),
        "action": actions[index],
        "supported_novel_question_count": len(candidates),
    }


def build_adapter(*, run_id: str, output_dir: Path) -> PerRequestSeedStructuredAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=245.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=PROJECTED_COST_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=MAX_RETRIES,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=0.002,
        openrouter_max_output_tokens=PROPOSAL_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536),
        config,
    )


def call_batch(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
    response_format: dict[str, Any],
    max_tokens: int,
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=TEMPERATURE,
        response_format=response_format,
        max_new_tokens=max_tokens,
    )
    if len(responses) != len(messages):
        raise ValueError("adapter returned the wrong response count")
    return list(responses)


def usage_gates(adapter: StructuredAdapter) -> tuple[dict[str, Any], dict[str, bool]]:
    usage = summarize_usage(adapter.usage_snapshot())
    accepted = usage["adapter_requests"]
    attempts = usage["http_attempts"]
    retries = usage["retry_count"]
    gates = {
        "exact_twenty_accepted_requests": accepted == EXPECTED_REQUESTS,
        "attempts_between_twenty_and_twenty_four": EXPECTED_REQUESTS
        <= attempts
        <= EXPECTED_REQUESTS + MAX_RETRIES,
        "attempts_equal_accepted_plus_retries": attempts == accepted + retries,
        "at_most_four_retries": retries <= MAX_RETRIES,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "within_smoke_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD + 1e-12,
    }
    return usage, gates


def run_smoke(
    *,
    output_dir: Path,
    adapter: StructuredAdapter,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    bindings = validate_bindings()
    cigs = load_smoke_cigs()
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    raw_bank: dict[str, Any] = {}
    privacy: list[dict[str, Any]] = []

    root_messages = []
    for cig in cigs:
        messages, audit = proposal_messages_for(cig, [])
        root_messages.append(messages)
        privacy.append(audit)
    root_proposal_seeds = [ROOT_PROPOSAL_SEED_START + task for task in range(2)]
    raw_root_proposals = call_batch(
        adapter,
        root_messages,
        root_proposal_seeds,
        proposal_response_format(),
        PROPOSAL_MAX_TOKENS,
    )
    raw_bank.update(
        {"root_proposal_seeds": root_proposal_seeds, "root_proposals": raw_root_proposals}
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    root_proposals = [parse_proposal(raw) for raw in raw_root_proposals]

    root_evaluator_messages = []
    for cig, proposal in zip(cigs, root_proposals, strict=True):
        messages, audit = evaluator_messages_for(cig, proposal, proposal["questions"])
        root_evaluator_messages.append(messages)
        privacy.append(audit)
    root_evaluator_seeds = [ROOT_EVALUATOR_SEED_START + task for task in range(2)]
    raw_root_evaluations = call_batch(
        adapter,
        root_evaluator_messages,
        root_evaluator_seeds,
        evaluator_response_format(4),
        EVALUATOR_MAX_TOKENS,
    )
    raw_bank.update(
        {
            "root_evaluator_seeds": root_evaluator_seeds,
            "root_evaluations": raw_root_evaluations,
        }
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    root_supports = [
        parse_evaluation(raw, proposal, proposal["questions"])
        for raw, proposal in zip(raw_root_evaluations, root_proposals, strict=True)
    ]
    root_selections = [
        select_root_question(cig, support)
        for cig, support in zip(cigs, root_supports, strict=True)
    ]
    branch_groups = []
    for cig, support, selected in zip(cigs, root_supports, root_selections, strict=True):
        groups = reply_groups(support, selected["index"])
        if len(groups) < BRANCHES:
            raise ValueError("selected root question has fewer than two reply groups")
        chosen = groups[:BRANCHES]
        if any(group["mass"] < MIN_BRANCH_MASS for group in chosen):
            raise ValueError("selected root branch mass is below minimum")
        if not all(
            source_reply_is_real(cig, selected["action"]["facet"], group["reply"])
            for group in chosen
        ):
            raise ValueError("selected root branch reply is not a real facet value")
        branch_groups.append(chosen)

    branch_messages = []
    branch_proposal_seeds = []
    branch_layout = []
    for task, (cig, selected, groups) in enumerate(
        zip(cigs, root_selections, branch_groups, strict=True)
    ):
        for branch, group in enumerate(groups):
            seed = BRANCH_PROPOSAL_SEED_START + 10 * task + branch
            dialogue = [
                {"role": "assistant", "content": selected["question"]},
                {"role": "user", "content": group["reply"]},
            ]
            conditioned_messages, conditioned_audit = proposal_messages_for(cig, dialogue)
            blind_messages, blind_audit = proposal_messages_for(cig, [])
            branch_messages.extend([conditioned_messages, blind_messages])
            branch_proposal_seeds.extend([seed, seed])
            privacy.extend([conditioned_audit, blind_audit])
            branch_layout.extend(
                [
                    {"task": task, "branch": branch, "arm": "conditioned"},
                    {"task": task, "branch": branch, "arm": "answer_free"},
                ]
            )
    raw_branch_proposals = call_batch(
        adapter,
        branch_messages,
        branch_proposal_seeds,
        proposal_response_format(),
        PROPOSAL_MAX_TOKENS,
    )
    raw_bank.update(
        {
            "branch_proposal_layout": branch_layout,
            "branch_proposal_seeds": branch_proposal_seeds,
            "branch_proposals": raw_branch_proposals,
        }
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    branch_proposals = [parse_proposal(raw) for raw in raw_branch_proposals]

    branch_evaluator_messages = []
    branch_evaluator_seeds = []
    evaluator_layout = []
    for position, (layout, proposal) in enumerate(
        zip(branch_layout, branch_proposals, strict=True)
    ):
        task = layout["task"]
        branch = layout["branch"]
        questions = [root_selections[task]["question"], *proposal["questions"]]
        messages, audit = evaluator_messages_for(cigs[task], proposal, questions)
        branch_evaluator_messages.append(messages)
        branch_evaluator_seeds.append(
            BRANCH_EVALUATOR_SEED_START + 10 * task + branch
        )
        evaluator_layout.append(dict(layout))
        privacy.append(audit)
    raw_branch_evaluations = call_batch(
        adapter,
        branch_evaluator_messages,
        branch_evaluator_seeds,
        evaluator_response_format(5),
        EVALUATOR_MAX_TOKENS,
    )
    raw_bank.update(
        {
            "branch_evaluator_layout": evaluator_layout,
            "branch_evaluator_seeds": branch_evaluator_seeds,
            "branch_evaluations": raw_branch_evaluations,
        }
    )
    checkpoint(private / "RAW_RESPONSES.json", raw_bank)
    checkpoint(private / "PRIVACY.json", {"audits": privacy})
    branch_supports = []
    for raw, proposal, layout in zip(
        raw_branch_evaluations, branch_proposals, evaluator_layout, strict=True
    ):
        task = layout["task"]
        questions = [root_selections[task]["question"], *proposal["questions"]]
        branch_supports.append(parse_evaluation(raw, proposal, questions))

    task_diagnostics = []
    own_minus_opposite = []
    own_minus_blind = []
    changed_actions = []
    all_own_masses = []
    posterior_checks = []
    structural_changes = []
    for task, cig in enumerate(cigs):
        branch_rows = []
        root_facet = root_selections[task]["action"]["facet"]
        conditioned_positions = [4 * task, 4 * task + 2]
        blind_positions = [4 * task + 1, 4 * task + 3]
        for branch in range(BRANCHES):
            reply = branch_groups[task][branch]["reply"]
            own = branch_supports[conditioned_positions[branch]]
            opposite = branch_supports[conditioned_positions[1 - branch]]
            blind = branch_supports[blind_positions[branch]]
            own_mass = reply_mass(own, 0, reply)
            opposite_mass = reply_mass(opposite, 0, reply)
            blind_mass = reply_mass(blind, 0, reply)
            conditioned_posterior, conditioned_update = condition_on_reply(own, 0, reply)
            blind_posterior, blind_update = condition_on_reply(blind, 0, reply)
            conditioned_second = select_child_question(
                cig, conditioned_posterior, root_facet
            )
            blind_second = select_child_question(cig, blind_posterior, root_facet)
            changed = (
                conditioned_second["action"]["facet"]
                != blind_second["action"]["facet"]
            )
            structural = (
                branch_proposals[conditioned_positions[branch]]["diagnostic"][
                    "structural_sha256"
                ]
                != branch_proposals[blind_positions[branch]]["diagnostic"][
                    "structural_sha256"
                ]
                and branch_proposals[conditioned_positions[branch]]["diagnostic"][
                    "structural_sha256"
                ]
                != branch_proposals[conditioned_positions[1 - branch]]["diagnostic"][
                    "structural_sha256"
                ]
            )
            own_delta = own_mass - opposite_mass
            blind_delta = own_mass - blind_mass
            own_minus_opposite.append(own_delta)
            own_minus_blind.append(blind_delta)
            all_own_masses.append(own_mass)
            changed_actions.append(changed)
            structural_changes.append(structural)
            posterior_checks.extend([conditioned_update, blind_update])
            branch_rows.append(
                {
                    "branch_index": branch,
                    "root_branch_mass": branch_groups[task][branch]["mass"],
                    "conditioned_own_mass": own_mass,
                    "opposite_conditioned_mass": opposite_mass,
                    "paired_answer_free_mass": blind_mass,
                    "own_minus_opposite": own_delta,
                    "own_minus_answer_free": blind_delta,
                    "conditioned_matching_particles": conditioned_update[
                        "matching_particle_count"
                    ],
                    "answer_free_matching_particles": blind_update[
                        "matching_particle_count"
                    ],
                    "conditioned_second_eig": conditioned_second["eig"],
                    "answer_free_second_eig": blind_second["eig"],
                    "selected_child_facet_changed": changed,
                    "conditioned_support_changed_from_controls": structural,
                    "conditioned_proposal_sha256": branch_proposals[
                        conditioned_positions[branch]
                    ]["diagnostic"]["structural_sha256"],
                    "answer_free_proposal_sha256": branch_proposals[
                        blind_positions[branch]
                    ]["diagnostic"]["structural_sha256"],
                }
            )
        task_diagnostics.append(
            {
                "task_index": task,
                "root_eig": root_selections[task]["eig"],
                "root_supported_question_count": root_selections[task][
                    "supported_question_count"
                ],
                "root_proposal_sha256": root_proposals[task]["diagnostic"][
                    "structural_sha256"
                ],
                "branches": branch_rows,
            }
        )

    usage, transport_gates = usage_gates(adapter)
    own_opposite_mean = sum(own_minus_opposite) / len(own_minus_opposite)
    own_blind_mean = sum(own_minus_blind) / len(own_minus_blind)
    mechanics_gates = {
        "exact_two_untouched_tasks": len(cigs) == 2,
        "exact_two_root_proposals_and_evaluations": len(root_proposals) == 2
        and len(root_supports) == 2,
        "exact_eight_branch_proposals_and_evaluations": len(branch_proposals) == 8
        and len(branch_supports) == 8,
        "all_proposals_exclude_weights_likelihoods_and_lineage": all(
            not proposal["diagnostic"]["contains_weights_or_likelihoods"]
            and not proposal["diagnostic"]["contains_lineage"]
            for proposal in [*root_proposals, *branch_proposals]
        ),
        "all_evaluators_are_exact_index_permutations": all(
            support["diagnostic"]["particle_index_permutation_exact"] is True
            for support in [*root_supports, *branch_supports]
        ),
        "all_prompt_privacy_audits_pass": len(privacy) == EXPECTED_REQUESTS
        and all(audit.get("passed") is True for audit in privacy),
        "proposal_pair_seeds_are_adjacent_and_equal": all(
            branch_proposal_seeds[index] == branch_proposal_seeds[index + 1]
            for index in range(0, 8, 2)
        ),
        "evaluator_pair_seeds_are_adjacent_and_equal": all(
            branch_evaluator_seeds[index] == branch_evaluator_seeds[index + 1]
            for index in range(0, 8, 2)
        ),
        "all_root_eig_positive": all(row["eig"] > 0 for row in root_selections),
        "all_root_branch_masses_at_least_010": all(
            group["mass"] >= MIN_BRANCH_MASS
            for groups in branch_groups
            for group in groups
        ),
        "all_own_masses_at_least_010": all(
            value >= MIN_BRANCH_MASS for value in all_own_masses
        ),
        "at_least_three_positive_own_minus_opposite": sum(
            value > 0 for value in own_minus_opposite
        )
        >= MIN_POSITIVE_SIGNAL_COUNT,
        "mean_own_minus_opposite_at_least_010": own_opposite_mean
        >= MIN_OWN_MINUS_OPPOSITE_MEAN,
        "at_least_three_positive_own_minus_answer_free": sum(
            value > 0 for value in own_minus_blind
        )
        >= MIN_POSITIVE_SIGNAL_COUNT,
        "mean_own_minus_answer_free_at_least_005": own_blind_mean
        >= MIN_OWN_MINUS_BLIND_MEAN,
        "all_exact_updates_normalized": all(
            abs(row["posterior_probability_sum"] - 1.0) <= 1e-9
            and abs(row["posterior_predictive_matched_reply"] - 1.0) <= 1e-9
            for row in posterior_checks
        ),
        "at_least_one_selected_child_facet_changes": any(changed_actions),
        "at_least_one_conditioned_support_changes_from_both_controls": any(
            structural_changes
        ),
        "all_selected_child_scores_positive": all(
            branch["conditioned_second_eig"] > 0
            and branch["answer_free_second_eig"] > 0
            for task in task_diagnostics
            for branch in task["branches"]
        ),
    }
    gates = {**transport_gates, **mechanics_gates}
    gates["all_pass"] = all(gates.values())
    status = "passed" if gates["all_pass"] else "mechanics_failed"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": (
            "separate_proposal_evaluator_v1_development_preregistration_only"
            if gates["all_pass"]
            else "nothing"
        ),
        "bindings": bindings,
        "schedule": {
            "root_proposals": 2,
            "root_evaluators": 2,
            "conditioned_branch_proposals": 4,
            "answer_free_branch_proposals": 4,
            "conditioned_branch_evaluators": 4,
            "answer_free_branch_evaluators": 4,
            "expected_accepted_requests": EXPECTED_REQUESTS,
        },
        "signal_summary": {
            "own_minus_opposite": own_minus_opposite,
            "own_minus_opposite_mean": own_opposite_mean,
            "own_minus_answer_free": own_minus_blind,
            "own_minus_answer_free_mean": own_blind_mean,
            "selected_child_facet_change_count": sum(changed_actions),
            "structural_change_count": sum(structural_changes),
        },
        "task_diagnostics": task_diagnostics,
        "gates": gates,
        "usage": usage,
        "daily_budget_status": dict(daily_budget_status or {}),
        "endpoint_outcomes_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    adapter = build_adapter(
        run_id="regretbench-proposal-evaluator-v1-exact20-smoke",
        output_dir=args.output_dir,
    )
    result = run_smoke(output_dir=args.output_dir, adapter=adapter)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
