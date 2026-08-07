#!/usr/bin/env python3
"""Run dynamic depth-two BED over DeepSeek-generated RegretBench beliefs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Protocol, Sequence

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
REGRETBENCH_SRC = REPO_ROOT / "external/RegretBench/src"
for path in (REPO_ROOT, REGRETBENCH_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from helpers import Config, ModelSpec
from regretbench.mapping.semantic_action_mapper import SemanticActionMapper
from regretbench.schemas.cig import CIG
from scripts import regretbench_deepseek_support_recovery as recovery
from scripts.bongard_openworld_luna_naive_first_link import LunaReasoningAdapter
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.number_game_qwen_history_blind_serving_smoke import (
    PerRequestSeedStructuredAdapter,
)
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "regretbench-deepseek-dynamic-depth2-policy-1"
MODEL_ID = recovery.MODEL_ID
TEMPERATURE = recovery.TEMPERATURE
MAX_TOKENS = recovery.MAX_TOKENS
HYPOTHESES = 8
QUESTIONS = 4
MIN_UNIQUE = 8
BRANCH_DRAWS = 2
PRIMARY_POLICIES = (
    "dynamic_depth2",
    "history_blind_depth2",
    "myopic_width",
    "fixed_depth2",
    "random",
)
POLICIES = (
    *PRIMARY_POLICIES,
    "naive_thinking",
)
PLANNING_REQUESTS = 64 + 64 * QUESTIONS * HYPOTHESES * BRANCH_DRAWS * 2
MAX_ACTUAL_REQUESTS = 64 * QUESTIONS * 2
NAIVE_FORMAL_REQUESTS = 128
NAIVE_ENDPOINT_REQUESTS = 128
MAX_PRIMARY_DEEPSEEK_REQUESTS = PLANNING_REQUESTS + MAX_ACTUAL_REQUESTS
MAX_DEEPSEEK_REQUESTS = MAX_PRIMARY_DEEPSEEK_REQUESTS + NAIVE_ENDPOINT_REQUESTS
MAX_REQUESTS = MAX_DEEPSEEK_REQUESTS + NAIVE_FORMAL_REQUESTS
RUN_BUDGET_USD = 3.50
PROJECTED_COST_USD = 3.10
SMOKE_BUDGET_USD = 0.20
SMOKE_PROJECTED_COST_USD = 0.02
NAIVE_SMOKE_BUDGET_USD = 0.20
NAIVE_SMOKE_PROJECTED_COST_USD = 0.10
CONCURRENCY = 128
MAX_REQUEST_COST_USD = 0.0015
NAIVE_MODEL_ID = "openai/gpt-5.6-luna"
NAIVE_REASONING_EFFORT = "medium"
NAIVE_MAX_TOKENS = 8_192
NAIVE_MAX_REQUEST_COST_USD = 0.008
NAIVE_TEMPERATURE = 0.0
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 202608150000
INITIAL_SEED_START = 202608089000
BRANCH_SEED_START = 202608100000
ACTUAL_FIRST_SEED_START = 202608110000
ACTUAL_FINAL_SEED_START = 202608120000
TRUTH_SEED_START = 202608130000
RANDOM_SEED_START = 202608140000
SMOKE_INITIAL_SEED_START = 202608088000
SMOKE_BRANCH_SEED_START = 202608088100
NAIVE_SMOKE_FIRST_SEED_START = 202608088200
NAIVE_SMOKE_SECOND_SEED_START = 202608088300
NAIVE_SMOKE_REDRAW_SEED_START = 202608088400
NAIVE_FIRST_SEED_START = 202608160000
NAIVE_SECOND_SEED_START = 202608170000
NAIVE_FIRST_SUPPORT_SEED_START = 202608180000
NAIVE_FINAL_SUPPORT_SEED_START = 202608190000
PREREGISTRATION = (
    REPO_ROOT
    / "results/nonmyopic/"
    "REGRETBENCH_DEEPSEEK_DYNAMIC_DEPTH2_POLICY_PREREGISTRATION.md"
)
PREREGISTRATION_SHA256 = (
    "03c0f5bd48bd021d696e8641c29942644306b2fd3532157c52af3fc01b6d0f54"
)
ACTION_NOVELTY_AMENDMENT = (
    REPO_ROOT
    / "results/nonmyopic/REGRETBENCH_DISTINCT_ACTION_AMENDMENT_20260807.md"
)
ACTION_NOVELTY_AMENDMENT_SHA256 = (
    "8d375fca72f4da30265a9068df27aefceefb14c5474fff2661cbd1513f056160"
)
VALID_TRAJECTORY_AMENDMENT = (
    REPO_ROOT
    / "results/nonmyopic/"
    "REGRETBENCH_VALID_TRAJECTORY_ENDPOINT_AMENDMENT_20260807.md"
)
VALID_TRAJECTORY_AMENDMENT_SHA256 = (
    "57dacb5e671282b825f3fa375dfbd088dbd4d79f3387df59f5600f6e995edbf3"
)
PROBABILITY_FLOOR = 1e-12
SUPPORT_RECOVERY_CORE_SHA256 = (
    "7e227e4d3a125b817dd45c31ce6b1fc94c24bae6ee982ce59f2a9082065752c2"
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


def validate_protocol_binding() -> None:
    recovery.validate_source_bindings()
    if recovery.sha256_file(Path(recovery.__file__).resolve()) != (
        SUPPORT_RECOVERY_CORE_SHA256
    ):
        raise ValueError("support-recovery core binding changed")
    if recovery.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("dynamic policy preregistration changed")
    if (
        recovery.sha256_file(ACTION_NOVELTY_AMENDMENT)
        != ACTION_NOVELTY_AMENDMENT_SHA256
    ):
        raise ValueError("distinct-action amendment changed")
    if (
        recovery.sha256_file(VALID_TRAJECTORY_AMENDMENT)
        != VALID_TRAJECTORY_AMENDMENT_SHA256
    ):
        raise ValueError("valid-trajectory endpoint amendment changed")


def enriched_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_enriched_support",
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
                            "required": [
                                "interpretation",
                                "final_answer",
                                "prior_weight",
                                "predicted_replies",
                            ],
                            "properties": {
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
                                        "maxLength": 200,
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


SYSTEM_PROMPT = """You maintain an explicit semantic belief state for an ambiguous factual question.
Return exactly eight plausible, meaningfully distinct interpretations and exactly four ranked, distinct, single-dimension clarification questions. Each hypothesis must include the concise factual final answer under that interpretation, a nonnegative prior weight, and exactly four concise predicted user replies aligned by index to the four questions. A predicted reply must answer that clarification as a cooperative user with that interpretation would, without volunteering the final factual answer unless the clarification directly requires it. Use the complete supplied dialogue, do not repeat an already answered question, and do not ask for the entity name, final answer, or an omnibus list. Return only the required JSON object."""


def messages_for(
    cig: CIG, dialogue: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = recovery.public_payload(cig, dialogue)
    audit = recovery.privacy_audit(cig, payload)
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": recovery.canonical_json(payload)},
    ], audit


def naive_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "regretbench_naive_question",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["question"],
                "properties": {
                    "question": {
                        "type": "string",
                        "minLength": 2,
                        "maxLength": 240,
                    }
                },
            },
        },
    }


def naive_messages_for(
    cig: CIG, dialogue: Sequence[Mapping[str, str]]
) -> tuple[list[dict[str, str]], dict[str, Any]]:
    payload = recovery.public_payload(cig, dialogue)
    audit = recovery.privacy_audit(cig, payload)
    turn = "first" if not dialogue else "second and final"
    system = (
        "Choose the "
        + turn
        + " single-dimension clarification question that will best identify "
        "which interpretation of the ambiguous factual question the user means. "
        "Reason carefully using only the supplied prompt and dialogue. Do not ask "
        "for the entity name, the final factual answer, or an omnibus list. Do not "
        "repeat an answered question. Return only the strict JSON object."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": recovery.canonical_json(payload)},
    ], audit


def parse_naive_question(raw: str) -> str:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"question"}:
        raise ValueError("naive response has wrong fields")
    question = value["question"]
    if not isinstance(question, str) or not question.strip().endswith("?"):
        raise ValueError("naive response is not an interrogative question")
    return question.strip()


def distinct_supported_actions(
    first: Mapping[str, Any], second: Mapping[str, Any]
) -> bool:
    return bool(
        first["supported"]
        and second["supported"]
        and first["facet"] is not None
        and second["facet"] is not None
        and first["facet"] != second["facet"]
    )


def parse_enriched_support(raw: str) -> dict[str, Any]:
    value = json.loads(raw)
    if not isinstance(value, dict) or set(value) != {"hypotheses", "questions"}:
        raise ValueError("enriched response has wrong top-level fields")
    questions = value["questions"]
    if not isinstance(questions, list) or len(questions) != QUESTIONS:
        raise ValueError("enriched response must contain four questions")
    cleaned_questions = []
    question_keys = set()
    for index, question in enumerate(questions):
        if not isinstance(question, str) or not question.strip().endswith("?"):
            raise ValueError(f"question {index} is not a question")
        cleaned = question.strip()
        key = recovery.normalize_text(cleaned)
        if not key or key in question_keys:
            raise ValueError("enriched response has duplicate questions")
        question_keys.add(key)
        cleaned_questions.append(cleaned)

    raw_hypotheses = value["hypotheses"]
    if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != HYPOTHESES:
        raise ValueError("enriched response must contain eight hypotheses")
    hypotheses = []
    seen = set()
    for index, item in enumerate(raw_hypotheses):
        if not isinstance(item, dict) or set(item) != {
            "interpretation",
            "final_answer",
            "prior_weight",
            "predicted_replies",
        }:
            raise ValueError(f"enriched hypothesis {index} has wrong fields")
        interpretation = item["interpretation"]
        answer = item["final_answer"]
        weight = item["prior_weight"]
        replies = item["predicted_replies"]
        if (
            not isinstance(interpretation, str)
            or not interpretation.strip()
            or not isinstance(answer, str)
            or not answer.strip()
            or isinstance(weight, bool)
            or not isinstance(weight, (int, float))
            or not math.isfinite(float(weight))
            or float(weight) < 0
            or not isinstance(replies, list)
            or len(replies) != QUESTIONS
            or any(not isinstance(reply, str) or not reply.strip() for reply in replies)
        ):
            raise ValueError(f"enriched hypothesis {index} has invalid values")
        key = (
            recovery.normalize_text(interpretation),
            recovery.normalize_text(answer),
        )
        if key in seen:
            raise ValueError("enriched support contains duplicate hypotheses")
        seen.add(key)
        hypotheses.append(
            {
                "interpretation": interpretation.strip(),
                "final_answer": answer.strip(),
                "probability": float(weight),
                "predicted_replies": [reply.strip() for reply in replies],
            }
        )
    if len(hypotheses) != MIN_UNIQUE:
        raise ValueError("enriched support must contain eight unique hypotheses")
    total = sum(item["probability"] for item in hypotheses)
    if total <= 0:
        raise ValueError("enriched support weights sum to zero")
    for item in hypotheses:
        item["probability"] /= total
    support = {
        "hypotheses": hypotheses,
        "questions": cleaned_questions,
    }
    support["diagnostic"] = {
        "codec_mode": "strict_json",
        "raw_hypothesis_count": len(raw_hypotheses),
        "valid_unique_count": len(hypotheses),
        "question_count": len(cleaned_questions),
        "informative_question_count": sum(
            question_eig(support, question_index) > 1e-12
            for question_index in range(QUESTIONS)
        ),
    }
    return support


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(value * math.log(value) for value in probabilities if value > 0)


def _normalized_weights(
    support: Mapping[str, Any], indexes: Sequence[int] | None = None
) -> tuple[list[int], list[float]]:
    selected = list(indexes) if indexes is not None else list(
        range(len(support["hypotheses"]))
    )
    weights = [support["hypotheses"][index]["probability"] for index in selected]
    total = sum(weights)
    if total <= 0:
        return selected, [1.0 / len(selected)] * len(selected)
    return selected, [value / total for value in weights]


def question_eig(
    support: Mapping[str, Any],
    question_index: int,
    indexes: Sequence[int] | None = None,
) -> float:
    selected, weights = _normalized_weights(support, indexes)
    masses: dict[str, float] = {}
    for index, weight in zip(selected, weights, strict=True):
        key = recovery.normalize_text(
            support["hypotheses"][index]["predicted_replies"][question_index]
        )
        masses[key] = masses.get(key, 0.0) + weight
    return _entropy(list(masses.values()))


def select_question(
    support: Mapping[str, Any], *, exclude: int | None = None
) -> int:
    candidates = [index for index in range(QUESTIONS) if index != exclude]
    return min(
        candidates,
        key=lambda index: (-question_eig(support, index), index),
    )


def _answer_matches(left: str, right: str) -> bool:
    return recovery.lexical_alias_match(left, right)


def matching_reply_indexes(
    support: Mapping[str, Any], question_index: int, observed_reply: str
) -> list[int]:
    observed = recovery.normalize_text(observed_reply)
    if not observed:
        return []
    return [
        index
        for index, hypothesis in enumerate(support["hypotheses"])
        if recovery.normalize_text(
            hypothesis["predicted_replies"][question_index]
        )
        == observed
    ]


def branch_truth_metrics(
    support: Mapping[str, Any], truth_answer: str
) -> dict[str, Any]:
    truth_indexes = [
        index
        for index, item in enumerate(support["hypotheses"])
        if _answer_matches(item["final_answer"], truth_answer)
    ]
    if not truth_indexes:
        return {
            "truth_mass": 0.0,
            "second_question_index": select_question(support),
            "expected_brier": 1.0,
            "expected_log_loss": -math.log(PROBABILITY_FLOOR),
        }
    truth_mass = sum(
        support["hypotheses"][index]["probability"] for index in truth_indexes
    )
    second = select_question(support)
    all_outcome_mass: dict[str, float] = {}
    truth_outcome_mass: dict[str, float] = {}
    for index, item in enumerate(support["hypotheses"]):
        outcome = recovery.normalize_text(item["predicted_replies"][second])
        probability = item["probability"]
        all_outcome_mass[outcome] = all_outcome_mass.get(outcome, 0.0) + probability
        if index in truth_indexes:
            truth_outcome_mass[outcome] = (
                truth_outcome_mass.get(outcome, 0.0) + probability
            )
    expected_brier = 0.0
    expected_log = 0.0
    for outcome, joint_truth_mass in truth_outcome_mass.items():
        probability_given_truth = joint_truth_mass / truth_mass
        posterior_truth = joint_truth_mass / all_outcome_mass[outcome]
        expected_brier += probability_given_truth * (1.0 - posterior_truth) ** 2
        expected_log += probability_given_truth * -math.log(
            max(PROBABILITY_FLOOR, posterior_truth)
        )
    return {
        "truth_mass": truth_mass,
        "second_question_index": second,
        "expected_brier": expected_brier,
        "expected_log_loss": expected_log,
    }


def dynamic_root_risks(
    initial: Mapping[str, Any],
    branch_rows: Sequence[Mapping[str, Any]],
    *,
    arm: str,
) -> list[dict[str, float]]:
    lookup = {
        (row["root_index"], row["hypothesis_index"], row["draw"]): row[arm]
        for row in branch_rows
    }
    risks = []
    for root in range(QUESTIONS):
        brier = 0.0
        log_loss = 0.0
        coverage = 0.0
        for hypothesis_index, hypothesis in enumerate(initial["hypotheses"]):
            local = [
                branch_truth_metrics(
                    lookup[(root, hypothesis_index, draw)],
                    hypothesis["final_answer"],
                )
                for draw in range(BRANCH_DRAWS)
            ]
            weight = hypothesis["probability"]
            brier += weight * float(np.mean([item["expected_brier"] for item in local]))
            log_loss += weight * float(
                np.mean([item["expected_log_loss"] for item in local])
            )
            coverage += weight * float(
                np.mean([item["truth_mass"] > 0 for item in local])
            )
        risks.append(
            {
                "brier": brier,
                "log_loss": log_loss,
                "coverage": coverage,
            }
        )
    return risks


def fixed_root_risks(initial: Mapping[str, Any]) -> list[dict[str, float]]:
    hypotheses = initial["hypotheses"]
    risks = []
    for root in range(QUESTIONS):
        mean_brier = 0.0
        mean_log = 0.0
        for truth_index, truth in enumerate(hypotheses):
            root_reply = recovery.normalize_text(truth["predicted_replies"][root])
            root_indexes = [
                index
                for index, item in enumerate(hypotheses)
                if recovery.normalize_text(item["predicted_replies"][root])
                == root_reply
            ]
            second_candidates = [index for index in range(QUESTIONS) if index != root]
            second = min(
                second_candidates,
                key=lambda index: (-question_eig(initial, index, root_indexes), index),
            )
            second_reply = recovery.normalize_text(
                truth["predicted_replies"][second]
            )
            posterior_indexes = [
                index
                for index in root_indexes
                if recovery.normalize_text(
                    hypotheses[index]["predicted_replies"][second]
                )
                == second_reply
            ]
            denominator = sum(hypotheses[index]["probability"] for index in posterior_indexes)
            numerator = sum(
                hypotheses[index]["probability"]
                for index in posterior_indexes
                if _answer_matches(
                    hypotheses[index]["final_answer"], truth["final_answer"]
                )
            )
            truth_mass = numerator / denominator if denominator > 0 else 0.0
            mean_brier += truth["probability"] * (1.0 - truth_mass) ** 2
            mean_log += truth["probability"] * -math.log(
                max(PROBABILITY_FLOOR, truth_mass)
            )
        risks.append({"brier": mean_brier, "log_loss": mean_log})
    return risks


def choose_roots(
    initial: Mapping[str, Any],
    conditioned: Sequence[Mapping[str, float]],
    blind: Sequence[Mapping[str, float]],
    fixed: Sequence[Mapping[str, float]],
    *,
    random_seed: int,
) -> dict[str, int]:
    return {
        "dynamic_depth2": min(
            range(QUESTIONS), key=lambda index: (conditioned[index]["brier"], index)
        ),
        "history_blind_depth2": min(
            range(QUESTIONS), key=lambda index: (blind[index]["brier"], index)
        ),
        "myopic_width": min(
            range(QUESTIONS),
            key=lambda index: (-question_eig(initial, index), index),
        ),
        "fixed_depth2": min(
            range(QUESTIONS), key=lambda index: (fixed[index]["brier"], index)
        ),
        "random": random.Random(random_seed).randrange(QUESTIONS),
    }


def branch_seed(task_index: int, hypothesis: int, draw: int) -> int:
    return BRANCH_SEED_START + task_index * 16 + hypothesis * 2 + draw


def actual_first_seed(task_index: int) -> int:
    return ACTUAL_FIRST_SEED_START + task_index


def actual_final_seed(task_index: int) -> int:
    return ACTUAL_FINAL_SEED_START + task_index


def build_adapter(
    *, stage: str, run_id: str, output_dir: Path
) -> PerRequestSeedStructuredAdapter:
    if stage not in {"smoke", "development"}:
        raise ValueError("invalid policy adapter stage")
    run_budget = SMOKE_BUDGET_USD if stage == "smoke" else RUN_BUDGET_USD
    projected = (
        SMOKE_PROJECTED_COST_USD if stage == "smoke" else PROJECTED_COST_USD
    )
    concurrency = 10 if stage == "smoke" else CONCURRENCY
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=275.0,
        openrouter_run_budget_usd=run_budget,
        openrouter_projected_cost_usd=projected,
        openrouter_concurrency=concurrency,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return PerRequestSeedStructuredAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=65_536),
        config,
    )


def build_naive_adapter(
    *, stage: str, run_id: str, output_dir: Path
) -> LunaReasoningAdapter:
    if stage not in {"smoke", "development"}:
        raise ValueError("invalid naive adapter stage")
    run_budget = (
        NAIVE_SMOKE_BUDGET_USD if stage == "smoke" else RUN_BUDGET_USD
    )
    projected = (
        NAIVE_SMOKE_PROJECTED_COST_USD
        if stage == "smoke"
        else PROJECTED_COST_USD
    )
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=275.0,
        openrouter_run_budget_usd=run_budget,
        openrouter_projected_cost_usd=projected,
        openrouter_concurrency=10 if stage == "smoke" else 64,
        openrouter_max_retries=0,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=NAIVE_MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=NAIVE_MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return LunaReasoningAdapter(
        ModelSpec(
            model=NAIVE_MODEL_ID,
            backend="openrouter",
            max_model_len=1_000_000,
            reasoning_effort=NAIVE_REASONING_EFFORT,
        ),
        config,
        request_seed=NAIVE_SMOKE_FIRST_SEED_START,
    )


def _call(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=TEMPERATURE,
        response_format=enriched_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("adapter returned the wrong response count")
    return list(responses)


def _call_naive(
    adapter: StructuredAdapter,
    messages: Sequence[list[dict[str, str]]],
    seeds: Sequence[int],
) -> list[str]:
    responses = adapter.chat_complete_seeded_messages_batched_structured(
        messages,
        seeds,
        temperature=NAIVE_TEMPERATURE,
        response_format=naive_response_format(),
        max_new_tokens=NAIVE_MAX_TOKENS,
    )
    if len(responses) != len(messages):
        raise ValueError("naive adapter returned the wrong response count")
    return list(responses)


def validate_support_predecessors(
    *, support_smoke_result: Path, support_development_result: Path
) -> dict[str, Any]:
    smoke = recovery.validate_smoke_result(support_smoke_result)
    development = json.loads(support_development_result.read_text(encoding="utf-8"))
    protocol = development.get("protocol") or {}
    if (
        development.get("status") != "passed"
        or development.get("authorizes")
        != "separately_preregistered_development_policy_only"
        or development.get("interface_version") != recovery.INTERFACE_VERSION
        or protocol.get("stage") != "development"
        or protocol.get("model") != MODEL_ID
        or protocol.get("split_hash") != recovery.SPLIT_HASHES["development"]
        or protocol.get("support_recovery_endpoint_accessed") is not True
        or protocol.get("policy_endpoint_opened") is not False
        or not (development.get("mechanics_gates") or {}).get("all_pass")
        or not ((development.get("science") or {}).get("gates") or {}).get("all_pass")
        or protocol.get("smoke_predecessor", {}).get("sha256") != smoke["sha256"]
    ):
        raise ValueError("support-recovery development does not authorize policy")
    return {
        "support_smoke_sha256": smoke["sha256"],
        "support_development_sha256": recovery.sha256_file(
            support_development_result
        ),
    }


def validate_policy_smoke(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    if (
        result.get("status") != "passed"
        or result.get("interface_version") != INTERFACE_VERSION
        or protocol.get("stage") != "smoke"
        or protocol.get("model") != MODEL_ID
        or protocol.get("expected_requests") != 10
        or protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or protocol.get("distinct_action_amendment_sha256")
        != ACTION_NOVELTY_AMENDMENT_SHA256
        or protocol.get("valid_trajectory_amendment_sha256")
        != VALID_TRAJECTORY_AMENDMENT_SHA256
        or protocol.get("efficacy_used_for_authorization") is not False
        or not (result.get("gates") or {}).get("all_pass")
    ):
        raise ValueError("enriched policy smoke does not authorize development")
    return {"path": str(path), "sha256": recovery.sha256_file(path)}


def _usage(adapter: StructuredAdapter) -> dict[str, Any]:
    snapshot = adapter.usage_snapshot()
    usage = summarize_usage(snapshot)
    usage["forced_final_requests"] = int(
        snapshot.get("forced_final_requests", 0)
    )
    usage["forced_final_successes"] = int(
        snapshot.get("forced_final_successes", 0)
    )
    return usage


def _empty_usage() -> dict[str, Any]:
    return {
        "adapter_requests": 0,
        "http_attempts": 0,
        "retry_count": 0,
        "provider_error_retries": 0,
        "adapter_reasoning_tokens": 0,
        "forced_exits": 0,
        "run_cost_usd": 0.0,
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "forced_final_requests": 0,
        "forced_final_successes": 0,
    }


def _combined_usage(*rows: Mapping[str, Any]) -> dict[str, Any]:
    combined = _empty_usage()
    for key in combined:
        combined[key] = sum(row.get(key, 0) for row in rows)
    return combined


def validate_naive_smoke(path: Path) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    if (
        result.get("status") != "passed"
        or result.get("interface_version") != INTERFACE_VERSION
        or protocol.get("stage") != "naive_smoke"
        or protocol.get("model") != NAIVE_MODEL_ID
        or protocol.get("reasoning_effort") != NAIVE_REASONING_EFFORT
        or protocol.get("expected_requests") != 10
        or protocol.get("preregistration_sha256") != PREREGISTRATION_SHA256
        or protocol.get("distinct_action_amendment_sha256")
        != ACTION_NOVELTY_AMENDMENT_SHA256
        or protocol.get("valid_trajectory_amendment_sha256")
        != VALID_TRAJECTORY_AMENDMENT_SHA256
        or protocol.get("efficacy_used_for_authorization") is not False
        or not (result.get("gates") or {}).get("all_pass")
    ):
        raise ValueError("naive-thinking smoke does not authorize development")
    return {"path": str(path), "sha256": recovery.sha256_file(path)}


def run_smoke(
    *,
    output_dir: Path,
    run_id: str,
    support_smoke_result: Path,
    support_development_result: Path,
    adapter: StructuredAdapter | None = None,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_protocol_binding()
    predecessors = validate_support_predecessors(
        support_smoke_result=support_smoke_result,
        support_development_result=support_development_result,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    cigs = recovery.load_stage_cigs("smoke")
    adapter = adapter or build_adapter(stage="smoke", run_id=run_id, output_dir=output_dir)
    initial_messages = []
    privacy = []
    for cig in cigs:
        messages, audit = messages_for(cig, [])
        initial_messages.append(messages)
        privacy.append(audit)
    initial_seeds = [SMOKE_INITIAL_SEED_START + index for index in range(4)]
    raw_initial = _call(adapter, initial_messages, initial_seeds)
    initial = [parse_enriched_support(raw) for raw in raw_initial]
    branch_messages = []
    branch_seeds = []
    mappings = []
    truths = []
    for index, (cig, support) in enumerate(zip(cigs[:3], initial[:3], strict=True)):
        question = support["questions"][0]
        _, truth = recovery.sample_truth(cig, recovery.STAGES["smoke"]["truth_seed_start"] + index)
        truths.append(truth)
        mapping = recovery.map_and_answer(cig, question, truth)
        mappings.append(mapping)
        answer = mapping["answer"]
        conditioned, audit_conditioned = messages_for(
            cig,
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": answer},
            ],
        )
        blind, audit_blind = messages_for(cig, [])
        seed = SMOKE_BRANCH_SEED_START + index
        branch_messages.extend([conditioned, blind])
        branch_seeds.extend([seed, seed])
        privacy.extend([audit_conditioned, audit_blind])
    raw_branches = _call(adapter, branch_messages, branch_seeds)
    branches = [parse_enriched_support(raw) for raw in raw_branches]
    second_mappings = []
    second_reply_matches = []
    for cig, truth, conditioned in zip(
        cigs[:3], truths, branches[::2], strict=True
    ):
        second = select_question(conditioned)
        mapping = recovery.map_and_answer(
            cig, conditioned["questions"][second], truth
        )
        second_mappings.append(mapping)
        second_reply_matches.append(
            bool(matching_reply_indexes(conditioned, second, mapping["answer"]))
        )
    checkpoint(
        private / "RAW_RESPONSES.json",
        {"initial": raw_initial, "branches": raw_branches},
    )
    usage = summarize_usage(adapter.usage_snapshot())
    all_supports = [*initial, *branches]
    gates = {
        "exact_ten_responses": len(all_supports) == 10,
        "exact_ten_requests": usage["adapter_requests"] == 10,
        "exact_ten_http_attempts": usage["http_attempts"] == 10,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_strict_and_exactly_eight_unique": all(
            support["diagnostic"]["codec_mode"] == "strict_json"
            and support["diagnostic"]["valid_unique_count"] == MIN_UNIQUE
            for support in all_supports
        ),
        "every_initial_has_two_informative_roots": all(
            support["diagnostic"]["informative_question_count"] >= 2
            for support in initial
        ),
        "every_branch_has_an_informative_followup": all(
            support["diagnostic"]["informative_question_count"] >= 1
            for support in branches
        ),
        "all_three_first_questions_supported": all(
            item["supported"] for item in mappings
        ),
        "all_three_second_questions_supported": all(
            item["supported"] for item in second_mappings
        ),
        "all_three_second_actions_are_novel": all(
            distinct_supported_actions(first, second)
            for first, second in zip(
                mappings, second_mappings, strict=True
            )
        ),
        "all_three_exact_second_replies_match_generated_likelihoods": all(
            second_reply_matches
        ),
        "all_privacy_audits_pass": len(privacy) == 10
        and all(item["passed"] for item in privacy),
        "within_smoke_budget": usage["run_cost_usd"] <= SMOKE_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if gates["all_pass"] else "mechanics_failed",
        "authorizes": "dynamic_policy_development_only" if gates["all_pass"] else "nothing",
        "protocol": {
            "stage": "smoke",
            "model": MODEL_ID,
            "reasoning": "disabled_excluded",
            "expected_requests": 10,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "distinct_action_amendment_sha256": (
                ACTION_NOVELTY_AMENDMENT_SHA256
            ),
            "valid_trajectory_amendment_sha256": (
                VALID_TRAJECTORY_AMENDMENT_SHA256
            ),
            "predecessors": predecessors,
            "efficacy_used_for_authorization": False,
            "policy_endpoint_opened": False,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "gates": gates,
        "supports": [support["diagnostic"] for support in all_supports],
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def run_naive_smoke(
    *,
    output_dir: Path,
    run_id: str,
    support_smoke_result: Path,
    support_development_result: Path,
    policy_smoke_result: Path,
    adapter: StructuredAdapter | None = None,
    daily_budget_status: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    validate_protocol_binding()
    predecessors = validate_support_predecessors(
        support_smoke_result=support_smoke_result,
        support_development_result=support_development_result,
    )
    enriched_smoke = validate_policy_smoke(policy_smoke_result)
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    cigs = recovery.load_stage_cigs("smoke")
    adapter = adapter or build_naive_adapter(
        stage="smoke", run_id=run_id, output_dir=output_dir
    )
    privacy = []
    first_messages = []
    for cig in cigs:
        messages, audit = naive_messages_for(cig, [])
        first_messages.append(messages)
        privacy.append(audit)
    first_seeds = [NAIVE_SMOKE_FIRST_SEED_START + index for index in range(4)]
    raw_first = _call_naive(adapter, first_messages, first_seeds)
    first_questions = [parse_naive_question(raw) for raw in raw_first]
    second_messages = []
    first_mappings = []
    second_truths = []
    for index, (cig, question) in enumerate(
        zip(cigs, first_questions, strict=True)
    ):
        _, truth = recovery.sample_truth(
            cig, recovery.STAGES["smoke"]["truth_seed_start"] + index
        )
        second_truths.append(truth)
        mapping = recovery.map_and_answer(cig, question, truth)
        first_mappings.append(mapping)
        messages, audit = naive_messages_for(
            cig,
            [
                {"role": "assistant", "content": question},
                {"role": "user", "content": mapping["answer"]},
            ],
        )
        second_messages.append(messages)
        privacy.append(audit)
    second_seeds = [NAIVE_SMOKE_SECOND_SEED_START + index for index in range(4)]
    raw_second = _call_naive(adapter, second_messages, second_seeds)
    second_questions = [parse_naive_question(raw) for raw in raw_second]
    second_mappings = [
        recovery.map_and_answer(cig, question, truth)
        for cig, question, truth in zip(
            cigs, second_questions, second_truths, strict=True
        )
    ]
    redraw_messages = []
    for cig in cigs[:2]:
        messages, audit = naive_messages_for(cig, [])
        redraw_messages.append(messages)
        privacy.append(audit)
    redraw_seeds = [NAIVE_SMOKE_REDRAW_SEED_START + index for index in range(2)]
    raw_redraw = _call_naive(adapter, redraw_messages, redraw_seeds)
    redraw_questions = [parse_naive_question(raw) for raw in raw_redraw]
    checkpoint(
        private / "RAW_RESPONSES.json",
        {
            "first": raw_first,
            "second": raw_second,
            "redraw": raw_redraw,
            "first_seeds": first_seeds,
            "second_seeds": second_seeds,
            "redraw_seeds": redraw_seeds,
        },
    )
    usage = _usage(adapter)
    gates = {
        "exact_ten_questions": len(first_questions)
        + len(second_questions)
        + len(redraw_questions)
        == 10,
        "exact_ten_requests": usage["adapter_requests"] == 10,
        "exact_ten_http_attempts": usage["http_attempts"] == 10,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "positive_reasoning_tokens": usage["adapter_reasoning_tokens"] > 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "zero_forced_final_requests": usage["forced_final_requests"] == 0,
        "all_eight_executed_questions_supported": all(
            item["supported"] for item in [*first_mappings, *second_mappings]
        ),
        "all_four_second_actions_are_novel": all(
            distinct_supported_actions(first, second)
            for first, second in zip(
                first_mappings, second_mappings, strict=True
            )
        ),
        "all_privacy_audits_pass": len(privacy) == 10
        and all(item["passed"] for item in privacy),
        "within_naive_smoke_budget": usage["run_cost_usd"]
        <= NAIVE_SMOKE_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": "passed" if gates["all_pass"] else "mechanics_failed",
        "authorizes": "dynamic_policy_development_only"
        if gates["all_pass"]
        else "nothing",
        "protocol": {
            "stage": "naive_smoke",
            "model": NAIVE_MODEL_ID,
            "reasoning_effort": NAIVE_REASONING_EFFORT,
            "expected_requests": 10,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "distinct_action_amendment_sha256": (
                ACTION_NOVELTY_AMENDMENT_SHA256
            ),
            "valid_trajectory_amendment_sha256": (
                VALID_TRAJECTORY_AMENDMENT_SHA256
            ),
            "support_predecessors": predecessors,
            "enriched_smoke": enriched_smoke,
            "efficacy_used_for_authorization": False,
            "policy_endpoint_opened": False,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "gates": gates,
        "question_sha256": [
            hashlib.sha256(question.encode()).hexdigest()
            for question in [
                *first_questions,
                *second_questions,
                *redraw_questions,
            ]
        ],
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def truth_mass_for_aliases(support: Mapping[str, Any], aliases: str) -> float:
    alternatives = [item.strip() for item in aliases.split("|") if item.strip()]
    return sum(
        hypothesis["probability"]
        for hypothesis in support["hypotheses"]
        if any(
            recovery.lexical_alias_match(hypothesis["final_answer"], alias)
            for alias in alternatives
        )
    )


def realized_terminal_metrics(
    support: Mapping[str, Any],
    *,
    question_index: int,
    observed_reply: str,
    aliases: str,
) -> dict[str, Any]:
    outcome_indexes = matching_reply_indexes(
        support, question_index, observed_reply
    )
    denominator = sum(
        support["hypotheses"][index]["probability"]
        for index in outcome_indexes
    )
    alternatives = [item.strip() for item in aliases.split("|") if item.strip()]
    numerator = sum(
        support["hypotheses"][index]["probability"]
        for index in outcome_indexes
        if any(
            _answer_matches(
                support["hypotheses"][index]["final_answer"], alias
            )
            for alias in alternatives
        )
    )
    truth_mass = numerator / denominator if denominator > 0.0 else 0.0
    return {
        "reply_matched": bool(outcome_indexes),
        "matched_hypothesis_count": len(outcome_indexes),
        "truth_mass": truth_mass,
        "brier": (1.0 - truth_mass) ** 2,
        "log_loss": -math.log(max(PROBABILITY_FLOOR, truth_mass)),
        "covered": truth_mass > 0.0,
    }


def realized_path_metrics(
    first_support: Mapping[str, Any],
    final_support: Mapping[str, Any],
    *,
    question_index: int,
    observed_reply: str,
    aliases: str,
    first_mapping: Mapping[str, Any],
    second_mapping: Mapping[str, Any],
) -> dict[str, Any]:
    valid = distinct_supported_actions(first_mapping, second_mapping)
    raw_first_mass = truth_mass_for_aliases(first_support, aliases)
    raw_terminal = realized_terminal_metrics(
        first_support,
        question_index=question_index,
        observed_reply=observed_reply,
        aliases=aliases,
    )
    raw_terminal_mass = float(raw_terminal["truth_mass"])
    terminal_mass = raw_terminal_mass if valid else 0.0
    raw_fresh_mass = truth_mass_for_aliases(final_support, aliases)
    fresh_mass = raw_fresh_mass if valid else 0.0
    first_mass = raw_first_mass if first_mapping["supported"] else 0.0
    return {
        "valid_two_action_trajectory": valid,
        "raw_truth_mass_after_first": raw_first_mass,
        "truth_mass_after_first": first_mass,
        "second_reply_likelihood_matched": raw_terminal["reply_matched"],
        "second_reply_matched_hypotheses": raw_terminal[
            "matched_hypothesis_count"
        ],
        "raw_truth_mass_final": raw_terminal_mass,
        "truth_mass_final": terminal_mass,
        "brier": (1.0 - terminal_mass) ** 2,
        "log_loss": -math.log(max(PROBABILITY_FLOOR, terminal_mass)),
        "covered": terminal_mass > 0.0,
        "raw_fresh_truth_mass_final": raw_fresh_mass,
        "fresh_truth_mass_final": fresh_mass,
        "fresh_brier": (1.0 - fresh_mass) ** 2,
        "fresh_log_loss": -math.log(max(PROBABILITY_FLOOR, fresh_mass)),
        "fresh_covered": fresh_mass > 0.0,
    }


def _rankdata(values: Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    order = np.argsort(array, kind="mergesort")
    ranks = np.empty(len(array), dtype=float)
    cursor = 0
    while cursor < len(array):
        end = cursor + 1
        while end < len(array) and array[order[end]] == array[order[cursor]]:
            end += 1
        rank = (cursor + end - 1) / 2.0
        ranks[order[cursor:end]] = rank
        cursor = end
    return ranks


def spearman(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) < 3 or len(left) != len(right):
        return None
    left_rank = _rankdata(left)
    right_rank = _rankdata(right)
    if float(np.std(left_rank)) <= 1e-12 or float(np.std(right_rank)) <= 1e-12:
        return None
    return float(np.corrcoef(left_rank, right_rank)[0, 1])


def paired_bootstrap(
    differences: Sequence[float], *, samples: int, seed: int
) -> dict[str, Any]:
    values = np.asarray(differences, dtype=float)
    if values.size == 0:
        return {
            "mean": None,
            "sample_sd": None,
            "ci95": [None, None],
            "probability_improvement": None,
        }
    rng = np.random.default_rng(seed)
    indexes = rng.integers(0, values.size, size=(samples, values.size))
    means = values[indexes].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "sample_sd": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "ci95": [float(value) for value in np.quantile(means, [0.025, 0.975])],
        "probability_improvement": float(np.mean(means < 0.0)),
        "samples": samples,
        "seed": seed,
    }


def comparison_summary(
    tasks: Sequence[Mapping[str, Any]],
    baseline: str,
    *,
    samples: int,
    seed: int,
    brier_key: str = "brier",
    log_loss_key: str = "log_loss",
) -> dict[str, Any]:
    brier = [
        task["policies"]["dynamic_depth2"][brier_key]
        - task["policies"][baseline][brier_key]
        for task in tasks
    ]
    log_loss = [
        task["policies"]["dynamic_depth2"][log_loss_key]
        - task["policies"][baseline][log_loss_key]
        for task in tasks
    ]
    return {
        "baseline": baseline,
        "brier_dynamic_minus_baseline": paired_bootstrap(
            brier, samples=samples, seed=seed
        ),
        "log_loss_dynamic_minus_baseline": paired_bootstrap(
            log_loss, samples=samples, seed=seed + 1
        ),
        "wins_ties_losses": {
            "wins": sum(value < -1e-12 for value in brier),
            "ties": sum(abs(value) <= 1e-12 for value in brier),
            "losses": sum(value > 1e-12 for value in brier),
        },
    }


def correlation_summary(
    predicted: Sequence[float],
    realized: Sequence[float],
    *,
    samples: int,
) -> dict[str, Any]:
    point = spearman(predicted, realized)
    if point is None:
        return {
            "spearman": None,
            "ci95": [None, None],
            "probability_positive": None,
            "n": len(predicted),
        }
    rng = np.random.default_rng(BOOTSTRAP_SEED + 99)
    correlations = []
    for _ in range(samples):
        indexes = rng.integers(0, len(predicted), size=len(predicted))
        value = spearman(
            [predicted[index] for index in indexes],
            [realized[index] for index in indexes],
        )
        if value is not None:
            correlations.append(value)
    if not correlations:
        return {
            "spearman": point,
            "ci95": [None, None],
            "probability_positive": None,
            "n": len(predicted),
        }
    return {
        "spearman": point,
        "ci95": [
            float(value)
            for value in np.quantile(np.asarray(correlations), [0.025, 0.975])
        ],
        "probability_positive": float(
            np.mean(np.asarray(correlations) > 0.0)
        ),
        "n": len(predicted),
        "samples": samples,
        "seed": BOOTSTRAP_SEED + 99,
    }


def scientific_summary(
    tasks: Sequence[Mapping[str, Any]], *, samples: int = BOOTSTRAP_SAMPLES
) -> dict[str, Any]:
    if len(tasks) != 64:
        raise ValueError("scientific summary requires all 64 tasks")
    baselines = [
        "myopic_width",
        "history_blind_depth2",
        "fixed_depth2",
        "random",
    ]
    comparisons = {
        baseline: comparison_summary(
            tasks,
            baseline,
            samples=samples,
            seed=BOOTSTRAP_SEED + index * 10,
        )
        for index, baseline in enumerate(baselines)
    }
    fresh_baselines = list(baselines)
    if all("naive_thinking" in task["policies"] for task in tasks):
        fresh_baselines.append("naive_thinking")
    fresh_comparisons = {
        baseline: comparison_summary(
            tasks,
            baseline,
            samples=samples,
            seed=BOOTSTRAP_SEED + 500 + index * 10,
            brier_key="fresh_brier",
            log_loss_key="fresh_log_loss",
        )
        for index, baseline in enumerate(fresh_baselines)
    }
    disagreements = {
        baseline: sum(
            task["selected_roots"]["dynamic_depth2"]
            != task["selected_roots"][baseline]
            for task in tasks
        )
        for baseline in (
            "myopic_width",
            "history_blind_depth2",
            "fixed_depth2",
            "random",
        )
    }
    predicted_gain = [
        task["conditioned_root_risks"][task["selected_roots"]["myopic_width"]][
            "brier"
        ]
        - task["conditioned_root_risks"][
            task["selected_roots"]["dynamic_depth2"]
        ]["brier"]
        for task in tasks
    ]
    changed = [
        index
        for index, task in enumerate(tasks)
        if task["selected_roots"]["dynamic_depth2"]
        != task["selected_roots"]["myopic_width"]
    ]
    realized_gain = [
        tasks[index]["policies"]["myopic_width"]["brier"]
        - tasks[index]["policies"]["dynamic_depth2"]["brier"]
        for index in changed
    ]
    correlation = correlation_summary(
        [predicted_gain[index] for index in changed],
        realized_gain,
        samples=samples,
    )
    myopic = comparisons["myopic_width"]
    blind = comparisons["history_blind_depth2"]
    fixed = comparisons["fixed_depth2"]
    gates = {
        "dynamic_myopic_differ_at_least_16": disagreements["myopic_width"] >= 16,
        "dynamic_blind_differ_at_least_12": disagreements[
            "history_blind_depth2"
        ]
        >= 12,
        "dynamic_fixed_differ_at_least_12": disagreements["fixed_depth2"] >= 12,
        "predicted_gain_over_myopic_at_least_001": float(np.mean(predicted_gain))
        >= 0.01,
        "dynamic_myopic_brier_gain_at_least_002": myopic[
            "brier_dynamic_minus_baseline"
        ]["mean"]
        <= -0.02,
        "dynamic_myopic_probability_at_least_090": myopic[
            "brier_dynamic_minus_baseline"
        ]["probability_improvement"]
        >= 0.90,
        "dynamic_myopic_wins_exceed_losses": myopic["wins_ties_losses"]["wins"]
        > myopic["wins_ties_losses"]["losses"],
        "dynamic_blind_brier_gain_at_least_0015": blind[
            "brier_dynamic_minus_baseline"
        ]["mean"]
        <= -0.015,
        "dynamic_blind_probability_at_least_080": blind[
            "brier_dynamic_minus_baseline"
        ]["probability_improvement"]
        >= 0.80,
        "dynamic_blind_wins_exceed_losses": blind["wins_ties_losses"]["wins"]
        > blind["wins_ties_losses"]["losses"],
        "dynamic_fixed_brier_gain_at_least_001": fixed[
            "brier_dynamic_minus_baseline"
        ]["mean"]
        <= -0.01,
        "dynamic_fixed_probability_at_least_080": fixed[
            "brier_dynamic_minus_baseline"
        ]["probability_improvement"]
        >= 0.80,
        "dynamic_fixed_wins_exceed_losses": fixed["wins_ties_losses"]["wins"]
        > fixed["wins_ties_losses"]["losses"],
        "dynamic_log_loss_nonworse_myopic": myopic[
            "log_loss_dynamic_minus_baseline"
        ]["mean"]
        <= 0.0,
        "dynamic_log_loss_nonworse_blind": blind[
            "log_loss_dynamic_minus_baseline"
        ]["mean"]
        <= 0.0,
        "dynamic_log_loss_nonworse_fixed": fixed[
            "log_loss_dynamic_minus_baseline"
        ]["mean"]
        <= 0.0,
        "predicted_realized_spearman_at_least_015": correlation["spearman"]
        is not None
        and correlation["spearman"] >= 0.15,
        "spearman_probability_positive_at_least_080": correlation[
            "probability_positive"
        ]
        is not None
        and correlation["probability_positive"] >= 0.80,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "comparisons": comparisons,
        "root_disagreements": disagreements,
        "mean_conditioned_predicted_gain_over_myopic": float(
            np.mean(predicted_gain)
        ),
        "predicted_to_realized_dynamic_myopic": correlation,
        "fresh_regeneration_comparisons_descriptive": fresh_comparisons,
        "gates": gates,
    }


def seed_schedule_gates(
    *,
    branch_manifest: Sequence[Mapping[str, Any]],
    paired_branch_seeds: Sequence[int],
    first_manifest: Sequence[Mapping[str, Any]],
    final_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    branch_groups: dict[tuple[int, int, int], list[Mapping[str, Any]]] = {}
    for row in branch_manifest:
        key = (row["task_index"], row["hypothesis_index"], row["draw"])
        branch_groups.setdefault(key, []).append(row)
    branch_group_seeds = [
        {int(row["seed"]) for row in rows} for rows in branch_groups.values()
    ]

    def realized_groups(
        rows: Sequence[Mapping[str, Any]],
    ) -> dict[int, list[Mapping[str, Any]]]:
        grouped: dict[int, list[Mapping[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(int(row["task_index"]), []).append(row)
        return grouped

    first_groups = realized_groups(first_manifest)
    final_groups = realized_groups(final_manifest)
    return {
        "conditioned_blind_pairs_share_exact_seed": len(paired_branch_seeds)
        == 2 * len(branch_manifest)
        and all(
            paired_branch_seeds[2 * index]
            == paired_branch_seeds[2 * index + 1]
            == row["seed"]
            for index, row in enumerate(branch_manifest)
        ),
        "simulated_roots_use_task_hypothesis_draw_crn": len(branch_groups)
        == 64 * HYPOTHESES * BRANCH_DRAWS
        and all(
            len(rows) == QUESTIONS
            and {int(row["root_index"]) for row in rows}
            == set(range(QUESTIONS))
            and len(seeds) == 1
            for rows, seeds in zip(
                branch_groups.values(), branch_group_seeds, strict=True
            )
        ),
        "simulated_crn_seed_formula_exact": all(
            int(row["seed"])
            == branch_seed(
                int(row["task_index"]),
                int(row["hypothesis_index"]),
                int(row["draw"]),
            )
            for row in branch_manifest
        ),
        "simulated_crn_seeds_distinct_across_groups": len(
            {next(iter(seeds)) for seeds in branch_group_seeds}
        )
        == len(branch_groups),
        "realized_first_refresh_uses_task_crn": len(first_groups) == 64
        and all(
            len({int(row["seed"]) for row in rows}) == 1
            for rows in first_groups.values()
        ),
        "realized_first_seed_formula_exact": all(
            int(row["seed"]) == actual_first_seed(int(row["task_index"]))
            for row in first_manifest
        ),
        "realized_first_seeds_distinct_across_tasks": len(
            {int(rows[0]["seed"]) for rows in first_groups.values()}
        )
        == len(first_groups),
        "realized_final_refresh_uses_task_crn": len(final_groups) == 64
        and all(
            len({int(row["seed"]) for row in rows}) == 1
            for rows in final_groups.values()
        ),
        "realized_final_seed_formula_exact": all(
            int(row["seed"]) == actual_final_seed(int(row["task_index"]))
            for row in final_manifest
        ),
        "realized_final_seeds_distinct_across_tasks": len(
            {int(rows[0]["seed"]) for rows in final_groups.values()}
        )
        == len(final_groups),
    }


def mechanics_gates(
    *,
    primary_deepseek_usage: Mapping[str, Any],
    expected_primary_deepseek_requests: int,
    primary_supports: Sequence[Mapping[str, Any]],
    initial_supports: Sequence[Mapping[str, Any]],
    simulated_supports: Sequence[Mapping[str, Any]],
    primary_privacy: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
    combined_cost: float,
    branch_manifest: Sequence[Mapping[str, Any]],
    paired_branch_seeds: Sequence[int],
    first_manifest: Sequence[Mapping[str, Any]],
    final_manifest: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    gates = {
        "exact_64_tasks": len(tasks) == 64,
        "expected_requests_within_frozen_maximum": PLANNING_REQUESTS
        <= expected_primary_deepseek_requests
        <= MAX_PRIMARY_DEEPSEEK_REQUESTS,
        "exact_deepseek_response_count": len(primary_supports)
        == expected_primary_deepseek_requests,
        "exact_deepseek_accepted_requests": primary_deepseek_usage[
            "adapter_requests"
        ]
        == expected_primary_deepseek_requests,
        "exact_deepseek_http_attempts": primary_deepseek_usage["http_attempts"]
        == expected_primary_deepseek_requests,
        "deepseek_zero_retries": primary_deepseek_usage["retry_count"] == 0,
        "deepseek_zero_provider_error_retries": primary_deepseek_usage[
            "provider_error_retries"
        ]
        == 0,
        "deepseek_zero_reasoning_tokens": primary_deepseek_usage[
            "adapter_reasoning_tokens"
        ]
        == 0,
        "deepseek_zero_forced_exits": primary_deepseek_usage["forced_exits"]
        == 0,
        "all_supports_strict_and_exactly_eight_unique": all(
            support["diagnostic"]["codec_mode"] == "strict_json"
            and support["diagnostic"]["valid_unique_count"] == MIN_UNIQUE
            and support["diagnostic"]["question_count"] == QUESTIONS
            for support in primary_supports
        ),
        "every_initial_has_two_informative_roots": all(
            support["diagnostic"]["informative_question_count"] >= 2
            for support in initial_supports
        ),
        "ninety_percent_simulated_have_informative_followup": bool(
            np.mean(
                [
                    support["diagnostic"]["informative_question_count"] >= 1
                    for support in simulated_supports
                ]
            )
            >= 0.90
        ),
        "all_privacy_audits_pass": len(primary_privacy)
        == expected_primary_deepseek_requests
        and all(item["passed"] for item in primary_privacy),
        "every_policy_has_48_supported_first_actions": all(
            sum(task["policies"][policy]["first_supported"] for task in tasks)
            >= 48
            for policy in PRIMARY_POLICIES
        ),
        "every_policy_has_40_supported_second_actions": all(
            sum(task["policies"][policy]["second_supported"] for task in tasks)
            >= 40
            for policy in PRIMARY_POLICIES
        ),
        "every_policy_has_40_novel_second_actions": all(
            sum(
                task["policies"][policy]["second_action_novel"]
                for task in tasks
            )
            >= 40
            for policy in PRIMARY_POLICIES
        ),
        "every_policy_has_40_matchable_second_replies": all(
            sum(
                task["policies"][policy][
                    "second_reply_likelihood_matched"
                ]
                for task in tasks
            )
            >= 40
            for policy in PRIMARY_POLICIES
        ),
        "within_combined_policy_budget": combined_cost <= RUN_BUDGET_USD,
        **seed_schedule_gates(
            branch_manifest=branch_manifest,
            paired_branch_seeds=paired_branch_seeds,
            first_manifest=first_manifest,
            final_manifest=final_manifest,
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def naive_baseline_diagnostics(
    *,
    enabled: bool,
    attempted: bool,
    error: Mapping[str, str] | None,
    luna_usage: Mapping[str, Any],
    endpoint_usage: Mapping[str, Any],
    supports: Sequence[Mapping[str, Any]],
    privacy: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    available = (
        enabled
        and attempted
        and error is None
        and len(tasks) == 64
        and all("naive_thinking" in task["policies"] for task in tasks)
    )
    gates = {
        "formal_baseline_available": available,
        "exact_luna_accepted_requests": luna_usage["adapter_requests"]
        == NAIVE_FORMAL_REQUESTS,
        "exact_luna_http_attempts": luna_usage["http_attempts"]
        == NAIVE_FORMAL_REQUESTS,
        "luna_zero_retries": luna_usage["retry_count"] == 0,
        "luna_zero_provider_error_retries": luna_usage[
            "provider_error_retries"
        ]
        == 0,
        "luna_positive_reasoning_tokens": luna_usage["adapter_reasoning_tokens"]
        > 0,
        "luna_zero_forced_exits": luna_usage["forced_exits"] == 0,
        "luna_zero_forced_final_requests": luna_usage["forced_final_requests"]
        == 0,
        "exact_endpoint_accepted_requests": endpoint_usage["adapter_requests"]
        == NAIVE_ENDPOINT_REQUESTS,
        "exact_endpoint_http_attempts": endpoint_usage["http_attempts"]
        == NAIVE_ENDPOINT_REQUESTS,
        "endpoint_zero_retries": endpoint_usage["retry_count"] == 0,
        "endpoint_zero_provider_error_retries": endpoint_usage[
            "provider_error_retries"
        ]
        == 0,
        "endpoint_zero_reasoning_tokens": endpoint_usage[
            "adapter_reasoning_tokens"
        ]
        == 0,
        "endpoint_zero_forced_exits": endpoint_usage["forced_exits"] == 0,
        "all_endpoint_supports_strict_and_exactly_eight_unique": len(supports)
        == NAIVE_ENDPOINT_REQUESTS
        and all(
            support["diagnostic"]["codec_mode"] == "strict_json"
            and support["diagnostic"]["valid_unique_count"] == MIN_UNIQUE
            and support["diagnostic"]["question_count"] == QUESTIONS
            for support in supports
        ),
        "all_privacy_audits_pass": len(privacy)
        == NAIVE_FORMAL_REQUESTS + NAIVE_ENDPOINT_REQUESTS
        and all(item["passed"] for item in privacy),
        "at_least_48_supported_first_actions": available
        and sum(
            task["policies"]["naive_thinking"]["first_supported"]
            for task in tasks
        )
        >= 48,
        "at_least_40_supported_second_actions": available
        and sum(
            task["policies"]["naive_thinking"]["second_supported"]
            for task in tasks
        )
        >= 40,
        "at_least_40_novel_second_actions": available
        and sum(
            task["policies"]["naive_thinking"]["second_action_novel"]
            for task in tasks
        )
        >= 40,
    }
    return {
        "status": (
            "available"
            if available
            else "failed_closed"
            if attempted
            else "disabled_by_smoke"
        ),
        "enabled": enabled,
        "attempted": attempted,
        "error": dict(error) if error is not None else None,
        "gates": gates,
        "all_transport_and_schema_gates_pass": all(
            value
            for key, value in gates.items()
            if key
            not in {
                "at_least_48_supported_first_actions",
                "at_least_40_supported_second_actions",
                "at_least_40_novel_second_actions",
            }
        ),
        "coverage_targets_met": gates["at_least_48_supported_first_actions"]
        and gates["at_least_40_supported_second_actions"]
        and gates["at_least_40_novel_second_actions"],
        "can_affect_primary_status": False,
    }


def run_development(
    *,
    output_dir: Path,
    run_id: str,
    support_smoke_result: Path,
    support_development_result: Path,
    policy_smoke_result: Path,
    naive_smoke_result: Path,
    adapter: StructuredAdapter | None = None,
    naive_adapter: StructuredAdapter | None = None,
    naive_endpoint_adapter: StructuredAdapter | None = None,
    naive_baseline_enabled: bool = True,
    daily_budget_status: Mapping[str, Any] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    validate_protocol_binding()
    predecessors = validate_support_predecessors(
        support_smoke_result=support_smoke_result,
        support_development_result=support_development_result,
    )
    policy_smoke = validate_policy_smoke(policy_smoke_result)
    naive_smoke = (
        validate_naive_smoke(naive_smoke_result)
        if naive_baseline_enabled
        else {
            "path": str(naive_smoke_result),
            "status": "disabled_after_smoke_failure",
        }
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    cigs = recovery.load_stage_cigs("development")
    adapter = adapter or build_adapter(
        stage="development", run_id=run_id, output_dir=output_dir
    )
    if naive_baseline_enabled:
        naive_adapter = naive_adapter or build_naive_adapter(
            stage="development", run_id=run_id, output_dir=output_dir
        )
        naive_endpoint_adapter = naive_endpoint_adapter or build_adapter(
            stage="development", run_id=run_id, output_dir=output_dir
        )
    primary_privacy: list[dict[str, Any]] = []
    naive_privacy: list[dict[str, Any]] = []

    initial_messages = []
    for cig in cigs:
        messages, audit = messages_for(cig, [])
        initial_messages.append(messages)
        primary_privacy.append(audit)
    initial_seeds = [INITIAL_SEED_START + index for index in range(64)]
    raw_initial = _call(adapter, initial_messages, initial_seeds)
    initial_supports = [parse_enriched_support(raw) for raw in raw_initial]
    checkpoint(
        private / "RAW_INITIAL.json",
        {"seeds": initial_seeds, "responses": raw_initial},
    )

    branch_messages = []
    branch_seeds = []
    branch_manifest = []
    for task_index, (cig, support) in enumerate(
        zip(cigs, initial_supports, strict=True)
    ):
        for root in range(QUESTIONS):
            question = support["questions"][root]
            for hypothesis_index, hypothesis in enumerate(support["hypotheses"]):
                answer = hypothesis["predicted_replies"][root]
                dialogue = [
                    {"role": "assistant", "content": question},
                    {"role": "user", "content": answer},
                ]
                for draw in range(BRANCH_DRAWS):
                    seed = branch_seed(task_index, hypothesis_index, draw)
                    conditioned, conditioned_audit = messages_for(cig, dialogue)
                    blind, blind_audit = messages_for(cig, [])
                    branch_messages.extend([conditioned, blind])
                    branch_seeds.extend([seed, seed])
                    primary_privacy.extend([conditioned_audit, blind_audit])
                    branch_manifest.append(
                        {
                            "task_index": task_index,
                            "root_index": root,
                            "hypothesis_index": hypothesis_index,
                            "draw": draw,
                            "seed": seed,
                        }
                    )
    if len(branch_messages) != PLANNING_REQUESTS - 64:
        raise ValueError("branch request schedule changed")
    raw_branches = _call(adapter, branch_messages, branch_seeds)
    checkpoint(
        private / "RAW_BRANCHES.json",
        {
            "manifest": branch_manifest,
            "paired_seeds": branch_seeds,
            "responses": raw_branches,
        },
    )
    branch_by_task: list[list[dict[str, Any]]] = [[] for _ in cigs]
    simulated_supports = []
    for index, manifest in enumerate(branch_manifest):
        conditioned = parse_enriched_support(raw_branches[2 * index])
        blind = parse_enriched_support(raw_branches[2 * index + 1])
        simulated_supports.extend([conditioned, blind])
        branch_by_task[manifest["task_index"]].append(
            {
                **manifest,
                "conditioned": conditioned,
                "blind": blind,
            }
        )

    planning = []
    for task_index, initial in enumerate(initial_supports):
        conditioned = dynamic_root_risks(
            initial, branch_by_task[task_index], arm="conditioned"
        )
        blind = dynamic_root_risks(
            initial, branch_by_task[task_index], arm="blind"
        )
        fixed = fixed_root_risks(initial)
        selected = choose_roots(
            initial,
            conditioned,
            blind,
            fixed,
            random_seed=RANDOM_SEED_START + task_index,
        )
        planning.append(
            {
                "conditioned": conditioned,
                "blind": blind,
                "fixed": fixed,
                "root_eig": [question_eig(initial, index) for index in range(QUESTIONS)],
                "selected": selected,
            }
        )
    naive_attempted = False
    naive_error: dict[str, str] | None = None
    naive_first_seeds = [NAIVE_FIRST_SEED_START + index for index in range(64)]
    raw_naive_first: list[str] = []
    naive_first_questions: list[str] = []
    if naive_baseline_enabled:
        naive_attempted = True
        try:
            naive_first_messages = []
            for cig in cigs:
                messages, audit = naive_messages_for(cig, [])
                naive_first_messages.append(messages)
                naive_privacy.append(audit)
            if naive_adapter is None:
                raise RuntimeError("naive adapter is unavailable")
            raw_naive_first = _call_naive(
                naive_adapter, naive_first_messages, naive_first_seeds
            )
            naive_first_questions = [
                parse_naive_question(raw) for raw in raw_naive_first
            ]
        except Exception as exc:
            naive_error = {
                "stage": "first_question",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            naive_first_questions = []
    checkpoint(
        private / "FROZEN_SELECTIONS.json",
        {
            "selected_roots": [row["selected"] for row in planning],
            "naive_first_seeds": naive_first_seeds,
            "naive_first_question_sha256": [
                hashlib.sha256(question.encode()).hexdigest()
                for question in naive_first_questions
            ],
            "naive_baseline_enabled": naive_baseline_enabled,
            "naive_first_status": (
                "passed"
                if naive_error is None and naive_baseline_enabled
                else "unavailable"
            ),
            "naive_error": naive_error,
            "hidden_truth_accessed": False,
        },
    )

    truths = []
    first_messages = []
    first_seeds = []
    first_manifest = []
    naive_first_support_messages = []
    naive_first_support_seeds = []
    naive_first_manifest = []
    naive_second_messages = []
    naive_second_seeds = []
    for task_index, (cig, initial, plan) in enumerate(
        zip(cigs, initial_supports, planning, strict=True)
    ):
        truth_index, truth = recovery.sample_truth(cig, TRUTH_SEED_START + task_index)
        truths.append((truth_index, truth))
        if naive_baseline_enabled and naive_error is None:
            try:
                naive_question = naive_first_questions[task_index]
                naive_mapping = recovery.map_and_answer(cig, naive_question, truth)
                naive_dialogue = [
                    {"role": "assistant", "content": naive_question},
                    {"role": "user", "content": naive_mapping["answer"]},
                ]
                support_messages, support_audit = messages_for(
                    cig, naive_dialogue
                )
                naive_first_support_messages.append(support_messages)
                naive_first_support_seeds.append(
                    NAIVE_FIRST_SUPPORT_SEED_START + task_index
                )
                naive_privacy.append(support_audit)
                second_messages, second_audit = naive_messages_for(
                    cig, naive_dialogue
                )
                naive_second_messages.append(second_messages)
                naive_second_seeds.append(NAIVE_SECOND_SEED_START + task_index)
                naive_privacy.append(second_audit)
                naive_first_manifest.append(
                    {
                        "task_index": task_index,
                        "first_mapping": naive_mapping,
                        "dialogue": naive_dialogue,
                    }
                )
            except Exception as exc:
                naive_error = {
                    "stage": "first_environment_mapping",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        for root in sorted(set(plan["selected"].values())):
            question = initial["questions"][root]
            mapping = recovery.map_and_answer(cig, question, truth)
            dialogue = [
                {"role": "assistant", "content": question},
                {"role": "user", "content": mapping["answer"]},
            ]
            messages, audit = messages_for(cig, dialogue)
            seed = actual_first_seed(task_index)
            first_messages.append(messages)
            first_seeds.append(seed)
            primary_privacy.append(audit)
            first_manifest.append(
                {
                    "task_index": task_index,
                    "root_index": root,
                    "first_mapping": mapping,
                    "dialogue": dialogue,
                    "seed": seed,
                }
            )
    raw_first = _call(adapter, first_messages, first_seeds)
    raw_naive_first_support: list[str] = []
    naive_first_supports: list[dict[str, Any]] = []
    raw_naive_second: list[str] = []
    naive_second_questions: list[str] = []
    if naive_baseline_enabled and naive_error is None:
        try:
            if naive_endpoint_adapter is None or naive_adapter is None:
                raise RuntimeError("naive baseline adapters are unavailable")
            raw_naive_first_support = _call(
                naive_endpoint_adapter,
                naive_first_support_messages,
                naive_first_support_seeds,
            )
            naive_first_supports = [
                parse_enriched_support(raw) for raw in raw_naive_first_support
            ]
            raw_naive_second = _call_naive(
                naive_adapter, naive_second_messages, naive_second_seeds
            )
            naive_second_questions = [
                parse_naive_question(raw) for raw in raw_naive_second
            ]
        except Exception as exc:
            naive_error = {
                "stage": "first_endpoint_or_second_question",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    first_paths = {}
    final_messages = []
    final_seeds = []
    final_manifest = []
    for manifest, raw in zip(first_manifest, raw_first, strict=True):
        support = parse_enriched_support(raw)
        task_index = manifest["task_index"]
        root = manifest["root_index"]
        cig = cigs[task_index]
        truth = truths[task_index][1]
        second = select_question(support)
        question = support["questions"][second]
        second_mapping = recovery.map_and_answer(cig, question, truth)
        dialogue = [
            *manifest["dialogue"],
            {"role": "assistant", "content": question},
            {"role": "user", "content": second_mapping["answer"]},
        ]
        messages, audit = messages_for(cig, dialogue)
        seed = actual_final_seed(task_index)
        final_messages.append(messages)
        final_seeds.append(seed)
        primary_privacy.append(audit)
        first_paths[(task_index, root)] = {
            "support": support,
            "first_mapping": manifest["first_mapping"],
            "second_index": second,
            "second_mapping": second_mapping,
            "dialogue": dialogue,
        }
        final_manifest.append(
            {"task_index": task_index, "root_index": root, "seed": seed}
        )
    naive_final_messages = []
    naive_final_seeds = []
    naive_paths = []
    if naive_baseline_enabled and naive_error is None:
        try:
            for manifest, first_support, second_question in zip(
                naive_first_manifest,
                naive_first_supports,
                naive_second_questions,
                strict=True,
            ):
                task_index = manifest["task_index"]
                cig = cigs[task_index]
                truth = truths[task_index][1]
                second_mapping = recovery.map_and_answer(
                    cig, second_question, truth
                )
                dialogue = [
                    *manifest["dialogue"],
                    {"role": "assistant", "content": second_question},
                    {"role": "user", "content": second_mapping["answer"]},
                ]
                messages, audit = messages_for(cig, dialogue)
                naive_final_messages.append(messages)
                naive_final_seeds.append(
                    NAIVE_FINAL_SUPPORT_SEED_START + task_index
                )
                naive_privacy.append(audit)
                naive_paths.append(
                    {
                        "first_support": first_support,
                        "first_mapping": manifest["first_mapping"],
                        "second_mapping": second_mapping,
                        "dialogue": dialogue,
                    }
                )
        except Exception as exc:
            naive_error = {
                "stage": "second_environment_mapping",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    raw_final = _call(adapter, final_messages, final_seeds)
    final_paths = {
        (manifest["task_index"], manifest["root_index"]): parse_enriched_support(raw)
        for manifest, raw in zip(final_manifest, raw_final, strict=True)
    }
    raw_naive_final: list[str] = []
    naive_final_supports: list[dict[str, Any]] = []
    if naive_baseline_enabled and naive_error is None:
        try:
            if naive_endpoint_adapter is None:
                raise RuntimeError("naive endpoint adapter is unavailable")
            raw_naive_final = _call(
                naive_endpoint_adapter, naive_final_messages, naive_final_seeds
            )
            naive_final_supports = [
                parse_enriched_support(raw) for raw in raw_naive_final
            ]
        except Exception as exc:
            naive_error = {
                "stage": "final_endpoint",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
    checkpoint(
        private / "RAW_ACTUAL.json",
        {
            "first_manifest": first_manifest,
            "first_responses": raw_first,
            "final_manifest": final_manifest,
            "final_responses": raw_final,
            "naive_first_questions": raw_naive_first,
            "naive_first_support_seeds": naive_first_support_seeds,
            "naive_first_support_responses": raw_naive_first_support,
            "naive_second_questions": raw_naive_second,
            "naive_final_support_seeds": naive_final_seeds,
            "naive_final_support_responses": raw_naive_final,
            "naive_baseline_error": naive_error,
        },
    )

    naive_policy_rows: list[dict[str, Any]] = []
    if naive_baseline_enabled and naive_error is None:
        try:
            for task_index, (naive_path, naive_final) in enumerate(
                zip(naive_paths, naive_final_supports, strict=True)
            ):
                aliases = str(
                    (truths[task_index][1].slots or {})["answer_aliases"]
                )
                raw_first_mass = truth_mass_for_aliases(
                    naive_path["first_support"], aliases
                )
                raw_final_mass = truth_mass_for_aliases(naive_final, aliases)
                valid = distinct_supported_actions(
                    naive_path["first_mapping"],
                    naive_path["second_mapping"],
                )
                first_mass = (
                    raw_first_mass
                    if naive_path["first_mapping"]["supported"]
                    else 0.0
                )
                final_mass = raw_final_mass if valid else 0.0
                naive_policy_rows.append(
                    {
                        "endpoint_mode": "fresh_regeneration_descriptive",
                        "root_index": None,
                        "second_question_index": None,
                        "first_supported": naive_path["first_mapping"][
                            "supported"
                        ],
                        "second_supported": naive_path["second_mapping"][
                            "supported"
                        ],
                        "second_action_novel": distinct_supported_actions(
                            naive_path["first_mapping"],
                            naive_path["second_mapping"],
                        ),
                        "valid_two_action_trajectory": valid,
                        "raw_truth_mass_after_first": raw_first_mass,
                        "truth_mass_after_first": first_mass,
                        "raw_truth_mass_final": raw_final_mass,
                        "truth_mass_final": final_mass,
                        "brier": (1.0 - final_mass) ** 2,
                        "log_loss": -math.log(
                            max(PROBABILITY_FLOOR, final_mass)
                        ),
                        "covered": final_mass > 0.0,
                        "raw_fresh_truth_mass_final": raw_final_mass,
                        "fresh_truth_mass_final": final_mass,
                        "fresh_brier": (1.0 - final_mass) ** 2,
                        "fresh_log_loss": -math.log(
                            max(PROBABILITY_FLOOR, final_mass)
                        ),
                        "fresh_covered": final_mass > 0.0,
                    }
                )
        except Exception as exc:
            naive_error = {
                "stage": "endpoint_scoring",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            naive_policy_rows = []

    tasks = []
    private_controls = []
    for task_index, (cig, initial, plan) in enumerate(
        zip(cigs, initial_supports, planning, strict=True)
    ):
        truth_index, truth = truths[task_index]
        aliases = str((truth.slots or {})["answer_aliases"])
        policies = {}
        for policy, root in plan["selected"].items():
            first_path = first_paths[(task_index, root)]
            final = final_paths[(task_index, root)]
            path_metrics = realized_path_metrics(
                first_path["support"],
                final,
                question_index=first_path["second_index"],
                observed_reply=first_path["second_mapping"]["answer"],
                aliases=aliases,
                first_mapping=first_path["first_mapping"],
                second_mapping=first_path["second_mapping"],
            )
            policies[policy] = {
                "endpoint_mode": "aligned_generated_likelihood",
                "root_index": root,
                "second_question_index": first_path["second_index"],
                "first_supported": first_path["first_mapping"]["supported"],
                "second_supported": first_path["second_mapping"]["supported"],
                "second_action_novel": distinct_supported_actions(
                    first_path["first_mapping"],
                    first_path["second_mapping"],
                ),
                **path_metrics,
            }
        if naive_baseline_enabled and naive_error is None:
            policies["naive_thinking"] = naive_policy_rows[task_index]
        tasks.append(
            {
                "task_id": cig.cig_id,
                "selected_roots": dict(plan["selected"]),
                "root_eig": plan["root_eig"],
                "conditioned_root_risks": plan["conditioned"],
                "blind_root_risks": plan["blind"],
                "fixed_root_risks": plan["fixed"],
                "policies": policies,
            }
        )
        selected_questions = {
            policy: {
                "first": initial["questions"][root],
                "second": first_paths[(task_index, root)]["support"]["questions"]
                [first_paths[(task_index, root)]["second_index"]],
            }
            for policy, root in plan["selected"].items()
        }
        if naive_baseline_enabled and naive_error is None:
            selected_questions["naive_thinking"] = {
                "first": naive_first_questions[task_index],
                "second": naive_second_questions[task_index],
            }
        private_controls.append(
            {
                "task_id": cig.cig_id,
                "truth_index": truth_index,
                "aliases": aliases,
                "selected_questions": selected_questions,
            }
        )

    primary_deepseek_usage = summarize_usage(adapter.usage_snapshot())
    endpoint_usage = (
        summarize_usage(naive_endpoint_adapter.usage_snapshot())
        if naive_endpoint_adapter is not None
        else _empty_usage()
    )
    naive_usage = (
        _usage(naive_adapter) if naive_adapter is not None else _empty_usage()
    )
    deepseek_usage = _combined_usage(primary_deepseek_usage, endpoint_usage)
    expected_primary_deepseek_requests = (
        PLANNING_REQUESTS + len(first_manifest) + len(final_manifest)
    )
    combined_cost = (
        float(primary_deepseek_usage["run_cost_usd"])
        + float(endpoint_usage["run_cost_usd"])
        + float(naive_usage["run_cost_usd"])
    )
    usage = {
        "deepseek": deepseek_usage,
        "deepseek_primary": primary_deepseek_usage,
        "deepseek_naive_endpoint": endpoint_usage,
        "naive_luna": naive_usage,
        "combined_cost_usd": combined_cost,
        "combined_requests": deepseek_usage["adapter_requests"]
        + naive_usage["adapter_requests"],
        "combined_http_attempts": deepseek_usage["http_attempts"]
        + naive_usage["http_attempts"],
    }
    primary_supports = [
        *initial_supports,
        *simulated_supports,
        *[first_paths[key]["support"] for key in sorted(first_paths)],
        *[final_paths[key] for key in sorted(final_paths)],
    ]
    mechanics = mechanics_gates(
        primary_deepseek_usage=primary_deepseek_usage,
        expected_primary_deepseek_requests=expected_primary_deepseek_requests,
        primary_supports=primary_supports,
        initial_supports=initial_supports,
        simulated_supports=simulated_supports,
        primary_privacy=primary_privacy,
        tasks=tasks,
        combined_cost=combined_cost,
        branch_manifest=branch_manifest,
        paired_branch_seeds=branch_seeds,
        first_manifest=first_manifest,
        final_manifest=final_manifest,
    )
    baseline = naive_baseline_diagnostics(
        enabled=naive_baseline_enabled,
        attempted=naive_attempted,
        error=naive_error,
        luna_usage=naive_usage,
        endpoint_usage=endpoint_usage,
        supports=[*naive_first_supports, *naive_final_supports],
        privacy=naive_privacy,
        tasks=tasks,
    )
    science = (
        scientific_summary(tasks, samples=bootstrap_samples)
        if mechanics["all_pass"]
        else None
    )
    status = "mechanics_failed"
    if mechanics["all_pass"]:
        status = "passed" if science and science["gates"]["all_pass"] else "gated_null"
    result = {
        "schema_version": SCHEMA_VERSION,
        "interface_version": INTERFACE_VERSION,
        "status": status,
        "authorizes": "confirmation_preregistration_only" if status == "passed" else "nothing",
        "protocol": {
            "stage": "development",
            "model": MODEL_ID,
            "reasoning": "disabled_excluded",
            "task_count": 64,
            "branch_draws": BRANCH_DRAWS,
            "planning_requests": PLANNING_REQUESTS,
            "expected_primary_deepseek_requests": expected_primary_deepseek_requests,
            "actual_naive_endpoint_requests": endpoint_usage[
                "adapter_requests"
            ],
            "actual_naive_requests": naive_usage["adapter_requests"],
            "actual_combined_requests": usage["combined_requests"],
            "maximum_naive_endpoint_requests": NAIVE_ENDPOINT_REQUESTS,
            "maximum_naive_requests": NAIVE_FORMAL_REQUESTS,
            "maximum_primary_deepseek_requests": MAX_PRIMARY_DEEPSEEK_REQUESTS,
            "maximum_deepseek_requests": MAX_DEEPSEEK_REQUESTS,
            "maximum_requests": MAX_REQUESTS,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "distinct_action_amendment_sha256": (
                ACTION_NOVELTY_AMENDMENT_SHA256
            ),
            "valid_trajectory_amendment_sha256": (
                VALID_TRAJECTORY_AMENDMENT_SHA256
            ),
            "support_predecessors": predecessors,
            "policy_smoke": policy_smoke,
            "naive_smoke": naive_smoke,
            "naive_model": NAIVE_MODEL_ID,
            "naive_reasoning_effort": NAIVE_REASONING_EFFORT,
            "naive_is_descriptive_only": True,
            "naive_can_gate_or_abort_primary": False,
            "primary_endpoint": "aligned_generated_likelihood_truth_mass",
            "fresh_regeneration_endpoint": "secondary_descriptive",
            "naive_endpoint": "fresh_regeneration_descriptive",
            "conditioned_blind_same_seed": True,
            "conditioned_blind_adjacent": True,
            "simulated_common_seed_across_roots": True,
            "realized_common_seed_across_roots": True,
            "hidden_cig_exposed_to_model": False,
            "selection_frozen_before_truth_access": True,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "mechanics_gates": mechanics,
        "naive_baseline": baseline,
        "science": science,
        "tasks": tasks,
    }
    checkpoint(output_dir / "RESULT.json", result)
    checkpoint(private / "CONTROLS.json", {"tasks": private_controls})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("smoke", "naive_smoke", "development"), required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--support-smoke-result", type=Path, required=True)
    parser.add_argument("--support-development-result", type=Path, required=True)
    parser.add_argument("--policy-smoke-result", type=Path)
    parser.add_argument("--naive-smoke-result", type=Path)
    parser.add_argument("--disable-naive-baseline", action="store_true")
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter: PerRequestSeedStructuredAdapter | None = None
    naive_adapter: LunaReasoningAdapter | None = None
    naive_endpoint_adapter: PerRequestSeedStructuredAdapter | None = None
    try:
        live = read_live_credits()
        ledger = json.loads(args.daily_ledger.read_text(encoding="utf-8"))
        stage_cap = {
            "smoke": SMOKE_BUDGET_USD,
            "naive_smoke": NAIVE_SMOKE_BUDGET_USD,
            "development": RUN_BUDGET_USD,
        }[args.stage]
        daily = require_budget(
            ledger,
            projected_cost_usd=stage_cap,
            total_usage_usd=live["total_usage_usd"],
        )
        if live["balance_usd"] + 1e-12 < stage_cap:
            raise RuntimeError("live balance is below the full policy stage cap")
        daily.update(live)
        if args.stage == "smoke":
            adapter = build_adapter(
                stage="smoke", run_id=args.run_id, output_dir=output_dir
            )
            result = run_smoke(
                output_dir=output_dir,
                run_id=args.run_id,
                support_smoke_result=args.support_smoke_result,
                support_development_result=args.support_development_result,
                adapter=adapter,
                daily_budget_status=daily,
            )
        elif args.stage == "naive_smoke":
            if args.policy_smoke_result is None:
                raise ValueError("naive smoke requires the enriched policy smoke")
            naive_adapter = build_naive_adapter(
                stage="smoke", run_id=args.run_id, output_dir=output_dir
            )
            result = run_naive_smoke(
                output_dir=output_dir,
                run_id=args.run_id,
                support_smoke_result=args.support_smoke_result,
                support_development_result=args.support_development_result,
                policy_smoke_result=args.policy_smoke_result,
                adapter=naive_adapter,
                daily_budget_status=daily,
            )
        else:
            if args.policy_smoke_result is None:
                raise ValueError("development requires the enriched policy smoke")
            if args.naive_smoke_result is None and not args.disable_naive_baseline:
                raise ValueError(
                    "enabled naive baseline requires its smoke result"
                )
            adapter = build_adapter(
                stage="development", run_id=args.run_id, output_dir=output_dir
            )
            if not args.disable_naive_baseline:
                naive_adapter = build_naive_adapter(
                    stage="development", run_id=args.run_id, output_dir=output_dir
                )
                naive_endpoint_adapter = build_adapter(
                    stage="development", run_id=args.run_id, output_dir=output_dir
                )
            result = run_development(
                output_dir=output_dir,
                run_id=args.run_id,
                support_smoke_result=args.support_smoke_result,
                support_development_result=args.support_development_result,
                policy_smoke_result=args.policy_smoke_result,
                naive_smoke_result=args.naive_smoke_result or Path("disabled"),
                adapter=adapter,
                naive_adapter=naive_adapter,
                naive_endpoint_adapter=naive_endpoint_adapter,
                naive_baseline_enabled=not args.disable_naive_baseline,
                daily_budget_status=daily,
            )
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error_type": type(exc).__name__,
            "error": str(exc),
            "authorizes": "nothing",
            "confirmation_opened": False,
        }
        if adapter is not None:
            failure["deepseek_usage"] = summarize_usage(adapter.usage_snapshot())
        if naive_adapter is not None:
            failure["naive_usage"] = _usage(naive_adapter)
        if naive_endpoint_adapter is not None:
            failure["naive_endpoint_usage"] = summarize_usage(
                naive_endpoint_adapter.usage_snapshot()
            )
        checkpoint(output_dir / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
