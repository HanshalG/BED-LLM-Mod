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
POLICIES = (
    "dynamic_depth2",
    "history_blind_depth2",
    "myopic_width",
    "fixed_depth2",
    "random",
)
PLANNING_REQUESTS = 64 + 64 * QUESTIONS * HYPOTHESES * BRANCH_DRAWS * 2
MAX_ACTUAL_REQUESTS = 64 * QUESTIONS * 2
MAX_REQUESTS = PLANNING_REQUESTS + MAX_ACTUAL_REQUESTS
RUN_BUDGET_USD = 3.70
PROJECTED_COST_USD = 3.20
SMOKE_BUDGET_USD = 0.20
SMOKE_PROJECTED_COST_USD = 0.02
CONCURRENCY = 128
MAX_REQUEST_COST_USD = 0.0015
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
PREREGISTRATION = (
    REPO_ROOT
    / "results/nonmyopic/"
    "REGRETBENCH_DEEPSEEK_DYNAMIC_DEPTH2_POLICY_PREREGISTRATION.md"
)
PREREGISTRATION_SHA256 = (
    "12895164ad10a530b3dbae34b608f7ed5129cb989a5e91e25fa7800a66bbe147"
)
PROBABILITY_FLOOR = 1e-12


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
    if recovery.sha256_file(PREREGISTRATION) != PREREGISTRATION_SHA256:
        raise ValueError("dynamic policy preregistration changed")


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


def branch_seed(task_index: int, root: int, hypothesis: int, draw: int) -> int:
    return BRANCH_SEED_START + task_index * 64 + root * 16 + hypothesis * 2 + draw


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
        or protocol.get("efficacy_used_for_authorization") is not False
        or not (result.get("gates") or {}).get("all_pass")
    ):
        raise ValueError("enriched policy smoke does not authorize development")
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
    for index, (cig, support) in enumerate(zip(cigs[:3], initial[:3], strict=True)):
        question = support["questions"][0]
        answer = support["hypotheses"][0]["predicted_replies"][0]
        _, truth = recovery.sample_truth(cig, recovery.STAGES["smoke"]["truth_seed_start"] + index)
        mapping = recovery.map_and_answer(cig, question, truth)
        mappings.append(mapping)
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
) -> dict[str, Any]:
    brier = [
        task["policies"]["dynamic_depth2"]["brier"]
        - task["policies"][baseline]["brier"]
        for task in tasks
    ]
    log_loss = [
        task["policies"]["dynamic_depth2"]["log_loss"]
        - task["policies"][baseline]["log_loss"]
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
    comparisons = {
        baseline: comparison_summary(
            tasks,
            baseline,
            samples=samples,
            seed=BOOTSTRAP_SEED + index * 10,
        )
        for index, baseline in enumerate(
            ("myopic_width", "history_blind_depth2", "fixed_depth2", "random")
        )
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
        "gates": gates,
    }


def mechanics_gates(
    *,
    usage: Mapping[str, Any],
    expected_requests: int,
    all_supports: Sequence[Mapping[str, Any]],
    initial_supports: Sequence[Mapping[str, Any]],
    simulated_supports: Sequence[Mapping[str, Any]],
    privacy: Sequence[Mapping[str, Any]],
    tasks: Sequence[Mapping[str, Any]],
) -> dict[str, bool]:
    gates = {
        "exact_64_tasks": len(tasks) == 64,
        "expected_requests_within_frozen_maximum": PLANNING_REQUESTS
        <= expected_requests
        <= MAX_REQUESTS,
        "exact_response_count": len(all_supports) == expected_requests,
        "exact_accepted_requests": usage["adapter_requests"] == expected_requests,
        "exact_http_attempts": usage["http_attempts"] == expected_requests,
        "zero_retries": usage["retry_count"] == 0,
        "zero_provider_error_retries": usage["provider_error_retries"] == 0,
        "zero_reasoning_tokens": usage["adapter_reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_supports_strict_and_exactly_eight_unique": all(
            support["diagnostic"]["codec_mode"] == "strict_json"
            and support["diagnostic"]["valid_unique_count"] == MIN_UNIQUE
            and support["diagnostic"]["question_count"] == QUESTIONS
            for support in all_supports
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
        "all_privacy_audits_pass": len(privacy) == expected_requests
        and all(item["passed"] for item in privacy),
        "every_policy_has_48_supported_first_actions": all(
            sum(task["policies"][policy]["first_supported"] for task in tasks)
            >= 48
            for policy in POLICIES
        ),
        "every_policy_has_40_supported_second_actions": all(
            sum(task["policies"][policy]["second_supported"] for task in tasks)
            >= 40
            for policy in POLICIES
        ),
        "within_policy_budget": usage["run_cost_usd"] <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_development(
    *,
    output_dir: Path,
    run_id: str,
    support_smoke_result: Path,
    support_development_result: Path,
    policy_smoke_result: Path,
    adapter: StructuredAdapter | None = None,
    daily_budget_status: Mapping[str, Any] | None = None,
    bootstrap_samples: int = BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    validate_protocol_binding()
    predecessors = validate_support_predecessors(
        support_smoke_result=support_smoke_result,
        support_development_result=support_development_result,
    )
    policy_smoke = validate_policy_smoke(policy_smoke_result)
    output_dir.mkdir(parents=True, exist_ok=True)
    private = output_dir / "private"
    private.mkdir(parents=True, exist_ok=True)
    cigs = recovery.load_stage_cigs("development")
    adapter = adapter or build_adapter(
        stage="development", run_id=run_id, output_dir=output_dir
    )
    privacy: list[dict[str, Any]] = []

    initial_messages = []
    for cig in cigs:
        messages, audit = messages_for(cig, [])
        initial_messages.append(messages)
        privacy.append(audit)
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
                    seed = branch_seed(task_index, root, hypothesis_index, draw)
                    conditioned, conditioned_audit = messages_for(cig, dialogue)
                    blind, blind_audit = messages_for(cig, [])
                    branch_messages.extend([conditioned, blind])
                    branch_seeds.extend([seed, seed])
                    privacy.extend([conditioned_audit, blind_audit])
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
    checkpoint(
        private / "FROZEN_SELECTIONS.json",
        {
            "selected_roots": [row["selected"] for row in planning],
            "hidden_truth_accessed": False,
        },
    )

    truths = []
    first_messages = []
    first_seeds = []
    first_manifest = []
    for task_index, (cig, initial, plan) in enumerate(
        zip(cigs, initial_supports, planning, strict=True)
    ):
        truth_index, truth = recovery.sample_truth(cig, TRUTH_SEED_START + task_index)
        truths.append((truth_index, truth))
        for root in sorted(set(plan["selected"].values())):
            question = initial["questions"][root]
            mapping = recovery.map_and_answer(cig, question, truth)
            dialogue = [
                {"role": "assistant", "content": question},
                {"role": "user", "content": mapping["answer"]},
            ]
            messages, audit = messages_for(cig, dialogue)
            seed = ACTUAL_FIRST_SEED_START + task_index * QUESTIONS + root
            first_messages.append(messages)
            first_seeds.append(seed)
            privacy.append(audit)
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
        seed = ACTUAL_FINAL_SEED_START + task_index * QUESTIONS + root
        final_messages.append(messages)
        final_seeds.append(seed)
        privacy.append(audit)
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
    raw_final = _call(adapter, final_messages, final_seeds)
    final_paths = {
        (manifest["task_index"], manifest["root_index"]): parse_enriched_support(raw)
        for manifest, raw in zip(final_manifest, raw_final, strict=True)
    }
    checkpoint(
        private / "RAW_ACTUAL.json",
        {
            "first_manifest": first_manifest,
            "first_responses": raw_first,
            "final_manifest": final_manifest,
            "final_responses": raw_final,
        },
    )

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
            first_mass = truth_mass_for_aliases(first_path["support"], aliases)
            final_mass = truth_mass_for_aliases(final, aliases)
            policies[policy] = {
                "root_index": root,
                "second_question_index": first_path["second_index"],
                "first_supported": first_path["first_mapping"]["supported"],
                "second_supported": first_path["second_mapping"]["supported"],
                "truth_mass_after_first": first_mass,
                "truth_mass_final": final_mass,
                "brier": (1.0 - final_mass) ** 2,
                "log_loss": -math.log(max(PROBABILITY_FLOOR, final_mass)),
                "covered": final_mass > 0.0,
            }
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
        private_controls.append(
            {
                "task_id": cig.cig_id,
                "truth_index": truth_index,
                "aliases": aliases,
                "selected_questions": {
                    policy: {
                        "first": initial["questions"][root],
                        "second": first_paths[(task_index, root)]["support"][
                            "questions"
                        ][first_paths[(task_index, root)]["second_index"]],
                    }
                    for policy, root in plan["selected"].items()
                },
            }
        )

    usage = summarize_usage(adapter.usage_snapshot())
    expected_requests = PLANNING_REQUESTS + len(first_manifest) + len(final_manifest)
    all_supports = [
        *initial_supports,
        *simulated_supports,
        *[first_paths[key]["support"] for key in sorted(first_paths)],
        *[final_paths[key] for key in sorted(final_paths)],
    ]
    mechanics = mechanics_gates(
        usage=usage,
        expected_requests=expected_requests,
        all_supports=all_supports,
        initial_supports=initial_supports,
        simulated_supports=simulated_supports,
        privacy=privacy,
        tasks=tasks,
    )
    science = scientific_summary(tasks, samples=bootstrap_samples) if mechanics["all_pass"] else None
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
            "expected_requests": expected_requests,
            "maximum_requests": MAX_REQUESTS,
            "preregistration_sha256": PREREGISTRATION_SHA256,
            "support_predecessors": predecessors,
            "policy_smoke": policy_smoke,
            "conditioned_blind_same_seed": True,
            "conditioned_blind_adjacent": True,
            "hidden_cig_exposed_to_model": False,
            "selection_frozen_before_truth_access": True,
            "confirmation_opened": False,
        },
        "daily_budget_status": dict(daily_budget_status or {}),
        "usage": usage,
        "mechanics_gates": mechanics,
        "science": science,
        "tasks": tasks,
    }
    checkpoint(output_dir / "RESULT.json", result)
    checkpoint(private / "CONTROLS.json", {"tasks": private_controls})
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("smoke", "development"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--support-smoke-result", type=Path, required=True)
    parser.add_argument("--support-development-result", type=Path, required=True)
    parser.add_argument("--policy-smoke-result", type=Path)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    adapter: PerRequestSeedStructuredAdapter | None = None
    try:
        live = read_live_credits()
        ledger = json.loads(args.daily_ledger.read_text(encoding="utf-8"))
        stage_cap = (
            SMOKE_BUDGET_USD if args.stage == "smoke" else RUN_BUDGET_USD
        )
        daily = require_budget(
            ledger,
            projected_cost_usd=stage_cap,
            total_usage_usd=live["total_usage_usd"],
        )
        if live["balance_usd"] + 1e-12 < stage_cap:
            raise RuntimeError("live balance is below the full policy stage cap")
        daily.update(live)
        adapter = build_adapter(
            stage=args.stage, run_id=args.run_id, output_dir=output_dir
        )
        if args.stage == "smoke":
            result = run_smoke(
                output_dir=output_dir,
                run_id=args.run_id,
                support_smoke_result=args.support_smoke_result,
                support_development_result=args.support_development_result,
                adapter=adapter,
                daily_budget_status=daily,
            )
        else:
            if args.policy_smoke_result is None:
                raise ValueError("development requires the enriched policy smoke")
            result = run_development(
                output_dir=output_dir,
                run_id=args.run_id,
                support_smoke_result=args.support_smoke_result,
                support_development_result=args.support_development_result,
                policy_smoke_result=args.policy_smoke_result,
                adapter=adapter,
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
            failure["usage"] = summarize_usage(adapter.usage_snapshot())
        checkpoint(output_dir / "FAILURE.json", failure)
        print(json.dumps(failure, indent=2, sort_keys=True))
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
