#!/usr/bin/env python3
"""Prospectively validate model-criticism acquisition on InteractComp."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.atd_code_first_link_audit import spearman
from scripts.interactcomp_first_link_opportunity import (
    ChatModel,
    GENERATOR_MODEL_ID,
    Hypothesis,
    MIN_REFRESH_PARTICLES,
    PARTICLE_COUNT,
    QUESTION_COUNT,
    RESPONDER_MODEL_ID,
    _argmax,
    _checkpoint,
    _classification_messages,
    _decrypted_fields,
    _initial_messages,
    _question_messages,
    _refresh_messages,
    _responder_messages,
    _truth_mass,
    aggregate_usage,
    entropy_from_labels,
    normalize_entity,
    parse_classification,
    parse_hypothesis,
    parse_question,
    parse_responder_answer,
    verify_source,
)
from scripts.interactcomp_robust_support_development import (
    _distribution,
    _entropy_distribution,
    balanced_model_information,
)


INTERFACE_VERSION = "interactcomp-model-criticism-validation-2"
SEED = 24_382
RANDOM_CONTROL_SEED = 24_383
SCREEN_INDICES = (
    90,
    12,
    133,
    130,
    184,
    125,
    105,
    69,
    67,
    124,
    11,
    64,
    91,
    9,
    60,
    197,
)
EXPECTED_SCREEN_IDS = (
    91,
    13,
    134,
    131,
    185,
    126,
    106,
    70,
    68,
    125,
    12,
    65,
    92,
    10,
    61,
    198,
)
ENROLL_COUNT = 6
COLLAPSED_UNIQUE_MAX = 4
AUXILIARY_PROPOSALS = 16
AUXILIARY_RETAIN = 8
EXPECTED_REQUESTS = (
    len(SCREEN_INDICES) * PARTICLE_COUNT
    + ENROLL_COUNT * QUESTION_COUNT
    + ENROLL_COUNT * PARTICLE_COUNT
    + ENROLL_COUNT * AUXILIARY_PROPOSALS
    + ENROLL_COUNT * AUXILIARY_PROPOSALS
    + ENROLL_COUNT * AUXILIARY_RETAIN
    + ENROLL_COUNT * QUESTION_COUNT
    + ENROLL_COUNT * QUESTION_COUNT * PARTICLE_COUNT
)
MAX_COST_USD = 1.50


class ValidationExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def parse_semantic_distinctness(text: str) -> bool:
    value = text.strip().upper()
    if value == "D":
        return True
    if value == "S":
        return False
    raise ValueError("semantic distinctness must be exactly D or S")


def parse_classification_ascii_whitespace(text: str) -> str:
    compact = text.translate(
        {ord(character): None for character in " \t\r\n"}
    )
    return parse_classification(compact)


def unique_entity_count(hypotheses: Sequence[Hypothesis]) -> int:
    return len(
        {normalize_entity(hypothesis.entity) for hypothesis in hypotheses}
    )


def enroll_collapsed_tasks(
    populations: dict[int, list[Hypothesis]],
) -> list[int]:
    return [
        index
        for index in SCREEN_INDICES
        if unique_entity_count(populations[index]) <= COLLAPSED_UNIQUE_MAX
    ][:ENROLL_COUNT]


def _auxiliary_messages(
    *,
    task_id: int,
    question: str,
    initial: Sequence[Hypothesis],
    sample_index: int,
) -> list[dict[str, str]]:
    request = {
        "benchmark_task_id": task_id,
        "ambiguous_question": question,
        "excluded_current_support": [
            hypothesis.as_dict() for hypothesis in initial
        ],
        "outside_sample": sample_index,
    }
    return [
        {
            "role": "system",
            "content": (
                "Generate one plausible answer candidate outside the supplied "
                "current support. Look for a materially different interpretation "
                "or a shared assumption all current candidates may have missed. "
                "Do not output aliases, spelling variants, broader/narrower labels, "
                "or the same underlying entity as an excluded candidate. Different "
                "outside_sample values should explore different possibilities. "
                "Output exactly two lines:\n"
                "ENTITY: <candidate name>\nPROFILE: <candidate attributes>"
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _semantic_validation_messages(
    *,
    current: Sequence[Hypothesis],
    candidate: Hypothesis,
) -> list[dict[str, str]]:
    request = {
        "current_support": [hypothesis.as_dict() for hypothesis in current],
        "semantic_candidate": candidate.as_dict(),
    }
    return [
        {
            "role": "system",
            "content": (
                "Decide whether the semantic candidate is a materially distinct "
                "answer entity from every current-support entity. Reply D only if "
                "it is genuinely different. Reply S if it is the same entity, an "
                "alias or spelling variant, or merely a broader/narrower label for "
                "the same answer. Output exactly one character: D or S."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def root_scores(
    current_classifications: Sequence[str],
    auxiliary_classifications: Sequence[str],
) -> dict[str, list[float]]:
    if len(current_classifications) != PARTICLE_COUNT:
        raise ValueError("current classification width changed")
    if len(auxiliary_classifications) != AUXILIARY_RETAIN:
        raise ValueError("auxiliary classification width changed")
    current_eig = []
    model_criticism = []
    augmented_eig = []
    for root_index in range(QUESTION_COUNT):
        current_labels = [
            classification[root_index]
            for classification in current_classifications
        ]
        auxiliary_labels = [
            classification[root_index]
            for classification in auxiliary_classifications
        ]
        current_distribution = _distribution(current_labels)
        auxiliary_distribution = _distribution(auxiliary_labels)
        mixture = {
            label: 0.5 * current_distribution[label]
            + 0.5 * auxiliary_distribution[label]
            for label in ("Y", "N", "U")
        }
        current_eig.append(entropy_from_labels(current_labels))
        model_criticism.append(
            balanced_model_information(current_labels, auxiliary_labels)
        )
        augmented_eig.append(_entropy_distribution(mixture))
    return {
        "current_eig": current_eig,
        "model_criticism": model_criticism,
        "augmented_eig": augmented_eig,
    }


def score_validation_task(
    *,
    task_id: int,
    initial: Sequence[Hypothesis],
    questions: Sequence[str],
    current_classifications: Sequence[str],
    auxiliary: Sequence[Hypothesis],
    auxiliary_classifications: Sequence[str],
    true_responses: Sequence[str],
    refreshed: Sequence[Sequence[Hypothesis]],
    target: str,
) -> dict[str, Any]:
    scores = root_scores(
        current_classifications,
        auxiliary_classifications,
    )
    endpoints = [
        _truth_mass(root_hypotheses, target)
        for root_hypotheses in refreshed
    ]
    initial_endpoint = _truth_mass(initial, target)
    selected = {
        "model_criticism": _argmax(scores["model_criticism"]),
        "current_eig": _argmax(scores["current_eig"]),
        "augmented_eig": _argmax(scores["augmented_eig"]),
        "random": random.Random(RANDOM_CONTROL_SEED + task_id).randrange(
            QUESTION_COUNT
        ),
    }
    oracle_index = _argmax(endpoints)
    return {
        "task_id": task_id,
        "initial_unique_entity_count": unique_entity_count(initial),
        "unique_question_count": len(set(questions)),
        "non_unknown_true_response_count": sum(
            response != "U" for response in true_responses
        ),
        "semantic_distinct_auxiliary_count": len(auxiliary),
        "unique_auxiliary_entity_count": unique_entity_count(auxiliary),
        "refresh_particle_counts": [
            len(root_hypotheses) for root_hypotheses in refreshed
        ],
        "initial_truth_mass": initial_endpoint,
        "endpoint_range": max(endpoints) - min(endpoints),
        "oracle_root_index": oracle_index,
        "oracle_endpoint": endpoints[oracle_index],
        "selected_root_indices": selected,
        "selected_endpoints": {
            name: endpoints[index] for name, index in selected.items()
        },
        "score_endpoint_spearman": {
            name: spearman(values, endpoints)
            for name, values in scores.items()
        },
        "roots": [
            {
                "root_index": root_index,
                "current_eig": scores["current_eig"][root_index],
                "model_criticism": scores["model_criticism"][root_index],
                "augmented_eig": scores["augmented_eig"][root_index],
                "true_response": true_responses[root_index],
                "truth_mass": endpoints[root_index],
                "refresh_particle_count": len(refreshed[root_index]),
            }
            for root_index in range(QUESTION_COUNT)
        ],
    }


def _mean_finite(
    values: Sequence[float | None],
) -> float | None:
    finite = [value for value in values if value is not None]
    return statistics.fmean(finite) if finite else None


def _build_models(config: Config) -> tuple[ChatModel, ChatModel]:
    generator_spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    responder_spec = replace(
        config.model_pairs[0].answerer,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if generator_spec.model != GENERATOR_MODEL_ID:
        raise ValueError("validation config selects wrong generator")
    if responder_spec.model != RESPONDER_MODEL_ID:
        raise ValueError("validation config selects wrong responder")
    return (
        build_model_adapter(generator_spec, config),
        build_model_adapter(responder_spec, config),
    )


def run_validation(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    generator: ChatModel,
    responder: ChatModel,
) -> dict[str, Any]:
    _path, encrypted_rows = verify_source(source_root)
    if tuple(
        int(encrypted_rows[index]["id"]) for index in SCREEN_INDICES
    ) != EXPECTED_SCREEN_IDS:
        raise ValueError("fresh screen manifest does not match")
    visible = _decrypted_fields(encrypted_rows, ("question",))
    tasks = {
        index: {
            "task_id": int(visible[index]["id"]),
            "question": visible[index]["question"],
        }
        for index in SCREEN_INDICES
    }
    raw: dict[str, Any] = {
        "screen_indices": list(SCREEN_INDICES),
        "screen_ids": list(EXPECTED_SCREEN_IDS),
    }
    try:
        initial_messages = [
            _initial_messages(
                task_id=tasks[index]["task_id"],
                question=tasks[index]["question"],
                sample_index=sample_index,
            )
            for index in SCREEN_INDICES
            for sample_index in range(PARTICLE_COUNT)
        ]
        initial_raw = generator.chat_complete_messages_batched(
            initial_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["initial"] = initial_raw
        initial = {}
        for offset, index in enumerate(SCREEN_INDICES):
            initial[index] = [
                parse_hypothesis(response)
                for response in initial_raw[
                    offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
                ]
            ]
        enrolled = enroll_collapsed_tasks(initial)
        raw["enrolled_indices"] = enrolled
        raw["screen_unique_counts"] = {
            str(index): unique_entity_count(initial[index])
            for index in SCREEN_INDICES
        }
        if len(enrolled) != ENROLL_COUNT:
            usage = aggregate_usage(generator, responder)
            raw["screen_complete_without_endpoint_access"] = True
            _checkpoint(raw_path, raw)
            return {
                "schema_version": 1,
                "status": "screen_failed",
                "protocol": {
                    "interface_version": INTERFACE_VERSION,
                    "seed": SEED,
                    "screen_indices": list(SCREEN_INDICES),
                    "screen_ids": list(EXPECTED_SCREEN_IDS),
                    "enrolled_indices": enrolled,
                    "reasoning_requested": False,
                    "target_answers_loaded": False,
                    "hidden_contexts_loaded": False,
                },
                "metrics": {
                    "screen_count": len(SCREEN_INDICES),
                    "collapsed_support_count": len(enrolled),
                },
                "gates": {
                    "six_collapsed_supports_enrolled": False,
                    "all_pass": False,
                },
                "usage": usage,
            }

        question_messages = [
            _question_messages(
                task_id=tasks[index]["task_id"],
                question=tasks[index]["question"],
                hypotheses=initial[index],
                question_index=question_index,
            )
            for index in enrolled
            for question_index in range(QUESTION_COUNT)
        ]
        question_raw = generator.chat_complete_messages_batched(
            question_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=160,
        )
        raw["questions"] = question_raw
        questions = {}
        for offset, index in enumerate(enrolled):
            task_questions = [
                parse_question(response)
                for response in question_raw[
                    offset * QUESTION_COUNT : (offset + 1) * QUESTION_COUNT
                ]
            ]
            for question in task_questions:
                normalized_question = normalize_entity(question)
                if any(
                    normalize_entity(hypothesis.entity) in normalized_question
                    for hypothesis in initial[index]
                    if len(normalize_entity(hypothesis.entity)) >= 4
                ):
                    raise ValueError("question names a current candidate")
            questions[index] = task_questions

        current_classification_messages = [
            _classification_messages(
                hypothesis=hypothesis,
                questions=questions[index],
            )
            for index in enrolled
            for hypothesis in initial[index]
        ]
        current_classification_raw = generator.chat_complete_messages_batched(
            current_classification_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["current_classifications"] = current_classification_raw
        current_classifications = {}
        for offset, index in enumerate(enrolled):
            current_classifications[index] = [
                parse_classification_ascii_whitespace(response)
                for response in current_classification_raw[
                    offset * PARTICLE_COUNT : (offset + 1) * PARTICLE_COUNT
                ]
            ]

        auxiliary_messages = [
            _auxiliary_messages(
                task_id=tasks[index]["task_id"],
                question=tasks[index]["question"],
                initial=initial[index],
                sample_index=sample_index,
            )
            for index in enrolled
            for sample_index in range(AUXILIARY_PROPOSALS)
        ]
        auxiliary_raw = generator.chat_complete_messages_batched(
            auxiliary_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["auxiliary_proposals"] = auxiliary_raw
        proposals = {}
        for offset, index in enumerate(enrolled):
            proposals[index] = [
                parse_hypothesis(response)
                for response in auxiliary_raw[
                    offset
                    * AUXILIARY_PROPOSALS : (offset + 1)
                    * AUXILIARY_PROPOSALS
                ]
            ]

        validation_messages = [
            _semantic_validation_messages(
                current=initial[index],
                candidate=candidate,
            )
            for index in enrolled
            for candidate in proposals[index]
        ]
        validation_raw = generator.chat_complete_messages_batched(
            validation_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=8,
        )
        raw["semantic_validations"] = validation_raw
        auxiliary = {}
        for offset, index in enumerate(enrolled):
            decisions = [
                parse_semantic_distinctness(response)
                for response in validation_raw[
                    offset
                    * AUXILIARY_PROPOSALS : (offset + 1)
                    * AUXILIARY_PROPOSALS
                ]
            ]
            retained = [
                candidate
                for candidate, is_distinct in zip(
                    proposals[index],
                    decisions,
                    strict=True,
                )
                if is_distinct
            ][:AUXILIARY_RETAIN]
            if len(retained) != AUXILIARY_RETAIN:
                raise ValueError(
                    f"task {index} retains {len(retained)} semantically distinct "
                    f"auxiliary particles; {AUXILIARY_RETAIN} required"
                )
            auxiliary[index] = retained

        auxiliary_classification_messages = [
            _classification_messages(
                hypothesis=hypothesis,
                questions=questions[index],
            )
            for index in enrolled
            for hypothesis in auxiliary[index]
        ]
        auxiliary_classification_raw = generator.chat_complete_messages_batched(
            auxiliary_classification_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["auxiliary_classifications"] = auxiliary_classification_raw
        auxiliary_classifications = {}
        frozen_scores = {}
        for offset, index in enumerate(enrolled):
            auxiliary_classifications[index] = [
                parse_classification_ascii_whitespace(response)
                for response in auxiliary_classification_raw[
                    offset * AUXILIARY_RETAIN : (offset + 1) * AUXILIARY_RETAIN
                ]
            ]
            frozen_scores[index] = root_scores(
                current_classifications[index],
                auxiliary_classifications[index],
            )

        # Context is available only to the separate closed-mode responder.
        contexts = _decrypted_fields(encrypted_rows, ("context",))
        responder_messages = [
            _responder_messages(
                context=contexts[index]["context"],
                question=question,
            )
            for index in enrolled
            for question in questions[index]
        ]
        responder_raw = responder.chat_complete_messages_batched(
            responder_messages,
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["true_responses"] = responder_raw
        true_responses = {}
        for offset, index in enumerate(enrolled):
            true_responses[index] = [
                parse_responder_answer(response)
                for response in responder_raw[
                    offset * QUESTION_COUNT : (offset + 1) * QUESTION_COUNT
                ]
            ]

        refresh_messages = [
            _refresh_messages(
                task_id=tasks[index]["task_id"],
                original_question=tasks[index]["question"],
                clarification_question=questions[index][root_index],
                clarification_answer=true_responses[index][root_index],
                sample_index=sample_index,
            )
            for index in enrolled
            for root_index in range(QUESTION_COUNT)
            for sample_index in range(PARTICLE_COUNT)
        ]
        refresh_raw = generator.chat_complete_messages_batched(
            refresh_messages,
            temperature=0.7,
            block_size=config.openrouter_concurrency,
            max_new_tokens=400,
        )
        raw["refresh"] = refresh_raw
        refreshed = {}
        cursor = 0
        for index in enrolled:
            task_roots = []
            for _root_index in range(QUESTION_COUNT):
                root_hypotheses = []
                for _sample_index in range(PARTICLE_COUNT):
                    response = refresh_raw[cursor]
                    cursor += 1
                    try:
                        root_hypotheses.append(parse_hypothesis(response))
                    except Exception:
                        continue
                if len(root_hypotheses) < MIN_REFRESH_PARTICLES:
                    raise ValueError(
                        f"task {index} refresh retains "
                        f"{len(root_hypotheses)} particles"
                    )
                task_roots.append(root_hypotheses)
            refreshed[index] = task_roots
        raw["all_scores_and_target_blind_calls_complete"] = True
        _checkpoint(raw_path, raw)
        usage = aggregate_usage(generator, responder)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise ValidationExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(generator, responder),
        ) from exc

    # Exact target answers load only after all model calls and scores freeze.
    targets = _decrypted_fields(encrypted_rows, ("answer",))
    records = [
        score_validation_task(
            task_id=tasks[index]["task_id"],
            initial=initial[index],
            questions=questions[index],
            current_classifications=current_classifications[index],
            auxiliary=auxiliary[index],
            auxiliary_classifications=auxiliary_classifications[index],
            true_responses=true_responses[index],
            refreshed=refreshed[index],
            target=targets[index]["answer"],
        )
        for index in enrolled
    ]
    for index, record in zip(enrolled, records, strict=True):
        for name, values in frozen_scores[index].items():
            if [
                root[name] for root in record["roots"]
            ] != values:
                raise AssertionError("target loading changed a frozen score")

    mean_selected = {
        name: statistics.fmean(
            record["selected_endpoints"][name] for record in records
        )
        for name in ("model_criticism", "current_eig", "augmented_eig", "random")
    }
    mean_rho = {
        name: _mean_finite(
            [record["score_endpoint_spearman"][name] for record in records]
        )
        for name in ("model_criticism", "current_eig", "augmented_eig")
    }
    rankable_count = sum(record["endpoint_range"] > 0.0 for record in records)
    recovered_count = sum(record["oracle_endpoint"] >= 1 / PARTICLE_COUNT for record in records)
    omitted_count = sum(record["initial_truth_mass"] == 0.0 for record in records)
    dynamic_count = sum(record["endpoint_range"] >= 1 / PARTICLE_COUNT for record in records)
    robust_wins_current = sum(
        record["selected_endpoints"]["model_criticism"]
        > record["selected_endpoints"]["current_eig"]
        for record in records
    )
    robust_losses_current = sum(
        record["selected_endpoints"]["model_criticism"]
        < record["selected_endpoints"]["current_eig"]
        for record in records
    )
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_transport_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "six_collapsed_supports_enrolled": len(records) == ENROLL_COUNT,
        "all_enrolled_supports_collapsed": all(
            record["initial_unique_entity_count"] <= COLLAPSED_UNIQUE_MAX
            for record in records
        ),
        "at_least_three_unique_questions_each": all(
            record["unique_question_count"] >= 3 for record in records
        ),
        "at_least_two_non_unknown_responses_each": all(
            record["non_unknown_true_response_count"] >= 2
            for record in records
        ),
        "eight_semantically_distinct_auxiliary_each": all(
            record["semantic_distinct_auxiliary_count"] == AUXILIARY_RETAIN
            for record in records
        ),
        "all_refreshes_have_at_least_six_particles": all(
            min(record["refresh_particle_counts"]) >= MIN_REFRESH_PARTICLES
            for record in records
        ),
        "at_least_four_initial_target_omissions": omitted_count >= 4,
        "at_least_four_target_recoveries": recovered_count >= 4,
        "at_least_four_dynamic_endpoints": dynamic_count >= 4,
        "at_least_four_rankable_tasks": rankable_count >= 4,
        "mean_model_criticism_rho_at_least_0_20": (
            mean_rho["model_criticism"] is not None
            and mean_rho["model_criticism"] >= 0.20
        ),
        "model_criticism_gain_over_current_eig_at_least_0_02": (
            mean_selected["model_criticism"]
            - mean_selected["current_eig"]
            >= 0.02
        ),
        "model_criticism_gain_over_augmented_eig_at_least_0_02": (
            mean_selected["model_criticism"]
            - mean_selected["augmented_eig"]
            >= 0.02
        ),
        "model_criticism_gain_over_random_at_least_0_02": (
            mean_selected["model_criticism"] - mean_selected["random"]
            >= 0.02
        ),
        "model_criticism_wins_current_at_least_two": robust_wins_current >= 2,
        "model_criticism_loses_current_at_most_one": robust_losses_current <= 1,
        "model_criticism_selects_oracle_at_least_three": sum(
            record["selected_root_indices"]["model_criticism"]
            == record["oracle_root_index"]
            for record in records
        )
        >= 3,
        "cost_at_most_1_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "seed": SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "screen_indices": list(SCREEN_INDICES),
            "screen_ids": list(EXPECTED_SCREEN_IDS),
            "enrolled_indices": enrolled,
            "enrolled_ids": [tasks[index]["task_id"] for index in enrolled],
            "screen_count": len(SCREEN_INDICES),
            "enroll_count": ENROLL_COUNT,
            "collapsed_unique_max": COLLAPSED_UNIQUE_MAX,
            "auxiliary_proposals": AUXILIARY_PROPOSALS,
            "auxiliary_retain": AUXILIARY_RETAIN,
            "expected_requests": EXPECTED_REQUESTS,
            "generator_model": GENERATOR_MODEL_ID,
            "responder_model": RESPONDER_MODEL_ID,
            "reasoning_requested": False,
            "primary_score": "balanced_model_identity_mutual_information",
            "ascii_whitespace_compaction_only": True,
            "target_answers_loaded_after_all_model_calls": True,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "screen_count": len(SCREEN_INDICES),
            "enrolled_count": len(records),
            "initial_target_omission_count": omitted_count,
            "target_recovery_count": recovered_count,
            "dynamic_endpoint_count": dynamic_count,
            "rankable_task_count": rankable_count,
            "mean_selected_endpoint": mean_selected,
            "mean_score_endpoint_spearman": mean_rho,
            "model_criticism_wins_current_eig": robust_wins_current,
            "model_criticism_losses_current_eig": robust_losses_current,
        },
        "tasks": records,
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, role: str) -> None:
        self.role = role
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        patterns = ("YYYY", "NNNN", "UUUU", "YNUY")
        for messages in batch_messages:
            if self.role == "responder":
                responses.append("yes")
                continue
            request = json.loads(messages[-1]["content"])
            if "semantic_candidate" in request:
                responses.append("D")
            elif "questions" in request:
                entity = normalize_entity(request["candidate"]["entity"])
                responses.append(patterns[int(entity[-1]) % len(patterns)])
            elif "candidate_support" in request:
                responses.append(
                    f"Does it have fixture property {request['question_index']}?"
                )
            elif "outside_sample" in request:
                sample = int(request["outside_sample"])
                responses.append(
                    f"ENTITY: Fixture Alternative {sample}\n"
                    f"PROFILE: Alternative profile {sample}."
                )
            elif "clarification" in request:
                sample = int(request["interpretation_sample"])
                responses.append(
                    f"ENTITY: Fixture Refresh {sample}\n"
                    f"PROFILE: Refreshed profile {sample}."
                )
            else:
                sample = int(request["interpretation_sample"]) % 4
                responses.append(
                    f"ENTITY: Fixture Current {sample}\n"
                    f"PROFILE: Current profile {sample}."
                )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
            "http_attempts": self.requests,
            "retry_count": 0,
            "forced_exits": 0,
        }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.50
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    if args.dry_run:
        generator: ChatModel = DeterministicFixtureModel("generator")
        responder: ChatModel = DeterministicFixtureModel("responder")
    else:
        generator, responder = _build_models(config)
    try:
        payload = run_validation(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            generator=generator,
            responder=responder,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ValidationExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "VALIDATION_FAILURE.json", failure)
        raise
    output = args.output_dir / "VALIDATION.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
