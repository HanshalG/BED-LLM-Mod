#!/usr/bin/env python3
"""Run the frozen ClariQ path-dependent dynamic-support mechanics smoke."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts import browsecomp_plus_semantic_mechanics as flat
from scripts.clariq_multisample_likelihood_development import (
    _build_model,
    _checkpoint,
    _load_split_archive,
    _load_tar_pickle,
    _usage,
)
from scripts.clariq_topic_level_train_opportunity import (
    _history_key,
    analyze_topics,
    verify_source,
)


INTERFACE_VERSION = "clariq-dynamic-support-smoke-1"
MANIFEST_SHA256 = (
    "8871d72aca825aa5b34cf52295eb93fe79caf631c0bba070d82abf6c3bec698d"
)
MODEL_ID = "openai/gpt-5.4"
MECHANICS_TOPIC_ID = "38"
HYPOTHESIS_COUNT = 8
EXPECTED_REQUESTS = 40
TEMPERATURE = 0.7
MAX_OUTPUT_TOKENS = 1024
MAX_COST_USD = 0.50
PROJECTED_COST_USD = 0.25
REQUEST_SEED = 24_412
RANDOM_CONTROL_SEED = 24_413
SHUFFLE_CONTROL_SEED = 24_414


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Support:
    question_ids: tuple[str, ...]
    hypotheses: tuple[str, ...]
    probabilities: tuple[float, ...]
    predictions: tuple[str, ...]

    def question_index(self, question_id: str) -> int:
        try:
            return self.question_ids.index(question_id)
        except ValueError as exc:
            raise ValueError(
                f"support does not predict question {question_id}"
            ) from exc


class DeterministicFixtureModel:
    def __init__(self) -> None:
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
        masses = (25, 20, 15, 12, 10, 8, 6, 4)
        for messages in batch_messages:
            payload = json.loads(messages[-1]["content"])
            questions = payload["questions"]
            salt = json.dumps(
                {
                    "request": payload["initial_request"],
                    "observed": payload.get("observed"),
                },
                sort_keys=True,
            )
            lines = []
            for hypothesis_index in range(HYPOTHESIS_COUNT):
                predictions = []
                for question in questions:
                    options = question["response_options"]
                    digest = hashlib.sha256(
                        (
                            salt
                            + question["question_id"]
                            + str(hypothesis_index)
                        ).encode("utf-8")
                    ).digest()
                    predictions.append(
                        options[digest[0] % len(options)]["code"]
                    )
                descriptor = hashlib.sha256(
                    f"{salt}:{hypothesis_index}".encode("utf-8")
                ).hexdigest()[:12]
                lines.append(
                    f"H{hypothesis_index + 1:02d}|"
                    f"{masses[hypothesis_index]}|"
                    f"{''.join(predictions)}|"
                    f"plausible intent {descriptor}"
                )
            responses.append("\n".join(lines))
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
        }


def load_manifest(path: Path, stage: str) -> dict[str, Any]:
    if hashlib.sha256(path.read_bytes()).hexdigest() != MANIFEST_SHA256:
        raise ValueError("ClariQ dynamic-support manifest hash changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest["status"] != "passed":
        raise ValueError("ClariQ dynamic-support manifest did not pass")
    tasks = manifest["stages"].get(stage)
    if not isinstance(tasks, list) or len(tasks) != 1:
        raise ValueError("ClariQ dynamic-support stage changed")
    task = tasks[0]
    if stage == "mechanics":
        if task["topic_id"] != MECHANICS_TOPIC_ID:
            raise ValueError("ClariQ mechanics topic changed")
        if task["expected_model_requests"] != EXPECTED_REQUESTS:
            raise ValueError("ClariQ mechanics request count changed")
    return task


def _question_lookup(task: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    return {
        str(question["question_id"]): dict(question)
        for question in task["question_bank"]
    }


def _question_payload(
    task: Mapping[str, Any],
    question_ids: Sequence[str],
) -> list[dict[str, Any]]:
    lookup = _question_lookup(task)
    return [lookup[question_id] for question_id in question_ids]


def support_messages(
    task: Mapping[str, Any],
    *,
    question_ids: Sequence[str],
    prior_support: Support | None = None,
    observed: Mapping[str, str] | None = None,
) -> list[dict[str, str]]:
    questions = _question_payload(task, question_ids)
    if observed is None:
        instruction = (
            "Generate an open-world categorical belief over the user's latent "
            "information need. Return exactly eight distinct plausible intent "
            "hypotheses. Assign each a nonnegative integer mass from 0 to 100; "
            "the masses are unnormalized and need not sum to 100, but at least "
            "one must be positive. For every supplied clarification question "
            "in order, predict the response-option code that would follow if "
            "that hypothesis were true."
        )
    else:
        instruction = (
            "Regenerate the open-world categorical belief after the observed "
            "clarification answer. Preserve, revise, add, or drop prior "
            "hypotheses as warranted; do not merely filter or copy the prior "
            "list. Return exactly eight distinct plausible remaining intent "
            "hypotheses with nonnegative integer masses from 0 to 100. Masses "
            "are unnormalized and need not sum to 100, but at least one must "
            "be positive. For every legal follow-up question in order, predict "
            "the response-option code that would follow if that hypothesis "
            "were true."
        )
    system = (
        instruction
        + " Output exactly eight ordered lines as "
        "H01|mass|response-codes|short intent hypothesis through H08. The "
        "response-code string must contain exactly one listed code per "
        "question in input order. Do not use the | character inside a "
        "hypothesis. Return no JSON, markdown, explanation, blank lines, or "
        "reasoning."
    )
    payload: dict[str, Any] = {
        "initial_request": task["initial_request"],
        "questions": questions,
        "question_order": list(question_ids),
        "observed": dict(observed) if observed is not None else None,
        "prior_support": (
            [
                {
                    "mass": round(probability * 100, 6),
                    "hypothesis": hypothesis,
                }
                for hypothesis, probability in zip(
                    prior_support.hypotheses,
                    prior_support.probabilities,
                    strict=True,
                )
            ]
            if prior_support is not None
            else None
        ),
        "exact_output_grammar": [
            (
                f"H{index:02d}|0..100 unnormalized mass|"
                f"{len(question_ids)} response codes|short intent hypothesis"
            )
            for index in range(1, HYPOTHESIS_COUNT + 1)
        ],
    }
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(payload, separators=(",", ":")),
        },
    ]


def parse_support(
    text: str,
    *,
    questions: Sequence[Mapping[str, Any]],
) -> Support:
    lines = flat._response_lines(text)
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError("wrong ClariQ support line count")
    question_ids = tuple(str(row["question_id"]) for row in questions)
    allowed_codes = [
        {str(option["code"]) for option in row["response_options"]}
        for row in questions
    ]
    masses: list[int] = []
    hypotheses: list[str] = []
    predictions: list[str] = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|")
        if len(parts) != 4 or parts[0] != f"H{index:02d}":
            raise ValueError("invalid ClariQ support line")
        masses.append(
            flat._canonical_integer(
                parts[1],
                minimum=0,
                maximum=100,
            )
        )
        prediction = parts[2]
        if len(prediction) != len(questions):
            raise ValueError("ClariQ response-code length changed")
        if any(
            code not in allowed
            for code, allowed in zip(
                prediction,
                allowed_codes,
                strict=True,
            )
        ):
            raise ValueError("ClariQ response code is outside its alphabet")
        predictions.append(prediction)
        hypothesis = " ".join(parts[3].split())
        if not hypothesis:
            raise ValueError("ClariQ hypothesis is empty")
        hypotheses.append(hypothesis)
    normalized = {flat.normalize_answer(value) for value in hypotheses}
    if "" in normalized or len(normalized) != HYPOTHESIS_COUNT:
        raise ValueError("ClariQ hypotheses must be normalized-distinct")
    total = sum(masses)
    if total <= 0:
        raise ValueError("ClariQ support must have positive total mass")
    return Support(
        question_ids=question_ids,
        hypotheses=tuple(hypotheses),
        probabilities=tuple(mass / total for mass in masses),
        predictions=tuple(predictions),
    )


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def response_distribution(
    support: Support,
    question_id: str,
    *,
    condition: tuple[str, str] | None = None,
) -> dict[str, float]:
    question_index = support.question_index(question_id)
    selected = list(range(len(support.hypotheses)))
    if condition is not None:
        condition_id, condition_code = condition
        condition_index = support.question_index(condition_id)
        selected = [
            index
            for index in selected
            if support.predictions[index][condition_index] == condition_code
        ]
    total = sum(support.probabilities[index] for index in selected)
    if total <= 0.0:
        return {}
    distribution: dict[str, float] = {}
    for index in selected:
        code = support.predictions[index][question_index]
        distribution[code] = (
            distribution.get(code, 0.0)
            + support.probabilities[index] / total
        )
    return distribution


def information_gain(
    support: Support,
    question_id: str,
    *,
    condition: tuple[str, str] | None = None,
) -> float:
    return entropy(
        list(
            response_distribution(
                support,
                question_id,
                condition=condition,
            ).values()
        )
    )


def _select(scores: Mapping[str, float]) -> str:
    return min(scores, key=lambda key: (-scores[key], key))


def policy_scores(
    task: Mapping[str, Any],
    initial_support: Support,
    branch_supports: Mapping[tuple[str, str], Support],
) -> dict[str, Any]:
    roots = {
        str(root["question_id"]): root for root in task["roots"]
    }
    myopic: dict[str, float] = {}
    fixed: dict[str, float] = {}
    dynamic: dict[str, float] = {}
    shuffled: dict[str, float] = {}
    fixed_followups: dict[str, dict[str, str]] = {}
    dynamic_followups: dict[str, dict[str, str]] = {}
    branch_dynamic_gains: dict[str, dict[str, float]] = {}
    branch_probabilities: dict[str, dict[str, float]] = {}
    for root_id, root in roots.items():
        immediate = information_gain(initial_support, root_id)
        myopic[root_id] = immediate
        root_distribution = response_distribution(initial_support, root_id)
        fixed_continuation = 0.0
        dynamic_continuation = 0.0
        fixed_followups[root_id] = {}
        dynamic_followups[root_id] = {}
        branch_dynamic_gains[root_id] = {}
        branch_probabilities[root_id] = {}
        for branch in root["branches"]:
            code = str(branch["response_code"])
            probability = root_distribution.get(code, 0.0)
            branch_probabilities[root_id][code] = probability
            followups = [
                str(value)
                for value in branch["legal_followup_question_ids"]
            ]
            fixed_options = {
                followup_id: information_gain(
                    initial_support,
                    followup_id,
                    condition=(root_id, code),
                )
                for followup_id in followups
            }
            fixed_id = _select(fixed_options)
            fixed_followups[root_id][code] = fixed_id
            fixed_continuation += probability * fixed_options[fixed_id]

            branch_support = branch_supports[(root_id, code)]
            dynamic_options = {
                followup_id: information_gain(
                    branch_support,
                    followup_id,
                )
                for followup_id in followups
            }
            dynamic_id = _select(dynamic_options)
            dynamic_followups[root_id][code] = dynamic_id
            gain = dynamic_options[dynamic_id]
            branch_dynamic_gains[root_id][code] = gain
            dynamic_continuation += probability * gain
        fixed[root_id] = immediate + fixed_continuation
        dynamic[root_id] = immediate + dynamic_continuation

        codes = sorted(branch_dynamic_gains[root_id])
        moved = list(codes)
        seed = int.from_bytes(
            hashlib.sha256(
                f"{SHUFFLE_CONTROL_SEED}:{root_id}".encode("utf-8")
            ).digest()[:8],
            "big",
        )
        random.Random(seed).shuffle(moved)
        if len(moved) > 1 and moved == codes:
            moved = moved[1:] + moved[:1]
        shuffled_continuation = sum(
            branch_probabilities[root_id][code]
            * branch_dynamic_gains[root_id][moved[index]]
            for index, code in enumerate(codes)
        )
        shuffled[root_id] = immediate + shuffled_continuation
    return {
        "myopic_scores": myopic,
        "fixed_depth_two_scores": fixed,
        "dynamic_depth_two_scores": dynamic,
        "shuffled_dynamic_scores": shuffled,
        "fixed_best_followups": fixed_followups,
        "dynamic_best_followups": dynamic_followups,
        "branch_dynamic_gains": branch_dynamic_gains,
        "branch_probabilities": branch_probabilities,
        "myopic_question_id": _select(myopic),
        "fixed_depth_two_question_id": _select(fixed),
        "dynamic_depth_two_question_id": _select(dynamic),
        "shuffled_dynamic_question_id": _select(shuffled),
    }


def spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    from scripts.clariq_multisample_likelihood_development import (
        spearman as old_spearman,
    )

    return old_spearman(values, targets)


def _support_payload(support: Support) -> dict[str, Any]:
    return {
        "question_ids": list(support.question_ids),
        "hypotheses": [
            {
                "hypothesis": hypothesis,
                "probability": probability,
                "response_codes": prediction,
            }
            for hypothesis, probability, prediction in zip(
                support.hypotheses,
                support.probabilities,
                support.predictions,
                strict=True,
            )
        ],
    }


def _official_profiles(
    task: Mapping[str, Any],
    synthetic: Mapping[Any, Mapping[str, Any]],
) -> tuple[list[str], dict[tuple[str, str], list[str]]]:
    topic_id = int(task["topic_id"])
    question_lookup = _question_lookup(task)
    answer_code = {
        question_id: {
            str(option["answer"]): str(option["code"])
            for option in question["response_options"]
        }
        for question_id, question in question_lookup.items()
    }
    question_id_by_text = {
        str(question["question"]): question_id
        for question_id, question in question_lookup.items()
    }
    states: dict[
        tuple[int, str, tuple[tuple[str, str], ...]], Any
    ] = {}
    answers: dict[
        tuple[
            tuple[int, str, tuple[tuple[str, str], ...]],
            str,
        ],
        str,
    ] = {}
    facets: set[str] = set()
    for row in synthetic.values():
        if int(row["topic_id"]) != topic_id:
            continue
        history = _history_key(row["conversation_context"])
        facet_id = str(row["facet_id"])
        state = (topic_id, facet_id, history)
        states[state] = row["context_id"]
        answers[(state, str(row["question"]))] = str(row["answer"])
        if not history:
            facets.add(facet_id)
    root_ids = tuple(str(root["question_id"]) for root in task["roots"])
    initial_profiles = []
    branch_profiles: dict[tuple[str, str], list[str]] = {}
    for facet_id in sorted(facets):
        state = (topic_id, facet_id, ())
        profile = "".join(
            answer_code[question_id][
                answers[(state, question_lookup[question_id]["question"])]
            ]
            for question_id in root_ids
        )
        initial_profiles.append(profile)
    for root in task["roots"]:
        root_id = str(root["question_id"])
        root_question = str(question_lookup[root_id]["question"])
        for branch in root["branches"]:
            code = str(branch["response_code"])
            followups = [
                str(value)
                for value in branch["legal_followup_question_ids"]
            ]
            profiles = []
            for facet_id in sorted(facets):
                initial_state = (topic_id, facet_id, ())
                root_answer = answers[(initial_state, root_question)]
                if answer_code[root_id][root_answer] != code:
                    continue
                successor = (
                    topic_id,
                    facet_id,
                    ((root_question, root_answer),),
                )
                profile = "".join(
                    answer_code[followup_id][
                        answers[
                            (
                                successor,
                                question_lookup[followup_id]["question"],
                            )
                        ]
                    ]
                    for followup_id in followups
                )
                profiles.append(profile)
            branch_profiles[(root_id, code)] = profiles
    return initial_profiles, branch_profiles


def run_smoke(
    config: Config,
    *,
    source_root: Path,
    manifest_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    task = load_manifest(manifest_path, "mechanics")
    question_ids = tuple(
        str(root["question_id"]) for root in task["roots"]
    )
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "manifest_sha256": MANIFEST_SHA256,
        "topic_id": task["topic_id"],
        "development_endpoints_loaded": False,
        "mechanics_endpoints_loaded": False,
    }
    try:
        initial_responses = model.chat_complete_messages_batched(
            [
                support_messages(
                    task,
                    question_ids=question_ids,
                )
            ],
            temperature=TEMPERATURE,
            block_size=config.openrouter_concurrency,
            max_new_tokens=MAX_OUTPUT_TOKENS,
        )
        raw["initial_responses"] = initial_responses
        _checkpoint(raw_path, raw)
        if len(initial_responses) != 1:
            raise ValueError("ClariQ initial response count changed")
        initial_support = parse_support(
            initial_responses[0],
            questions=_question_payload(task, question_ids),
        )

        branch_keys = [
            (str(root["question_id"]), str(branch["response_code"]))
            for root in task["roots"]
            for branch in root["branches"]
        ]
        random.Random(REQUEST_SEED).shuffle(branch_keys)
        root_lookup = {
            str(root["question_id"]): root for root in task["roots"]
        }
        question_lookup = _question_lookup(task)
        branch_lookup = {
            (str(root["question_id"]), str(branch["response_code"])): branch
            for root in task["roots"]
            for branch in root["branches"]
        }
        branch_responses = model.chat_complete_messages_batched(
            [
                support_messages(
                    task,
                    question_ids=branch_lookup[key][
                        "legal_followup_question_ids"
                    ],
                    prior_support=initial_support,
                    observed={
                        "question_id": key[0],
                        "question": question_lookup[key[0]]["question"],
                        "response_code": key[1],
                        "answer": branch_lookup[key]["answer"],
                    },
                )
                for key in branch_keys
            ],
            temperature=TEMPERATURE,
            block_size=config.openrouter_concurrency,
            max_new_tokens=MAX_OUTPUT_TOKENS,
        )
        raw["branch_keys"] = branch_keys
        raw["branch_responses"] = branch_responses
        _checkpoint(raw_path, raw)
        if len(branch_responses) != len(branch_keys):
            raise ValueError("ClariQ branch response count changed")
        branch_supports = {
            key: parse_support(
                response,
                questions=_question_payload(
                    task,
                    branch_lookup[key]["legal_followup_question_ids"],
                ),
            )
            for key, response in zip(
                branch_keys,
                branch_responses,
                strict=True,
            )
        }
        scores = policy_scores(task, initial_support, branch_supports)
        frozen = {
            "initial_support": _support_payload(initial_support),
            "branch_supports": {
                f"{root_id}:{code}": _support_payload(support)
                for (root_id, code), support in sorted(
                    branch_supports.items()
                )
            },
            "policy": scores,
        }
        raw["frozen_before_endpoint"] = frozen
        _checkpoint(raw_path, raw)
        usage = _usage(model)
        if usage["adapter_cost_usd"] > MAX_COST_USD:
            raise ValueError("ClariQ smoke exceeded its cost cap")
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    # Every LLM response, support, score, and selected root is frozen above.
    paths = verify_source(source_root)
    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )["NDCG20"]
    question_id_by_text = {
        (int(task["topic_id"]), str(question["question"])): str(
            question["question_id"]
        )
        for question in task["question_bank"]
    }
    external = analyze_topics(
        synthetic,
        evaluation,
        question_id_by_text,
        [str(task["topic_id"])],
    )
    if len(external["records"]) != 1:
        raise ValueError("ClariQ mechanics endpoint did not reproduce")
    endpoint_record = external["records"][0]
    endpoints = {
        str(root["question_id"]): float(root["terminal_utility"])
        for root in endpoint_record["roots"]
    }
    if set(endpoints) != set(question_ids):
        raise ValueError("ClariQ endpoint root set changed")
    random_id = random.Random(RANDOM_CONTROL_SEED).choice(
        sorted(question_ids)
    )
    selected_ids = {
        "myopic": scores["myopic_question_id"],
        "fixed_depth_two": scores["fixed_depth_two_question_id"],
        "dynamic_depth_two": scores["dynamic_depth_two_question_id"],
        "shuffled_dynamic": scores["shuffled_dynamic_question_id"],
        "random": random_id,
    }
    selected_endpoints = {
        name: endpoints[question_id]
        for name, question_id in selected_ids.items()
    }
    score_names = {
        "myopic": "myopic_scores",
        "fixed_depth_two": "fixed_depth_two_scores",
        "dynamic_depth_two": "dynamic_depth_two_scores",
        "shuffled_dynamic": "shuffled_dynamic_scores",
    }
    correlations = {
        name: spearman(
            [scores[key][question_id] for question_id in question_ids],
            [endpoints[question_id] for question_id in question_ids],
        )
        for name, key in score_names.items()
    }
    positive_initial_descriptions = {
        flat.normalize_answer(hypothesis)
        for hypothesis, probability in zip(
            initial_support.hypotheses,
            initial_support.probabilities,
            strict=True,
        )
        if probability > 0.0
    }
    branch_change_count = 0
    branch_profile_diversity_count = 0
    positive_continuation_count = 0
    for key, support in branch_supports.items():
        positive_descriptions = {
            flat.normalize_answer(hypothesis)
            for hypothesis, probability in zip(
                support.hypotheses,
                support.probabilities,
                strict=True,
            )
            if probability > 0.0
        }
        if positive_descriptions != positive_initial_descriptions:
            branch_change_count += 1
        prediction_profiles = {
            prediction
            for probability, prediction in zip(
                support.probabilities,
                support.predictions,
                strict=True,
            )
            if probability > 0.0
        }
        if len(prediction_profiles) >= 2:
            branch_profile_diversity_count += 1
        root_id, code = key
        if scores["branch_dynamic_gains"][root_id][code] > 1e-12:
            positive_continuation_count += 1

    initial_profiles, official_branch_profiles = _official_profiles(
        task,
        synthetic,
    )
    positive_initial_predictions = {
        prediction
        for probability, prediction in zip(
            initial_support.probabilities,
            initial_support.predictions,
            strict=True,
        )
        if probability > 0.0
    }
    initial_profile_coverage = sum(
        profile in positive_initial_predictions
        for profile in initial_profiles
    ) / len(initial_profiles)
    branch_coverages = []
    for key, profiles in official_branch_profiles.items():
        support = branch_supports[key]
        generated = {
            prediction
            for probability, prediction in zip(
                support.probabilities,
                support.predictions,
                strict=True,
            )
            if probability > 0.0
        }
        branch_coverages.extend(profile in generated for profile in profiles)
    branch_profile_coverage = (
        sum(branch_coverages) / len(branch_coverages)
        if branch_coverages
        else 0.0
    )

    dynamic_scores = list(scores["dynamic_depth_two_scores"].values())
    myopic_scores = list(scores["myopic_scores"].values())
    future_values = [
        scores["dynamic_depth_two_scores"][question_id]
        - scores["myopic_scores"][question_id]
        for question_id in question_ids
    ]
    dynamic_root = selected_ids["dynamic_depth_two"]
    benchmark_endpoint = max(
        selected_endpoints["myopic"],
        selected_endpoints["fixed_depth_two"],
    )
    metrics = {
        "branch_support_change_count": branch_change_count,
        "branch_profile_diversity_count": (
            branch_profile_diversity_count
        ),
        "positive_continuation_count": positive_continuation_count,
        "initial_profile_coverage": initial_profile_coverage,
        "branch_profile_coverage": branch_profile_coverage,
        "myopic_score_range": max(myopic_scores) - min(myopic_scores),
        "dynamic_future_range": max(future_values) - min(future_values),
        "dynamic_score_range": max(dynamic_scores) - min(dynamic_scores),
        "maximum_dynamic_fixed_score_difference": max(
            abs(
                scores["dynamic_depth_two_scores"][question_id]
                - scores["fixed_depth_two_scores"][question_id]
            )
            for question_id in question_ids
        ),
        "selected_question_ids": selected_ids,
        "selected_terminal_ndcg20": selected_endpoints,
        "score_terminal_ndcg20_spearman": correlations,
        "dynamic_gain_over_myopic": (
            selected_endpoints["dynamic_depth_two"]
            - selected_endpoints["myopic"]
        ),
        "dynamic_gain_over_fixed": (
            selected_endpoints["dynamic_depth_two"]
            - selected_endpoints["fixed_depth_two"]
        ),
        "dynamic_gain_over_shuffled": (
            selected_endpoints["dynamic_depth_two"]
            - selected_endpoints["shuffled_dynamic"]
        ),
    }
    gates = {
        "exact_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_http_attempts": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_supports_parse": (
            len(branch_supports) + 1 == EXPECTED_REQUESTS
        ),
        "branch_support_changes_at_least_35": branch_change_count >= 35,
        "branch_profile_diversity_at_least_30": (
            branch_profile_diversity_count >= 30
        ),
        "positive_continuations_at_least_30": (
            positive_continuation_count >= 30
        ),
        "myopic_score_range_at_least_0_05": (
            metrics["myopic_score_range"] >= 0.05
        ),
        "dynamic_future_range_at_least_0_05": (
            metrics["dynamic_future_range"] >= 0.05
        ),
        "dynamic_fixed_scores_differ": (
            metrics["maximum_dynamic_fixed_score_difference"] >= 0.02
        ),
        "dynamic_changes_root_from_myopic": (
            dynamic_root != selected_ids["myopic"]
        ),
        "dynamic_spearman_at_least_0_20": (
            correlations["dynamic_depth_two"] >= 0.20
        ),
        "dynamic_spearman_beats_myopic_and_fixed_by_0_02": (
            correlations["dynamic_depth_two"]
            >= max(
                correlations["myopic"],
                correlations["fixed_depth_two"],
            )
            + 0.02
        ),
        "dynamic_endpoint_strictly_beats_myopic_and_fixed": (
            selected_endpoints["dynamic_depth_two"]
            > benchmark_endpoint + 1e-12
        ),
        "dynamic_endpoint_nonworse_than_shuffled": (
            selected_endpoints["dynamic_depth_two"]
            >= selected_endpoints["shuffled_dynamic"] - 1e-12
        ),
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "model": MODEL_ID,
            "topic_id": task["topic_id"],
            "hypothesis_count": HYPOTHESIS_COUNT,
            "temperature": TEMPERATURE,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
            "expected_requests": EXPECTED_REQUESTS,
            "request_seed": REQUEST_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "shuffle_control_seed": SHUFFLE_CONTROL_SEED,
            "reasoning_requested": False,
            "mechanics_selection_is_endpoint_disclosed": True,
            "all_scores_frozen_before_endpoint": True,
            "development_endpoints_loaded": False,
            "holdout_content_or_endpoints_loaded": False,
            "repairs_or_reissues": 0,
        },
        "task": task,
        "initial_support": _support_payload(initial_support),
        "branch_supports": {
            f"{root_id}:{code}": _support_payload(support)
            for (root_id, code), support in sorted(branch_supports.items())
        },
        "policy": scores,
        "endpoint_by_question_id": endpoints,
        "metrics": metrics,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = EXPECTED_REQUESTS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_smoke(
            config,
            source_root=args.source_root,
            manifest_path=args.manifest,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "manifest_sha256": MANIFEST_SHA256,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "SMOKE_FAILURE.json", failure)
        raise
    _checkpoint(args.output_dir / "SMOKE.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
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
