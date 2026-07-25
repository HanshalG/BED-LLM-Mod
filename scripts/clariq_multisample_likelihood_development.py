#!/usr/bin/env python3
"""Evaluate multisample LLM likelihood planning on fresh ClariQ topics."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.clariq_topic_level_train_opportunity import (
    _load_split_archive,
    _load_tar_pickle,
    analyze_topics,
    build_split,
    verify_source,
)
from scripts.pscon_binary_query_serving_smoke import parse_labels


INTERFACE_VERSION = "clariq-multisample-likelihood-development-1"
MODEL_ID = "openai/gpt-5.4"
SELECTED_TOPIC_IDS = ("46", "177", "117")
EXPECTED_FACET_COUNTS = {"46": 3, "177": 4, "117": 3}
EXPECTED_QUESTION_COUNTS = {"46": 14, "177": 14, "117": 15}
SAMPLES_PER_QUESTION = 5
EXPECTED_REQUESTS = 215
REQUEST_SEED = 24_400
RANDOM_CONTROL_SEED = 24_401
TEMPERATURE = 0.7
SMOOTHING_ALPHA = 0.5
OUTCOMES = ("Y", "N", "U")
MAX_COST_USD = 0.50


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class DevelopmentExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


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
        for request_index, messages in enumerate(batch_messages):
            payload = json.loads(messages[-1]["content"])
            question = payload["clarification_question"]
            digest = hashlib.sha256(question.encode()).digest()
            responses.append(
                "".join(
                    OUTCOMES[
                        (
                            digest[facet_index % len(digest)]
                            + request_index
                            + facet_index
                        )
                        % len(OUTCOMES)
                    ]
                    for facet_index in range(len(payload["candidate_facets"]))
                )
            )
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


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _usage(model: ChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(
            snapshot.get("adapter_reasoning_tokens", 0)
        ),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "model": snapshot,
    }


def _nonthinking_spec(spec: Any) -> Any:
    return replace(
        spec,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_model(config: Config) -> ChatModel:
    spec = _nonthinking_spec(config.model_pairs[0].questioner)
    if spec.model != MODEL_ID:
        raise ValueError("ClariQ development config selects the wrong model")
    return build_model_adapter(spec, config)


def load_visible_tasks(train_path: Path) -> dict[str, dict[str, Any]]:
    import csv

    with train_path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    split = build_split(row["topic_id"] for row in rows)
    if any(topic_id not in split["development"] for topic_id in SELECTED_TOPIC_IDS):
        raise ValueError("selected ClariQ topic is outside development split")

    tasks: dict[str, dict[str, Any]] = {}
    for topic_id in SELECTED_TOPIC_IDS:
        selected = [row for row in rows if row["topic_id"] == topic_id]
        facets: dict[str, str] = {}
        questions: dict[str, str] = {}
        requests = {row["initial_request"] for row in selected}
        if len(requests) != 1:
            raise ValueError("ClariQ topic has multiple initial requests")
        for row in selected:
            old_facet = facets.setdefault(row["facet_id"], row["facet_desc"])
            old_question = questions.setdefault(
                row["question_id"], row["question"]
            )
            if old_facet != row["facet_desc"]:
                raise ValueError("ClariQ facet description changed")
            if old_question != row["question"]:
                raise ValueError("ClariQ question text changed")
        if len(facets) != EXPECTED_FACET_COUNTS[topic_id]:
            raise ValueError("ClariQ facet count changed")
        if len(questions) != EXPECTED_QUESTION_COUNTS[topic_id]:
            raise ValueError("ClariQ question count changed")
        tasks[topic_id] = {
            "topic_id": topic_id,
            "initial_request": requests.pop(),
            "facets": [
                {"facet_id": facet_id, "description": facets[facet_id]}
                for facet_id in sorted(facets)
            ],
            "questions": [
                {
                    "question_id": question_id,
                    "question": questions[question_id],
                }
                for question_id in sorted(questions)
            ],
        }
    return tasks


def likelihood_messages(
    task: dict[str, Any],
    question: dict[str, str],
) -> list[dict[str, str]]:
    payload = {
        "initial_request": task["initial_request"],
        "clarification_question": question["question"],
        "candidate_facets": task["facets"],
    }
    return [
        {
            "role": "system",
            "content": (
                "Act as a semantic response model for clarification. For each "
                "candidate user facet, predict the coarse answer to the supplied "
                "question. Use Y when that facet implies an affirmative answer, N "
                "when it implies a negative answer, and U when the facet does not "
                "determine a yes/no answer or the question is open-ended. Output "
                "exactly one Y/N/U character per facet in input order, with no "
                "spaces, punctuation, explanation, or extra text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def build_likelihood(
    maps: Sequence[str],
    facet_count: int,
    *,
    alpha: float = SMOOTHING_ALPHA,
) -> list[list[float]]:
    if not maps or any(len(value) != facet_count for value in maps):
        raise ValueError("likelihood maps have inconsistent shape")
    denominator = len(maps) + alpha * len(OUTCOMES)
    table = []
    for facet_index in range(facet_count):
        counts = Counter(value[facet_index] for value in maps)
        table.append(
            [
                (counts[outcome] + alpha) / denominator
                for outcome in OUTCOMES
            ]
        )
    return table


def entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0.0
    )


def information_gain(
    prior: Sequence[float],
    likelihood: Sequence[Sequence[float]],
) -> tuple[float, list[float], list[list[float]]]:
    predictive = [
        sum(
            prior[facet_index] * likelihood[facet_index][outcome_index]
            for facet_index in range(len(prior))
        )
        for outcome_index in range(len(OUTCOMES))
    ]
    posteriors = []
    expected_entropy = 0.0
    for outcome_index, probability in enumerate(predictive):
        if probability <= 0.0:
            posterior = [0.0] * len(prior)
        else:
            posterior = [
                prior[facet_index]
                * likelihood[facet_index][outcome_index]
                / probability
                for facet_index in range(len(prior))
            ]
            expected_entropy += probability * entropy(posterior)
        posteriors.append(posterior)
    return entropy(prior) - expected_entropy, predictive, posteriors


def policy_scores(
    likelihoods: dict[str, list[list[float]]],
) -> dict[str, Any]:
    if len(likelihoods) < 2:
        raise ValueError("ClariQ policy requires at least two questions")
    facet_count = len(next(iter(likelihoods.values())))
    prior = [1.0 / facet_count] * facet_count
    myopic_scores = {}
    depth_two_scores = {}
    best_followups = {}
    for question_id, likelihood in likelihoods.items():
        immediate, predictive, posteriors = information_gain(
            prior, likelihood
        )
        myopic_scores[question_id] = immediate
        continuation_value = 0.0
        branch_followups = []
        for probability, posterior in zip(predictive, posteriors):
            options = [
                (
                    information_gain(posterior, candidate_likelihood)[0],
                    candidate_id,
                )
                for candidate_id, candidate_likelihood in likelihoods.items()
                if candidate_id != question_id
            ]
            best_gain, best_id = min(
                options,
                key=lambda item: (-item[0], item[1]),
            )
            continuation_value += probability * best_gain
            branch_followups.append(best_id)
        depth_two_scores[question_id] = immediate + continuation_value
        best_followups[question_id] = dict(
            zip(OUTCOMES, branch_followups)
        )

    def selected(scores: dict[str, float]) -> str:
        return min(scores, key=lambda key: (-scores[key], key))

    return {
        "myopic_scores": myopic_scores,
        "depth_two_scores": depth_two_scores,
        "best_followups": best_followups,
        "myopic_question_id": selected(myopic_scores),
        "depth_two_question_id": selected(depth_two_scores),
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for index in order[start:end]:
            ranks[index] = average
        start = end
    return ranks


def spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    if len(values) != len(targets) or len(values) < 2:
        raise ValueError("Spearman inputs have inconsistent shape")
    first = _average_ranks(values)
    second = _average_ranks(targets)
    first_mean = sum(first) / len(first)
    second_mean = sum(second) / len(second)
    numerator = sum(
        (left - first_mean) * (right - second_mean)
        for left, right in zip(first, second)
    )
    first_scale = math.sqrt(
        sum((value - first_mean) ** 2 for value in first)
    )
    second_scale = math.sqrt(
        sum((value - second_mean) ** 2 for value in second)
    )
    if first_scale == 0.0 or second_scale == 0.0:
        return 0.0
    return numerator / (first_scale * second_scale)


def run_development(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    model: ChatModel,
    tasks_override: dict[str, dict[str, Any]] | None = None,
    expected_requests: int = EXPECTED_REQUESTS,
    interface_version: str = INTERFACE_VERSION,
) -> dict[str, Any]:
    paths = verify_source(source_root)
    tasks = (
        tasks_override
        if tasks_override is not None
        else load_visible_tasks(paths["train"])
    )
    topic_ids = tuple(tasks)
    facet_counts = {
        topic_id: len(task["facets"]) for topic_id, task in tasks.items()
    }
    request_keys = [
        (topic_id, question["question_id"], sample_index)
        for topic_id in topic_ids
        for question in tasks[topic_id]["questions"]
        for sample_index in range(SAMPLES_PER_QUESTION)
    ]
    if len(request_keys) != expected_requests:
        raise ValueError("ClariQ request count changed")
    random.Random(REQUEST_SEED).shuffle(request_keys)
    question_lookup = {
        (topic_id, question["question_id"]): question
        for topic_id, task in tasks.items()
        for question in task["questions"]
    }
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "request_keys": request_keys,
        "development_endpoints_loaded": False,
    }
    try:
        responses = model.chat_complete_messages_batched(
            [
                likelihood_messages(
                    tasks[topic_id],
                    question_lookup[(topic_id, question_id)],
                )
                for topic_id, question_id, _sample_index in request_keys
            ],
            temperature=TEMPERATURE,
            block_size=config.openrouter_concurrency,
            max_new_tokens=16,
        )
        raw["responses"] = responses
        _checkpoint(raw_path, raw)
        if len(responses) != expected_requests:
            raise ValueError("ClariQ likelihood response count changed")
        parsed = [
            parse_labels(
                response,
                facet_counts[topic_id],
            )
            for response, (topic_id, _question_id, _sample_index) in zip(
                responses, request_keys
            )
        ]
        maps: dict[str, dict[str, list[str]]] = {
            topic_id: {
                question["question_id"]: [""] * SAMPLES_PER_QUESTION
                for question in tasks[topic_id]["questions"]
            }
            for topic_id in topic_ids
        }
        for value, (topic_id, question_id, sample_index) in zip(
            parsed, request_keys
        ):
            maps[topic_id][question_id][sample_index] = value
        if any(
            not value
            for topic in maps.values()
            for samples in topic.values()
            for value in samples
        ):
            raise ValueError("ClariQ likelihood sample is missing")

        likelihoods = {
            topic_id: {
                question_id: build_likelihood(
                    samples,
                    facet_counts[topic_id],
                )
                for question_id, samples in topic_maps.items()
            }
            for topic_id, topic_maps in maps.items()
        }
        policies = {
            topic_id: policy_scores(topic_likelihoods)
            for topic_id, topic_likelihoods in likelihoods.items()
        }
        single_sample_selections = {}
        for topic_id, topic_maps in maps.items():
            selections = []
            for sample_index in range(SAMPLES_PER_QUESTION):
                sample_likelihoods = {
                    question_id: build_likelihood(
                        [samples[sample_index]],
                        facet_counts[topic_id],
                    )
                    for question_id, samples in topic_maps.items()
                }
                selections.append(
                    policy_scores(sample_likelihoods)[
                        "depth_two_question_id"
                    ]
                )
            single_sample_selections[topic_id] = selections
        frozen = {
            "maps": maps,
            "policies": policies,
            "single_sample_depth_two_selections": single_sample_selections,
        }
        raw["frozen_before_endpoint"] = frozen
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise DevelopmentExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    # All LLM outputs, likelihoods, scores, and roots are frozen above.
    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation_payload = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )
    evaluation = evaluation_payload["NDCG20"]
    question_ids = {
        (int(topic_id), question["question"]): question["question_id"]
        for topic_id, task in tasks.items()
        for question in task["questions"]
    }
    external = analyze_topics(
        synthetic,
        evaluation,
        question_ids,
        topic_ids,
    )
    external_by_topic = {
        record["topic_id"]: record for record in external["records"]
    }
    rng = random.Random(RANDOM_CONTROL_SEED)
    rows = []
    pooled_depth_scores = []
    pooled_myopic_scores = []
    pooled_targets = []
    for topic_id in topic_ids:
        policy = policies[topic_id]
        record = external_by_topic.get(topic_id)
        endpoint_by_question = (
            {
                root["question_id"]: root["terminal_utility"]
                for root in record["roots"]
            }
            if record is not None
            else {}
        )
        valid_ids = sorted(endpoint_by_question)
        random_id = rng.choice(valid_ids) if valid_ids else None
        myopic_id = policy["myopic_question_id"]
        depth_id = policy["depth_two_question_id"]
        for question_id in valid_ids:
            pooled_depth_scores.append(
                policy["depth_two_scores"][question_id]
            )
            pooled_myopic_scores.append(
                policy["myopic_scores"][question_id]
            )
            pooled_targets.append(endpoint_by_question[question_id])
        modal_count = Counter(
            single_sample_selections[topic_id]
        ).most_common(1)[0][1]
        modal_maps = {
            question_id: "".join(
                Counter(
                    value[facet_index] for value in samples
                ).most_common(1)[0][0]
                for facet_index in range(facet_counts[topic_id])
            )
            for question_id, samples in maps[topic_id].items()
        }
        eig_values = list(policy["myopic_scores"].values())
        rows.append(
            {
                "topic_id": topic_id,
                "myopic_question_id": myopic_id,
                "depth_two_question_id": depth_id,
                "random_question_id": random_id,
                "myopic_oracle_tail": endpoint_by_question.get(myopic_id),
                "depth_two_oracle_tail": endpoint_by_question.get(depth_id),
                "random_oracle_tail": endpoint_by_question.get(random_id),
                "distinct_modal_partition_count": len(
                    set(modal_maps.values())
                ),
                "myopic_eig_range": max(eig_values) - min(eig_values),
                "single_sample_depth_two_modal_count": modal_count,
                "external_greedy_question_id": (
                    record["greedy_question_id"] if record else None
                ),
                "external_depth_two_question_id": (
                    record["depth_two_question_id"] if record else None
                ),
                "external_terminal_gain": (
                    record["terminal_gain"] if record else None
                ),
            }
        )

    endpoint_complete = all(
        row["myopic_oracle_tail"] is not None
        and row["depth_two_oracle_tail"] is not None
        and row["random_oracle_tail"] is not None
        for row in rows
    )
    depth_differences = [
        row["depth_two_oracle_tail"] - row["myopic_oracle_tail"]
        for row in rows
        if row["depth_two_oracle_tail"] is not None
        and row["myopic_oracle_tail"] is not None
    ]
    random_differences = [
        row["depth_two_oracle_tail"] - row["random_oracle_tail"]
        for row in rows
        if row["depth_two_oracle_tail"] is not None
        and row["random_oracle_tail"] is not None
    ]
    depth_rho = (
        spearman(pooled_depth_scores, pooled_targets)
        if len(pooled_targets) >= 2
        else 0.0
    )
    myopic_rho = (
        spearman(pooled_myopic_scores, pooled_targets)
        if len(pooled_targets) >= 2
        else 0.0
    )
    metrics = {
        "depth_two_root_change_count": sum(
            row["myopic_question_id"] != row["depth_two_question_id"]
            for row in rows
        ),
        "depth_two_wins_over_myopic": sum(
            value > 1e-12 for value in depth_differences
        ),
        "depth_two_losses_to_myopic": sum(
            value < -1e-12 for value in depth_differences
        ),
        "depth_two_ties_with_myopic": sum(
            abs(value) <= 1e-12 for value in depth_differences
        ),
        "mean_depth_two_gain_over_myopic": (
            sum(depth_differences) / len(depth_differences)
            if depth_differences
            else 0.0
        ),
        "mean_depth_two_gain_over_random": (
            sum(random_differences) / len(random_differences)
            if random_differences
            else 0.0
        ),
        "pooled_depth_two_score_oracle_tail_spearman": depth_rho,
        "pooled_myopic_score_oracle_tail_spearman": myopic_rho,
    }
    gates = {
        "exact_physical_requests": (
            usage["physical_requests"] == expected_requests
        ),
        "exact_http_attempts": usage["http_attempts"] == expected_requests,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_maps_parse": len(parsed) == expected_requests,
        "three_modal_partitions_per_topic": all(
            row["distinct_modal_partition_count"] >= 3 for row in rows
        ),
        "eig_range_at_least_0_05_per_topic": all(
            row["myopic_eig_range"] >= 0.05 for row in rows
        ),
        "depth_two_changes_root": (
            metrics["depth_two_root_change_count"] >= 1
        ),
        "single_sample_modal_count_at_least_3": all(
            row["single_sample_depth_two_modal_count"] >= 3 for row in rows
        ),
        "all_selected_roots_have_endpoints": endpoint_complete,
        "depth_two_losses_at_most_1": (
            metrics["depth_two_losses_to_myopic"] <= 1
        ),
        "depth_two_wins_at_least_1": (
            metrics["depth_two_wins_over_myopic"] >= 1
        ),
        "mean_depth_two_gain_at_least_0_003": (
            metrics["mean_depth_two_gain_over_myopic"] >= 0.003
        ),
        "mean_depth_two_gain_over_random_nonnegative": (
            metrics["mean_depth_two_gain_over_random"] >= 0.0
        ),
        "depth_two_spearman_at_least_0_20": depth_rho >= 0.20,
        "cost_at_most_0_50": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "model": MODEL_ID,
            "topic_ids": list(topic_ids),
            "samples_per_question": SAMPLES_PER_QUESTION,
            "temperature": TEMPERATURE,
            "smoothing_alpha": SMOOTHING_ALPHA,
            "request_seed": REQUEST_SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "expected_requests": expected_requests,
            "reasoning_requested": False,
            "development_endpoints_loaded_after_scores_froze": True,
            "holdout_endpoints_loaded": False,
            "repairs_or_reissues": 0,
        },
        "tasks": tasks,
        "maps": maps,
        "policies": policies,
        "single_sample_depth_two_selections": single_sample_selections,
        "rows": rows,
        "metrics": metrics,
        "gates": gates,
        "usage": usage,
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
    config.openrouter_projected_cost_usd = 0.20
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        payload = run_development(
            config,
            source_root=args.source_root,
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
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, DevelopmentExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "DEVELOPMENT_FAILURE.json", failure)
        raise
    output = args.output_dir / "DEVELOPMENT.json"
    _checkpoint(output, payload)
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
