#!/usr/bin/env python3
"""Run the paired DiscoverLLM priority-world mechanics smoke."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import discoverllm_priority_world_manifest as v1
from scripts import discoverllm_priority_world_manifest_v2 as manifest_v2


INTERFACE_VERSION = "discoverllm-priority-world-mechanics-1"
MODEL_ID = "openai/gpt-5.4"
MANIFEST_SHA256 = (
    "9edfd3b20f762491db78087c95bccb1d345063af3423d4d7ccf6c481aa97ad3a"
)
OBSERVATION_SEED = 24_413
EXPECTED_TASKS = 3
EXPECTED_REQUESTS = 15
PROJECTED_COST_USD = 0.40
MAX_COST_USD = 0.75
LIKELIHOOD_TEMPERATURE = 20.0
ACTION_LABELS = ("A", "B")
WORLD_LABELS = ("W1", "W2", "W3", "W4")
OBSERVATION_LABELS = ("O1", "O2", "O3", "O4")


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class MechanicsExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class MechanicsTask:
    key: str
    conversation: tuple[dict[str, str], ...]
    actions: tuple[str, str]
    worlds: tuple[dict[str, Any], ...]


def _checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _strict_object(text: str, expected_keys: set[str]) -> dict[str, Any]:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != expected_keys:
        raise ValueError("response is not an exact flat JSON object")
    return value


def _parse_text_object(
    text: str,
    expected_keys: set[str],
    *,
    maximum_length: int,
) -> dict[str, str]:
    value = _strict_object(text, expected_keys)
    for item in value.values():
        if (
            not isinstance(item, str)
            or not item.strip()
            or len(item) > maximum_length
        ):
            raise ValueError("response contains an invalid text value")
    return {key: item.strip() for key, item in value.items()}


def _parse_score_object(
    text: str,
    expected_keys: set[str],
) -> dict[str, int]:
    value = _strict_object(text, expected_keys)
    parsed: dict[str, int] = {}
    for key, item in value.items():
        if (
            not isinstance(item, str)
            or not item.isdigit()
            or (len(item) > 1 and item.startswith("0"))
        ):
            raise ValueError("likelihood score is not a canonical digit string")
        score = int(item)
        if not 0 <= score <= 100:
            raise ValueError("likelihood score is outside 0 through 100")
        parsed[key] = score
    return parsed


def _semantic_tree(node: dict[str, Any]) -> dict[str, Any]:
    return {
        "criterion": str(node["text"]),
        "subcriteria": [
            _semantic_tree(child) for child in node.get("children", []) or []
        ],
    }


def _load_tasks(
    paths: dict[str, Path],
    manifest_path: Path,
) -> list[MechanicsTask]:
    if v1.sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("DiscoverLLM priority-world manifest hash changed")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_keys = frozen["splits"]["mechanics"]["artifact_ids"]
    if tuple(task_keys) != manifest_v2.EXPECTED_MECHANICS_IDS:
        raise ValueError("DiscoverLLM mechanics task IDs changed")

    rows_by_domain = {
        domain: manifest_v2._load_earliest_turn_rows(path)
        for domain, path in sorted(paths.items())
    }
    tasks: list[MechanicsTask] = []
    for key in task_keys:
        domain, artifact_id = key.split(":", 1)
        candidates = [
            row
            for row in rows_by_domain[domain]
            if row["artifact_id"] == artifact_id
        ]
        if len(candidates) != 2:
            raise ValueError(f"{key} no longer has two candidate actions")
        histories = [candidate["criteria_history"] for candidate in candidates]
        if not histories[0] or histories[0] != histories[1]:
            raise ValueError(f"{key} candidate pre-action histories changed")
        roots = v1.eligible_world_roots(histories[0][-1])
        selected_ids = manifest_v2._world_ids(domain, artifact_id, roots)
        roots_by_id = {str(root["id"]): root for root in roots}
        worlds = tuple(_semantic_tree(roots_by_id[root_id]) for root_id in selected_ids)
        conversation = tuple(
            {
                "role": str(message["role"]),
                "content": str(message["content"]),
            }
            for message in candidates[0]["prompt"]
        )
        tasks.append(
            MechanicsTask(
                key=key,
                conversation=conversation,
                actions=tuple(
                    candidate["completion"] for candidate in candidates
                ),
                worlds=worlds,
            )
        )
    if len(tasks) != EXPECTED_TASKS:
        raise ValueError("DiscoverLLM mechanics task count changed")
    return tasks


def _messages(stage: str, instruction: str, payload: dict[str, Any]):
    return [
        {
            "role": "system",
            "content": (
                f"STAGE={stage}. {instruction} Return one strict flat JSON "
                "object with exactly the requested keys and no other text."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def _root_observation_keys() -> set[str]:
    return {
        f"{action}_{world}"
        for action in ACTION_LABELS
        for world in WORLD_LABELS
    }


def _branch_keys() -> set[str]:
    return {
        f"{action}_{observation}"
        for action in ACTION_LABELS
        for observation in OBSERVATION_LABELS
    }


def _likelihood_keys() -> set[str]:
    return {
        f"{action}_{observation}_{world}"
        for action in ACTION_LABELS
        for observation in OBSERVATION_LABELS
        for world in WORLD_LABELS
    }


def _root_observation_messages(task: MechanicsTask):
    payload = {
        "conversation": task.conversation,
        "candidate_actions": {
            label: action for label, action in zip(ACTION_LABELS, task.actions)
        },
        "private_priority_worlds": {
            label: world for label, world in zip(WORLD_LABELS, task.worlds)
        },
        "output_keys": sorted(_root_observation_keys()),
    }
    return _messages(
        "ROOT_OBSERVATIONS",
        (
            "Simulate each action/world cell independently. The listed world is "
            "the user's sole private priority. After reading the conversation "
            "and candidate assistant response, write the user's natural next "
            "feedback or request in at most 70 words. Do not mention labels, "
            "experiments, hidden worlds, or scoring."
        ),
        payload,
    )


def _observation_mapping(task_key: str) -> dict[str, tuple[int, ...]]:
    result = {}
    for action in ACTION_LABELS:
        indices = list(range(len(WORLD_LABELS)))
        digest = hashlib.sha256(
            f"{OBSERVATION_SEED}:{task_key}:{action}".encode("utf-8")
        ).digest()[:8]
        random.Random(int.from_bytes(digest, "big")).shuffle(indices)
        result[action] = tuple(indices)
    return result


def _shuffle_root_observations(
    task: MechanicsTask,
    raw: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict[str, tuple[int, ...]]]:
    mapping = _observation_mapping(task.key)
    shuffled: dict[str, dict[str, str]] = {}
    for action in ACTION_LABELS:
        shuffled[action] = {}
        for observation_index, world_index in enumerate(mapping[action]):
            shuffled[action][OBSERVATION_LABELS[observation_index]] = raw[
                f"{action}_{WORLD_LABELS[world_index]}"
            ]
    return shuffled, mapping


def _root_likelihood_messages(
    task: MechanicsTask,
    observations: dict[str, dict[str, str]],
):
    payload = {
        "conversation": task.conversation,
        "candidate_actions": {
            label: action for label, action in zip(ACTION_LABELS, task.actions)
        },
        "candidate_priority_worlds": {
            label: world for label, world in zip(WORLD_LABELS, task.worlds)
        },
        "observed_user_feedback": observations,
        "output_keys": sorted(_likelihood_keys()),
    }
    return _messages(
        "ROOT_LIKELIHOODS",
        (
            "Act as an independent semantic likelihood assessor. For every "
            "action, observed feedback, and candidate priority world, rate how "
            "likely that feedback would be if that world were the user's sole "
            "private priority. Scores are comparative likelihood weights from "
            "0 to 100, encoded as canonical decimal digit strings. Observation "
            "labels are shuffled and do not identify their generating world."
        ),
        payload,
    )


def _followup_messages(
    task: MechanicsTask,
    observations: dict[str, dict[str, str]],
):
    branches = {}
    for action_index, action in enumerate(ACTION_LABELS):
        for observation in OBSERVATION_LABELS:
            branches[f"{action}_{observation}"] = {
                "conversation": task.conversation,
                "root_assistant_response": task.actions[action_index],
                "observed_user_feedback": observations[action][observation],
            }
    return _messages(
        "FOLLOWUP_POLICY",
        (
            "For every branch, write the assistant's best next response for "
            "learning and satisfying the user's still-uncertain priority. Use "
            "only that branch's visible history. Do not assume access to any "
            "private criterion or observation-to-world mapping. Keep each "
            "response under 90 words and make it concrete enough to elicit "
            "diagnostic feedback."
        ),
        {"branches": branches, "output_keys": sorted(_branch_keys())},
    )


def _followup_observation_messages(
    task: MechanicsTask,
    observations: dict[str, dict[str, str]],
    mapping: dict[str, tuple[int, ...]],
    followups: dict[str, str],
):
    branches = {}
    for action_index, action in enumerate(ACTION_LABELS):
        for observation_index, observation in enumerate(OBSERVATION_LABELS):
            key = f"{action}_{observation}"
            true_world = mapping[action][observation_index]
            branches[key] = {
                "conversation": task.conversation,
                "root_assistant_response": task.actions[action_index],
                "prior_user_feedback": observations[action][observation],
                "assistant_followup": followups[key],
                "private_priority_label": WORLD_LABELS[true_world],
                "private_priority": task.worlds[true_world],
            }
    return _messages(
        "FOLLOWUP_OBSERVATIONS",
        (
            "Simulate every branch independently. The listed private priority "
            "is the user's sole hidden priority. Continue consistently from "
            "the visible branch history and write the user's natural next "
            "feedback or request in at most 70 words. Do not mention labels, "
            "experiments, hidden priorities, or scoring."
        ),
        {"branches": branches, "output_keys": sorted(_branch_keys())},
    )


def _followup_likelihood_messages(
    task: MechanicsTask,
    observations: dict[str, dict[str, str]],
    followups: dict[str, str],
    second_observations: dict[str, str],
):
    branches = {}
    for action_index, action in enumerate(ACTION_LABELS):
        for observation in OBSERVATION_LABELS:
            key = f"{action}_{observation}"
            branches[key] = {
                "conversation": task.conversation,
                "root_assistant_response": task.actions[action_index],
                "prior_user_feedback": observations[action][observation],
                "assistant_followup": followups[key],
                "new_user_feedback": second_observations[key],
            }
    payload = {
        "branches": branches,
        "candidate_priority_worlds": {
            label: world for label, world in zip(WORLD_LABELS, task.worlds)
        },
        "output_keys": sorted(_likelihood_keys()),
    }
    return _messages(
        "FOLLOWUP_LIKELIHOODS",
        (
            "Act as an independent semantic likelihood assessor. For every "
            "branch and candidate priority world, rate how likely the new user "
            "feedback would be given the complete visible branch history if "
            "that world were the user's sole priority. Scores are comparative "
            "likelihood weights from 0 to 100, encoded as canonical decimal "
            "digit strings. Observation labels do not identify truth."
        ),
        payload,
    )


def _normalize(weights: Sequence[float]) -> list[float]:
    total = sum(weights)
    if not math.isfinite(total) or total <= 0:
        raise ValueError("posterior normalization failed")
    return [weight / total for weight in weights]


def _entropy(probabilities: Sequence[float]) -> float:
    return -sum(
        probability * math.log(probability)
        for probability in probabilities
        if probability > 0
    )


def _likelihood_weight(score: int) -> float:
    return math.exp((score - 50.0) / LIKELIHOOD_TEMPERATURE)


def _analyze_task(
    mapping: dict[str, tuple[int, ...]],
    root_scores: dict[str, int],
    followup_scores: dict[str, int],
) -> dict[str, Any]:
    prior = [0.25] * 4
    start_entropy = _entropy(prior)
    actions: dict[str, Any] = {}
    for action in ACTION_LABELS:
        root_posteriors = []
        terminal_posteriors = []
        truth_indices = []
        for observation_index, observation in enumerate(OBSERVATION_LABELS):
            root_likelihoods = [
                _likelihood_weight(
                    root_scores[f"{action}_{observation}_{world}"]
                )
                for world in WORLD_LABELS
            ]
            root_posterior = _normalize(
                [
                    prior[index] * root_likelihoods[index]
                    for index in range(4)
                ]
            )
            followup_likelihoods = [
                _likelihood_weight(
                    followup_scores[f"{action}_{observation}_{world}"]
                )
                for world in WORLD_LABELS
            ]
            terminal_posterior = _normalize(
                [
                    root_posterior[index] * followup_likelihoods[index]
                    for index in range(4)
                ]
            )
            root_posteriors.append(root_posterior)
            terminal_posteriors.append(terminal_posterior)
            truth_indices.append(mapping[action][observation_index])
        actions[action] = {
            "root_eig": start_entropy
            - statistics.mean(map(_entropy, root_posteriors)),
            "depth_two_eig": start_entropy
            - statistics.mean(map(_entropy, terminal_posteriors)),
            "root_truth_log_posterior": statistics.mean(
                math.log(max(posterior[truth], 1e-300))
                for posterior, truth in zip(root_posteriors, truth_indices)
            ),
            "terminal_truth_log_posterior": statistics.mean(
                math.log(max(posterior[truth], 1e-300))
                for posterior, truth in zip(terminal_posteriors, truth_indices)
            ),
            "terminal_map_accuracy": statistics.mean(
                max(range(4), key=lambda index: posterior[index]) == truth
                for posterior, truth in zip(terminal_posteriors, truth_indices)
            ),
        }
    myopic = max(ACTION_LABELS, key=lambda action: actions[action]["root_eig"])
    nonmyopic = max(
        ACTION_LABELS,
        key=lambda action: actions[action]["depth_two_eig"],
    )
    return {
        "actions": actions,
        "myopic_action": myopic,
        "nonmyopic_action": nonmyopic,
        "root_changed": myopic != nonmyopic,
        "delayed_reversal": (
            myopic != nonmyopic
            and actions[nonmyopic]["root_eig"]
            <= actions[myopic]["root_eig"] - 0.01
            and actions[nonmyopic]["terminal_truth_log_posterior"]
            > actions[myopic]["terminal_truth_log_posterior"]
        ),
    }


def _rank(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    cursor = 0
    while cursor < len(ordered):
        end = cursor + 1
        while end < len(ordered) and math.isclose(
            values[ordered[end]],
            values[ordered[cursor]],
            abs_tol=1e-12,
        ):
            end += 1
        rank = (cursor + end - 1) / 2.0
        for position in range(cursor, end):
            ranks[ordered[position]] = rank
        cursor = end
    return ranks


def _pearson(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or len(left) < 2:
        raise ValueError("correlation inputs have incompatible lengths")
    left_mean = statistics.mean(left)
    right_mean = statistics.mean(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = math.sqrt(
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    )
    return numerator / denominator if denominator > 0 else 0.0


def _spearman(left: Sequence[float], right: Sequence[float]) -> float:
    return _pearson(_rank(left), _rank(right))


def run_mechanics(
    config: Config,
    *,
    paths: dict[str, Path],
    manifest_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    tasks = _load_tasks(paths, manifest_path)
    raw: dict[str, Any] = {"task_ids": [task.key for task in tasks]}
    try:
        root_raw = model.chat_complete_messages_batched(
            [_root_observation_messages(task) for task in tasks],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=2_200,
        )
        raw["root_observations"] = root_raw
        _checkpoint(raw_path, raw)
        root_outputs = [
            _parse_text_object(
                response,
                _root_observation_keys(),
                maximum_length=800,
            )
            for response in root_raw
        ]
        shuffled_and_mappings = [
            _shuffle_root_observations(task, output)
            for task, output in zip(tasks, root_outputs)
        ]
        observations = [value[0] for value in shuffled_and_mappings]
        mappings = [value[1] for value in shuffled_and_mappings]

        root_likelihood_raw = model.chat_complete_messages_batched(
            [
                _root_likelihood_messages(task, task_observations)
                for task, task_observations in zip(tasks, observations)
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=700,
        )
        raw["root_likelihoods"] = root_likelihood_raw
        _checkpoint(raw_path, raw)
        root_likelihoods = [
            _parse_score_object(response, _likelihood_keys())
            for response in root_likelihood_raw
        ]

        followup_raw = model.chat_complete_messages_batched(
            [
                _followup_messages(task, task_observations)
                for task, task_observations in zip(tasks, observations)
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=2_200,
        )
        raw["followups"] = followup_raw
        _checkpoint(raw_path, raw)
        followups = [
            _parse_text_object(
                response,
                _branch_keys(),
                maximum_length=1_000,
            )
            for response in followup_raw
        ]

        second_observation_raw = model.chat_complete_messages_batched(
            [
                _followup_observation_messages(
                    task,
                    task_observations,
                    mapping,
                    task_followups,
                )
                for task, task_observations, mapping, task_followups in zip(
                    tasks,
                    observations,
                    mappings,
                    followups,
                )
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=2_200,
        )
        raw["followup_observations"] = second_observation_raw
        _checkpoint(raw_path, raw)
        second_observations = [
            _parse_text_object(
                response,
                _branch_keys(),
                maximum_length=800,
            )
            for response in second_observation_raw
        ]

        followup_likelihood_raw = model.chat_complete_messages_batched(
            [
                _followup_likelihood_messages(
                    task,
                    task_observations,
                    task_followups,
                    task_second_observations,
                )
                for (
                    task,
                    task_observations,
                    task_followups,
                    task_second_observations,
                ) in zip(tasks, observations, followups, second_observations)
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=700,
        )
        raw["followup_likelihoods"] = followup_likelihood_raw
        _checkpoint(raw_path, raw)
        followup_likelihoods = [
            _parse_score_object(response, _likelihood_keys())
            for response in followup_likelihood_raw
        ]
        raw["all_stages_parsed"] = True
        _checkpoint(raw_path, raw)
        usage = model.usage_snapshot()
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}",
            model.usage_snapshot(),
        ) from exc

    analyses = [
        _analyze_task(mapping, root_scores, followup_scores)
        for mapping, root_scores, followup_scores in zip(
            mappings,
            root_likelihoods,
            followup_likelihoods,
        )
    ]
    root_values = [
        analysis["actions"][action]["root_eig"]
        for analysis in analyses
        for action in ACTION_LABELS
    ]
    depth_values = [
        analysis["actions"][action]["depth_two_eig"]
        for analysis in analyses
        for action in ACTION_LABELS
    ]
    terminal_values = [
        analysis["actions"][action]["terminal_truth_log_posterior"]
        for analysis in analyses
        for action in ACTION_LABELS
    ]
    root_rho = _spearman(root_values, terminal_values)
    depth_rho = _spearman(depth_values, terminal_values)
    dynamic_tasks = sum(
        abs(
            analysis["actions"]["A"]["root_eig"]
            - analysis["actions"]["B"]["root_eig"]
        )
        >= 0.02
        and abs(
            analysis["actions"]["A"]["depth_two_eig"]
            - analysis["actions"]["B"]["depth_two_eig"]
        )
        >= 0.02
        for analysis in analyses
    )
    nonmyopic_terminal = [
        analysis["actions"][analysis["nonmyopic_action"]][
            "terminal_truth_log_posterior"
        ]
        for analysis in analyses
    ]
    myopic_terminal = [
        analysis["actions"][analysis["myopic_action"]][
            "terminal_truth_log_posterior"
        ]
        for analysis in analyses
    ]
    nonmyopic_map = [
        analysis["actions"][analysis["nonmyopic_action"]][
            "terminal_map_accuracy"
        ]
        for analysis in analyses
    ]
    myopic_map = [
        analysis["actions"][analysis["myopic_action"]][
            "terminal_map_accuracy"
        ]
        for analysis in analyses
    ]
    gates = {
        "exact_15_requests": int(usage.get("adapter_requests", 0))
        == EXPECTED_REQUESTS,
        "exact_15_http_attempts": int(usage.get("http_attempts", 0))
        == EXPECTED_REQUESTS,
        "zero_retries": int(usage.get("retry_count", 0)) == 0,
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "cost_at_most_0_75": float(usage.get("adapter_cost_usd", 0.0))
        <= MAX_COST_USD,
        "at_least_two_dynamic_tasks": dynamic_tasks >= 2,
        "at_least_one_root_change": sum(
            analysis["root_changed"] for analysis in analyses
        )
        >= 1,
        "at_least_one_strict_delayed_reversal": sum(
            analysis["delayed_reversal"] for analysis in analyses
        )
        >= 1,
        "nonmyopic_mean_terminal_truth_log_strictly_higher": (
            statistics.mean(nonmyopic_terminal)
            > statistics.mean(myopic_terminal)
        ),
        "nonmyopic_mean_terminal_map_no_lower": (
            statistics.mean(nonmyopic_map) >= statistics.mean(myopic_map)
        ),
        "depth_score_terminal_rho_at_least_0_20": depth_rho >= 0.20,
        "depth_rho_advantage_at_least_0_10": (
            depth_rho >= root_rho + 0.10
        ),
    }
    gates["all_pass"] = all(gates.values())
    public_tasks = []
    for task, analysis in zip(tasks, analyses):
        public_tasks.append(
            {
                "task_id": task.key,
                "myopic_action": analysis["myopic_action"],
                "nonmyopic_action": analysis["nonmyopic_action"],
                "root_changed": analysis["root_changed"],
                "delayed_reversal": analysis["delayed_reversal"],
                "actions": analysis["actions"],
            }
        )
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "temperature": 0.0,
            "manifest_sha256": MANIFEST_SHA256,
            "observation_shuffle_seed": OBSERVATION_SEED,
            "likelihood_temperature": LIKELIHOOD_TEMPERATURE,
            "expected_requests": EXPECTED_REQUESTS,
            "released_scores_read": False,
            "released_winner_labels_read": False,
            "truth_map_exposed_to_likelihood_scorer": False,
            "truth_map_exposed_to_policy": False,
        },
        "tasks": public_tasks,
        "summary": {
            "dynamic_tasks": dynamic_tasks,
            "root_changes": sum(
                analysis["root_changed"] for analysis in analyses
            ),
            "strict_delayed_reversals": sum(
                analysis["delayed_reversal"] for analysis in analyses
            ),
            "mean_nonmyopic_minus_myopic_terminal_truth_log_posterior": (
                statistics.mean(
                    nonmyopic - myopic
                    for nonmyopic, myopic in zip(
                        nonmyopic_terminal,
                        myopic_terminal,
                    )
                )
            ),
            "mean_nonmyopic_minus_myopic_terminal_map_accuracy": (
                statistics.mean(
                    nonmyopic - myopic
                    for nonmyopic, myopic in zip(
                        nonmyopic_map,
                        myopic_map,
                    )
                )
            ),
            "root_eig_terminal_truth_log_spearman": root_rho,
            "depth_two_eig_terminal_truth_log_spearman": depth_rho,
        },
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    @staticmethod
    def _stage(messages: list[dict[str, str]]) -> str:
        return messages[0]["content"].split("STAGE=", 1)[1].split(".", 1)[0]

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            stage = self._stage(messages)
            payload = json.loads(messages[-1]["content"])
            keys = payload["output_keys"]
            if stage == "ROOT_OBSERVATIONS":
                value = {
                    key: (
                        f"{key.split('_')[0]} feedback from "
                        f"{key.split('_')[1]}"
                    )
                    for key in keys
                }
            elif stage in {"ROOT_LIKELIHOODS", "FOLLOWUP_LIKELIHOODS"}:
                value = {}
                observations = (
                    payload.get("observed_user_feedback")
                    if stage == "ROOT_LIKELIHOODS"
                    else {
                        action: {
                            observation: payload["branches"][
                                f"{action}_{observation}"
                            ]["new_user_feedback"]
                            for observation in OBSERVATION_LABELS
                        }
                        for action in ACTION_LABELS
                    }
                )
                for key in keys:
                    action, observation, world = key.split("_")
                    text = observations[action][observation]
                    true_world = next(
                        label for label in WORLD_LABELS if label in text
                    )
                    if stage == "ROOT_LIKELIHOODS":
                        high, low = (
                            (90, 10) if action == "A" else (65, 35)
                        )
                    else:
                        high, low = (
                            (50, 50) if action == "A" else (99, 1)
                        )
                    value[key] = str(high if world == true_world else low)
            elif stage == "FOLLOWUP_POLICY":
                value = {
                    key: f"Diagnostic continuation for {key}?"
                    for key in keys
                }
            elif stage == "FOLLOWUP_OBSERVATIONS":
                value = {}
                for key in keys:
                    true_world = payload["branches"][key][
                        "private_priority_label"
                    ]
                    action = key.split("_")[0]
                    value[key] = (
                        f"{action} followup feedback from {true_world}"
                    )
            else:  # pragma: no cover - defensive
                raise AssertionError(stage)
            responses.append(
                json.dumps(value, ensure_ascii=True, separators=(",", ":"))
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


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("DiscoverLLM mechanics config selects wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--creative-writing", type=Path, required=True)
    parser.add_argument("--technical-writing", type=Path, required=True)
    parser.add_argument("--svg-drawing", type=Path, required=True)
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
    config.openrouter_concurrency = EXPECTED_TASKS
    config.openrouter_max_output_tokens = 2_200
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        result = run_mechanics(
            config,
            paths={
                "creative_writing": args.creative_writing,
                "technical_writing": args.technical_writing,
                "svg_drawing": args.svg_drawing,
            },
            manifest_path=args.manifest,
            raw_path=raw_path,
            model=model,
        )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, MechanicsExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "MECHANICS_FAILURE.json", failure)
        raise
    output_path = args.output_dir / "MECHANICS.json"
    _checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "summary": result["summary"],
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
