#!/usr/bin/env python3
"""Run the DiscoverLLM native-progress realistic serving gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Protocol

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import discoverllm_priority_world_manifest as manifest_v1
from scripts import discoverllm_priority_world_manifest_v2 as manifest_v2
from scripts import discoverllm_priority_world_mechanics as cardinal
from scripts import discoverllm_priority_world_ordinal_serving as ordinal
from scripts import discoverllm_priority_world_tier_serving as tier


INTERFACE_VERSION = "discoverllm-native-progress-serving-1"
MODEL_ID = "openai/gpt-5.4"
SERVING_TASK_ID = "creative_writing:artifact_305"
RESERVED_MECHANICS_TASK_IDS = (
    "creative_writing:artifact_100",
    "technical_writing:artifact_37",
    "technical_writing:artifact_359",
)
ACTION_LABELS = ("D1", "D2", "R1", "R2")
WORLD_LABELS = cardinal.WORLD_LABELS
OBSERVATION_LABELS = cardinal.OBSERVATION_LABELS
WORLD_PRESENTATION_SEED = 24_416
OBSERVATION_SEED = 24_417
EXPECTED_REQUESTS = 8
MAX_TRANSPORT_RETRIES = 3
PROJECTED_COST_USD = 0.18
MAX_COST_USD = 0.30


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class ServingExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class ProgressWorld:
    current: dict[str, Any]
    next: dict[str, Any] | None
    after_next: dict[str, Any] | None


@dataclass(frozen=True)
class ProgressTask:
    key: str
    conversation: tuple[dict[str, str], ...]
    worlds: tuple[ProgressWorld, ...]


def _load_task_by_id(
    paths: dict[str, Path],
    manifest_path: Path,
    task_id: str,
) -> ProgressTask:
    if manifest_v1.sha256_file(manifest_path) != cardinal.MANIFEST_SHA256:
        raise ValueError("DiscoverLLM priority-world manifest hash changed")
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    if task_id not in set(frozen["splits"]["development"]["artifact_ids"]):
        raise ValueError("native-progress task is outside development")
    for domain, expected_sha256 in manifest_v2.SOURCE_SHA256.items():
        if manifest_v1.sha256_file(paths[domain]) != expected_sha256:
            raise ValueError(f"DiscoverLLM {domain} Parquet hash changed")

    domain, artifact_id = task_id.split(":", 1)
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("native-progress serving requires pyarrow") from exc
    table = parquet.read_table(
        paths[domain],
        columns=[
            "artifact_id",
            "turn_id",
            "assistant_index",
            "prompt",
            "criteria_history",
        ],
        filters=[("artifact_id", "=", artifact_id)],
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError("native-progress artifact is absent")
    earliest_turn = min(int(row["turn_id"]) for row in rows)
    candidates = [
        {
            **row,
            "assistant_index": int(row["assistant_index"]),
            "criteria_history": (
                json.loads(row["criteria_history"])
                if isinstance(row["criteria_history"], str)
                else row["criteria_history"]
            ),
        }
        for row in rows
        if int(row["turn_id"]) == earliest_turn
    ]
    candidates.sort(key=lambda row: row["assistant_index"])
    if len(candidates) != 2:
        raise ValueError("native-progress task no longer has two source rows")
    histories = [candidate["criteria_history"] for candidate in candidates]
    if not histories[0] or histories[0] != histories[1]:
        raise ValueError("native-progress source histories changed")
    roots = manifest_v1.eligible_world_roots(histories[0][-1])
    if len(roots) < len(WORLD_LABELS):
        raise ValueError("native-progress task has too few ordered roots")
    semantic_roots = [cardinal._semantic_tree(root) for root in roots]
    worlds = []
    for index in range(len(WORLD_LABELS)):
        worlds.append(
            ProgressWorld(
                current=semantic_roots[index],
                next=(
                    semantic_roots[index + 1]
                    if index + 1 < len(semantic_roots)
                    else None
                ),
                after_next=(
                    semantic_roots[index + 2]
                    if index + 2 < len(semantic_roots)
                    else None
                ),
            )
        )
    conversation = tuple(
        {
            "role": str(message["role"]),
            "content": str(message["content"]),
        }
        for message in candidates[0]["prompt"]
    )
    return ProgressTask(
        key=task_id,
        conversation=conversation,
        worlds=tuple(worlds),
    )


def _load_task(
    paths: dict[str, Path],
    manifest_path: Path,
) -> ProgressTask:
    frozen = json.loads(manifest_path.read_text(encoding="utf-8"))
    development_ids = tuple(frozen["splits"]["development"]["artifact_ids"])
    expected_prefix = (
        ordinal.SERVING_TASK_ID,
        tier.SERVING_TASK_ID,
        *tier.RESERVED_MECHANICS_TASK_IDS,
        SERVING_TASK_ID,
        *RESERVED_MECHANICS_TASK_IDS,
    )
    if development_ids[: len(expected_prefix)] != expected_prefix:
        raise ValueError("native-progress task reservation changed")
    return _load_task_by_id(paths, manifest_path, SERVING_TASK_ID)


def _world_payload(world: ProgressWorld) -> dict[str, Any]:
    return {
        "current_priority": world.current,
        "next_priority_if_advanced": world.next,
        "priority_after_next": world.after_next,
    }


def _world_presentation(
    task: ProgressTask,
    stage: str,
) -> dict[str, dict[str, Any]]:
    labels = list(WORLD_LABELS)
    digest = hashlib.sha256(
        f"{WORLD_PRESENTATION_SEED}:{task.key}:{stage}".encode("utf-8")
    ).digest()[:8]
    random.Random(int.from_bytes(digest, "big")).shuffle(labels)
    by_label = dict(zip(WORLD_LABELS, task.worlds))
    return {label: _world_payload(by_label[label]) for label in labels}


def _messages(
    stage: str,
    instruction: str,
    payload: dict[str, Any],
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": f"STAGE={stage}. {instruction}"},
        {
            "role": "user",
            "content": json.dumps(
                payload,
                ensure_ascii=True,
                separators=(",", ":"),
            ),
        },
    ]


def _action_messages(task: ProgressTask) -> list[dict[str, str]]:
    return _messages(
        "ACTION_BANK",
        (
            "Generate four distinct shared next assistant messages for the "
            "conversation and candidate-priority support. D1 must be one broad "
            "dialog-act clarification question; D2 one specific contrastive "
            "dialog-act question; R1 one concise artifact attempt; R2 two or "
            "more concise artifact alternatives. Do not mention labels, hidden "
            "worlds, experiments, or scoring. Each message is at most 90 words. "
            "Return exactly four lines and no other text: LABEL|message, in "
            "the order D1,D2,R1,R2. Messages may not contain a pipe character."
        ),
        {
            "conversation": task.conversation,
            "candidate_current_priorities": _world_presentation(
                task,
                "ACTION_BANK",
            ),
            "output_labels": list(ACTION_LABELS),
        },
    )


def _parse_actions(text: str) -> dict[str, str]:
    lines = text.strip().splitlines()
    if len(lines) != len(ACTION_LABELS):
        raise ValueError("action bank has the wrong line count")
    parsed = {}
    for expected, line in zip(ACTION_LABELS, lines):
        if line.count("|") != 1:
            raise ValueError("action line has invalid separators")
        label, content = line.split("|")
        if label != expected or not content.strip() or len(content) > 1_200:
            raise ValueError("action line is invalid")
        parsed[label] = content.strip()
    if len(set(parsed.values())) != len(ACTION_LABELS):
        raise ValueError("action bank contains duplicate messages")
    return parsed


def _root_transition_keys() -> set[str]:
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


def _followup_transition_keys() -> set[str]:
    return {
        f"{branch}_{world}"
        for branch in _branch_keys()
        for world in WORLD_LABELS
    }


def _transition_messages(
    stage: str,
    task: ProgressTask,
    actions: dict[str, str],
    *,
    branches: dict[str, Any] | None = None,
) -> list[dict[str, str]]:
    output_keys = (
        sorted(_root_transition_keys())
        if branches is None
        else sorted(_followup_transition_keys())
    )
    payload: dict[str, Any] = {
        "conversation": task.conversation,
        "candidate_worlds": _world_presentation(task, stage),
        "output_keys": output_keys,
    }
    if branches is None:
        payload["assistant_actions"] = actions
    else:
        payload["branches"] = branches
    return _messages(
        stage,
        (
            "Apply DiscoverLLM's native transition semantics independently to "
            "every requested cell. Classify the last assistant message as D "
            "(pure dialog act with zero artifact content) or R (artifact or "
            "artifact alternatives; artifact takes precedence). For D, output "
            "P only if it directly and explicitly probes the complete active "
            "priority subtree, otherwise N; dialog never advances the current "
            "root. For R, output S only if it fully satisfies every leaf of "
            "the active priority subtree, otherwise N; only R|S advances to "
            "the next root. If the candidate world is already terminal, use "
            "D|T or R|T. Return exactly the requested number of lines and no "
            "other text, each KEY|D|P, KEY|D|N, KEY|R|S, KEY|R|N, KEY|D|T, "
            "or KEY|R|T."
        ),
        payload,
    )


def _parse_transitions(
    text: str,
    expected_keys: set[str],
) -> dict[str, tuple[str, str]]:
    lines = text.strip().splitlines()
    if len(lines) != len(expected_keys):
        raise ValueError("transition response has the wrong line count")
    parsed = {}
    for line in lines:
        parts = line.split("|")
        if len(parts) != 3:
            raise ValueError("transition line has invalid separators")
        key, kind, outcome = parts
        if key not in expected_keys or key in parsed:
            raise ValueError("transition line has invalid or duplicate key")
        if kind == "D" and outcome not in {"P", "N", "T"}:
            raise ValueError("dialog transition outcome is invalid")
        if kind == "R" and outcome not in {"S", "N", "T"}:
            raise ValueError("artifact transition outcome is invalid")
        if kind not in {"D", "R"}:
            raise ValueError("transition kind is invalid")
        parsed[key] = (kind, outcome)
    if set(parsed) != expected_keys:
        raise ValueError("transition response is missing keys")
    return parsed


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


def _active_index(transition: tuple[str, str], start: int = 0) -> int:
    kind, outcome = transition
    if kind == "R" and outcome == "S":
        return start + 1
    return start


def _priority_at(world: ProgressWorld, index: int) -> dict[str, Any] | None:
    return (world.current, world.next, world.after_next)[index] if index < 3 else None


def _feedback_guidance(
    transition: tuple[str, str],
    active_priority: dict[str, Any] | None,
) -> str:
    kind, outcome = transition
    if outcome == "T" or active_priority is None:
        return "Briefly express that the request is satisfied."
    if kind == "D" and outcome == "P":
        return "Clearly and directly articulate the active priority."
    if kind == "D":
        return "Respond vaguely about the active priority without naming its details."
    if outcome == "S":
        return "Acknowledge progress, then vaguely introduce the newly active priority."
    return "Clearly explain what remains wrong about the active priority."


def _root_feedback_messages(
    task: ProgressTask,
    actions: dict[str, str],
    transitions: dict[str, tuple[str, str]],
) -> list[dict[str, str]]:
    branches = {}
    for action in ACTION_LABELS:
        for world_index, world_label in enumerate(WORLD_LABELS):
            world = task.worlds[world_index]
            transition = transitions[f"{action}_{world_label}"]
            active_index = _active_index(transition)
            active = _priority_at(world, active_index)
            branches[f"{action}_{world_label}"] = {
                "conversation": task.conversation,
                "assistant_response": actions[action],
                "private_original_priority": world.current,
                "private_active_priority": active,
                "transition": "|".join(transition),
                "response_guidance": _feedback_guidance(
                    transition,
                    active,
                ),
            }
    return cardinal._messages(
        "ROOT_PROGRESS_FEEDBACK",
        (
            "Simulate each branch independently as the human user. Follow the "
            "private transition and response guidance exactly, remain natural, "
            "and use at most 55 words. Do not mention labels, hidden states, "
            "experiments, transitions, or scoring."
        ),
        {"branches": branches, "output_keys": sorted(branches)},
    )


def _shuffle_feedback(
    task: ProgressTask,
    feedback: dict[str, str],
) -> tuple[dict[str, dict[str, str]], dict[str, tuple[int, ...]]]:
    mapping = _observation_mapping(task.key)
    observations = {}
    for action in ACTION_LABELS:
        observations[action] = {}
        for observation_index, world_index in enumerate(mapping[action]):
            observations[action][OBSERVATION_LABELS[observation_index]] = (
                feedback[f"{action}_{WORLD_LABELS[world_index]}"]
            )
    return observations, mapping


def _tier_messages(
    stage: str,
    task: ProgressTask,
    payload: dict[str, Any],
    output_keys: set[str],
) -> list[dict[str, str]]:
    return _messages(
        stage,
        (
            "Act as an independent semantic likelihood assessor. For every "
            "requested observed branch, assign each candidate starting world "
            "one likelihood tier: H=strongly compatible, M=plausible but "
            "ambiguous, L=weak or inconsistent. Account for the provided "
            "native transition trajectory. Worlds may share a tier. World "
            "presentation order is shuffled. Return exactly the requested "
            "number of lines and no other text. Each line must be "
            "KEY|W1:X,W2:X,W3:X,W4:X in that exact world order, where X is "
            "H, M, or L."
        ),
        {
            **payload,
            "candidate_worlds": _world_presentation(task, stage),
            "output_keys": sorted(output_keys),
        },
    )


def _root_tier_messages(
    task: ProgressTask,
    actions: dict[str, str],
    transitions: dict[str, tuple[str, str]],
    observations: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    return _tier_messages(
        "ROOT_PROGRESS_TIERS",
        task,
        {
            "conversation": task.conversation,
            "assistant_actions": actions,
            "candidate_root_transitions": {
                key: "|".join(value)
                for key, value in sorted(transitions.items())
            },
            "observed_user_feedback": observations,
        },
        _branch_keys(),
    )


def _followup_messages(
    task: ProgressTask,
    actions: dict[str, str],
    observations: dict[str, dict[str, str]],
) -> list[dict[str, str]]:
    branches = {}
    for action in ACTION_LABELS:
        for observation in OBSERVATION_LABELS:
            key = f"{action}_{observation}"
            branches[key] = {
                "conversation": task.conversation,
                "root_assistant_response": actions[action],
                "observed_user_feedback": observations[action][observation],
            }
    return cardinal._messages(
        "FOLLOWUP_PROGRESS_POLICY",
        (
            "For every branch, write the assistant's best next response for "
            "learning and satisfying the user's still-uncertain priority. Use "
            "only the visible branch history. Do not assume private worlds or "
            "transition labels. Each response is at most 90 words."
        ),
        {"branches": branches, "output_keys": sorted(branches)},
    )


def _candidate_active_state(
    world: ProgressWorld,
    transition: tuple[str, str],
) -> tuple[int, dict[str, Any] | None]:
    index = _active_index(transition)
    return index, _priority_at(world, index)


def _followup_transition_branches(
    task: ProgressTask,
    actions: dict[str, str],
    root_transitions: dict[str, tuple[str, str]],
    observations: dict[str, dict[str, str]],
    followups: dict[str, str],
) -> dict[str, Any]:
    branches = {}
    for action in ACTION_LABELS:
        for observation in OBSERVATION_LABELS:
            branch = f"{action}_{observation}"
            for world_index, world_label in enumerate(WORLD_LABELS):
                world = task.worlds[world_index]
                root_transition = root_transitions[f"{action}_{world_label}"]
                active_index, active = _candidate_active_state(
                    world,
                    root_transition,
                )
                branches[f"{branch}_{world_label}"] = {
                    "root_assistant_response": actions[action],
                    "observed_user_feedback": observations[action][observation],
                    "assistant_followup": followups[branch],
                    "candidate_active_priority": active,
                    "candidate_active_offset": active_index,
                    "candidate_terminal": active is None,
                }
    return branches


def _followup_feedback_messages(
    task: ProgressTask,
    actions: dict[str, str],
    mapping: dict[str, tuple[int, ...]],
    root_transitions: dict[str, tuple[str, str]],
    observations: dict[str, dict[str, str]],
    followups: dict[str, str],
    followup_transitions: dict[str, tuple[str, str]],
) -> list[dict[str, str]]:
    branches = {}
    for action in ACTION_LABELS:
        for observation_index, observation in enumerate(OBSERVATION_LABELS):
            branch = f"{action}_{observation}"
            world_index = mapping[action][observation_index]
            world_label = WORLD_LABELS[world_index]
            world = task.worlds[world_index]
            root_transition = root_transitions[f"{action}_{world_label}"]
            active_index, _ = _candidate_active_state(
                world,
                root_transition,
            )
            follow_transition = followup_transitions[
                f"{branch}_{world_label}"
            ]
            next_index = _active_index(
                follow_transition,
                start=active_index,
            )
            active = _priority_at(world, next_index)
            branches[branch] = {
                "conversation": task.conversation,
                "root_assistant_response": actions[action],
                "prior_user_feedback": observations[action][observation],
                "assistant_followup": followups[branch],
                "private_active_priority": active,
                "transition": "|".join(follow_transition),
                "response_guidance": _feedback_guidance(
                    follow_transition,
                    active,
                ),
            }
    return cardinal._messages(
        "FOLLOWUP_PROGRESS_FEEDBACK",
        (
            "Simulate each branch independently as the human user. Follow the "
            "private transition and response guidance exactly, remain natural, "
            "and use at most 55 words. Do not mention labels, hidden states, "
            "experiments, transitions, or scoring."
        ),
        {"branches": branches, "output_keys": sorted(branches)},
    )


def _followup_tier_messages(
    task: ProgressTask,
    actions: dict[str, str],
    root_transitions: dict[str, tuple[str, str]],
    observations: dict[str, dict[str, str]],
    followups: dict[str, str],
    followup_transitions: dict[str, tuple[str, str]],
    second_feedback: dict[str, str],
) -> list[dict[str, str]]:
    return _tier_messages(
        "FOLLOWUP_PROGRESS_TIERS",
        task,
        {
            "conversation": task.conversation,
            "assistant_actions": actions,
            "candidate_root_transitions": {
                key: "|".join(value)
                for key, value in sorted(root_transitions.items())
            },
            "observed_user_feedback": observations,
            "assistant_followups": followups,
            "candidate_followup_transitions": {
                key: "|".join(value)
                for key, value in sorted(followup_transitions.items())
            },
            "new_user_feedback": second_feedback,
        },
        _branch_keys(),
    )


def _action_transition_types_match(
    transitions: dict[str, tuple[str, str]],
) -> bool:
    return all(
        kind == ("D" if action.startswith("D") else "R")
        for key, (kind, _) in transitions.items()
        for action in [key.split("_", 1)[0]]
    )


def _serving_gates(
    usage: dict[str, Any],
    root_transitions: dict[str, tuple[str, str]],
) -> dict[str, bool]:
    requests = int(usage.get("adapter_requests", -1))
    retries = int(usage.get("retry_count", -1))
    attempts = int(usage.get("http_attempts", -1))
    gates = {
        "exact_8_logical_requests": requests == EXPECTED_REQUESTS,
        "bounded_transport_retries": 0 <= retries <= MAX_TRANSPORT_RETRIES,
        "http_attempts_match_requests_plus_retries": (
            attempts == requests + retries
        ),
        "zero_reasoning_tokens": int(
            usage.get("adapter_reasoning_tokens", 0)
        )
        == 0,
        "zero_forced_exits": int(usage.get("forced_exits", 0)) == 0,
        "cost_at_most_0_30": float(
            usage.get("adapter_cost_usd", float("inf"))
        )
        <= MAX_COST_USD,
        "root_action_types_match_slots": _action_transition_types_match(
            root_transitions
        ),
    }
    gates["all_pass"] = all(gates.values())
    return gates


def run_serving(
    config: Config,
    *,
    paths: dict[str, Path],
    manifest_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    del config
    task = _load_task(paths, manifest_path)
    raw: dict[str, Any] = {"task_id": task.key}
    try:
        action_raw = model.chat_complete_messages_batched(
            [_action_messages(task)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=900,
        )
        raw["actions"] = action_raw
        ordinal._checkpoint(raw_path, raw)
        actions = _parse_actions(action_raw[0])

        root_transition_raw = model.chat_complete_messages_batched(
            [_transition_messages("ROOT_PROGRESS_TRANSITIONS", task, actions)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=800,
        )
        raw["root_transitions"] = root_transition_raw
        ordinal._checkpoint(raw_path, raw)
        root_transitions = _parse_transitions(
            root_transition_raw[0],
            _root_transition_keys(),
        )

        root_feedback_raw = model.chat_complete_messages_batched(
            [_root_feedback_messages(task, actions, root_transitions)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=3_200,
        )
        raw["root_feedback"] = root_feedback_raw
        ordinal._checkpoint(raw_path, raw)
        root_feedback = cardinal._parse_text_object(
            root_feedback_raw[0],
            _root_transition_keys(),
            maximum_length=700,
        )
        observations, mapping = _shuffle_feedback(task, root_feedback)

        root_tier_raw = model.chat_complete_messages_batched(
            [
                _root_tier_messages(
                    task,
                    actions,
                    root_transitions,
                    observations,
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=600,
        )
        raw["root_tiers"] = root_tier_raw
        ordinal._checkpoint(raw_path, raw)
        root_tiers = tier._parse_tiers(
            root_tier_raw[0],
            _branch_keys(),
        )

        followup_raw = model.chat_complete_messages_batched(
            [_followup_messages(task, actions, observations)],
            temperature=0.0,
            block_size=1,
            max_new_tokens=3_200,
        )
        raw["followups"] = followup_raw
        ordinal._checkpoint(raw_path, raw)
        followups = cardinal._parse_text_object(
            followup_raw[0],
            _branch_keys(),
            maximum_length=1_100,
        )

        followup_branches = _followup_transition_branches(
            task,
            actions,
            root_transitions,
            observations,
            followups,
        )
        followup_transition_raw = model.chat_complete_messages_batched(
            [
                _transition_messages(
                    "FOLLOWUP_PROGRESS_TRANSITIONS",
                    task,
                    actions,
                    branches=followup_branches,
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=2_400,
        )
        raw["followup_transitions"] = followup_transition_raw
        ordinal._checkpoint(raw_path, raw)
        followup_transitions = _parse_transitions(
            followup_transition_raw[0],
            _followup_transition_keys(),
        )

        second_feedback_raw = model.chat_complete_messages_batched(
            [
                _followup_feedback_messages(
                    task,
                    actions,
                    mapping,
                    root_transitions,
                    observations,
                    followups,
                    followup_transitions,
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=3_200,
        )
        raw["followup_feedback"] = second_feedback_raw
        ordinal._checkpoint(raw_path, raw)
        second_feedback = cardinal._parse_text_object(
            second_feedback_raw[0],
            _branch_keys(),
            maximum_length=700,
        )

        followup_tier_raw = model.chat_complete_messages_batched(
            [
                _followup_tier_messages(
                    task,
                    actions,
                    root_transitions,
                    observations,
                    followups,
                    followup_transitions,
                    second_feedback,
                )
            ],
            temperature=0.0,
            block_size=1,
            max_new_tokens=600,
        )
        raw["followup_tiers"] = followup_tier_raw
        ordinal._checkpoint(raw_path, raw)
        followup_tiers = tier._parse_tiers(
            followup_tier_raw[0],
            _branch_keys(),
        )
        usage = model.usage_snapshot()
    except Exception as exc:
        ordinal._checkpoint(raw_path, raw)
        raise ServingExecutionError(
            f"{type(exc).__name__}: {exc}",
            model.usage_snapshot(),
        ) from exc

    gates = _serving_gates(usage, root_transitions)
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "temperature": 0.0,
            "task_id": task.key,
            "reserved_mechanics_task_ids": list(
                RESERVED_MECHANICS_TASK_IDS
            ),
            "manifest_sha256": cardinal.MANIFEST_SHA256,
            "world_presentation_seed": WORLD_PRESENTATION_SEED,
            "observation_seed": OBSERVATION_SEED,
            "action_slots": list(ACTION_LABELS),
            "tier_weights": tier.TIER_WEIGHTS,
            "expected_logical_requests": EXPECTED_REQUESTS,
            "maximum_transport_retries": MAX_TRANSPORT_RETRIES,
            "semantic_reissues_allowed": False,
            "released_scores_read": False,
            "released_winner_labels_read": False,
            "truth_map_exposed_to_likelihood_scorer": False,
            "truth_map_exposed_to_policy": False,
            "semantic_content_emitted": False,
        },
        "parse_counts": {
            "actions": len(actions),
            "root_transitions": len(root_transitions),
            "root_feedback": len(root_feedback),
            "root_tiers": len(root_tiers),
            "followups": len(followups),
            "followup_transitions": len(followup_transitions),
            "followup_feedback": len(second_feedback),
            "followup_tiers": len(followup_tiers),
        },
        "root_transition_summary": {
            "dialog_probe_cells": sum(
                value == ("D", "P") for value in root_transitions.values()
            ),
            "artifact_advance_cells": sum(
                value == ("R", "S") for value in root_transitions.values()
            ),
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
            keys = payload.get("output_keys", [])
            if stage == "ACTION_BANK":
                response = "\n".join(
                    (
                        "D1|What broad direction matters most?",
                        "D2|Which of these priorities should lead?",
                        "R1|Here is one concise draft artifact.",
                        "R2|Option one is concise. Option two is detailed.",
                    )
                )
            elif stage in {
                "ROOT_PROGRESS_TRANSITIONS",
                "FOLLOWUP_PROGRESS_TRANSITIONS",
            }:
                lines = []
                for key in keys:
                    action = key.split("_", 1)[0]
                    lines.append(
                        f"{key}|D|P"
                        if action.startswith("D")
                        else f"{key}|R|S"
                    )
                response = "\n".join(lines)
            elif stage in {
                "ROOT_PROGRESS_FEEDBACK",
                "FOLLOWUP_PROGRESS_FEEDBACK",
                "FOLLOWUP_PROGRESS_POLICY",
            }:
                value = {}
                for key in keys:
                    if stage == "FOLLOWUP_PROGRESS_POLICY":
                        value[key] = f"What should I address next for {key}?"
                    else:
                        branch = payload["branches"][key]
                        private = branch.get("private_active_priority")
                        marker = (
                            private["criterion"]
                            if isinstance(private, dict)
                            else "terminal"
                        )
                        value[key] = f"Feedback about {marker}"
                response = json.dumps(value, separators=(",", ":"))
            elif stage in {
                "ROOT_PROGRESS_TIERS",
                "FOLLOWUP_PROGRESS_TIERS",
            }:
                lines = []
                feedback = (
                    payload["observed_user_feedback"]
                    if stage == "ROOT_PROGRESS_TIERS"
                    else payload["new_user_feedback"]
                )
                for key in keys:
                    text = (
                        feedback[key.split("_")[0]][key.split("_")[1]]
                        if stage == "ROOT_PROGRESS_TIERS"
                        else feedback[key]
                    )
                    truth = next(
                        (
                            world
                            for world in WORLD_LABELS
                            if f"Criterion {world}" in text
                        ),
                        "W1",
                    )
                    assignments = ",".join(
                        f"{world}:{'H' if world == truth else 'L'}"
                        for world in WORLD_LABELS
                    )
                    lines.append(f"{key}|{assignments}")
                response = "\n".join(lines)
            else:  # pragma: no cover
                raise AssertionError(stage)
            responses.append(response)
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
        raise ValueError("native-progress config selects wrong model")
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
    config.openrouter_concurrency = 1
    config.openrouter_max_retries = MAX_TRANSPORT_RETRIES
    config.openrouter_max_output_tokens = 3_200
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel = (
        DeterministicFixtureModel() if args.dry_run else _build_model(config)
    )
    try:
        result = run_serving(
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
        if isinstance(exc, ServingExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        ordinal._checkpoint(args.output_dir / "SERVING_FAILURE.json", failure)
        raise
    output_path = args.output_dir / "SERVING.json"
    ordinal._checkpoint(output_path, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output_path),
                "parse_counts": result["parse_counts"],
                "root_transition_summary": result[
                    "root_transition_summary"
                ],
                "gates": result["gates"],
                "usage": result["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
