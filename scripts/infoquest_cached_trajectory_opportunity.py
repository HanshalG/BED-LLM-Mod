#!/usr/bin/env python3
"""Audit path-dependent opportunity in released InfoQuest trajectories."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

from scripts import infoquest_llm_bed_manifest as source_manifest


INTERFACE_VERSION = "infoquest-cached-trajectory-opportunity-1"
BASELINE_FILENAMES = [
    (
        "chats-gemini_2_0_flash_001-tiiuae_Falcon3_7B_Instruct-"
        "AtlaAI_Selene_1_Mini_Llama_3_1_8B__mt30-0.jsonl"
    ),
    (
        "chats-gemini_2_0_flash_001-tiiuae_Falcon3_7B_Instruct-"
        "AtlaAI_Selene_1_Mini_Llama_3_1_8B__mt30-1.jsonl"
    ),
    (
        "chats-gemini_2_0_flash_001-tiiuae_Falcon3_7B_Instruct-"
        "AtlaAI_Selene_1_Mini_Llama_3_1_8B__mt30-2.jsonl"
    ),
]
BASELINE_SHA256 = [
    "454039500859fd9a36146908a2a2137cfcf6b358d89826c7cffea8445d07f0c1",
    "518a39ec19dd3b6cdfbdf2fec81c8d1f6d6811091ff18b873d335628d39eb9cb",
    "296c07e45afa87bc3dafe559b62f1c836ae1903a736c81410602e9ebf5197d15",
]
QUARANTINED_IDS = {4}
MAX_TURNS = 30
CHECKLIST_SIZE = 5
THRESHOLDS = {
    "multi_turn_fraction": 0.95,
    "initial_at_least_two_unresolved_fraction": 0.80,
    "delayed_gain_at_least_two_fraction": 0.75,
    "mean_delayed_gain": 2.0,
    "novel_uptake_advantage_per_transition": 0.25,
    "novel_uptake_positive_trajectory_fraction": 0.60,
    "turn_range_at_least_two_task_world_fraction": 0.40,
}
STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "been",
    "being",
    "but",
    "by",
    "can",
    "could",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "i",
    "if",
    "in",
    "into",
    "is",
    "it",
    "its",
    "may",
    "me",
    "my",
    "of",
    "on",
    "or",
    "our",
    "should",
    "so",
    "that",
    "the",
    "their",
    "them",
    "then",
    "there",
    "these",
    "they",
    "this",
    "to",
    "was",
    "we",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "will",
    "with",
    "would",
    "you",
    "your",
}


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise ValueError(f"{path.name}:{line_number} is blank")
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"{path.name}:{line_number} is not valid JSON"
                ) from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path.name}:{line_number} is not an object")
            rows.append(value)
    return rows


def _rows_by_id(
    path: Path,
    *,
    expected_sha256: str,
    expected_records: int = source_manifest.EXPECTED_RECORDS,
) -> dict[int, dict[str, Any]]:
    if source_manifest.sha256_file(path) != expected_sha256:
        raise ValueError(f"{path.name} hash changed")
    rows = _load_jsonl(path)
    if len(rows) != expected_records:
        raise ValueError(f"{path.name} record count changed")
    if any(
        not isinstance(row.get("id"), int) or isinstance(row.get("id"), bool)
        for row in rows
    ):
        raise ValueError(f"{path.name} contains a non-integer id")
    by_id = {row["id"]: row for row in rows}
    if len(by_id) != expected_records:
        raise ValueError(f"{path.name} contains duplicate ids")
    if set(by_id) != set(range(expected_records)):
        raise ValueError(f"{path.name} ids are not exactly 0..499")
    return by_id


def _tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 3 and token not in STOPWORDS
    }


def _require_message(
    message: Any,
    context: str,
    *,
    allow_empty: bool = False,
) -> dict[str, str]:
    if not isinstance(message, dict) or set(message) != {"role", "content"}:
        raise ValueError(f"{context} is not an exact role/content message")
    if not isinstance(message["role"], str):
        raise ValueError(f"{context}.role is not a string")
    if not isinstance(message["content"], str):
        raise ValueError(f"{context}.content is not a string")
    if not allow_empty and not message["content"].strip():
        raise ValueError(f"{context}.content is empty")
    return message


def _extract_trajectory(
    row: dict[str, Any],
    *,
    world: int,
    context: str,
    allow_empty_later: bool = False,
) -> tuple[list[str], list[str], list[dict[str, Any]]]:
    expected_keys = {
        "id",
        "user_history1",
        "user_history2",
        "evaluations1",
        "evaluations2",
        "generation_time1",
        "generation_time2",
    }
    if set(row) != expected_keys:
        raise ValueError(f"{context} has unexpected fields")
    history = row[f"user_history{world}"]
    evaluations = row[f"evaluations{world}"]
    if not isinstance(history, list) or not isinstance(evaluations, list):
        raise ValueError(f"{context} history/evaluations are not lists")
    if len(history) < 4 or len(history) % 2 != 0:
        raise ValueError(f"{context} history has an invalid length")
    messages = []
    for index, message in enumerate(history):
        messages.append(
            _require_message(
                message,
                f"{context}.history[{index}]",
                allow_empty=allow_empty_later and index >= 2,
            )
        )
    if messages[0]["role"] != "system":
        raise ValueError(f"{context} does not begin with a system message")
    expected_roles = [
        "user" if index % 2 else "assistant"
        for index in range(1, len(messages))
    ]
    if [message["role"] for message in messages[1:]] != expected_roles:
        raise ValueError(f"{context} messages do not alternate exactly")
    policy_messages = [message["content"] for message in messages[1::2]]
    observations = [message["content"] for message in messages[2::2]]
    if len(policy_messages) != len(evaluations):
        raise ValueError(f"{context} policy/evaluation counts differ")
    if len(observations) != len(policy_messages) - 1:
        raise ValueError(f"{context} observation count is invalid")
    if not 2 <= len(evaluations) <= MAX_TURNS:
        raise ValueError(f"{context} has an invalid number of turns")

    rewards: list[int] = []
    for turn, evaluation in enumerate(evaluations):
        evaluation_context = f"{context}.evaluations[{turn}]"
        expected_evaluation_keys = {
            "done",
            "generation_time",
            "invalid_responses",
            "questions",
            "total_reward",
        }
        if (
            not isinstance(evaluation, dict)
            or set(evaluation) != expected_evaluation_keys
        ):
            raise ValueError(f"{evaluation_context} has unexpected fields")
        reward = evaluation["total_reward"]
        if (
            not isinstance(reward, int)
            or isinstance(reward, bool)
            or not 0 <= reward <= CHECKLIST_SIZE
        ):
            raise ValueError(f"{evaluation_context}.total_reward is invalid")
        if not isinstance(evaluation["done"], bool):
            raise ValueError(f"{evaluation_context}.done is not boolean")
        if (
            not isinstance(evaluation["questions"], dict)
            or not all(
                isinstance(question, str) and question.strip()
                for question in evaluation["questions"]
            )
        ):
            raise ValueError(f"{evaluation_context}.questions is invalid")
        invalid = evaluation["invalid_responses"]
        if (
            not isinstance(invalid, int)
            or isinstance(invalid, bool)
            or invalid < 0
        ):
            raise ValueError(
                f"{evaluation_context}.invalid_responses is invalid"
            )
        generation_time = evaluation["generation_time"]
        if (
            not isinstance(generation_time, (int, float))
            or isinstance(generation_time, bool)
            or generation_time < 0
        ):
            raise ValueError(f"{evaluation_context}.generation_time is invalid")
        rewards.append(reward)
    if any(previous > current for previous, current in zip(rewards, rewards[1:])):
        raise ValueError(f"{context} reward trace is not monotone")
    if any(evaluation["done"] for evaluation in evaluations[:-1]):
        raise ValueError(f"{context} is marked done before its final turn")
    final_done = evaluations[-1]["done"]
    if final_done != (rewards[-1] == CHECKLIST_SIZE):
        raise ValueError(f"{context} final done/reward fields disagree")
    if not final_done and len(evaluations) != MAX_TURNS:
        raise ValueError(f"{context} stops incomplete before the turn limit")
    return policy_messages, observations, evaluations


def trajectory_metrics(
    row: dict[str, Any],
    *,
    seed_message: str,
    run: int,
    record_id: int,
    world: int,
    allow_empty_later: bool = False,
) -> dict[str, Any]:
    policy, observations, evaluations = _extract_trajectory(
        row,
        world=world,
        context=f"run{run}.id{record_id}.world{world}",
        allow_empty_later=allow_empty_later,
    )
    rewards = [evaluation["total_reward"] for evaluation in evaluations]

    seen = _tokens(seed_message) | _tokens(policy[0])
    immediate_uptake = 0
    shifted_uptake = 0
    for turn, observation in enumerate(observations):
        current_novel = _tokens(observation) - seen
        next_policy_tokens = _tokens(policy[turn + 1])
        immediate_uptake += len(current_novel & next_policy_tokens)

        shifted_observation = observations[(turn + 1) % len(observations)]
        shifted_novel = _tokens(shifted_observation) - seen
        shifted_uptake += len(shifted_novel & next_policy_tokens)
        seen |= _tokens(observation) | next_policy_tokens

    later_messages = policy[1:] + observations
    return {
        "run": run,
        "record_id": record_id,
        "world": world,
        "turns": len(evaluations),
        "initial_reward": rewards[0],
        "final_reward": rewards[-1],
        "delayed_gain": rewards[-1] - rewards[0],
        "completed": rewards[-1] == CHECKLIST_SIZE,
        "reward_trace": rewards,
        "transition_count": len(observations),
        "immediate_novel_token_uptake": immediate_uptake,
        "shifted_novel_token_uptake": shifted_uptake,
        "later_message_count": len(later_messages),
        "empty_later_message_count": sum(
            not message.strip() for message in later_messages
        ),
        "first_action_sha256": source_manifest.canonical_sha256(policy[0]),
        "history_sha256": source_manifest.canonical_sha256(
            row[f"user_history{world}"]
        ),
        "evaluations_sha256": source_manifest.canonical_sha256(evaluations),
    }


def _fraction(values: Iterable[bool]) -> float:
    values = list(values)
    if not values:
        raise ValueError("cannot compute a fraction over zero values")
    return sum(values) / len(values)


def summarize_metrics(
    trajectories: Sequence[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, bool]]:
    if not trajectories:
        raise ValueError("no trajectories were provided")
    total_transitions = sum(row["transition_count"] for row in trajectories)
    if total_transitions <= 0:
        raise ValueError("no trajectory transitions were provided")
    immediate_total = sum(
        row["immediate_novel_token_uptake"] for row in trajectories
    )
    shifted_total = sum(
        row["shifted_novel_token_uptake"] for row in trajectories
    )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in trajectories:
        grouped[(row["record_id"], row["world"])].append(row)
    if any(len(rows) != 3 for rows in grouped.values()):
        raise ValueError("each task/world must have exactly three runs")

    metrics = {
        "trajectories": len(trajectories),
        "task_worlds": len(grouped),
        "total_transitions": total_transitions,
        "multi_turn_fraction": _fraction(
            row["turns"] >= 2 for row in trajectories
        ),
        "initial_at_least_two_unresolved_fraction": _fraction(
            row["initial_reward"] <= CHECKLIST_SIZE - 2
            for row in trajectories
        ),
        "delayed_gain_at_least_two_fraction": _fraction(
            row["delayed_gain"] >= 2 for row in trajectories
        ),
        "mean_initial_reward": sum(
            row["initial_reward"] for row in trajectories
        )
        / len(trajectories),
        "mean_final_reward": sum(row["final_reward"] for row in trajectories)
        / len(trajectories),
        "mean_delayed_gain": sum(row["delayed_gain"] for row in trajectories)
        / len(trajectories),
        "mean_turns": sum(row["turns"] for row in trajectories)
        / len(trajectories),
        "full_completion_fraction": _fraction(
            row["completed"] for row in trajectories
        ),
        "immediate_novel_token_uptake_total": immediate_total,
        "shifted_novel_token_uptake_total": shifted_total,
        "novel_uptake_advantage_per_transition": (
            immediate_total - shifted_total
        )
        / total_transitions,
        "novel_uptake_positive_trajectory_fraction": _fraction(
            row["immediate_novel_token_uptake"]
            > row["shifted_novel_token_uptake"]
            for row in trajectories
        ),
        "turn_range_at_least_two_task_world_fraction": _fraction(
            max(row["turns"] for row in rows)
            - min(row["turns"] for row in rows)
            >= 2
            for rows in grouped.values()
        ),
        "three_distinct_first_actions_task_world_fraction": _fraction(
            len({row["first_action_sha256"] for row in rows}) == 3
            for rows in grouped.values()
        ),
        "final_reward_varies_task_world_fraction": _fraction(
            len({row["final_reward"] for row in rows}) > 1
            for rows in grouped.values()
        ),
    }
    gates = {
        name: metrics[name] >= threshold
        for name, threshold in THRESHOLDS.items()
    }
    return metrics, gates


def build_audit(
    *,
    manifest_path: Path,
    settings_path: Path,
    trajectory_paths: Sequence[Path],
) -> dict[str, Any]:
    if len(trajectory_paths) != 3:
        raise ValueError("exactly three trajectory files are required")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("interface_version")
        != "infoquest-llm-bed-manifest-1"
    ):
        raise ValueError("unexpected InfoQuest manifest interface")
    if (
        manifest["source"]["revision"] != source_manifest.SOURCE_REVISION
        or manifest["selection"]["combined_splits_sha256"]
        != source_manifest.EXPECTED_COMBINED_SPLITS_HASH
    ):
        raise ValueError("InfoQuest manifest does not match the frozen source")
    opportunity = manifest["selection"]["splits"]["opportunity"]
    opportunity_ids = opportunity["record_ids"]
    if (
        opportunity["ordered_sha256"]
        != source_manifest.EXPECTED_SPLIT_HASHES["opportunity"]
        or source_manifest.canonical_sha256(opportunity_ids)
        != opportunity["ordered_sha256"]
    ):
        raise ValueError("InfoQuest opportunity split changed")
    if QUARANTINED_IDS & set(opportunity_ids):
        raise ValueError("a quarantined InfoQuest id entered opportunity")

    expected_settings_hash = source_manifest.SOURCE_SHA256["settings"]
    settings_rows = _rows_by_id(
        settings_path,
        expected_sha256=expected_settings_hash,
    )
    runs = [
        _rows_by_id(path, expected_sha256=expected_hash)
        for path, expected_hash in zip(trajectory_paths, BASELINE_SHA256)
    ]

    trajectories = []
    for run, rows_by_id in enumerate(runs):
        for record_id in opportunity_ids:
            setting = settings_rows[record_id]
            if setting["id"] != record_id:
                raise ValueError("InfoQuest setting id changed")
            seed_message = setting["seed_message"]
            if not isinstance(seed_message, str) or not seed_message.strip():
                raise ValueError("InfoQuest seed message is empty")
            for world in (1, 2):
                trajectories.append(
                    trajectory_metrics(
                        rows_by_id[record_id],
                        seed_message=seed_message,
                        run=run,
                        record_id=record_id,
                        world=world,
                    )
                )
    metrics, scientific_gates = summarize_metrics(trajectories)
    structural_gates = {
        "exactly_three_hash_pinned_runs": True,
        "each_run_has_exact_ids_0_through_499": True,
        "opportunity_split_hash_matches": True,
        "quarantined_ids_absent": True,
        "exactly_480_opportunity_trajectories": (
            len(trajectories) == 3 * 80 * 2
        ),
        "all_histories_and_evaluations_validate": True,
        "all_reward_traces_monotone": True,
        "semantic_content_emitted": False,
        "openrouter_calls_zero": True,
        "oatml_jobs_zero": True,
    }
    gates = {**structural_gates, **scientific_gates}
    return {
        "interface_version": INTERFACE_VERSION,
        "source": {
            "repository": source_manifest.SOURCE_REPOSITORY,
            "revision": source_manifest.SOURCE_REVISION,
            "manifest_sha256": source_manifest.sha256_file(manifest_path),
            "settings_sha256": expected_settings_hash,
            "trajectory_files": [
                {
                    "filename": filename,
                    "sha256": digest,
                }
                for filename, digest in zip(
                    BASELINE_FILENAMES,
                    BASELINE_SHA256,
                )
            ],
        },
        "access": {
            "split": "opportunity",
            "record_ids": opportunity_ids,
            "ordered_sha256": opportunity["ordered_sha256"],
            "quarantined_ids": sorted(QUARANTINED_IDS),
            "development_read": False,
            "effective_holdout_read": False,
        },
        "thresholds": THRESHOLDS,
        "metrics": metrics,
        "gates": gates,
        "passed": all(gates.values()),
        "trajectories": trajectories,
        "semantic_content_emitted": False,
        "seed_message_content_emitted": False,
        "hidden_setting_content_emitted": False,
        "message_content_emitted": False,
        "checklist_content_emitted": False,
        "causal_policy_efficacy_claimed": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument(
        "--trajectory",
        type=Path,
        action="append",
        required=True,
        help="Pass exactly three times in run-index order.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_audit(
        manifest_path=args.manifest,
        settings_path=args.settings,
        trajectory_paths=args.trajectory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
