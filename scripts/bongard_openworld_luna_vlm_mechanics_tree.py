#!/usr/bin/env python3
"""Run the shared four-task Luna semantic-belief mechanics tree."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Callable, Mapping, Protocol, Sequence
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from helpers import Config, ModelSpec
from scripts import bongard_openworld_luna_vlm_serving_smoke as serving
from scripts import bongard_openworld_vlm_bed as bed
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint
from scripts.number_game_deepseek_planner_serving_smoke import summarize_usage
from scripts.openrouter_daily_budget import read_live_credits, require_budget


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-luna-vlm-mechanics-tree-12"
MODEL_ID = serving.MODEL_ID
MODEL_SEED = 2_026_081_021
RANDOM_SEED = 2_026_081_022
ROOT_REQUESTS = 4
DYNAMIC_BRANCH_REQUESTS = 64
HISTORY_BLIND_BRANCH_REQUESTS = 64
BRANCH_REQUESTS = DYNAMIC_BRANCH_REQUESTS
FIRST_STAGE_REQUESTS = (
    ROOT_REQUESTS + DYNAMIC_BRANCH_REQUESTS + HISTORY_BLIND_BRANCH_REQUESTS
)
MAX_FINAL_REQUESTS = 44
MAX_REQUESTS = FIRST_STAGE_REQUESTS + MAX_FINAL_REQUESTS
FINAL_SEED_OFFSET = 1_000_000
CONCURRENCY = 24
MAX_TOKENS = serving.MAX_TOKENS
TEMPERATURE = 0.0
RUN_BUDGET_USD = 1.75
MIN_MATERIAL_BRANCH_PAIRS = 24
MIN_MYOPIC_BRIER = 0.03
MIN_MYOPIC_LOG_LOSS = 0.15
MIN_ACTION_MARGIN_NATS = 1e-6
SCORE_OBJECTIVE = "endpoint_predictive_information_gain_nats"
POLICIES = (
    "myopic_width",
    "fixed_depth2",
    "fixed_score_dynamic_update",
    "dynamic_depth2",
    "history_blind_update_matched_first",
    "shuffled_dynamic_depth2",
    "history_blind_depth2",
    "random",
)
SCORE_POLICIES = (
    "myopic_width",
    "fixed_depth2",
    "dynamic_depth2",
    "shuffled_dynamic_depth2",
    "history_blind_depth2",
)


class StructuredModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, Any]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def chat_complete_seeded_messages_batched_structured(
        self,
        batch_messages: Sequence[list[dict[str, Any]]],
        seeds: Sequence[int],
        *,
        temperature: float,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class BeliefCase:
    case_id: str
    task: bed.VisualTask
    history: tuple[tuple[str, bool], ...]
    kind: str
    candidate_id: str | None = None
    simulated_label: bool | None = None


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def history_key(history: Sequence[tuple[str, bool]]) -> str:
    return ";".join(
        f"{image_id}:{int(label)}" for image_id, label in sorted(history)
    )


def message_sha256(messages: Sequence[Mapping[str, Any]]) -> str:
    return hashlib.sha256(
        bed.canonical_json(messages).encode("utf-8")
    ).hexdigest()


def _request_seed_key(case: BeliefCase) -> tuple[str, ...]:
    if case.kind == "root":
        return ("root", case.task.task_id)
    if case.kind in {"branch", "history_blind"}:
        if case.candidate_id is None or case.simulated_label is None:
            raise ValueError("branch seed key lacks a candidate or label")
        return (
            "branch_pair",
            case.task.task_id,
            case.candidate_id,
            str(int(case.simulated_label)),
        )
    if case.kind == "final":
        return ("final", case.task.task_id, history_key(case.history))
    raise ValueError(f"unknown belief case kind {case.kind!r}")


def request_seeds_for_cases(
    cases: Sequence[BeliefCase],
    *,
    base_seed: int,
    final_stage: bool = False,
) -> list[int]:
    if final_stage:
        if any(case.kind != "final" for case in cases):
            raise ValueError("final-stage seed control accepts only final cases")
        # Common random numbers remove per-history seed luck from terminal
        # policy comparisons while retaining independent seeds across tasks.
        keys = [("final_task", case.task.task_id) for case in cases]
    else:
        keys = [_request_seed_key(case) for case in cases]
    unique_keys = sorted(set(keys))
    offset = FINAL_SEED_OFFSET if final_stage else 0
    seed_by_key = {
        key: base_seed + offset + index
        for index, key in enumerate(unique_keys)
    }
    seeds = [seed_by_key[key] for key in keys]
    if len(set(seed_by_key.values())) != len(unique_keys):
        raise AssertionError("distinct request seed keys collided")
    return seeds


def first_stage_cases(tasks: Sequence[bed.VisualTask]) -> list[BeliefCase]:
    ordered = sorted(tasks, key=lambda task: task.task_id)
    if len(ordered) != ROOT_REQUESTS:
        raise ValueError("full mechanics tree requires exactly four tasks")
    roots = [
        BeliefCase(
            case_id=f"{task.task_id}-root",
            task=task,
            history=task.initial_history,
            kind="root",
        )
        for task in ordered
    ]
    paired_branches = []
    for task in ordered:
        for candidate_id in task.candidate_ids:
            for label in (False, True):
                suffix = "positive" if label else "negative"
                paired_branches.extend(
                    [
                        BeliefCase(
                            case_id=f"{task.task_id}-{candidate_id}-{suffix}",
                            task=task,
                            history=tuple(
                                sorted(
                                    (*task.initial_history, (candidate_id, label))
                                )
                            ),
                            kind="branch",
                            candidate_id=candidate_id,
                            simulated_label=label,
                        ),
                        BeliefCase(
                            case_id=(
                                f"{task.task_id}-history-blind-"
                                f"{candidate_id}-{suffix}"
                            ),
                            task=task,
                            history=task.initial_history,
                            kind="history_blind",
                            candidate_id=candidate_id,
                            simulated_label=label,
                        ),
                    ]
                )
    cases = roots + paired_branches
    if len(cases) != FIRST_STAGE_REQUESTS:
        raise AssertionError("first-stage request count changed")
    return cases


def paired_request_diagnostics(
    *,
    cases: Sequence[BeliefCase],
    messages: Sequence[Sequence[Mapping[str, Any]]],
    seeds: Sequence[int],
    batch_size: int = CONCURRENCY,
) -> dict[str, Any]:
    if not (len(cases) == len(messages) == len(seeds)):
        raise ValueError("paired request manifest lengths differ")
    if batch_size <= 0:
        raise ValueError("paired request batch size must be positive")
    root_cases = [case for case in cases if case.kind == "root"]
    roots = {
        case.task.task_id: (case, message, seed, index)
        for index, (case, message, seed) in enumerate(
            zip(cases, messages, seeds, strict=True)
        )
        if case.kind == "root"
    }
    dynamic = {
        (case.task.task_id, str(case.candidate_id), bool(case.simulated_label)): (
            case,
            message,
            seed,
            index,
        )
        for index, (case, message, seed) in enumerate(
            zip(cases, messages, seeds, strict=True)
        )
        if case.kind == "branch"
    }
    blind = {
        (case.task.task_id, str(case.candidate_id), bool(case.simulated_label)): (
            case,
            message,
            seed,
            index,
        )
        for index, (case, message, seed) in enumerate(
            zip(cases, messages, seeds, strict=True)
        )
        if case.kind == "history_blind"
    }
    expected_pairs = {
        (case.task.task_id, candidate, label)
        for case in cases
        if case.kind == "root"
        for candidate in case.task.candidate_ids
        for label in (False, True)
    }
    expected_task_ids = {task_id for task_id, _, _ in expected_pairs}
    exact_keys = (
        len(dynamic) == len(blind) == len(expected_pairs)
        and set(dynamic) == set(blind) == expected_pairs
    )
    pair_rows = []
    if exact_keys:
        for key in sorted(expected_pairs):
            (
                dynamic_case,
                dynamic_message,
                dynamic_seed,
                dynamic_index,
            ) = dynamic[key]
            blind_case, blind_message, blind_seed, blind_index = blind[key]
            task_id, candidate_id, label = key
            root_case, root_message, root_seed, _ = roots[task_id]
            blind_payload = bed.request_payload(blind_message)
            dynamic_payload = bed.request_payload(dynamic_message)
            expected_payload = json.loads(json.dumps(blind_payload))
            expected_payload["observed_labels"] = sorted(
                [
                    *blind_payload["observed_labels"],
                    {
                        "image_id": candidate_id,
                        "label": bed.LABELS[label],
                    },
                ],
                key=lambda row: row["image_id"],
            )
            pair_rows.append(
                {
                    "task_id": task_id,
                    "candidate_id": candidate_id,
                    "simulated_label": bed.LABELS[label],
                    "paired_seed": dynamic_seed,
                    "same_requested_seed": dynamic_seed == blind_seed,
                    "adjacent_dynamic_then_blind": (
                        blind_index == dynamic_index + 1
                    ),
                    "same_dispatch_batch": (
                        dynamic_index // batch_size == blind_index // batch_size
                    ),
                    "blind_history_is_initial": (
                        blind_case.history == blind_case.task.initial_history
                    ),
                    "dynamic_history_adds_exact_answer": (
                        dynamic_case.history
                        == tuple(
                            sorted(
                                (
                                    *dynamic_case.task.initial_history,
                                    (candidate_id, label),
                                )
                            )
                        )
                    ),
                    "blind_prompt_matches_root": (
                        bed.canonical_json(blind_message)
                        == bed.canonical_json(root_message)
                        and blind_case.history == root_case.history
                    ),
                    "prompts_differ_only_by_simulated_answer": (
                        bed.canonical_json(dynamic_payload)
                        == bed.canonical_json(expected_payload)
                    ),
                    "root_seed_is_distinct": root_seed != dynamic_seed,
                    "dynamic_request_sha256": message_sha256(dynamic_message),
                    "blind_request_sha256": message_sha256(blind_message),
                }
            )
    pair_seeds = [row["paired_seed"] for row in pair_rows]
    gates = {
        "exact_root_count": (
            len(root_cases) == len(roots) == len(expected_task_ids)
        ),
        "exact_dynamic_and_blind_pair_maps": exact_keys,
        "all_pairs_share_requested_seed": all(
            row["same_requested_seed"] for row in pair_rows
        ),
        "distinct_pairs_use_distinct_seeds": (
            len(pair_seeds) == len(set(pair_seeds)) == len(expected_pairs)
        ),
        "each_pair_is_adjacent_dynamic_then_blind": all(
            row["adjacent_dynamic_then_blind"] for row in pair_rows
        ),
        "each_pair_shares_one_dispatch_batch": all(
            row["same_dispatch_batch"] for row in pair_rows
        ),
        "all_blind_histories_are_initial_only": all(
            row["blind_history_is_initial"] for row in pair_rows
        ),
        "all_blind_prompts_match_ordinary_root_prompt": all(
            row["blind_prompt_matches_root"] for row in pair_rows
        ),
        "all_dynamic_prompts_add_only_the_simulated_answer": all(
            row["dynamic_history_adds_exact_answer"]
            and row["prompts_differ_only_by_simulated_answer"]
            for row in pair_rows
        ),
        "root_and_pair_seeds_are_distinct": all(
            row["root_seed_is_distinct"] for row in pair_rows
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "pair_count": len(pair_rows),
        "unique_pair_seed_count": len(set(pair_seeds)),
        "gates": gates,
        "pairs": pair_rows,
    }


def rotated_candidate_mapping(task: bed.VisualTask) -> dict[str, str]:
    candidates = tuple(sorted(task.candidate_ids))
    mapping = {
        candidate: candidates[(index + 1) % len(candidates)]
        for index, candidate in enumerate(candidates)
    }
    if (
        set(mapping) != set(candidates)
        or set(mapping.values()) != set(candidates)
        or any(source == target for source, target in mapping.items())
    ):
        raise AssertionError("shuffled branch mapping has a fixed point")
    return mapping


def shuffled_continuation_control(
    *,
    task: bed.VisualTask,
    myopic_scores: Mapping[str, float],
    dynamic_scores: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, str], dict[str, dict[str, float]]]:
    """Permute complete continuation values without mismatching query sets."""
    candidates = tuple(sorted(task.candidate_ids))
    if set(myopic_scores) != set(candidates) or set(dynamic_scores) != set(
        candidates
    ):
        raise ValueError("shuffled control score maps do not match candidates")
    mapping = rotated_candidate_mapping(task)
    dynamic_future = {
        candidate: dynamic_scores[candidate] - myopic_scores[candidate]
        for candidate in candidates
    }
    if any(not math.isfinite(value) for value in dynamic_future.values()):
        raise ValueError("dynamic continuation values must be finite")
    shuffled_future = {
        candidate: dynamic_future[mapping[candidate]]
        for candidate in candidates
    }
    shuffled_scores = {
        candidate: myopic_scores[candidate] + shuffled_future[candidate]
        for candidate in candidates
    }
    if sorted(dynamic_future.values()) != sorted(shuffled_future.values()):
        raise AssertionError("shuffled continuation values are not a permutation")
    return (
        shuffled_scores,
        mapping,
        {
            "dynamic_expected_continuation_utility": dynamic_future,
            "shuffled_expected_continuation_utility": shuffled_future,
        },
    )


def selection_margin(scores: Mapping[str, float], selected: str) -> float:
    alternatives = [value for key, value in scores.items() if key != selected]
    if selected not in scores or not alternatives:
        raise ValueError("selection margin requires a selected action and alternative")
    return scores[selected] - max(alternatives)


def _random_choices(task: bed.VisualTask) -> tuple[str, str]:
    offset = int.from_bytes(
        hashlib.sha256(task.task_id.encode()).digest()[:8], "big"
    )
    rng = random.Random(RANDOM_SEED + offset)
    return tuple(rng.sample(list(task.candidate_ids), 2))  # type: ignore[return-value]


def plan_task_policies(
    *,
    task: bed.VisualTask,
    root: bed.SemanticBelief,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
    history_blind_branches: Mapping[
        tuple[str, bool], bed.SemanticBelief
    ],
) -> dict[str, Any]:
    candidates = tuple(task.candidate_ids)
    endpoint_ids = tuple(task.endpoint_ids)
    myopic_scores = bed.candidate_endpoint_eigs(
        root, candidates, endpoint_ids
    )
    fixed_scores = bed.fixed_support_endpoint_depth_two_scores(
        root, candidates, endpoint_ids
    )
    dynamic_scores = bed.dynamic_support_endpoint_depth_two_scores(
        root, candidates, endpoint_ids, branches
    )
    history_blind_scores = bed.history_blind_endpoint_depth_two_scores(
        root, candidates, endpoint_ids, history_blind_branches
    )
    hypothesis_eig_diagnostics = bed.candidate_eigs(root, candidates)
    (
        shuffled_scores,
        shuffled_mapping,
        continuation_values,
    ) = shuffled_continuation_control(
        task=task,
        myopic_scores=myopic_scores,
        dynamic_scores=dynamic_scores,
    )
    first_by_policy = {
        "myopic_width": bed.select_best(myopic_scores),
        "fixed_depth2": bed.select_best(fixed_scores),
        "fixed_score_dynamic_update": bed.select_best(fixed_scores),
        "dynamic_depth2": bed.select_best(dynamic_scores),
        "history_blind_update_matched_first": bed.select_best(dynamic_scores),
        "shuffled_dynamic_depth2": bed.select_best(shuffled_scores),
        "history_blind_depth2": bed.select_best(history_blind_scores),
    }
    random_first, random_second = _random_choices(task)
    first_by_policy["random"] = random_first

    policies = {}
    for policy in POLICIES:
        first = first_by_policy[policy]
        first_label = bool(task.actual_labels[first])
        remaining = tuple(
            candidate for candidate in candidates if candidate != first
        )
        if policy == "random":
            second = random_second
            second_scores = None
        elif policy == "fixed_depth2":
            fixed_weights = bed.updated_weights_for_label(
                root, first, first_label
            )
            second_scores = bed.candidate_endpoint_eigs(
                root, remaining, endpoint_ids, weights=fixed_weights
            )
            second = bed.select_best(second_scores)
        elif policy == "history_blind_update_matched_first":
            blind = history_blind_branches[(first, first_label)]
            if blind.history != root.history:
                raise ValueError(
                    "matched history-blind updater must contain only root history"
                )
            blind_weights = bed.updated_weights_for_label(
                blind, first, first_label
            )
            second_scores = bed.candidate_endpoint_eigs(
                blind, remaining, endpoint_ids, weights=blind_weights
            )
            second = bed.select_best(second_scores)
        else:
            realized_branch = branches[(first, first_label)]
            second_scores = bed.candidate_endpoint_eigs(
                realized_branch, remaining, endpoint_ids
            )
            second = bed.select_best(second_scores)
        second_label = bool(task.actual_labels[second])
        final_history = tuple(
            sorted(
                (
                    *task.initial_history,
                    (first, first_label),
                    (second, second_label),
                )
            )
        )
        policies[policy] = {
            "first_image_id": first,
            "first_label": bed.LABELS[first_label],
            "second_image_id": second,
            "second_label": bed.LABELS[second_label],
            "final_history": final_history,
            "final_history_key": history_key(final_history),
            "first_score": (
                None
                if policy == "random"
                else {
                    "myopic_width": myopic_scores,
                    "fixed_depth2": fixed_scores,
                    "fixed_score_dynamic_update": fixed_scores,
                    "dynamic_depth2": dynamic_scores,
                    "history_blind_update_matched_first": dynamic_scores,
                    "shuffled_dynamic_depth2": shuffled_scores,
                    "history_blind_depth2": history_blind_scores,
                }[policy][first]
            ),
            "first_score_margin": (
                None
                if policy == "random"
                else selection_margin(
                    {
                        "myopic_width": myopic_scores,
                        "fixed_depth2": fixed_scores,
                        "fixed_score_dynamic_update": fixed_scores,
                        "dynamic_depth2": dynamic_scores,
                        "history_blind_update_matched_first": dynamic_scores,
                        "shuffled_dynamic_depth2": shuffled_scores,
                        "history_blind_depth2": history_blind_scores,
                    }[policy],
                    first,
                )
            ),
            "second_scores": second_scores,
            "second_score_margin": (
                None
                if second_scores is None
                else selection_margin(second_scores, second)
            ),
        }
    return {
        "score_objective": SCORE_OBJECTIVE,
        "root_hypothesis_eig_diagnostics": hypothesis_eig_diagnostics,
        "root_scores": {
            "myopic_width": myopic_scores,
            "fixed_depth2": fixed_scores,
            "fixed_score_dynamic_update": fixed_scores,
            "dynamic_depth2": dynamic_scores,
            "history_blind_update_matched_first": dynamic_scores,
            "shuffled_dynamic_depth2": shuffled_scores,
            "history_blind_depth2": history_blind_scores,
        },
        "shuffled_branch_mapping": shuffled_mapping,
        "continuation_values": continuation_values,
        "policies": policies,
    }


def final_cases(
    tasks: Sequence[bed.VisualTask],
    plans: Mapping[str, Mapping[str, Any]],
) -> list[BeliefCase]:
    cases = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        seen = set()
        for policy in (
            "dynamic_depth2",
            "history_blind_depth2",
            "history_blind_update_matched_first",
            "myopic_width",
            "fixed_depth2",
            "fixed_score_dynamic_update",
            "shuffled_dynamic_depth2",
            "random",
        ):
            policy_row = plans[task.task_id]["policies"][policy]
            key = policy_row["final_history_key"]
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                BeliefCase(
                    case_id=(
                        f"{task.task_id}-final-"
                        f"{hashlib.sha256(key.encode()).hexdigest()[:12]}"
                    ),
                    task=task,
                    history=policy_row["final_history"],
                    kind="final",
                )
            )
    if not len(tasks) <= len(cases) <= len(tasks) * len(POLICIES):
        raise ValueError("distinct final history count is outside task/policy bounds")
    return cases


def all_first_action_paths(
    *,
    task: bed.VisualTask,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> dict[str, dict[str, Any]]:
    paths = {}
    for first in task.candidate_ids:
        first_label = bool(task.actual_labels[first])
        remaining = tuple(
            candidate for candidate in task.candidate_ids if candidate != first
        )
        second_scores = bed.candidate_endpoint_eigs(
            branches[(first, first_label)], remaining, task.endpoint_ids
        )
        second = bed.select_best(second_scores)
        second_label = bool(task.actual_labels[second])
        final_history = tuple(
            sorted(
                (
                    *task.initial_history,
                    (first, first_label),
                    (second, second_label),
                )
            )
        )
        paths[first] = {
            "first_image_id": first,
            "first_label": bed.LABELS[first_label],
            "second_image_id": second,
            "second_label": bed.LABELS[second_label],
            "second_scores": second_scores,
            "final_history": final_history,
            "final_history_key": history_key(final_history),
        }
    return paths


def fixed_score_dynamic_update_is_exact(tree: Mapping[str, Any]) -> bool:
    matched = tree["policies"]["fixed_score_dynamic_update"]
    fixed = tree["policies"]["fixed_depth2"]
    return (
        matched["first_image_id"] == fixed["first_image_id"]
        and tree["root_scores"]["fixed_score_dynamic_update"]
        == tree["root_scores"]["fixed_depth2"]
        and matched["final_history_key"]
        == tree["all_first_action_paths"][matched["first_image_id"]][
            "final_history_key"
        ]
    )


def history_blind_update_matched_first_is_exact(
    tree: Mapping[str, Any],
) -> bool:
    matched = tree["policies"]["history_blind_update_matched_first"]
    dynamic = tree["policies"]["dynamic_depth2"]
    return (
        matched["first_image_id"] == dynamic["first_image_id"]
        and matched["first_label"] == dynamic["first_label"]
        and matched["first_score"] == dynamic["first_score"]
        and matched["first_score_margin"] == dynamic["first_score_margin"]
        and tree["root_scores"]["history_blind_update_matched_first"]
        == tree["root_scores"]["dynamic_depth2"]
        and matched["second_scores"] is not None
    )


def all_action_final_cases(
    *,
    tasks: Sequence[bed.VisualTask],
    plans: Mapping[str, Mapping[str, Any]],
    action_paths: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> list[BeliefCase]:
    cases = []
    for task in sorted(tasks, key=lambda item: item.task_id):
        histories = [
            plans[task.task_id]["policies"][policy]["final_history"]
            for policy in (
                "dynamic_depth2",
                "history_blind_depth2",
                "history_blind_update_matched_first",
                "myopic_width",
                "fixed_depth2",
                "fixed_score_dynamic_update",
                "shuffled_dynamic_depth2",
                "random",
            )
        ] + [
            action_paths[task.task_id][first]["final_history"]
            for first in sorted(action_paths[task.task_id])
        ]
        seen = set()
        for history in histories:
            key = history_key(history)
            if key in seen:
                continue
            seen.add(key)
            cases.append(
                BeliefCase(
                    case_id=(
                        f"{task.task_id}-final-"
                        f"{hashlib.sha256(key.encode()).hexdigest()[:12]}"
                    ),
                    task=task,
                    history=history,
                    kind="final",
                )
            )
    if not len(tasks) * 4 <= len(cases) <= len(tasks) * 11:
        raise ValueError("all-action final history count is outside bounds")
    return cases


def task_preserving_dispatch_batches(
    cases: Sequence[BeliefCase], *, batch_size: int = CONCURRENCY
) -> list[dict[str, Any]]:
    """Pack contiguous final-task groups without crossing dispatch boundaries."""
    if batch_size <= 0:
        raise ValueError("terminal dispatch batch size must be positive")
    if any(case.kind != "final" for case in cases):
        raise ValueError("terminal dispatch accepts only final cases")
    if not cases:
        return []

    task_groups: list[dict[str, Any]] = []
    seen_task_ids: set[str] = set()
    start = 0
    while start < len(cases):
        task_id = cases[start].task.task_id
        if task_id in seen_task_ids:
            raise ValueError("terminal task cases are not contiguous")
        seen_task_ids.add(task_id)
        stop = start + 1
        while stop < len(cases) and cases[stop].task.task_id == task_id:
            stop += 1
        if stop - start > batch_size:
            raise ValueError("one terminal task group exceeds dispatch batch size")
        task_groups.append(
            {"task_id": task_id, "start": start, "stop": stop}
        )
        start = stop

    batches: list[dict[str, Any]] = []
    batch_start = task_groups[0]["start"]
    batch_stop = batch_start
    batch_tasks: list[str] = []
    for group in task_groups:
        group_size = group["stop"] - group["start"]
        if batch_tasks and group["stop"] - batch_start > batch_size:
            batches.append(
                {
                    "batch_index": len(batches),
                    "start": batch_start,
                    "stop": batch_stop,
                    "request_count": batch_stop - batch_start,
                    "task_ids": batch_tasks,
                }
            )
            batch_start = group["start"]
            batch_tasks = []
        batch_stop = group["stop"]
        batch_tasks.append(group["task_id"])
        if group_size <= 0:
            raise AssertionError("empty terminal task group")
    batches.append(
        {
            "batch_index": len(batches),
            "start": batch_start,
            "stop": batch_stop,
            "request_count": batch_stop - batch_start,
            "task_ids": batch_tasks,
        }
    )
    return batches


def final_request_diagnostics(
    *,
    cases: Sequence[BeliefCase],
    seeds: Sequence[int],
    plans: Mapping[str, Mapping[str, Any]],
    dispatch_batches: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Audit task-level terminal common-random-number pairing."""
    if len(cases) != len(seeds):
        raise ValueError("final request manifest lengths differ")
    index_by_key = {
        (case.task.task_id, history_key(case.history)): index
        for index, case in enumerate(cases)
    }
    exact_unique_cases = (
        len(index_by_key) == len(cases)
        and all(case.kind == "final" for case in cases)
    )
    batches = list(
        task_preserving_dispatch_batches(cases)
        if dispatch_batches is None
        else dispatch_batches
    )
    batch_by_index: dict[int, int] = {}
    valid_batches = bool(batches)
    expected_start = 0
    for batch_index, batch in enumerate(batches):
        start = int(batch.get("start", -1))
        stop = int(batch.get("stop", -1))
        request_count = int(batch.get("request_count", -1))
        valid_batches = valid_batches and (
            batch.get("batch_index") == batch_index
            and start == expected_start
            and start < stop <= len(cases)
            and request_count == stop - start
            and request_count <= CONCURRENCY
            and batch.get("task_ids")
            == list(dict.fromkeys(case.task.task_id for case in cases[start:stop]))
        )
        for index in range(max(0, start), min(len(cases), stop)):
            if index in batch_by_index:
                valid_batches = False
            batch_by_index[index] = batch_index
        expected_start = stop
    valid_batches = valid_batches and expected_start == len(cases)

    rows = []
    if exact_unique_cases and set(plans) == {
        case.task.task_id for case in cases
    }:
        for task_id in sorted(plans):
            task_indices = [
                index
                for index, case in enumerate(cases)
                if case.task.task_id == task_id
            ]
            dynamic_key = plans[task_id]["policies"]["dynamic_depth2"][
                "final_history_key"
            ]
            blind_key = plans[task_id]["policies"]["history_blind_depth2"][
                "final_history_key"
            ]
            matched_update_key = plans[task_id]["policies"][
                "history_blind_update_matched_first"
            ]["final_history_key"]
            dynamic_index = index_by_key[(task_id, dynamic_key)]
            blind_index = index_by_key[(task_id, blind_key)]
            matched_update_index = index_by_key[(task_id, matched_update_key)]
            rows.append(
                {
                    "task_id": task_id,
                    "final_request_count": len(task_indices),
                    "shared_task_seed": len({seeds[index] for index in task_indices})
                    == 1,
                    "task_seed": seeds[task_indices[0]],
                    "dispatch_batch_index": batch_by_index.get(task_indices[0]),
                    "task_cases_are_contiguous": task_indices
                    == list(range(min(task_indices), max(task_indices) + 1)),
                    "task_cases_share_one_dispatch_batch": len(
                        {batch_by_index.get(index) for index in task_indices}
                    )
                    == 1,
                    "dynamic_history_equals_history_blind": (
                        dynamic_key == blind_key
                    ),
                    "dynamic_then_history_blind_are_adjacent_when_distinct": (
                        dynamic_key == blind_key
                        or blind_index == dynamic_index + 1
                    ),
                    "dynamic_and_history_blind_share_seed": (
                        seeds[dynamic_index] == seeds[blind_index]
                    ),
                    "dynamic_and_history_blind_share_dispatch_batch": (
                        batch_by_index.get(dynamic_index)
                        == batch_by_index.get(blind_index)
                    ),
                    "dynamic_history_equals_matched_history_blind_update": (
                        dynamic_key == matched_update_key
                    ),
                    "dynamic_and_matched_history_blind_update_share_seed": (
                        seeds[dynamic_index] == seeds[matched_update_index]
                    ),
                    "dynamic_and_matched_history_blind_update_share_dispatch_batch": (
                        batch_by_index.get(dynamic_index)
                        == batch_by_index.get(matched_update_index)
                    ),
                }
            )
    task_seeds = [row["task_seed"] for row in rows]
    gates = {
        "exact_unique_final_case_map": exact_unique_cases,
        "exact_task_plan_coverage": len(rows) == len(plans) > 0,
        "dispatch_batches_exactly_cover_cases_within_limit": valid_batches,
        "all_final_histories_within_task_share_one_seed": all(
            row["shared_task_seed"] for row in rows
        ),
        "distinct_tasks_use_distinct_terminal_seeds": (
            len(task_seeds) == len(set(task_seeds))
        ),
        "all_task_final_case_groups_are_contiguous": all(
            row["task_cases_are_contiguous"] for row in rows
        ),
        "all_task_final_case_groups_share_one_dispatch_batch": all(
            row["task_cases_share_one_dispatch_batch"] for row in rows
        ),
        "dynamic_and_history_blind_terminal_pairs_are_ordered": all(
            row["dynamic_then_history_blind_are_adjacent_when_distinct"]
            for row in rows
        ),
        "dynamic_and_history_blind_terminal_pairs_share_seed": all(
            row["dynamic_and_history_blind_share_seed"] for row in rows
        ),
        "dynamic_and_history_blind_terminal_pairs_share_dispatch_batch": all(
            row["dynamic_and_history_blind_share_dispatch_batch"] for row in rows
        ),
        "dynamic_and_matched_history_blind_update_terminal_pairs_share_seed": all(
            row["dynamic_and_matched_history_blind_update_share_seed"]
            for row in rows
        ),
        "dynamic_and_matched_history_blind_update_terminal_pairs_share_dispatch_batch": all(
            row[
                "dynamic_and_matched_history_blind_update_share_dispatch_batch"
            ]
            for row in rows
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "task_count": len(rows),
        "dispatch_batch_count": len(batches),
        "dispatch_batches": batches,
        "gates": gates,
        "tasks": rows,
    }


def _average_ranks(values: Sequence[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for position in range(start, end):
            ranks[order[position]] = rank
        start = end
    return ranks


def spearman_correlation(left: Sequence[float], right: Sequence[float]) -> float:
    left_ranks = _average_ranks(left)
    right_ranks = _average_ranks(right)
    left_mean = sum(left_ranks) / len(left_ranks)
    right_mean = sum(right_ranks) / len(right_ranks)
    numerator = sum(
        (a - left_mean) * (b - right_mean)
        for a, b in zip(left_ranks, right_ranks, strict=True)
    )
    left_scale = math.sqrt(sum((value - left_mean) ** 2 for value in left_ranks))
    right_scale = math.sqrt(sum((value - right_mean) ** 2 for value in right_ranks))
    if left_scale == 0 or right_scale == 0:
        return 0.0
    return numerator / (left_scale * right_scale)


def verify_serving_result(
    path: Path,
    *,
    tasks: Sequence[bed.VisualTask] | None = None,
) -> dict[str, Any]:
    result = json.loads(path.read_text(encoding="utf-8"))
    protocol = result.get("protocol") or {}
    if (
        result.get("status") != "passed"
        or protocol.get("interface_version") != serving.INTERFACE_VERSION
        or protocol.get("model") != MODEL_ID
        or protocol.get("actual_candidate_labels_accessed") is not False
        or protocol.get("endpoint_labels_accessed") is not False
        or not result.get("gates")
        or not all(result["gates"].values())
    ):
        raise ValueError("serving result is not the frozen clean pass")
    raw_path = path.parent / "private/RAW_RESPONSES.json"
    if sha256_file(raw_path) != result.get("raw_responses_sha256"):
        raise ValueError("serving raw response hash changed")
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    cases = serving.build_smoke_cases(tasks or bed.load_mechanics_tasks())
    if raw.get("case_ids") != [case.case_id for case in cases]:
        raise ValueError("serving raw case order changed")
    responses = raw.get("responses") or []
    if len(responses) != serving.EXPECTED_REQUESTS:
        raise ValueError("serving raw response count changed")
    messages = [
        bed.build_belief_messages(case.task, case.history) for case in cases
    ]
    prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, message)
        for case, message in zip(cases, messages, strict=True)
    ]
    beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(cases, responses, strict=True)
    ]
    metrics = serving.serving_metrics(cases, beliefs)
    gates = serving.serving_gates(
        cases=cases,
        beliefs=beliefs,
        metrics=metrics,
        prompt_errors=prompt_errors,
        usage=result["usage"],
    )
    if bed.canonical_json(metrics) != bed.canonical_json(result.get("metrics")):
        raise ValueError("serving metrics do not replay")
    if gates != result.get("gates") or not gates["all_pass"]:
        raise ValueError("serving gates do not replay")
    return {
        "result_sha256": sha256_file(path),
        "raw_responses_sha256": sha256_file(raw_path),
        "cost_usd": float(result["usage"]["run_cost_usd"]),
        "verified": True,
    }


def _branch_diagnostics(
    *,
    task: bed.VisualTask,
    branches: Mapping[tuple[str, bool], bed.SemanticBelief],
) -> list[dict[str, Any]]:
    return [
        serving.branch_sensitivity(
            branches[(candidate, False)],
            branches[(candidate, True)],
            candidate_id=candidate,
        )
        for candidate in task.candidate_ids
    ]


def _pooled_policy_metrics(trees: Sequence[dict[str, Any]]) -> dict[str, Any]:
    pooled = {}
    for policy in POLICIES:
        rows = [tree["policies"][policy]["endpoint"] for tree in trees]
        pooled[policy] = {
            "mean_brier": sum(row["mean_brier"] for row in rows) / len(rows),
            "mean_log_loss": sum(row["mean_log_loss"] for row in rows) / len(rows),
            "mean_accuracy": sum(row["accuracy"] for row in rows) / len(rows),
            "mean_truth_probability": sum(
                row["mean_truth_probability"] for row in rows
            )
            / len(rows),
        }
    return pooled


def mechanics_gates(
    *,
    trees: Sequence[dict[str, Any]],
    branch_diagnostics: Sequence[dict[str, Any]],
    branch_label_obedience: Mapping[str, Any],
    terminal_label_obedience: Mapping[str, Any],
    history_blind_branch_count: int,
    paired_requests: Mapping[str, Any],
    final_pairing: Mapping[str, Any],
    final_case_count: int,
    usage: Mapping[str, Any],
    prompt_errors: Sequence[Sequence[str]],
    serving_verification: Mapping[str, Any],
) -> dict[str, bool]:
    expected_requests = FIRST_STAGE_REQUESTS + final_case_count
    pooled = _pooled_policy_metrics(trees)
    dynamic_changes = sum(
        tree["policies"]["dynamic_depth2"]["first_image_id"]
        != tree["policies"]["myopic_width"]["first_image_id"]
        for tree in trees
    )
    robust_dynamic_changes = sum(
        (
            dynamic_first := tree["policies"]["dynamic_depth2"][
                "first_image_id"
            ]
        )
        != (
            myopic_first := tree["policies"]["myopic_width"][
                "first_image_id"
            ]
        )
        and tree["root_scores"]["dynamic_depth2"][dynamic_first]
        - tree["root_scores"]["dynamic_depth2"][myopic_first]
        >= MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    robust_dynamic_blind_changes = sum(
        tree["policies"]["dynamic_depth2"]["first_image_id"]
        != tree["policies"]["history_blind_depth2"]["first_image_id"]
        and tree["policies"]["dynamic_depth2"]["first_score_margin"]
        >= MIN_ACTION_MARGIN_NATS
        and tree["policies"]["history_blind_depth2"]["first_score_margin"]
        >= MIN_ACTION_MARGIN_NATS
        for tree in trees
    )
    matched_fixed_control_exact = all(
        fixed_score_dynamic_update_is_exact(tree) for tree in trees
    )
    matched_history_blind_update_exact = all(
        history_blind_update_matched_first_is_exact(tree) for tree in trees
    )
    matched_history_blind_update_changes = sum(
        tree["policies"]["dynamic_depth2"]["second_image_id"]
        != tree["policies"]["history_blind_update_matched_first"][
            "second_image_id"
        ]
        for tree in trees
    )
    distinct_control_policies = sum(
        any(
            tree["policies"][policy]["final_history_key"]
            != tree["policies"]["myopic_width"]["final_history_key"]
            for tree in trees
        )
        for policy in POLICIES
        if policy != "myopic_width"
    )
    finite_scores = all(
        math.isfinite(value)
        for tree in trees
        for score_map in tree["root_scores"].values()
        for value in score_map.values()
    ) and all(
        row["second_scores"] is None
        or all(math.isfinite(value) for value in row["second_scores"].values())
        for tree in trees
        for row in tree["policies"].values()
    ) and all(
        math.isfinite(value)
        for tree in trees
        for value in tree["root_hypothesis_eig_diagnostics"].values()
    )
    finite_endpoints = all(
        math.isfinite(value)
        for values in pooled.values()
        for value in values.values()
    ) and all(
        math.isfinite(value)
        for tree in trees
        for path in tree["all_first_action_paths"].values()
        for key, value in path["endpoint"].items()
        if key != "rows"
    )
    finite_ranking = all(
        math.isfinite(value)
        for tree in trees
        for value in tree["ranking_fidelity"].values()
    )
    shuffled_control_exact = all(
        all(
            math.isclose(
                tree["continuation_values"]["shuffled_expected_continuation_utility"][
                    target
                ],
                tree["continuation_values"]["dynamic_expected_continuation_utility"][
                    source
                ],
                rel_tol=0.0,
                abs_tol=1e-12,
            )
            for target, source in tree["shuffled_branch_mapping"].items()
        )
        for tree in trees
    )
    gates = {
        "serving_result_independently_replays": serving_verification.get("verified") is True,
        **serving.transport_retry_gates(
            usage, expected_requests=expected_requests
        ),
        "zero_reasoning_tokens": usage.get("adapter_reasoning_tokens") == 0,
        "zero_forced_exits": usage.get("forced_exits") == 0,
        "all_root_branch_and_final_responses_parse": (
            len(trees) == ROOT_REQUESTS
            and len(branch_diagnostics) == DYNAMIC_BRANCH_REQUESTS // 2
            and history_blind_branch_count == HISTORY_BLIND_BRANCH_REQUESTS
            and 16 <= final_case_count <= MAX_FINAL_REQUESTS
        ),
        "exact_paired_seed_and_prompt_difference_accounting": (
            paired_requests.get("pair_count") == DYNAMIC_BRANCH_REQUESTS
            and (paired_requests.get("gates") or {}).get("all_pass") is True
        ),
        "terminal_histories_use_task_level_common_random_numbers": (
            final_pairing.get("task_count") == ROOT_REQUESTS
            and (final_pairing.get("gates") or {}).get("all_pass") is True
        ),
        "all_scores_are_finite_and_executable": finite_scores,
        "all_policies_use_endpoint_predictive_information_gain": all(
            tree.get("score_objective") == SCORE_OBJECTIVE for tree in trees
        ),
        "at_least_24_of_32_branch_pairs_change_unobserved_beliefs": sum(
            row["material"] for row in branch_diagnostics
        )
        >= MIN_MATERIAL_BRANCH_PAIRS,
        "simulated_branch_labels_beat_constant_half_brier_in_both_classes": (
            serving.branch_label_obedience_passes(branch_label_obedience)
        ),
        "terminal_beliefs_retain_both_queried_labels_better_than_constant_half": (
            serving.terminal_label_obedience_passes(
                terminal_label_obedience
            )
        ),
        "dynamic_depth2_changes_at_least_one_myopic_first_action": dynamic_changes >= 1,
        "dynamic_action_change_clears_numerical_tie_margin": (
            robust_dynamic_changes >= 1
        ),
        "dynamic_and_history_blind_change_a_nontied_first_action": (
            robust_dynamic_blind_changes >= 1
        ),
        "fixed_score_dynamic_update_exactly_matches_fixed_first_and_dynamic_second": (
            matched_fixed_control_exact
        ),
        "history_blind_update_matched_first_exactly_matches_dynamic_first": (
            matched_history_blind_update_exact
        ),
        "dynamic_and_matched_history_blind_update_change_at_least_one_second_action": (
            matched_history_blind_update_changes >= 1
        ),
        "shuffled_control_exactly_permutes_complete_continuation_values": (
            shuffled_control_exact
        ),
        "at_least_two_controls_have_a_distinct_final_history": distinct_control_policies >= 2,
        "all_distinct_all_action_final_supports_generated_once_and_mapped": (
            16 <= final_case_count <= MAX_FINAL_REQUESTS
        ),
        "all_eight_realized_first_action_continuations_are_scored": all(
            len(tree["all_first_action_paths"]) == 8 for tree in trees
        ),
        "all_ranking_fidelity_diagnostics_are_finite": finite_ranking,
        "root_candidate_brier_beats_constant_half": (
            sum(tree["root_candidate_brier"] for tree in trees) / len(trees)
            < 0.25
        ),
        "myopic_endpoint_is_not_saturated": (
            pooled["myopic_width"]["mean_brier"] >= MIN_MYOPIC_BRIER
            or pooled["myopic_width"]["mean_log_loss"] >= MIN_MYOPIC_LOG_LOSS
        ),
        "all_endpoint_metrics_are_finite": finite_endpoints,
        "all_prompts_hide_bound_source_truth": not any(prompt_errors),
        "cost_at_most_1_75": float(usage.get("run_cost_usd", math.inf)) <= RUN_BUDGET_USD,
    }
    gates["all_pass"] = all(gates.values())
    return gates


def _adapter(*, output_dir: Path, run_id: str) -> serving.LunaVisionAdapter:
    config = Config(
        task="animals",
        run_id=run_id,
        log_path=output_dir / "run.log",
        openrouter_budget_usd=500.0,
        openrouter_run_budget_usd=RUN_BUDGET_USD,
        openrouter_projected_cost_usd=RUN_BUDGET_USD,
        openrouter_concurrency=CONCURRENCY,
        openrouter_max_retries=4,
        openrouter_backoff_seconds=1.0,
        openrouter_request_timeout_seconds=300.0,
        openrouter_max_request_cost_usd=serving.MAX_REQUEST_COST_USD,
        openrouter_max_output_tokens=MAX_TOKENS,
        openrouter_spend_path="results/path_e/openrouter_spend.json",
    )
    return serving.LunaVisionAdapter(
        ModelSpec(model=MODEL_ID, backend="openrouter", max_model_len=1_000_000),
        config,
        request_seed=MODEL_SEED,
    )


def run_mechanics(
    *,
    output_dir: Path,
    run_id: str,
    serving_result: Path,
    tasks: Sequence[bed.VisualTask] | None = None,
    adapter: StructuredModel | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(tasks or bed.load_mechanics_tasks(), key=lambda task: task.task_id)
    serving_verification = verify_serving_result(serving_result, tasks=tasks)
    model = adapter or _adapter(output_dir=output_dir, run_id=run_id)

    stage_cases = first_stage_cases(tasks)
    stage_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in stage_cases
    ]
    stage_seeds = request_seeds_for_cases(
        stage_cases, base_seed=MODEL_SEED
    )
    paired_requests = paired_request_diagnostics(
        cases=stage_cases,
        messages=stage_messages,
        seeds=stage_seeds,
    )
    if not paired_requests["gates"]["all_pass"]:
        raise ValueError("paired history-blind request audit failed")
    stage_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(stage_cases, stage_messages, strict=True)
    ]
    if any(stage_prompt_errors):
        raise ValueError("first-stage hidden-state prompt audit failed")
    stage_responses = model.chat_complete_seeded_messages_batched_structured(
        stage_messages,
        stage_seeds,
        temperature=TEMPERATURE,
        response_format=bed.belief_response_format(),
        max_new_tokens=MAX_TOKENS,
    )
    stage_beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(stage_cases, stage_responses, strict=True)
    ]
    roots = {
        case.task.task_id: belief
        for case, belief in zip(stage_cases, stage_beliefs, strict=True)
        if case.kind == "root"
    }
    branches_by_task: dict[
        str, dict[tuple[str, bool], bed.SemanticBelief]
    ] = {task.task_id: {} for task in tasks}
    history_blind_by_task: dict[
        str, dict[tuple[str, bool], bed.SemanticBelief]
    ] = {task.task_id: {} for task in tasks}
    for case, belief in zip(stage_cases, stage_beliefs, strict=True):
        if case.kind == "branch":
            branches_by_task[case.task.task_id][
                (str(case.candidate_id), bool(case.simulated_label))
            ] = belief
        elif case.kind == "history_blind":
            history_blind_by_task[case.task.task_id][
                (str(case.candidate_id), bool(case.simulated_label))
            ] = belief

    branch_obedience = serving.branch_label_obedience(
        [
            (candidate_id, label, belief)
            for branches in branches_by_task.values()
            for (candidate_id, label), belief in branches.items()
        ]
    )

    plans = {
        task.task_id: plan_task_policies(
            task=task,
            root=roots[task.task_id],
            branches=branches_by_task[task.task_id],
            history_blind_branches=history_blind_by_task[task.task_id],
        )
        for task in tasks
    }
    action_paths = {
        task.task_id: all_first_action_paths(
            task=task, branches=branches_by_task[task.task_id]
        )
        for task in tasks
    }
    endpoint_accessed_after_selection = True
    selected_final_cases = all_action_final_cases(
        tasks=tasks, plans=plans, action_paths=action_paths
    )
    selected_final_messages = [
        bed.build_belief_messages(case.task, case.history)
        for case in selected_final_cases
    ]
    final_seeds = request_seeds_for_cases(
        selected_final_cases,
        base_seed=MODEL_SEED,
        final_stage=True,
    )
    final_dispatch_batches = task_preserving_dispatch_batches(
        selected_final_cases
    )
    final_pairing = final_request_diagnostics(
        cases=selected_final_cases,
        seeds=final_seeds,
        plans=plans,
        dispatch_batches=final_dispatch_batches,
    )
    if not final_pairing["gates"]["all_pass"]:
        raise ValueError("terminal common-random-number audit failed")
    final_prompt_errors = [
        bed.prompt_hidden_state_errors(case.task, case.history, messages)
        for case, messages in zip(
            selected_final_cases, selected_final_messages, strict=True
        )
    ]
    if any(final_prompt_errors):
        raise ValueError("final hidden-state prompt audit failed")
    final_responses = []
    for batch in final_dispatch_batches:
        start = int(batch["start"])
        stop = int(batch["stop"])
        response_batch = model.chat_complete_seeded_messages_batched_structured(
            selected_final_messages[start:stop],
            final_seeds[start:stop],
            temperature=TEMPERATURE,
            response_format=bed.belief_response_format(),
            max_new_tokens=MAX_TOKENS,
        )
        if len(response_batch) != stop - start:
            raise ValueError("model returned the wrong terminal batch size")
        final_responses.extend(response_batch)
    final_beliefs = [
        bed.parse_belief_response(
            response,
            image_ids=case.task.image_ids,
            history=case.history,
        )
        for case, response in zip(
            selected_final_cases, final_responses, strict=True
        )
    ]
    terminal_obedience = serving.terminal_label_obedience(
        [
            (case.task.initial_history, belief)
            for case, belief in zip(
                selected_final_cases, final_beliefs, strict=True
            )
        ]
    )
    final_by_task_history = {
        (case.task.task_id, history_key(case.history)): belief
        for case, belief in zip(
            selected_final_cases, final_beliefs, strict=True
        )
    }

    trees = []
    all_branch_diagnostics = []
    for task in tasks:
        task_plan = plans[task.task_id]
        diagnostics = _branch_diagnostics(
            task=task, branches=branches_by_task[task.task_id]
        )
        all_branch_diagnostics.extend(diagnostics)
        policy_rows = {}
        endpoint_labels = {
            image_id: bool(task.actual_labels[image_id])
            for image_id in task.endpoint_ids
        }
        for policy in POLICIES:
            plan = task_plan["policies"][policy]
            final_belief = final_by_task_history[
                (task.task_id, plan["final_history_key"])
            ]
            policy_rows[policy] = {
                **{
                    key: value
                    for key, value in plan.items()
                    if key != "final_history"
                },
                "endpoint": bed.endpoint_metrics(
                    final_belief, endpoint_labels
                ),
                "final_belief": bed.public_belief_summary(final_belief),
            }
        action_rows = {}
        for first, path in action_paths[task.task_id].items():
            final_belief = final_by_task_history[
                (task.task_id, path["final_history_key"])
            ]
            action_rows[first] = {
                **{
                    key: value
                    for key, value in path.items()
                    if key != "final_history"
                },
                "endpoint": bed.endpoint_metrics(final_belief, endpoint_labels),
                "final_belief": bed.public_belief_summary(final_belief),
            }
        first_ids = tuple(sorted(action_rows))
        endpoint_utility = [
            -action_rows[first]["endpoint"]["mean_brier"]
            for first in first_ids
        ]
        ranking_fidelity = {
            score_name: spearman_correlation(
                [task_plan["root_scores"][score_name][first] for first in first_ids],
                endpoint_utility,
            )
            for score_name in SCORE_POLICIES
        }
        root_candidate_brier = sum(
            (
                bed.predictive_probability(roots[task.task_id], image_id)
                - float(task.actual_labels[image_id])
            )
            ** 2
            for image_id in task.candidate_ids
        ) / len(task.candidate_ids)
        trees.append(
            {
                "task_id": task.task_id,
                "score_objective": task_plan["score_objective"],
                "root_hypothesis_eig_diagnostics": task_plan[
                    "root_hypothesis_eig_diagnostics"
                ],
                "root_scores": task_plan["root_scores"],
                "shuffled_branch_mapping": task_plan[
                    "shuffled_branch_mapping"
                ],
                "continuation_values": task_plan["continuation_values"],
                "root_belief": bed.public_belief_summary(
                    roots[task.task_id]
                ),
                "branch_diagnostics": diagnostics,
                "policies": policy_rows,
                "all_first_action_paths": action_rows,
                "ranking_fidelity": ranking_fidelity,
                "root_candidate_brier": root_candidate_brier,
            }
        )

    usage = summarize_usage(model.usage_snapshot())
    gates = mechanics_gates(
        trees=trees,
        branch_diagnostics=all_branch_diagnostics,
        branch_label_obedience=branch_obedience,
        terminal_label_obedience=terminal_obedience,
        history_blind_branch_count=sum(
            len(branches) for branches in history_blind_by_task.values()
        ),
        paired_requests=paired_requests,
        final_pairing=final_pairing,
        final_case_count=len(selected_final_cases),
        usage=usage,
        prompt_errors=[*stage_prompt_errors, *final_prompt_errors],
        serving_verification=serving_verification,
    )
    raw_path = output_dir / "private/RAW_RESPONSES.json"
    checkpoint(
        raw_path,
        {
            "first_stage_case_ids": [case.case_id for case in stage_cases],
            "first_stage_request_sha256": [
                message_sha256(messages) for messages in stage_messages
            ],
            "first_stage_request_seeds": stage_seeds,
            "first_stage_responses": stage_responses,
            "final_case_ids": [case.case_id for case in selected_final_cases],
            "final_request_sha256": [
                message_sha256(messages) for messages in selected_final_messages
            ],
            "final_request_seeds": final_seeds,
            "final_request_pairing": final_pairing,
            "final_responses": final_responses,
            "actual_candidate_labels_accessed_after_root_selection": True,
            "endpoint_labels_accessed_after_all_query_selection": (
                endpoint_accessed_after_selection
            ),
            "development_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
    )
    pooled = _pooled_policy_metrics(trees)
    ranking_fidelity = {
        score_name: sum(
            tree["ranking_fidelity"][score_name] for tree in trees
        )
        / len(trees)
        for score_name in SCORE_POLICIES
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "mechanics_pass" if gates["all_pass"] else "gated_null",
        "scientific_opportunity_status": "untested",
        "authorizes_development": False,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_VLM_MECHANICS_PREREGISTRATION.md"
            ),
            "amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_FULL_MECHANICS_AMENDMENT.md"
            ),
            "semantic_validity_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SEMANTIC_VALIDITY_AMENDMENT.md"
            ),
            "shuffled_control_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_SHUFFLED_CONTROL_AMENDMENT.md"
            ),
            "history_blind_control_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_HISTORY_BLIND_CONTROL_AMENDMENT.md"
            ),
            "matched_realized_updater_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_MATCHED_REALIZED_UPDATER_AMENDMENT_20260808.md"
            ),
            "terminal_crn_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_TERMINAL_CRN_AMENDMENT.md"
            ),
            "terminal_obedience_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_LUNA_TERMINAL_OBEDIENCE_AMENDMENT.md"
            ),
            "endpoint_predictive_utility_amendment": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md"
            ),
            "score_objective": SCORE_OBJECTIVE,
            "model": MODEL_ID,
            "model_seed": MODEL_SEED,
            "random_seed": RANDOM_SEED,
            "reasoning": False,
            "first_stage_requests": FIRST_STAGE_REQUESTS,
            "root_requests": ROOT_REQUESTS,
            "conditioned_branch_requests": DYNAMIC_BRANCH_REQUESTS,
            "history_blind_branch_requests": HISTORY_BLIND_BRANCH_REQUESTS,
            "distinct_final_history_requests": len(selected_final_cases),
            "expected_total_requests": (
                FIRST_STAGE_REQUESTS + len(selected_final_cases)
            ),
            "run_budget_usd": RUN_BUDGET_USD,
            "development_accessed": False,
            "confirmation_accessed": False,
            "sealed_test_accessed": False,
        },
        "serving_verification": serving_verification,
        "usage": usage,
        "pooled_policy_metrics": pooled,
        "mean_root_candidate_brier": sum(
            tree["root_candidate_brier"] for tree in trees
        )
        / len(trees),
        "mean_ranking_fidelity": ranking_fidelity,
        "branch_label_obedience": branch_obedience,
        "terminal_label_obedience": terminal_obedience,
        "paired_request_diagnostics": paired_requests,
        "final_request_pairing": final_pairing,
        "comparisons_vs_myopic": {
            policy: {
                "brier_difference": (
                    pooled[policy]["mean_brier"]
                    - pooled["myopic_width"]["mean_brier"]
                ),
                "log_loss_difference": (
                    pooled[policy]["mean_log_loss"]
                    - pooled["myopic_width"]["mean_log_loss"]
                ),
            }
            for policy in POLICIES
            if policy != "myopic_width"
        },
        "gates": gates,
        "trees": trees,
        "raw_responses_sha256": sha256_file(raw_path),
    }
    checkpoint(output_dir / "RESULT.json", result)
    return result


def validate_daily_ledger(
    ledger: Mapping[str, Any], *, now: datetime | None = None
) -> None:
    timezone = ZoneInfo(serving.TIMEZONE)
    local_now = now.astimezone(timezone) if now else datetime.now(timezone)
    if ledger.get("date") != local_now.date().isoformat():
        raise RuntimeError("full mechanics requires the current daily ledger")
    if ledger.get("timezone") != serving.TIMEZONE:
        raise RuntimeError("daily ledger timezone changed")
    if float(ledger.get("daily_cap_usd", 0.0)) != 5.0:
        raise RuntimeError("daily ledger no longer has the exact $5 cap")
    smoke = ledger.get("bongard_luna_vlm_serving_smoke") or {}
    if (
        smoke.get("status") != "passed"
        or smoke.get("interface_version") != serving.INTERFACE_VERSION
        or smoke.get("model") != MODEL_ID
    ):
        raise RuntimeError("daily ledger lacks the passed Luna serving gate")
    if "bongard_luna_vlm_mechanics_tree" in ledger:
        raise RuntimeError("full mechanics tree is already recorded")


def reconcile_ledger(
    *,
    ledger: Mapping[str, Any],
    measured_cost_usd: float,
    live_after: Mapping[str, float],
    status: str,
) -> dict[str, Any]:
    updated = json.loads(json.dumps(ledger))
    opening = float(updated["opening_total_usage_usd"])
    previous = float(updated.get("recorded_actual_spend_usd", 0.0))
    posted = max(0.0, float(live_after["total_usage_usd"]) - opening)
    local = previous + measured_cost_usd
    recorded = max(posted, local)
    previous_block = float(
        (updated.get("bongard_luna_vlm_mechanics_tree") or {}).get(
            "actual_cost_usd", 0.0
        )
    )
    block_cost = max(previous_block, measured_cost_usd)
    updated["recorded_actual_spend_usd"] = recorded
    updated["bongard_luna_vlm_mechanics_tree"] = {
        "status": status,
        "actual_cost_usd": block_cost,
        "maximum_cost_usd": RUN_BUDGET_USD,
        "interface_version": INTERFACE_VERSION,
        "model": MODEL_ID,
    }
    updated["reconciliation"] = {
        "live_total_credits_usd": float(live_after["total_credits_usd"]),
        "live_total_usage_usd": float(live_after["total_usage_usd"]),
        "live_balance_usd": float(live_after["balance_usd"]),
        "posted_spend_since_opening_usd": posted,
        "locally_measured_spend_usd": local,
        "recorded_spend_is_max_of_posted_and_local": True,
        "remaining_daily_allowance_usd": max(0.0, 5.0 - recorded),
    }
    return updated


def execute_mechanics(
    *,
    output_dir: Path,
    run_id: str,
    serving_result: Path,
    ledger_path: Path,
    now: datetime | None = None,
    live_reader: Callable[[], dict[str, float]] = read_live_credits,
    mechanics_runner: Callable[..., dict[str, Any]] = run_mechanics,
) -> dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"output directory is not empty: {output_dir}")
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    validate_daily_ledger(ledger, now=now)
    verification = verify_serving_result(serving_result)
    observed_projection = (
        verification["cost_usd"]
        / serving.EXPECTED_REQUESTS
        * MAX_REQUESTS
        * 1.5
    )
    if observed_projection > RUN_BUDGET_USD + 1e-12:
        raise RuntimeError(
            f"serving cost projects ${observed_projection:.6f}, above full cap"
        )
    live = live_reader()
    require_budget(
        ledger,
        projected_cost_usd=RUN_BUDGET_USD,
        total_usage_usd=live["total_usage_usd"],
        now=now,
    )
    if live["balance_usd"] + 1e-12 < RUN_BUDGET_USD:
        raise RuntimeError("OpenRouter balance is below the full mechanics cap")
    try:
        result = mechanics_runner(
            output_dir=output_dir,
            run_id=run_id,
            serving_result=serving_result,
        )
    except Exception as exc:
        reconciliation_error = None
        try:
            reconciled = reconcile_ledger(
                ledger=ledger,
                measured_cost_usd=0.0,
                live_after=live_reader(),
                status="failed_closed_posted_spend_reconciled",
            )
            checkpoint(ledger_path, reconciled)
        except Exception as reconciliation_exc:
            reconciliation_error = (
                f"{type(reconciliation_exc).__name__}: {reconciliation_exc}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        checkpoint(
            output_dir / "FAILURE.json",
            {
                "schema_version": SCHEMA_VERSION,
                "interface_version": INTERFACE_VERSION,
                "status": "failed_closed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "ledger_reconciliation_error": reconciliation_error,
            },
        )
        raise
    measured = float(result["usage"]["run_cost_usd"])
    local = reconcile_ledger(
        ledger=ledger,
        measured_cost_usd=measured,
        live_after=live,
        status=result["status"],
    )
    checkpoint(ledger_path, local)
    final = reconcile_ledger(
        ledger=local,
        measured_cost_usd=0.0,
        live_after=live_reader(),
        status=result["status"],
    )
    checkpoint(ledger_path, final)
    checkpoint(
        output_dir / "EXECUTION.json",
        {
            "schema_version": SCHEMA_VERSION,
            "interface_version": INTERFACE_VERSION,
            "status": "complete_reconciled",
            "result_sha256": sha256_file(output_dir / "RESULT.json"),
            "ledger_sha256": sha256_file(ledger_path),
            "recorded_daily_spend_usd": final["recorded_actual_spend_usd"],
            "remaining_daily_allowance_usd": final["reconciliation"][
                "remaining_daily_allowance_usd"
            ],
        },
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--serving-result", type=Path, required=True)
    parser.add_argument("--daily-ledger", type=Path, required=True)
    args = parser.parse_args()
    result = execute_mechanics(
        output_dir=args.output_dir,
        run_id=args.run_id,
        serving_result=args.serving_result,
        ledger_path=args.daily_ledger,
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "usage": result["usage"],
                "pooled_policy_metrics": result["pooled_policy_metrics"],
                "comparisons_vs_myopic": result["comparisons_vs_myopic"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
