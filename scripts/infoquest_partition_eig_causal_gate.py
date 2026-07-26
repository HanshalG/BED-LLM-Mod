#!/usr/bin/env python3
"""Evaluate exact EIG over LLM-generated InfoQuest response partitions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts import infoquest_discrete_action_causal_gate as discrete
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-partition-eig-causal-1"
ACTION_KEYS = ("a", "b", "c", "d")
EXPECTED_SERVING_REQUESTS = 7
EXPECTED_MECHANICS_REQUESTS = 159
SERVING_MAX_COST_USD = 0.12
MECHANICS_MAX_COST_USD = 0.95

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


@dataclass(frozen=True)
class PartitionBelief:
    hypotheses: tuple[str, ...]
    weights: tuple[int, ...]
    profiles: tuple[tuple[int, ...], ...]
    eig_scores: tuple[float, ...]
    selected_action_index: int
    selected_question: str

    def as_policy(self) -> base.RefreshPolicy:
        return base.RefreshPolicy(
            hypotheses=self.hypotheses,
            followup=self.selected_question,
        )


def partition_eig(
    weights: Sequence[int],
    labels: Sequence[int],
) -> float:
    if len(weights) != base.SUPPORT_SIZE or len(labels) != base.SUPPORT_SIZE:
        raise ValueError("partition EIG requires eight weights and labels")
    total = sum(weights)
    if total <= 0:
        raise ValueError("partition weights have zero total")
    cluster_mass: dict[int, float] = {}
    for weight, label in zip(weights, labels):
        cluster_mass[label] = cluster_mass.get(label, 0.0) + weight / total
    return -sum(
        probability * math.log(probability)
        for probability in cluster_mass.values()
        if probability > 0
    )


def _partition_request(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
) -> dict[str, Any]:
    return discrete._choice_request(
        seed_message,
        initial,
        asked_root_index,
        answer,
    )


def dynamic_partition_messages(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
    *,
    max_cluster_label: int = 3,
    compact_arrays: bool = False,
) -> list[dict[str, str]]:
    if not 0 <= max_cluster_label < base.SUPPORT_SIZE:
        raise ValueError("cluster label maximum is outside support")
    output_instruction = (
        "Output only one JSON object with exactly six fields: h is an array "
        "of eight hypothesis strings; w is an array of eight integer weights; "
        "and a, b, c, and d are arrays of eight integer cluster labels. "
        "COMPACT_ARRAYS=true."
        if compact_arrays
        else (
            "Output only one JSON object with exactly h1..h8, w1..w8, "
            "a1..a8, b1..b8, c1..c8, and d1..d8."
        )
    )
    weight_instruction = (
        "Assign each context a posterior plausibility in the eight-integer w "
        "array, with every value from 1 to 100. "
        if compact_arrays
        else (
            "Assign each context an integer posterior plausibility w1..w8 "
            "from 1 to 100. "
        )
    )
    profile_instruction = (
        "For each candidate action A, B, C, and D, predict the answer under "
        "every context and cluster semantically indistinguishable answers: "
        f"put integer labels 0..{max_cluster_label} in the corresponding "
        "eight-value arrays a, b, c, and d. "
        if compact_arrays
        else (
            "For each candidate action A, B, C, and D, predict the answer "
            "under every context and cluster semantically indistinguishable "
            f"answers: output integer labels 0..{max_cluster_label} in fields "
            "a1..a8, b1..b8, c1..c8, and d1..d8. "
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "STAGE=PARTITION_DYNAMIC. Regenerate eight distinct concrete "
                "latent contexts from the complete observed history. "
                f"{weight_instruction}{profile_instruction}"
                "Cluster labels are local to each action. Use the "
                "same label within an action iff the answers would convey the "
                "same information. Do not choose an action. "
                f"{output_instruction}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _partition_request(
                    seed_message,
                    initial,
                    asked_root_index,
                    answer,
                ),
                separators=(",", ":"),
            ),
        },
    ]


def fixed_partition_messages(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
    *,
    max_cluster_label: int = 3,
    compact_arrays: bool = False,
) -> list[dict[str, str]]:
    if not 0 <= max_cluster_label < base.SUPPORT_SIZE:
        raise ValueError("cluster label maximum is outside support")
    output_instruction = (
        "Output only one JSON object with exactly six fields: h is the array "
        "of eight verbatim hypothesis strings; w is an array of eight integer "
        "weights; and a, b, c, and d are arrays of eight integer cluster "
        "labels. COMPACT_ARRAYS=true."
        if compact_arrays
        else (
            "Output only one JSON object with exactly h1..h8, w1..w8, "
            "a1..a8, b1..b8, c1..c8, and d1..d8."
        )
    )
    support_instruction = (
        "Keep the supplied eight hypotheses exactly fixed and copy them "
        "verbatim into the h array without adding, removing, rewriting, or "
        "reordering. "
        if compact_arrays
        else (
            "Keep the supplied eight hypotheses exactly fixed and copy "
            "h1..h8 verbatim without adding, removing, rewriting, or "
            "reordering. "
        )
    )
    weight_instruction = (
        "Using the complete observed history, assign posterior plausibilities "
        "in the eight-integer w array, with every value from 1 to 100. "
        if compact_arrays
        else (
            "Using the complete observed history, assign integer posterior "
            "plausibilities w1..w8 from 1 to 100. "
        )
    )
    profile_instruction = (
        "For each candidate action A, B, C, and D, predict the answer under "
        "every hypothesis and cluster semantically indistinguishable answers "
        f"using integer labels 0..{max_cluster_label} in the corresponding "
        "eight-value arrays a, b, c, and d. "
        if compact_arrays
        else (
            "For each candidate action A, B, C, and D, predict the answer "
            "under every hypothesis and cluster semantically indistinguishable "
            f"answers using integer labels 0..{max_cluster_label} in a1..a8, "
            "b1..b8, c1..c8, and d1..d8. "
        )
    )
    return [
        {
            "role": "system",
            "content": (
                "STAGE=PARTITION_FIXED. "
                f"{support_instruction}{weight_instruction}"
                f"{profile_instruction}"
                "Labels are local to each action. Do not choose an "
                f"action. {output_instruction}"
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _partition_request(
                    seed_message,
                    initial,
                    asked_root_index,
                    answer,
                ),
                separators=(",", ":"),
            ),
        },
    ]


def _parse_bounded_integer(
    value: Any,
    *,
    minimum: int,
    maximum: int,
    name: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} is not an integer")
    if not minimum <= value <= maximum:
        raise ValueError(f"{name} is outside [{minimum}, {maximum}]")
    return value


def parse_partition_belief(
    response: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    *,
    require_fixed_support: bool,
    max_cluster_label: int = 3,
    compact_arrays: bool = False,
) -> PartitionBelief:
    if not 0 <= max_cluster_label < base.SUPPORT_SIZE:
        raise ValueError("cluster label maximum is outside support")
    if compact_arrays:
        value = base._parse_exact_object(
            response,
            {"h", "w", *ACTION_KEYS},
        )
        for key in ("h", "w", *ACTION_KEYS):
            if not isinstance(value[key], list):
                raise ValueError(f"{key} is not an array")
            if len(value[key]) != base.SUPPORT_SIZE:
                raise ValueError(f"{key} array does not have eight values")
        hypothesis_values = value["h"]
        weight_values = value["w"]
        profile_values = [value[action] for action in ACTION_KEYS]
    else:
        expected = {
            *(f"h{index}" for index in range(1, base.SUPPORT_SIZE + 1)),
            *(f"w{index}" for index in range(1, base.SUPPORT_SIZE + 1)),
            *(
                f"{action}{index}"
                for action in ACTION_KEYS
                for index in range(1, base.SUPPORT_SIZE + 1)
            ),
        }
        value = base._parse_exact_object(response, expected)
        hypothesis_values = [
            value[f"h{index}"]
            for index in range(1, base.SUPPORT_SIZE + 1)
        ]
        weight_values = [
            value[f"w{index}"]
            for index in range(1, base.SUPPORT_SIZE + 1)
        ]
        profile_values = [
            [
                value[f"{action}{index}"]
                for index in range(1, base.SUPPORT_SIZE + 1)
            ]
            for action in ACTION_KEYS
        ]
    hypotheses = tuple(base._clean_text(item) for item in hypothesis_values)
    if len({base._normalize(item) for item in hypotheses}) != base.SUPPORT_SIZE:
        raise ValueError("partition hypotheses are not distinct")
    if require_fixed_support and hypotheses != initial.hypotheses:
        raise ValueError("fixed partition changed the initial support")
    weights = tuple(
        _parse_bounded_integer(
            weight_values[index - 1],
            minimum=1,
            maximum=100,
            name=f"w{index}",
        )
        for index in range(1, base.SUPPORT_SIZE + 1)
    )
    profiles = tuple(
        tuple(
            _parse_bounded_integer(
                profile_values[action_index][index - 1],
                minimum=0,
                maximum=max_cluster_label,
                name=f"{action}{index}",
            )
            for index in range(1, base.SUPPORT_SIZE + 1)
        )
        for action_index, action in enumerate(ACTION_KEYS)
    )
    scores = tuple(partition_eig(weights, profile) for profile in profiles)
    selected = max(range(len(scores)), key=scores.__getitem__)
    candidates = discrete.candidate_bank(initial, asked_root_index)
    return PartitionBelief(
        hypotheses=hypotheses,
        weights=weights,
        profiles=profiles,
        eig_scores=scores,
        selected_action_index=selected,
        selected_question=candidates[selected],
    )


def _public_metrics(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    dynamic_beliefs: Sequence[Sequence[PartitionBelief]],
    fixed_beliefs: Sequence[Sequence[PartitionBelief]],
    dynamic_answers: Sequence[Sequence[str]],
    fixed_answers: Sequence[Sequence[str]],
    judgments: Sequence[base.ChecklistJudgment],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    dynamic_policies = [
        [belief.as_policy() for belief in row] for row in dynamic_beliefs
    ]
    fixed_policies = [
        [belief.as_policy() for belief in row] for row in fixed_beliefs
    ]
    metrics, fixture_metrics, gates = discrete._public_metrics(
        fixtures,
        initials,
        root_answers,
        dynamic_policies,
        fixed_policies,
        dynamic_answers,
        fixed_answers,
        judgments,
    )
    gates.pop("at_least_20_dynamic_followups_differ_from_fixed")
    dynamic_informative = 0
    fixed_informative = 0
    dynamic_selected_scores = []
    fixed_selected_scores = []
    for fixture_index in range(len(fixtures)):
        dynamic_score_row = []
        fixed_score_row = []
        for root_index in range(base.ROOT_COUNT):
            dynamic_belief = dynamic_beliefs[fixture_index][root_index]
            fixed_belief = fixed_beliefs[fixture_index][root_index]
            dynamic_informative += (
                sum(score > 1e-9 for score in dynamic_belief.eig_scores) >= 2
            )
            fixed_informative += (
                sum(score > 1e-9 for score in fixed_belief.eig_scores) >= 2
            )
            dynamic_score = dynamic_belief.eig_scores[
                dynamic_belief.selected_action_index
            ]
            fixed_score = fixed_belief.eig_scores[
                fixed_belief.selected_action_index
            ]
            dynamic_selected_scores.append(dynamic_score)
            fixed_selected_scores.append(fixed_score)
            dynamic_score_row.append(dynamic_score)
            fixed_score_row.append(fixed_score)
        fixture_metrics[fixture_index]["dynamic_action_eig_nats"] = (
            dynamic_score_row
        )
        fixture_metrics[fixture_index]["fixed_action_eig_nats"] = fixed_score_row
    metrics["dynamic_cells_with_at_least_2_informative_actions"] = (
        dynamic_informative
    )
    metrics["fixed_cells_with_at_least_2_informative_actions"] = (
        fixed_informative
    )
    metrics["mean_dynamic_selected_eig_nats"] = (
        sum(dynamic_selected_scores) / len(dynamic_selected_scores)
    )
    metrics["mean_fixed_selected_eig_nats"] = (
        sum(fixed_selected_scores) / len(fixed_selected_scores)
    )
    gates["at_least_24_dynamic_cells_have_2_informative_actions"] = (
        dynamic_informative >= 24
    )
    gates["at_least_24_fixed_cells_have_2_informative_actions"] = (
        fixed_informative >= 24
    )
    gates["at_least_8_dynamic_actions_differ_from_fixed"] = (
        metrics["dynamic_choice_differs_from_fixed_cells"] >= 8
    )
    return metrics, fixture_metrics, gates


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        seed_message = "Synthetic ambiguous request."
        initial_raw = refresh._complete(
            models.generator,
            [base.initial_messages(seed_message)],
            max_new_tokens=1_300,
        )
        refresh._checkpoint_stage(raw_path, raw, "initial", initial_raw)
        initial = base.parse_initial(initial_raw[0])

        simulator_system = (
            "You are a hidden user. Answer the latest specific question in "
            "one concise sentence and reveal at most one detail."
        )
        root_raw = refresh._complete(
            models.simulator,
            [base.simulator_root_messages(simulator_system, initial.roots[0])],
            max_new_tokens=160,
        )
        refresh._checkpoint_stage(raw_path, raw, "root_answer", root_raw)
        root_answer = base._clean_text(root_raw[0], maximum=1_000)

        dynamic_raw = refresh._complete(
            models.generator,
            [
                dynamic_partition_messages(
                    seed_message,
                    initial,
                    0,
                    root_answer,
                )
            ],
            max_new_tokens=1_300,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "dynamic_partition",
            dynamic_raw,
        )
        dynamic = parse_partition_belief(
            dynamic_raw[0],
            initial,
            0,
            require_fixed_support=False,
        )

        fixed_raw = refresh._complete(
            models.generator,
            [
                fixed_partition_messages(
                    seed_message,
                    initial,
                    0,
                    root_answer,
                )
            ],
            max_new_tokens=1_300,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "fixed_partition",
            fixed_raw,
        )
        fixed = parse_partition_belief(
            fixed_raw[0],
            initial,
            0,
            require_fixed_support=True,
        )

        followup_raw = refresh._complete(
            models.simulator,
            [
                base.simulator_followup_messages(
                    simulator_system,
                    initial.roots[0],
                    root_answer,
                    dynamic.selected_question,
                ),
                base.simulator_followup_messages(
                    simulator_system,
                    initial.roots[0],
                    root_answer,
                    fixed.selected_question,
                ),
            ],
            max_new_tokens=160,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "followup_answers",
            followup_raw,
        )
        followup_answers = [
            base._clean_text(value, maximum=1_000) for value in followup_raw
        ]

        fixture = base.WorldFixture(
            fixture_id="SYNTHETIC",
            record_id=-1,
            world=1,
            seed_message=seed_message,
            simulator_system=simulator_system,
            truth_packet={},
            checklist=tuple(f"Checklist item {index}" for index in range(5)),
        )
        checklist_raw = refresh._complete(
            models.checklist_judge,
            [
                base.checklist_judge_messages(
                    fixture,
                    initial,
                    [root_answer] * base.ROOT_COUNT,
                    [dynamic.as_policy()] * base.ROOT_COUNT,
                    [followup_answers[0]] * base.ROOT_COUNT,
                    [fixed.selected_question] * base.ROOT_COUNT,
                    [followup_answers[1]] * base.ROOT_COUNT,
                )
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgment",
            checklist_raw,
        )
        base.parse_checklist_judgment(checklist_raw[0])
        usage = refresh.aggregate_usage(models)
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    gates = {
        "exact_7_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_7_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_12": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
        "all_stage_parsers_pass": True,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "serving",
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "exact_eig_scorer": True,
            "models": {
                "semantic_support_likelihood": refresh.GENERATOR_MODEL_ID,
                "simulator": refresh.SIMULATOR_MODEL_ID,
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
            "reasoning_requested": False,
            "scientific_endpoint_evaluated": False,
            "repairs_or_reissues": 0,
            "all_batches_checkpointed_before_parse": True,
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[base.WorldFixture],
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "private_fixtures": [
            {
                "fixture_id": fixture.fixture_id,
                "seed_message": fixture.seed_message,
                "simulator_system": fixture.simulator_system,
                "checklist": fixture.checklist,
            }
            for fixture in fixtures
        ],
    }
    try:
        fixtures_by_record = {
            fixture.record_id: fixture for fixture in fixtures
        }
        initial_raw = refresh._complete(
            models.generator,
            [
                base.initial_messages(
                    fixtures_by_record[record_id].seed_message
                )
                for record_id in base.MECHANICS_IDS
            ],
            max_new_tokens=1_300,
        )
        refresh._checkpoint_stage(raw_path, raw, "initial", initial_raw)
        initials = {
            record_id: base.parse_initial(response)
            for record_id, response in zip(base.MECHANICS_IDS, initial_raw)
        }

        root_requests = []
        for fixture in fixtures:
            root_requests.extend(
                base.simulator_root_messages(
                    fixture.simulator_system,
                    root,
                )
                for root in initials[fixture.record_id].roots
            )
        root_raw = refresh._complete(
            models.simulator,
            root_requests,
            max_new_tokens=220,
        )
        refresh._checkpoint_stage(raw_path, raw, "root_answers", root_raw)
        root_flat = [
            base._clean_text(value, maximum=1_200) for value in root_raw
        ]
        root_answers = [
            root_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(root_flat), base.ROOT_COUNT)
        ]

        dynamic_requests = []
        fixed_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index in range(base.ROOT_COUNT):
                answer = root_answers[fixture_index][root_index]
                dynamic_requests.append(
                    dynamic_partition_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        answer,
                    )
                )
                fixed_requests.append(
                    fixed_partition_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        answer,
                    )
                )

        dynamic_raw = refresh._complete(
            models.generator,
            dynamic_requests,
            max_new_tokens=1_400,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "dynamic_partitions",
            dynamic_raw,
        )
        dynamic_flat = []
        for response_index, response in enumerate(dynamic_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            dynamic_flat.append(
                parse_partition_belief(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=False,
                )
            )
        dynamic_beliefs = [
            dynamic_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(dynamic_flat), base.ROOT_COUNT)
        ]

        fixed_raw = refresh._complete(
            models.generator,
            fixed_requests,
            max_new_tokens=1_400,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "fixed_partitions",
            fixed_raw,
        )
        fixed_flat = []
        for response_index, response in enumerate(fixed_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            fixed_flat.append(
                parse_partition_belief(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=True,
                )
            )
        fixed_beliefs = [
            fixed_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(fixed_flat), base.ROOT_COUNT)
        ]

        followup_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index, root in enumerate(initial.roots):
                common = (
                    fixture.simulator_system,
                    root,
                    root_answers[fixture_index][root_index],
                )
                followup_requests.append(
                    base.simulator_followup_messages(
                        *common,
                        dynamic_beliefs[fixture_index][
                            root_index
                        ].selected_question,
                    )
                )
                followup_requests.append(
                    base.simulator_followup_messages(
                        *common,
                        fixed_beliefs[fixture_index][
                            root_index
                        ].selected_question,
                    )
                )
        followup_raw = refresh._complete(
            models.simulator,
            followup_requests,
            max_new_tokens=220,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "followup_answers",
            followup_raw,
        )
        followup_clean = [
            base._clean_text(value, maximum=1_200) for value in followup_raw
        ]
        dynamic_answers: list[list[str]] = []
        fixed_answers: list[list[str]] = []
        cursor = 0
        for _fixture in fixtures:
            dynamic_row = []
            fixed_row = []
            for _root in range(base.ROOT_COUNT):
                dynamic_row.append(followup_clean[cursor])
                fixed_row.append(followup_clean[cursor + 1])
                cursor += 2
            dynamic_answers.append(dynamic_row)
            fixed_answers.append(fixed_row)

        checklist_raw = refresh._complete(
            models.checklist_judge,
            [
                base.checklist_judge_messages(
                    fixture,
                    initials[fixture.record_id],
                    root_answers[fixture_index],
                    [
                        belief.as_policy()
                        for belief in dynamic_beliefs[fixture_index]
                    ],
                    dynamic_answers[fixture_index],
                    [
                        belief.selected_question
                        for belief in fixed_beliefs[fixture_index]
                    ],
                    fixed_answers[fixture_index],
                )
                for fixture_index, fixture in enumerate(fixtures)
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgments",
            checklist_raw,
        )
        judgments = [
            base.parse_checklist_judgment(value) for value in checklist_raw
        ]
        usage = refresh.aggregate_usage(models)
        metrics, fixture_metrics, scientific_gates = _public_metrics(
            fixtures,
            initials,
            root_answers,
            dynamic_beliefs,
            fixed_beliefs,
            dynamic_answers,
            fixed_answers,
            judgments,
        )
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    mechanics_gates = {
        "exact_159_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_159_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_95": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
        "exact_6_fixtures_30_root_cells": (
            metrics["fixtures"] == 6
            and metrics["root_world_cells"] == 30
        ),
    }
    gates = {**mechanics_gates, **scientific_gates}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "mechanics",
            "mechanics_ids": list(base.MECHANICS_IDS),
            "support_size": base.SUPPORT_SIZE,
            "root_count": base.ROOT_COUNT,
            "candidate_actions_per_cell": len(ACTION_KEYS),
            "response_clusters_per_action": 4,
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "models": {
                "semantic_support_likelihood": refresh.GENERATOR_MODEL_ID,
                "simulator": refresh.SIMULATOR_MODEL_ID,
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
            "exact_eig_scorer": True,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "compute_matched_semantic_outputs": True,
            "shared_discrete_action_bank": True,
            "all_batches_checkpointed_before_parse": True,
            "opportunity_or_later_split_read": False,
            "causal_policy_efficacy_claimed": False,
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


class DeterministicGenerator(refresh.DeterministicGenerator):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        stages = [
            messages[0]["content"].split("STAGE=", 1)[1].split(".", 1)[0]
            for messages in batch_messages
        ]
        if all(
            stage in {"PARTITION_DYNAMIC", "PARTITION_FIXED"}
            for stage in stages
        ):
            responses = []
            for stage, messages in zip(stages, batch_messages):
                request = json.loads(messages[-1]["content"])
                root = request["clarification_question"]
                if stage == "PARTITION_FIXED":
                    hypotheses = request["initial_hypotheses"]
                    selected = 3
                else:
                    seed = request["ambiguous_seed_message"]
                    hypotheses = [
                        (
                            f"{seed} Refreshed context {index} after {root} "
                            f"has goal {index} and constraint {index}."
                        )
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    ]
                    match = re.search(r"detail ([1-5])", root)
                    root_number = int(match.group(1)) if match else 1
                    selected = (root_number - 1) % len(ACTION_KEYS)
                value: dict[str, Any] = {
                    **{
                        f"h{index}": hypotheses[index - 1]
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                    **{
                        f"w{index}": 10
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                }
                for action_index, action in enumerate(ACTION_KEYS):
                    if action_index == selected:
                        labels = (0, 0, 1, 1, 2, 2, 3, 3)
                    elif action_index == (selected + 1) % len(ACTION_KEYS):
                        labels = (0, 0, 0, 0, 1, 1, 1, 1)
                    else:
                        labels = (0,) * base.SUPPORT_SIZE
                    for index, label in enumerate(labels, start=1):
                        value[f"{action}{index}"] = label
                responses.append(json.dumps(value, separators=(",", ":")))
            self.requests += len(responses)
            return responses
        return super().chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("fixture", "serving", "mechanics"),
        required=True,
    )
    parser.add_argument("--config", type=Path)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path)
    parser.add_argument("--run-id")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixtures, public_fixture = base.build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.stage == "fixture":
        base._checkpoint(args.output_dir / "FIXTURE.json", public_fixture)
        print(json.dumps(public_fixture, indent=2, sort_keys=True))
        return
    if args.config is None or args.private_raw_dir is None or not args.run_id:
        parser.error(
            "--config, --private-raw-dir, and --run-id are required "
            "for serving/mechanics"
        )

    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.07 if args.stage == "serving" else 0.70
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else 30
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 1_500
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    models = (
        ModelBundle(
            generator=DeterministicGenerator("generator"),
            simulator=base.DeterministicFixtureModel("simulator"),
            checklist_judge=base.DeterministicFixtureModel(
                "checklist_judge"
            ),
        )
        if args.dry_run
        else refresh._build_models(config)
    )

    try:
        result = (
            run_serving_gate(models, raw_path=raw_path)
            if args.stage == "serving"
            else run_mechanics_gate(fixtures, models, raw_path=raw_path)
        )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        result["protocol"]["fixture_sha256"] = public_fixture[
            "private_fixture_sha256"
        ]
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        base._checkpoint(args.output_dir / "GATE_FAILURE.json", failure)
        raise

    output_name = (
        "SERVING.json" if args.stage == "serving" else "MECHANICS.json"
    )
    base._checkpoint(args.output_dir / output_name, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
