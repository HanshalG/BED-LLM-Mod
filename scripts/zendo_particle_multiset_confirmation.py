#!/usr/bin/env python3
"""Confirm Zendo path-dependent particle scoring across fresh public rules."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.zendo_final_readiness_belief_smoke import (
    MAX_NEW_TOKENS,
    _branch_signature,
    _build_model,
    _checkpoint,
    _usage_snapshot,
    build_pathways,
    deterministic_random_audit_bank,
    deterministic_scene_pool,
    fixed_support_depth_two_scores,
    parse_filtered_particle_population,
    parse_particle_population,
    parse_scorer,
    scorer_messages,
    select_root_scenes,
)
from scripts.zendo_path_dependent_belief_gate import (
    MODEL_ID,
    OBSERVATION_ACCURACY,
    PARTICLE_COUNT,
    RULE_ORDER,
    SOURCE_COMMIT,
    _argmax,
    _canonical_json,
    _sha256,
    _signature_count,
    behavior_agreements,
    branch_key,
    eig,
    initial_messages,
    posterior_weights,
    raw_official_scene,
    refresh_messages,
    spearman,
    truth_function,
    verify_source,
    weighted_agreement,
)


INTERFACE_VERSION = "zendo-particle-multiset-confirmation-1"
TASKS = ("upsilon", "iota", "kappa", "omega", "nu", "xi", "psi")
BASE_SEED = 24371
SEED_STRIDE = 7919
ROOT_COUNT = 4
EXPECTED_REQUESTS = len(TASKS) * (1 + 2 * ROOT_COUNT + 3)
RUN_COST_CAP_USD = 1.50
PROJECTED_COST_USD = 1.10
MIN_UNIQUE_ASTS = 8


class ConfirmationExecutionError(RuntimeError):
    """A failed-closed confirmation with usage accounting."""

    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass
class TaskState:
    name: str
    task_index: int
    selection_seed: int
    official_case: dict[str, Any]
    initial_scene: dict[str, Any]
    pool: list[dict[str, Any]]
    hypotheses: list[dict[str, Any]]
    initial_weights: list[float]
    roots: list[dict[str, Any]]
    root_selection: list[dict[str, Any]]
    branches: dict[str, list[dict[str, Any]]] | None = None
    pathways: list[dict[str, Any]] | None = None
    particle_reports: dict[str, dict[str, Any]] | None = None


def first_executable_positive_scene(
    official_case: dict[str, Any],
) -> tuple[int, dict[str, Any]]:
    for index, raw in enumerate(official_case["t"]):
        try:
            return index, raw_official_scene(raw)
        except ValueError:
            continue
    raise ValueError("task has no positive scene within the six-block DSL")


def task_seed(task_index: int) -> int:
    return BASE_SEED + SEED_STRIDE * task_index


def root_only_messages(
    *,
    hypotheses: Sequence[dict[str, Any]],
    weights: Sequence[float],
    roots: Sequence[dict[str, Any]],
    root_selection: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    packet = {
        "current_belief": [
            {
                "rule_text": hypothesis["rule_text"],
                "probability": weight,
            }
            for hypothesis, weight in zip(hypotheses, weights, strict=True)
        ],
        "candidate_roots": [
            {
                "label": chr(ord("A") + index),
                "root_scene": root,
                "predicted_probability_good": row[
                    "initial_probability_yes"
                ],
            }
            for index, (root, row) in enumerate(
                zip(roots, root_selection, strict=True)
            )
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You assess single experiments in a Zendo concept-learning game. "
                "Return one exact JSON object only, with no markdown, comments, "
                "reasoning, or extra fields."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    "A hidden executable rule classifies block scenes as good or bad.",
                    "Score each root A-D from 0 to 100 for expected readiness to state",
                    "a behaviorally correct executable rule immediately after that one",
                    "experiment. Judge current semantic coverage and how diagnostic its",
                    "two possible labels would be. Repeated rules are particle",
                    "multiplicity. You are not shown the hidden rule, future regenerated",
                    "beliefs, continuations, or exact information-gain values.",
                    "Do not favor label order.",
                    'Return exactly {"root_scores":[A,B,C,D]} using JSON integers.',
                    "ROOT_ONLY_STATE=" + _canonical_json(packet),
                ]
            ),
        },
    ]


def deranged_pathways(
    pathways: Sequence[dict[str, Any]],
    *,
    shift: int,
) -> list[dict[str, Any]]:
    if shift <= 0 or shift >= ROOT_COUNT:
        raise ValueError("derangement shift must be one through three")
    return [
        {
            "label": pathway["label"],
            "root_scene": pathway["root_scene"],
            "branches": pathways[(index + shift) % ROOT_COUNT]["branches"],
        }
        for index, pathway in enumerate(pathways)
    ]


def pairwise_accuracy(
    scores: Sequence[float],
    endpoints: Sequence[float],
) -> float:
    earned = 0.0
    count = 0
    for left, right in itertools.combinations(range(len(scores)), 2):
        endpoint_delta = endpoints[left] - endpoints[right]
        if math.isclose(endpoint_delta, 0.0, abs_tol=1e-12):
            continue
        score_delta = scores[left] - scores[right]
        count += 1
        if math.isclose(score_delta, 0.0, abs_tol=1e-12):
            earned += 0.5
        elif score_delta * endpoint_delta > 0:
            earned += 1.0
    return earned / count if count else 0.5


def exact_sign_flip_pvalue(differences: Sequence[float]) -> float:
    observed = sum(differences) / len(differences)
    outcomes = [
        sum(sign * value for sign, value in zip(signs, differences, strict=True))
        / len(differences)
        for signs in itertools.product((-1.0, 1.0), repeat=len(differences))
    ]
    return sum(value >= observed - 1e-12 for value in outcomes) / len(outcomes)


def _policy_row(
    *,
    name: str,
    choice: int,
    endpoints: Sequence[float],
) -> dict[str, Any]:
    return {
        "policy": name,
        "selected_root": choice + 1,
        "realized_weighted_truth_agreement": endpoints[choice],
    }


def analyze_task(
    state: TaskState,
    *,
    aligned_scores: list[int],
    root_only_scores: list[int],
    shuffled_scores: list[int],
) -> dict[str, Any]:
    if state.branches is None or state.pathways is None:
        raise ValueError("task branches are not frozen")
    if state.particle_reports is None:
        raise ValueError("task particle reports are not frozen")
    audit = deterministic_random_audit_bank(
        task_index=state.task_index,
        selection_seed=state.selection_seed,
    )
    truth = truth_function(state.name)
    immediate_scores = [
        eig(state.hypotheses, state.initial_weights, root)
        for root in state.roots
    ]
    fixed_scores = fixed_support_depth_two_scores(
        state.hypotheses,
        state.initial_weights,
        state.roots,
        state.pool,
    )
    root_rows = []
    for root_index, root in enumerate(state.roots):
        root_outcome = truth(root)
        packet = next(
            packet
            for packet in state.pathways[root_index]["branches"]
            if packet["outcome"] == root_outcome
        )
        continuation = packet["continuation_scene"]
        continuation_outcome = truth(continuation)
        support = state.branches[branch_key(root_index, root_outcome)]
        final_weights = posterior_weights(
            support,
            [
                (state.initial_scene, True),
                (root, root_outcome),
                (continuation, continuation_outcome),
            ],
        )
        agreements = behavior_agreements(support, audit, truth)
        root_rows.append(
            {
                "root_index": root_index + 1,
                "actual_root_outcome": root_outcome,
                "actual_continuation_outcome": continuation_outcome,
                "immediate_eig": immediate_scores[root_index],
                "fixed_support_depth_two_score": fixed_scores[root_index],
                "aligned_score": aligned_scores[root_index],
                "root_only_score": root_only_scores[root_index],
                "shuffled_score": shuffled_scores[root_index],
                "realized_weighted_truth_agreement": weighted_agreement(
                    agreements, final_weights
                ),
                "realized_maximum_truth_agreement": max(agreements),
            }
        )
    endpoints = [
        row["realized_weighted_truth_agreement"] for row in root_rows
    ]
    choices = {
        "aligned": _argmax(aligned_scores),
        "root_only": _argmax(root_only_scores),
        "shuffled": _argmax(shuffled_scores),
        "myopic": _argmax(immediate_scores),
        "fixed": _argmax(fixed_scores),
        "random": random.Random(state.selection_seed + 17).randrange(
            ROOT_COUNT
        ),
    }
    policies = {
        name: _policy_row(name=name, choice=choice, endpoints=endpoints)
        for name, choice in choices.items()
    }
    unique_ast_counts = {
        "initial": len(
            {
                _canonical_json(hypothesis["rule"])
                for hypothesis in state.hypotheses
            }
        ),
        **{
            key: len(
                {
                    _canonical_json(hypothesis["rule"])
                    for hypothesis in support
                }
            )
            for key, support in sorted(state.branches.items())
        },
    }
    branch_signatures = {
        _branch_signature(support, audit)
        for support in state.branches.values()
    }
    aligned_choice = choices["aligned"]
    myopic_choice = choices["myopic"]
    return {
        "task_name": state.name,
        "selection_seed": state.selection_seed,
        "initial_support_signature_count": _signature_count(
            state.hypotheses, audit
        ),
        "population_unique_ast_counts": unique_ast_counts,
        "particle_reports": state.particle_reports,
        "valid_particle_count_minimum": min(
            report["valid_particle_count"]
            for report in state.particle_reports.values()
        ),
        "invalid_particle_count": sum(
            report["invalid_particle_count"]
            for report in state.particle_reports.values()
        ),
        "raw_particle_count": sum(
            report["raw_particle_count"]
            for report in state.particle_reports.values()
        ),
        "distinct_refreshed_support_count": len(branch_signatures),
        "all_continuations_positive_finite": all(
            math.isfinite(packet["continuation_expected_information_nats"])
            and packet["continuation_expected_information_nats"] > 0
            for pathway in state.pathways
            for packet in pathway["branches"]
        ),
        "root_prediction_signature_count": len(
            {
                tuple(row["prediction_signature"])
                for row in state.root_selection
            }
        ),
        "informative_root_count": sum(
            min(
                row["initial_probability_yes"],
                1 - row["initial_probability_yes"],
            )
            >= 0.10
            for row in state.root_selection
        ),
        "endpoint_range": max(endpoints) - min(endpoints),
        "aligned_immediate_sacrifice": (
            immediate_scores[myopic_choice] - immediate_scores[aligned_choice]
        ),
        "aligned_endpoint_spearman": spearman(aligned_scores, endpoints),
        "root_only_endpoint_spearman": spearman(
            root_only_scores, endpoints
        ),
        "shuffled_endpoint_spearman": spearman(
            shuffled_scores, endpoints
        ),
        "aligned_pairwise_accuracy": pairwise_accuracy(
            aligned_scores, endpoints
        ),
        "root_only_pairwise_accuracy": pairwise_accuracy(
            root_only_scores, endpoints
        ),
        "shuffled_pairwise_accuracy": pairwise_accuracy(
            shuffled_scores, endpoints
        ),
        "aligned_score_unique_maximum": (
            aligned_scores.count(max(aligned_scores)) == 1
        ),
        "root_only_score_unique_maximum": (
            root_only_scores.count(max(root_only_scores)) == 1
        ),
        "shuffled_score_unique_maximum": (
            shuffled_scores.count(max(shuffled_scores)) == 1
        ),
        "aligned_scores_vary": len(set(aligned_scores)) >= 2,
        "root_only_scores_vary": len(set(root_only_scores)) >= 2,
        "shuffled_scores_vary": len(set(shuffled_scores)) >= 2,
        "policies": policies,
        "root_rows": root_rows,
        "audit_sha256": hashlib.sha256(
            _canonical_json(audit).encode("utf-8")
        ).hexdigest(),
    }


def _values(
    tasks: Sequence[dict[str, Any]],
    left: str,
    right: str,
) -> list[float]:
    return [
        task["policies"][left]["realized_weighted_truth_agreement"]
        - task["policies"][right]["realized_weighted_truth_agreement"]
        for task in tasks
    ]


def _comparison(differences: Sequence[float]) -> dict[str, Any]:
    return {
        "mean_difference": sum(differences) / len(differences),
        "wins": sum(value > 1e-12 for value in differences),
        "losses": sum(value < -1e-12 for value in differences),
        "ties": sum(abs(value) <= 1e-12 for value in differences),
        "exact_one_sided_sign_flip_p": exact_sign_flip_pvalue(differences),
        "per_task_differences": list(differences),
    }


def summarize(
    tasks: list[dict[str, Any]],
    usage: dict[str, Any],
) -> dict[str, Any]:
    comparisons = {
        "aligned_vs_myopic": _comparison(
            _values(tasks, "aligned", "myopic")
        ),
        "aligned_vs_fixed": _comparison(
            _values(tasks, "aligned", "fixed")
        ),
        "aligned_vs_root_only": _comparison(
            _values(tasks, "aligned", "root_only")
        ),
        "aligned_vs_shuffled": _comparison(
            _values(tasks, "aligned", "shuffled")
        ),
        "aligned_vs_random": _comparison(
            _values(tasks, "aligned", "random")
        ),
    }
    generator = usage["generator"]
    mechanics = {
        "exact_84_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_84_http_attempts": (
            generator["http_attempts"] == EXPECTED_REQUESTS
        ),
        "zero_transport_retries": generator["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": generator["forced_exits"] == 0,
        "all_responses_parsed_without_repair": True,
        "all_populations_have_eight_unique_asts": all(
            min(task["population_unique_ast_counts"].values())
            >= MIN_UNIQUE_ASTS
            for task in tasks
        ),
        "all_populations_have_eight_valid_particles": all(
            task["valid_particle_count_minimum"] >= 8 for task in tasks
        ),
        "invalid_particle_fraction_at_most_0_15": (
            sum(task["invalid_particle_count"] for task in tasks)
            / sum(task["raw_particle_count"] for task in tasks)
            <= 0.15
        ),
        "all_initial_supports_have_six_signatures": all(
            task["initial_support_signature_count"] >= 6 for task in tasks
        ),
        "all_tasks_have_four_distinct_informative_roots": all(
            task["root_prediction_signature_count"] == ROOT_COUNT
            and task["informative_root_count"] == ROOT_COUNT
            for task in tasks
        ),
        "all_tasks_have_four_distinct_refreshed_supports": all(
            task["distinct_refreshed_support_count"] >= 4 for task in tasks
        ),
        "all_continuations_positive_finite": all(
            task["all_continuations_positive_finite"] for task in tasks
        ),
        "all_scorer_vectors_vary": all(
            task[f"{condition}_scores_vary"]
            for task in tasks
            for condition in ("aligned", "root_only", "shuffled")
        ),
        "aligned_unique_maximum_on_five_tasks": sum(
            task["aligned_score_unique_maximum"] for task in tasks
        )
        >= 5,
        "cost_at_most_1_50": (
            usage["adapter_cost_usd"] <= RUN_COST_CAP_USD
        ),
    }
    mean_aligned_spearman = sum(
        task["aligned_endpoint_spearman"] for task in tasks
    ) / len(tasks)
    mean_root_only_spearman = sum(
        task["root_only_endpoint_spearman"] for task in tasks
    ) / len(tasks)
    mean_shuffled_spearman = sum(
        task["shuffled_endpoint_spearman"] for task in tasks
    ) / len(tasks)
    aligned_pairwise = sum(
        task["aligned_pairwise_accuracy"] for task in tasks
    ) / len(tasks)
    root_only_pairwise = sum(
        task["root_only_pairwise_accuracy"] for task in tasks
    ) / len(tasks)
    shuffled_pairwise = sum(
        task["shuffled_pairwise_accuracy"] for task in tasks
    ) / len(tasks)
    science = {
        "endpoint_range_at_least_0_10_on_four_tasks": sum(
            task["endpoint_range"] >= 0.10 for task in tasks
        )
        >= 4,
        "aligned_differs_from_myopic_on_four_tasks": sum(
            task["policies"]["aligned"]["selected_root"]
            != task["policies"]["myopic"]["selected_root"]
            for task in tasks
        )
        >= 4,
        "immediate_sacrifice_at_least_0_01_on_three_tasks": sum(
            task["aligned_immediate_sacrifice"] >= 0.01
            for task in tasks
        )
        >= 3,
        "aligned_vs_myopic_mean_at_least_0_04": (
            comparisons["aligned_vs_myopic"]["mean_difference"] >= 0.04
        ),
        "aligned_vs_myopic_four_wins_at_most_two_losses": (
            comparisons["aligned_vs_myopic"]["wins"] >= 4
            and comparisons["aligned_vs_myopic"]["losses"] <= 2
        ),
        "aligned_vs_myopic_sign_flip_p_at_most_0_10": (
            comparisons["aligned_vs_myopic"][
                "exact_one_sided_sign_flip_p"
            ]
            <= 0.10
        ),
        "aligned_vs_fixed_mean_at_least_0_03": (
            comparisons["aligned_vs_fixed"]["mean_difference"] >= 0.03
        ),
        "aligned_vs_fixed_four_wins_at_most_two_losses": (
            comparisons["aligned_vs_fixed"]["wins"] >= 4
            and comparisons["aligned_vs_fixed"]["losses"] <= 2
        ),
        "aligned_vs_root_only_mean_at_least_0_03": (
            comparisons["aligned_vs_root_only"]["mean_difference"] >= 0.03
        ),
        "aligned_vs_root_only_four_wins_at_most_two_losses": (
            comparisons["aligned_vs_root_only"]["wins"] >= 4
            and comparisons["aligned_vs_root_only"]["losses"] <= 2
        ),
        "aligned_vs_shuffled_mean_at_least_0_03": (
            comparisons["aligned_vs_shuffled"]["mean_difference"] >= 0.03
        ),
        "aligned_vs_shuffled_four_wins_at_most_two_losses": (
            comparisons["aligned_vs_shuffled"]["wins"] >= 4
            and comparisons["aligned_vs_shuffled"]["losses"] <= 2
        ),
        "aligned_vs_random_mean_at_least_0_04": (
            comparisons["aligned_vs_random"]["mean_difference"] >= 0.04
        ),
        "aligned_vs_random_four_wins": (
            comparisons["aligned_vs_random"]["wins"] >= 4
        ),
        "mean_aligned_spearman_at_least_0_25": (
            mean_aligned_spearman >= 0.25
        ),
        "aligned_spearman_exceeds_root_only_by_0_15": (
            mean_aligned_spearman - mean_root_only_spearman >= 0.15
        ),
        "aligned_spearman_exceeds_shuffled_by_0_15": (
            mean_aligned_spearman - mean_shuffled_spearman >= 0.15
        ),
        "aligned_pairwise_exceeds_root_only_by_0_05": (
            aligned_pairwise - root_only_pairwise >= 0.05
        ),
        "aligned_pairwise_exceeds_shuffled_by_0_05": (
            aligned_pairwise - shuffled_pairwise >= 0.05
        ),
    }
    gates = {**mechanics, **science}
    gates["all_pass"] = all(gates.values())
    return {
        "mechanics_gates": mechanics,
        "science_gates": science,
        "gates": gates,
        "comparisons": comparisons,
        "mean_aligned_endpoint_spearman": mean_aligned_spearman,
        "mean_root_only_endpoint_spearman": mean_root_only_spearman,
        "mean_shuffled_endpoint_spearman": mean_shuffled_spearman,
        "mean_aligned_pairwise_accuracy": aligned_pairwise,
        "mean_root_only_pairwise_accuracy": root_only_pairwise,
        "mean_shuffled_pairwise_accuracy": shuffled_pairwise,
    }


def run_confirmation(
    config: Config,
    *,
    source_dir: Path,
    raw_checkpoint_path: Path,
    model_adapter: Any | None = None,
    interface_version: str = INTERFACE_VERSION,
    filter_invalid_particles: bool = False,
) -> dict[str, Any]:
    cases_path = verify_source(source_dir)
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    case_by_name = dict(zip(RULE_ORDER, cases, strict=True))
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "tasks": {
            name: {
                "initial_hypotheses": None,
                "branch_refreshes": {},
                "scorers": {},
                "particle_reports": {},
            }
            for name in TASKS
        },
    }
    try:
        initial_scenes = {}
        initial_indices = {}
        for name in TASKS:
            index, scene = first_executable_positive_scene(
                case_by_name[name]
            )
            initial_indices[name] = index
            initial_scenes[name] = scene
        initial_responses = model.chat_complete_messages_batched(
            [initial_messages(initial_scenes[name]) for name in TASKS],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for name, response in zip(TASKS, initial_responses, strict=True):
            raw["tasks"][name]["initial_hypotheses"] = response
        _checkpoint(raw_checkpoint_path, raw)
        states = []
        for name, response in zip(TASKS, initial_responses, strict=True):
            if filter_invalid_particles:
                hypotheses, report = parse_filtered_particle_population(
                    response
                )
            else:
                hypotheses = parse_particle_population(
                    response, allow_duplicate_asts=True
                )
                report = {
                    "raw_particle_count": len(hypotheses),
                    "valid_particle_count": len(hypotheses),
                    "invalid_particle_count": 0,
                    "invalid_rows": [],
                }
            raw["tasks"][name]["particle_reports"]["initial"] = report
            task_index = RULE_ORDER.index(name)
            seed = task_seed(task_index)
            initial_scene = initial_scenes[name]
            initial_weights = posterior_weights(
                hypotheses, [(initial_scene, True)]
            )
            pool = deterministic_scene_pool(
                initial_scene, selection_seed=seed
            )
            roots, root_selection = select_root_scenes(
                hypotheses, initial_weights, pool
            )
            states.append(
                TaskState(
                    name=name,
                    task_index=task_index,
                    selection_seed=seed,
                    official_case=case_by_name[name],
                    initial_scene=initial_scene,
                    pool=pool,
                    hypotheses=hypotheses,
                    initial_weights=initial_weights,
                    roots=roots,
                    root_selection=root_selection,
                    particle_reports={"initial": report},
                )
            )
        refresh_jobs = [
            (state, root_index, outcome)
            for state in states
            for root_index in range(ROOT_COUNT)
            for outcome in (False, True)
        ]
        refresh_responses = model.chat_complete_messages_batched(
            [
                refresh_messages(
                    state.initial_scene,
                    state.hypotheses,
                    state.roots[root_index],
                    outcome,
                )
                for state, root_index, outcome in refresh_jobs
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        for (state, root_index, outcome), response in zip(
            refresh_jobs, refresh_responses, strict=True
        ):
            raw["tasks"][state.name]["branch_refreshes"][
                branch_key(root_index, outcome)
            ] = response
        _checkpoint(raw_checkpoint_path, raw)
        for state in states:
            state.branches = {}
            assert state.particle_reports is not None
            for key, response in raw["tasks"][state.name][
                "branch_refreshes"
            ].items():
                if filter_invalid_particles:
                    population, report = (
                        parse_filtered_particle_population(response)
                    )
                else:
                    population = parse_particle_population(
                        response, allow_duplicate_asts=True
                    )
                    report = {
                        "raw_particle_count": len(population),
                        "valid_particle_count": len(population),
                        "invalid_particle_count": 0,
                        "invalid_rows": [],
                    }
                state.branches[key] = population
                state.particle_reports[key] = report
                raw["tasks"][state.name]["particle_reports"][key] = report
            state.pathways = build_pathways(
                initial_scene=state.initial_scene,
                initial_hypotheses=state.hypotheses,
                initial_weights=state.initial_weights,
                roots=state.roots,
                branches=state.branches,
                pool=state.pool,
            )
        scorer_jobs = [
            (state, condition)
            for state in states
            for condition in ("aligned", "root_only", "shuffled")
        ]
        scorer_prompts = []
        for state, condition in scorer_jobs:
            assert state.pathways is not None
            if condition == "aligned":
                prompt = scorer_messages(
                    state.pathways, particle_semantics="multiset"
                )
            elif condition == "root_only":
                prompt = root_only_messages(
                    hypotheses=state.hypotheses,
                    weights=state.initial_weights,
                    roots=state.roots,
                    root_selection=state.root_selection,
                )
            else:
                prompt = scorer_messages(
                    deranged_pathways(
                        state.pathways,
                        shift=1 + state.task_index % 3,
                    ),
                    particle_semantics="multiset",
                )
            scorer_prompts.append(prompt)
        scorer_responses = model.chat_complete_messages_batched(
            scorer_prompts,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=MAX_NEW_TOKENS,
        )
        parsed_scores: dict[str, dict[str, list[int]]] = {
            state.name: {} for state in states
        }
        for (state, condition), response in zip(
            scorer_jobs, scorer_responses, strict=True
        ):
            raw["tasks"][state.name]["scorers"][condition] = response
            parsed_scores[state.name][condition] = parse_scorer(response)
        _checkpoint(raw_checkpoint_path, raw)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_checkpoint_path, raw)
        raise ConfirmationExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc
    task_rows = [
        analyze_task(
            state,
            aligned_scores=parsed_scores[state.name]["aligned"],
            root_only_scores=parsed_scores[state.name]["root_only"],
            shuffled_scores=parsed_scores[state.name]["shuffled"],
        )
        for state in states
    ]
    summary = summarize(task_rows, usage)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "tasks": list(TASKS),
            "model": MODEL_ID,
            "base_seed": BASE_SEED,
            "seed_stride": SEED_STRIDE,
            "source_commit": SOURCE_COMMIT,
            "particle_count": PARTICLE_COUNT,
            "root_count": ROOT_COUNT,
            "observation_accuracy": OBSERVATION_ACCURACY,
            "expected_physical_requests": EXPECTED_REQUESTS,
            "scorer_conditions": ["aligned", "root_only", "shuffled"],
            "truth_hidden_until_all_populations_and_scores_frozen": True,
            "reasoning_requested": False,
            "repairs_or_scientific_retries": 0,
            "particle_semantics": "multiset",
            "particle_validation": (
                "filter_invalid_ast_samples"
                if filter_invalid_particles
                else "fail_on_any_invalid_ast"
            ),
            "audit_mode": "random_only",
            "initial_positive_indices": initial_indices,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "tasks": task_rows,
        "usage": usage,
    }


def run_cli(
    *,
    interface_version: str = INTERFACE_VERSION,
    filter_invalid_particles: bool = False,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = RUN_COST_CAP_USD
    config.openrouter_concurrency = 8
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        payload = run_confirmation(
            config,
            source_dir=args.source_dir,
            raw_checkpoint_path=raw_path,
            interface_version=interface_version,
            filter_invalid_particles=filter_invalid_particles,
        )
        payload["protocol"]["private_raw_sha256"] = _sha256(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": interface_version,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, ConfirmationExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256(raw_path)
        (args.output_dir / "FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / "CONFIRMATION.json"
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "summary": payload["summary"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run_cli()
