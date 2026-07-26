#!/usr/bin/env python3
"""Compare refreshed- and fixed-support InfoQuest continuations."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-refresh-continuation-1"
GENERATOR_MODEL_ID = base.GENERATOR_MODEL_ID
SIMULATOR_MODEL_ID = base.SIMULATOR_MODEL_ID
CHECKLIST_JUDGE_MODEL_ID = base.CHECKLIST_JUDGE_MODEL_ID
EXPECTED_SERVING_REQUESTS = 10
EXPECTED_MECHANICS_REQUESTS = 159
SERVING_MAX_COST_USD = 0.12
MECHANICS_MAX_COST_USD = 0.85


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class ModelBundle:
    generator: ChatModel
    simulator: ChatModel
    checklist_judge: ChatModel


def fixed_support_messages(
    seed_message: str,
    initial: base.InitialPolicy,
    root: str,
    answer: str,
) -> list[dict[str, str]]:
    request = {
        "ambiguous_seed_message": seed_message,
        "fixed_initial_hypotheses": list(initial.hypotheses),
        "clarification_question": root,
        "user_answer": answer,
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=FIXED_FULL. Keep the supplied eight hypotheses exactly "
                "fixed. Copy h1..h8 verbatim without adding, removing, "
                "rewriting, or reordering any hypothesis. Using that fixed "
                "support and the observed answer, choose one atomic follow-up "
                "question. Output only one JSON object with string fields "
                "h1..h8 and followup. The followup must contain one question "
                "mark, end with it, and contain neither and nor or."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def parse_fixed_support(
    response: str,
    initial: base.InitialPolicy,
) -> base.RefreshPolicy:
    policy = base.parse_refresh(response)
    if policy.hypotheses != initial.hypotheses:
        raise ValueError("fixed-support response changed the initial support")
    return policy


def _complete(
    model: ChatModel,
    messages: list[list[dict[str, str]]],
    *,
    max_new_tokens: int,
) -> list[str]:
    return model.chat_complete_messages_batched(
        messages,
        temperature=0.0,
        block_size=len(messages),
        max_new_tokens=max_new_tokens,
    )


def aggregate_usage(models: ModelBundle) -> dict[str, Any]:
    snapshots = {
        "generator": models.generator.usage_snapshot(),
        "simulator": models.simulator.usage_snapshot(),
        "checklist_judge": models.checklist_judge.usage_snapshot(),
    }
    return {
        "physical_requests": sum(
            int(value.get("adapter_requests", 0))
            for value in snapshots.values()
        ),
        "http_attempts": sum(
            int(value.get("http_attempts", 0))
            for value in snapshots.values()
        ),
        "retry_count": sum(
            int(value.get("retry_count", 0))
            for value in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(value.get("adapter_reasoning_tokens", 0))
            for value in snapshots.values()
        ),
        "forced_exits": sum(
            int(value.get("forced_exits", 0))
            for value in snapshots.values()
        ),
        "adapter_cost_usd": sum(
            float(value.get("adapter_cost_usd", 0.0))
            for value in snapshots.values()
        ),
        "models": snapshots,
    }


def _checkpoint_stage(
    raw_path: Path,
    raw: dict[str, Any],
    key: str,
    responses: Sequence[str],
) -> None:
    raw[key] = list(responses)
    base._checkpoint(raw_path, raw)


def _normalized_set(values: Sequence[str]) -> set[str]:
    return {base._normalize(value) for value in values}


def _public_metrics(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    refreshed: Sequence[Sequence[base.RefreshPolicy]],
    fixed: Sequence[Sequence[base.RefreshPolicy]],
    dynamic_answers: Sequence[Sequence[str]],
    fixed_answers: Sequence[Sequence[str]],
    judgments: Sequence[base.ChecklistJudgment],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    fixture_metrics: list[dict[str, Any]] = []
    all_differences: list[int] = []
    all_dynamic_incremental: list[int] = []
    all_fixed_incremental: list[int] = []
    changed_supports = 0
    novel_fractions: list[float] = []
    different_followups = 0
    positive_fixtures = 0
    fixtures_with_dynamic_range = 0

    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        initial_set = _normalized_set(initial.hypotheses)
        judgment = judgments[fixture_index]
        immediate = [sum(bits) for bits in judgment.immediate]
        dynamic_total = [sum(bits) for bits in judgment.dynamic]
        fixed_total = [sum(bits) for bits in judgment.fixed]
        dynamic_incremental = [
            total - first for total, first in zip(dynamic_total, immediate)
        ]
        fixed_incremental = [
            total - first for total, first in zip(fixed_total, immediate)
        ]
        differences = [
            dynamic_value - fixed_value
            for dynamic_value, fixed_value in zip(
                dynamic_total,
                fixed_total,
            )
        ]
        support_changed = []
        fixture_novel_fractions = []
        followup_different = []
        for root_index in range(base.ROOT_COUNT):
            dynamic_policy = refreshed[fixture_index][root_index]
            fixed_policy = fixed[fixture_index][root_index]
            dynamic_set = _normalized_set(dynamic_policy.hypotheses)
            changed = dynamic_set != initial_set
            novel_fraction = len(dynamic_set - initial_set) / base.SUPPORT_SIZE
            differs = (
                base._normalize(dynamic_policy.followup)
                != base._normalize(fixed_policy.followup)
            )
            support_changed.append(changed)
            fixture_novel_fractions.append(novel_fraction)
            followup_different.append(differs)
        fixture_mean_difference = sum(differences) / base.ROOT_COUNT
        if fixture_mean_difference > 0:
            positive_fixtures += 1
        if max(dynamic_total) - min(dynamic_total) >= 1:
            fixtures_with_dynamic_range += 1
        changed_supports += sum(support_changed)
        novel_fractions.extend(fixture_novel_fractions)
        different_followups += sum(followup_different)
        all_differences.extend(differences)
        all_dynamic_incremental.extend(dynamic_incremental)
        all_fixed_incremental.extend(fixed_incremental)
        fixture_metrics.append(
            {
                "fixture_id": fixture.fixture_id,
                "immediate_checklist_counts": immediate,
                "dynamic_checklist_counts": dynamic_total,
                "fixed_checklist_counts": fixed_total,
                "dynamic_incremental_counts": dynamic_incremental,
                "fixed_incremental_counts": fixed_incremental,
                "dynamic_minus_fixed_counts": differences,
                "mean_dynamic_minus_fixed": fixture_mean_difference,
                "refreshed_support_changed": support_changed,
                "refreshed_support_novel_fraction": fixture_novel_fractions,
                "dynamic_followup_differs_from_fixed": followup_different,
                "initial_support_sha256": base._sha256_value(
                    initial.hypotheses
                ),
                "root_hashes": [
                    base._sha256_value(root) for root in initial.roots
                ],
                "root_answer_hashes": [
                    base._sha256_value(value)
                    for value in root_answers[fixture_index]
                ],
                "dynamic_support_hashes": [
                    base._sha256_value(policy.hypotheses)
                    for policy in refreshed[fixture_index]
                ],
                "fixed_support_hashes": [
                    base._sha256_value(policy.hypotheses)
                    for policy in fixed[fixture_index]
                ],
                "dynamic_followup_hashes": [
                    base._sha256_value(policy.followup)
                    for policy in refreshed[fixture_index]
                ],
                "fixed_followup_hashes": [
                    base._sha256_value(policy.followup)
                    for policy in fixed[fixture_index]
                ],
                "dynamic_answer_hashes": [
                    base._sha256_value(value)
                    for value in dynamic_answers[fixture_index]
                ],
                "fixed_answer_hashes": [
                    base._sha256_value(value)
                    for value in fixed_answers[fixture_index]
                ],
            }
        )

    initial_support_hashes = {
        base._sha256_value(initials[record_id].hypotheses)
        for record_id in base.MECHANICS_IDS
    }
    wins = sum(value > 0 for value in all_differences)
    ties = sum(value == 0 for value in all_differences)
    losses = sum(value < 0 for value in all_differences)
    metrics = {
        "fixtures": len(fixtures),
        "root_world_cells": len(all_differences),
        "distinct_initial_supports": len(initial_support_hashes),
        "refreshed_support_changed_cells": changed_supports,
        "mean_refreshed_support_novel_fraction": (
            sum(novel_fractions) / len(novel_fractions)
        ),
        "dynamic_followup_differs_from_fixed_cells": different_followups,
        "mean_dynamic_incremental_checklist": (
            sum(all_dynamic_incremental) / len(all_dynamic_incremental)
        ),
        "mean_fixed_incremental_checklist": (
            sum(all_fixed_incremental) / len(all_fixed_incremental)
        ),
        "mean_dynamic_minus_fixed_checklist": (
            sum(all_differences) / len(all_differences)
        ),
        "dynamic_wins_ties_losses_vs_fixed": [wins, ties, losses],
        "fixtures_with_positive_mean_dynamic_minus_fixed": positive_fixtures,
        "fixtures_with_dynamic_endpoint_range_at_least_1": (
            fixtures_with_dynamic_range
        ),
    }
    gates = {
        "three_distinct_initial_supports": (
            metrics["distinct_initial_supports"] == 3
        ),
        "at_least_24_refreshed_supports_change": (
            metrics["refreshed_support_changed_cells"] >= 24
        ),
        "mean_novel_support_fraction_at_least_0_50": (
            metrics["mean_refreshed_support_novel_fraction"] >= 0.50
        ),
        "at_least_20_dynamic_followups_differ_from_fixed": (
            metrics["dynamic_followup_differs_from_fixed_cells"] >= 20
        ),
        "dynamic_incremental_mean_at_least_0_50": (
            metrics["mean_dynamic_incremental_checklist"] >= 0.50
        ),
        "dynamic_beats_fixed_by_at_least_0_15": (
            metrics["mean_dynamic_minus_fixed_checklist"] >= 0.15
        ),
        "dynamic_has_more_wins_than_losses": wins > losses,
        "at_least_4_fixtures_have_positive_dynamic_gain": (
            metrics["fixtures_with_positive_mean_dynamic_minus_fixed"] >= 4
        ),
        "at_least_4_fixtures_have_dynamic_endpoint_range": (
            metrics["fixtures_with_dynamic_endpoint_range_at_least_1"] >= 4
        ),
    }
    return metrics, fixture_metrics, gates


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        initial_raw = _complete(
            models.generator,
            [
                base.initial_messages(f"Synthetic ambiguous request {index}.")
                for index in range(2)
            ],
            max_new_tokens=1_200,
        )
        _checkpoint_stage(raw_path, raw, "initial", initial_raw)
        initials = [base.parse_initial(value) for value in initial_raw]

        synthetic_system = (
            "You are a hidden user. Answer the latest specific question in "
            "one concise sentence and reveal at most one detail."
        )
        root_raw = _complete(
            models.simulator,
            [
                base.simulator_root_messages(
                    synthetic_system,
                    initials[index].roots[0],
                )
                for index in range(2)
            ],
            max_new_tokens=160,
        )
        _checkpoint_stage(raw_path, raw, "root_answers", root_raw)
        root_answers = [
            base._clean_text(value, maximum=1_000) for value in root_raw
        ]

        refresh_raw = _complete(
            models.generator,
            [
                base.refresh_messages(
                    f"Synthetic ambiguous request {index}.",
                    initials[index],
                    initials[index].roots[0],
                    root_answers[index],
                )
                for index in range(2)
            ],
            max_new_tokens=1_100,
        )
        _checkpoint_stage(raw_path, raw, "refresh", refresh_raw)
        refreshed = [base.parse_refresh(value) for value in refresh_raw]

        fixed_raw = _complete(
            models.generator,
            [
                fixed_support_messages(
                    f"Synthetic ambiguous request {index}.",
                    initials[index],
                    initials[index].roots[0],
                    root_answers[index],
                )
                for index in range(2)
            ],
            max_new_tokens=1_100,
        )
        _checkpoint_stage(raw_path, raw, "fixed", fixed_raw)
        fixed = [
            parse_fixed_support(value, initials[index])
            for index, value in enumerate(fixed_raw)
        ]

        followup_raw = _complete(
            models.simulator,
            [
                base.simulator_followup_messages(
                    synthetic_system,
                    initials[0].roots[0],
                    root_answers[0],
                    refreshed[0].followup,
                )
            ],
            max_new_tokens=160,
        )
        _checkpoint_stage(raw_path, raw, "followup_answers", followup_raw)
        followup_answer = [base._clean_text(value) for value in followup_raw]

        fixture = base.WorldFixture(
            fixture_id="SYNTHETIC",
            record_id=-1,
            world=1,
            seed_message="Synthetic ambiguous request.",
            simulator_system=synthetic_system,
            truth_packet={},
            checklist=tuple(f"Checklist item {index}" for index in range(5)),
        )
        checklist_raw = _complete(
            models.checklist_judge,
            [
                base.checklist_judge_messages(
                    fixture,
                    initials[0],
                    [root_answers[0]] * base.ROOT_COUNT,
                    [refreshed[0]] * base.ROOT_COUNT,
                    [followup_answer[0]] * base.ROOT_COUNT,
                    [fixed[0].followup] * base.ROOT_COUNT,
                    [followup_answer[0]] * base.ROOT_COUNT,
                )
            ],
            max_new_tokens=280,
        )
        _checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgment",
            checklist_raw,
        )
        checklist = [
            base.parse_checklist_judgment(value) for value in checklist_raw
        ]
        usage = aggregate_usage(models)
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(models),
        ) from exc

    gates = {
        "exact_10_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_10_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_stage_parsers_pass": (
            len(initials) == 2
            and len(root_answers) == 2
            and len(refreshed) == 2
            and len(fixed) == 2
            and len(followup_answer) == 1
            and len(checklist) == 1
        ),
        "cost_at_most_0_12": (
            usage["adapter_cost_usd"] <= SERVING_MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "serving",
            "expected_requests": EXPECTED_SERVING_REQUESTS,
            "models": {
                "generator_refresh_fixed": GENERATOR_MODEL_ID,
                "simulator": SIMULATOR_MODEL_ID,
                "checklist_judge": CHECKLIST_JUDGE_MODEL_ID,
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
        initial_raw = _complete(
            models.generator,
            [
                base.initial_messages(
                    fixtures_by_record[record_id].seed_message
                )
                for record_id in base.MECHANICS_IDS
            ],
            max_new_tokens=1_300,
        )
        _checkpoint_stage(raw_path, raw, "initial", initial_raw)
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
        root_raw = _complete(
            models.simulator,
            root_requests,
            max_new_tokens=220,
        )
        _checkpoint_stage(raw_path, raw, "root_answers", root_raw)
        root_flat = [
            base._clean_text(value, maximum=1_200) for value in root_raw
        ]
        root_answers = [
            root_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(root_flat), base.ROOT_COUNT)
        ]

        refresh_requests = []
        fixed_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index, root in enumerate(initial.roots):
                answer = root_answers[fixture_index][root_index]
                refresh_requests.append(
                    base.refresh_messages(
                        fixture.seed_message,
                        initial,
                        root,
                        answer,
                    )
                )
                fixed_requests.append(
                    fixed_support_messages(
                        fixture.seed_message,
                        initial,
                        root,
                        answer,
                    )
                )
        refresh_raw = _complete(
            models.generator,
            refresh_requests,
            max_new_tokens=1_200,
        )
        _checkpoint_stage(raw_path, raw, "refresh", refresh_raw)
        refreshed_flat = [base.parse_refresh(value) for value in refresh_raw]
        refreshed = [
            refreshed_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(refreshed_flat), base.ROOT_COUNT)
        ]

        fixed_raw = _complete(
            models.generator,
            fixed_requests,
            max_new_tokens=1_200,
        )
        _checkpoint_stage(raw_path, raw, "fixed", fixed_raw)
        fixed_flat = []
        for response_index, response in enumerate(fixed_raw):
            fixture_index = response_index // base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            fixed_flat.append(parse_fixed_support(response, initial))
        fixed = [
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
                        refreshed[fixture_index][root_index].followup,
                    )
                )
                followup_requests.append(
                    base.simulator_followup_messages(
                        *common,
                        fixed[fixture_index][root_index].followup,
                    )
                )
        followup_raw = _complete(
            models.simulator,
            followup_requests,
            max_new_tokens=220,
        )
        _checkpoint_stage(raw_path, raw, "followup_answers", followup_raw)
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

        checklist_raw = _complete(
            models.checklist_judge,
            [
                base.checklist_judge_messages(
                    fixture,
                    initials[fixture.record_id],
                    root_answers[fixture_index],
                    refreshed[fixture_index],
                    dynamic_answers[fixture_index],
                    [
                        policy.followup
                        for policy in fixed[fixture_index]
                    ],
                    fixed_answers[fixture_index],
                )
                for fixture_index, fixture in enumerate(fixtures)
            ],
            max_new_tokens=320,
        )
        _checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgments",
            checklist_raw,
        )
        judgments = [
            base.parse_checklist_judgment(value) for value in checklist_raw
        ]
        usage = aggregate_usage(models)
        metrics, fixture_metrics, scientific_gates = _public_metrics(
            fixtures,
            initials,
            root_answers,
            refreshed,
            fixed,
            dynamic_answers,
            fixed_answers,
            judgments,
        )
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(models),
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
        "cost_at_most_0_85": (
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
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "models": {
                "generator_refresh_fixed": GENERATOR_MODEL_ID,
                "simulator": SIMULATOR_MODEL_ID,
                "checklist_judge": CHECKLIST_JUDGE_MODEL_ID,
            },
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "compute_matched_support_outputs": True,
            "all_batches_checkpointed_before_parse": True,
            "opportunity_or_later_split_read": False,
            "causal_policy_efficacy_claimed": False,
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


class DeterministicGenerator(base.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if all(
            "STAGE=INITIAL" in messages[0]["content"]
            for messages in batch_messages
        ):
            responses = []
            for messages in batch_messages:
                request = json.loads(messages[-1]["content"])
                seed = request["ambiguous_seed_message"]
                value = {
                    **{
                        f"h{index}": (
                            f"{seed} Concrete hidden context {index} has "
                            f"goal {index} and constraint {index}."
                        )
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                    **{
                        f"q{index}": f"What is hidden detail {index}?"
                        for index in range(1, base.ROOT_COUNT + 1)
                    },
                }
                responses.append(json.dumps(value, separators=(",", ":")))
            self.requests += len(responses)
            return responses
        if all(
            "STAGE=FIXED_FULL" in messages[0]["content"]
            for messages in batch_messages
        ):
            responses = []
            for messages in batch_messages:
                request = json.loads(messages[-1]["content"])
                hypotheses = request["fixed_initial_hypotheses"]
                value = {
                    **{
                        f"h{index}": hypotheses[index - 1]
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                    "followup": "What is the fixed hidden detail?",
                }
                responses.append(json.dumps(value, separators=(",", ":")))
            self.requests += len(responses)
            return responses
        return super().chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens,
        )


def _nonthinking_spec(spec: Any, model_id: str) -> Any:
    return replace(
        spec,
        model=model_id,
        backend="openrouter",
        thinking=None,
        reasoning_effort=None,
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_models(config: Config) -> ModelBundle:
    base_spec = config.model_pairs[0].questioner
    return ModelBundle(
        generator=build_model_adapter(
            _nonthinking_spec(base_spec, GENERATOR_MODEL_ID),
            config,
        ),
        simulator=build_model_adapter(
            _nonthinking_spec(base_spec, SIMULATOR_MODEL_ID),
            config,
        ),
        checklist_judge=build_model_adapter(
            _nonthinking_spec(base_spec, CHECKLIST_JUDGE_MODEL_ID),
            config,
        ),
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
        0.07 if args.stage == "serving" else 0.58
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
    config.openrouter_max_output_tokens = 1_400
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
        else _build_models(config)
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
