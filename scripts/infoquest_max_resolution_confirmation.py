#!/usr/bin/env python3
"""Confirm max-resolution information-need scoring on fresh InfoQuest records."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts import infoquest_cached_trajectory_opportunity as trajectory
from scripts import infoquest_information_need_gate as needs
from scripts import infoquest_llm_bed_manifest as manifest
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base
from scripts import infoquest_target_alignment_audit as alignment


INTERFACE_VERSION = "infoquest-max-resolution-confirmation-1"
MANIFEST_PUBLIC_SHA256 = (
    "392ae5ee33ee807994cb917ea8ff38ba6bd9ddaf95112ecb15e2c599783d32cb"
)
DEVELOPMENT_SPLIT_SHA256 = (
    "068587d494c71d5488da4f1d53ebe34be713c178083f6be98733085337c48b48"
)
CONFIRMATION_IDS = (443, 153, 210, 356, 237)
RANDOM_SEED = 24_418
EXPECTED_SERVING_REQUESTS = 10
EXPECTED_MECHANICS_REQUESTS = 115
SERVING_MAX_COST_USD = 0.10
MECHANICS_MAX_COST_USD = 0.60

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


def _sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest(path: Path) -> None:
    if _sha256_path(path) != MANIFEST_PUBLIC_SHA256:
        raise ValueError("InfoQuest manifest SHA-256 mismatch")
    value = json.loads(path.read_text())
    development = value.get("selection", {}).get("splits", {}).get(
        "development",
        {},
    )
    if development.get("ordered_sha256") != DEVELOPMENT_SPLIT_SHA256:
        raise ValueError("InfoQuest development split hash mismatch")
    record_ids = development.get("record_ids")
    if not isinstance(record_ids, list):
        raise ValueError("InfoQuest development IDs are missing")
    if tuple(record_ids[: len(CONFIRMATION_IDS)]) != CONFIRMATION_IDS:
        raise ValueError("InfoQuest confirmation IDs are not the frozen prefix")


def build_fresh_fixtures(
    *,
    settings_path: Path,
    baseline_path: Path,
) -> tuple[list[base.WorldFixture], dict[str, Any]]:
    settings = base._load_jsonl_by_id(
        settings_path,
        manifest.SOURCE_SHA256["settings"],
    )
    baseline = base._load_jsonl_by_id(
        baseline_path,
        trajectory.BASELINE_SHA256[0],
    )
    fixtures = []
    private_rows = []
    for record_id in CONFIRMATION_IDS:
        setting_row = settings[record_id]
        for world in (1, 2):
            setting = setting_row[f"setting{world}"]
            checklist = setting.get("checklist")
            if (
                not isinstance(checklist, list)
                or len(checklist) != base.ROOT_COUNT
                or not all(
                    isinstance(item, str) and item.strip()
                    for item in checklist
                )
            ):
                raise ValueError("fresh InfoQuest checklist is malformed")
            history = baseline[record_id][f"user_history{world}"]
            if (
                not isinstance(history, list)
                or not history
                or history[0].get("role") != "system"
                or not isinstance(history[0].get("content"), str)
                or not history[0]["content"].strip()
            ):
                raise ValueError("fresh InfoQuest simulator prompt is malformed")
            truth_packet = {
                key: setting[key]
                for key in (
                    "description",
                    "goal",
                    "obstacle",
                    "constraints",
                    "solution",
                    "persona",
                )
            }
            fixture = base.WorldFixture(
                fixture_id=f"D{record_id}W{world}",
                record_id=record_id,
                world=world,
                seed_message=setting_row["seed_message"],
                simulator_system=history[0]["content"],
                truth_packet=truth_packet,
                checklist=tuple(checklist),
            )
            fixtures.append(fixture)
            private_rows.append(
                {
                    "fixture_id": fixture.fixture_id,
                    "seed_message": fixture.seed_message,
                    "simulator_system": fixture.simulator_system,
                    "truth_packet": fixture.truth_packet,
                    "checklist": fixture.checklist,
                }
            )
    private_sha = base._sha256_value(private_rows)
    return fixtures, {
        "interface_version": INTERFACE_VERSION,
        "confirmation_ids": list(CONFIRMATION_IDS),
        "fixtures": len(fixtures),
        "settings_sha256": manifest.SOURCE_SHA256["settings"],
        "baseline_sha256": trajectory.BASELINE_SHA256[0],
        "private_fixture_sha256": private_sha,
        "fixture_hashes": {
            row["fixture_id"]: base._sha256_value(row)
            for row in private_rows
        },
        "semantic_content_emitted": False,
    }


def immediate_judgment_messages(
    fixture: base.WorldFixture,
    initial: base.InitialPolicy,
    root_answers: Sequence[str],
) -> list[dict[str, str]]:
    placeholders = [
        base.RefreshPolicy(
            hypotheses=initial.hypotheses,
            followup=initial.roots[index],
        )
        for index in range(base.ROOT_COUNT)
    ]
    return base.checklist_judge_messages(
        fixture,
        initial,
        root_answers,
        placeholders,
        root_answers,
        initial.roots,
        root_answers,
    )


class DeterministicFreshChecklist(base.DeterministicFixtureModel):
    BITS = ("10000", "11000", "11100", "11110", "11111")

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if all(
            "STAGE=CHECKLIST_JUDGE" in messages[0]["content"]
            for messages in batch_messages
        ):
            responses = [
                "\n".join(
                    f"Q{index}|{bits}|{bits}|{bits}"
                    for index, bits in enumerate(self.BITS, start=1)
                )
                for _ in batch_messages
            ]
            self.requests += len(responses)
            return responses
        return super().chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens,
        )


class DeterministicFreshGenerator(needs.DeterministicNeedGenerator):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        if all(
            "STAGE=INFORMATION_NEED_BELIEF" in messages[0]["content"]
            for messages in batch_messages
        ):
            responses = []
            for _messages in batch_messages:
                if self.selection_cursor < len(self.selected_actions):
                    selected = self.selected_actions[self.selection_cursor]
                else:
                    selected = self.selection_cursor % 4
                self.selection_cursor += 1
                distractor = (selected + 1) % 4
                profiles = [[10] * needs.NEED_COUNT for _ in range(4)]
                profiles[selected] = [0, 0, 0, 0, 99]
                profiles[distractor] = [70] * needs.NEED_COUNT
                response = {
                    "n": [
                        f"Missing decision-critical value {index}"
                        for index in range(1, needs.NEED_COUNT + 1)
                    ],
                    "w": [100, 80, 60, 40, 20],
                    **{
                        action: profiles[action_index]
                        for action_index, action in enumerate(
                            needs.partition.ACTION_KEYS
                        )
                    },
                }
                responses.append(json.dumps(response, separators=(",", ":")))
            self.requests += len(responses)
            return responses
        return super().chat_complete_messages_batched(
            batch_messages,
            temperature,
            block_size,
            max_new_tokens,
        )


def _synthetic_fixture(index: int) -> base.WorldFixture:
    return base.WorldFixture(
        fixture_id=f"S{index}",
        record_id=index,
        world=1,
        seed_message=f"Synthetic ambiguous request {index}.",
        simulator_system="Answer the user's one question briefly.",
        truth_packet={},
        checklist=tuple(f"Synthetic need {item}" for item in range(5)),
    )


def _dummy_initial(index: int) -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(f"Context {index}-{item}" for item in range(8)),
        roots=tuple(f"What is synthetic detail {item}?" for item in range(5)),
    )


def _pairwise(
    better: Sequence[int],
    baseline: Sequence[int],
) -> list[int]:
    wins = sum(a > b for a, b in zip(better, baseline))
    ties = sum(a == b for a, b in zip(better, baseline))
    losses = sum(a < b for a, b in zip(better, baseline))
    return [wins, ties, losses]


def _score_metrics(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    beliefs: Sequence[Sequence[needs.InformationNeedBelief]],
    judgments: Sequence[base.ChecklistJudgment],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    rng = random.Random(RANDOM_SEED)
    scorer_gains = {"max": [], "linear": [], "random": [], "oracle": []}
    max_rhos = []
    linear_rhos = []
    max_optimal = 0
    max_score_spread = 0
    target_spread = 0
    fixture_metrics = []

    for fixture_index, fixture in enumerate(fixtures):
        judgment = judgments[fixture_index]
        max_fixture = []
        linear_fixture = []
        random_fixture = []
        oracle_fixture = []
        for root_index in range(base.ROOT_COUNT):
            candidate_indices = [
                index
                for index in range(base.ROOT_COUNT)
                if index != root_index
            ]
            gains = [
                alignment.additive_gain(
                    judgment.immediate[root_index],
                    judgment.immediate[index],
                )
                for index in candidate_indices
            ]
            belief = beliefs[fixture_index][root_index]
            max_scores = tuple(
                float(max(profile))
                for profile in belief.resolution_probabilities
            )
            linear_scores = belief.scores
            max_action = max(
                range(len(max_scores)),
                key=max_scores.__getitem__,
            )
            linear_action = max(
                range(len(linear_scores)),
                key=linear_scores.__getitem__,
            )
            random_action = rng.randrange(len(gains))
            oracle = max(gains)
            max_gain = gains[max_action]
            linear_gain = gains[linear_action]
            random_gain = gains[random_action]

            scorer_gains["max"].append(max_gain)
            scorer_gains["linear"].append(linear_gain)
            scorer_gains["random"].append(random_gain)
            scorer_gains["oracle"].append(oracle)
            max_fixture.append(max_gain)
            linear_fixture.append(linear_gain)
            random_fixture.append(random_gain)
            oracle_fixture.append(oracle)
            max_optimal += max_gain == oracle
            max_score_spread += max(max_scores) > min(max_scores)
            target_spread += max(gains) > min(gains)
            max_rho = alignment.spearman(max_scores, gains)
            linear_rho = alignment.spearman(linear_scores, gains)
            if max_rho is not None:
                max_rhos.append(max_rho)
            if linear_rho is not None:
                linear_rhos.append(linear_rho)

        fixture_metrics.append(
            {
                "fixture_id": fixture.fixture_id,
                "mean_max_target_gain": sum(max_fixture) / 5,
                "mean_linear_target_gain": sum(linear_fixture) / 5,
                "mean_random_target_gain": sum(random_fixture) / 5,
                "mean_oracle_target_gain": sum(oracle_fixture) / 5,
                "max_minus_linear_target_gain": (
                    (sum(max_fixture) - sum(linear_fixture)) / 5
                ),
            }
        )

    cells = len(scorer_gains["max"])
    means = {
        name: sum(values) / cells for name, values in scorer_gains.items()
    }
    max_linear = _pairwise(
        scorer_gains["max"],
        scorer_gains["linear"],
    )
    max_random = _pairwise(
        scorer_gains["max"],
        scorer_gains["random"],
    )
    positive_fixtures = sum(
        item["max_minus_linear_target_gain"] > 0
        for item in fixture_metrics
    )
    metrics = {
        "fixtures": len(fixtures),
        "root_world_cells": cells,
        "candidate_actions": cells * 4,
        "cells_with_target_gain_spread": target_spread,
        "cells_with_max_score_spread": max_score_spread,
        "defined_max_target_spearman": len(max_rhos),
        "defined_linear_target_spearman": len(linear_rhos),
        "mean_max_target_spearman": (
            sum(max_rhos) / len(max_rhos) if max_rhos else None
        ),
        "mean_linear_target_spearman": (
            sum(linear_rhos) / len(linear_rhos) if linear_rhos else None
        ),
        "mean_max_selected_target_gain": means["max"],
        "mean_linear_selected_target_gain": means["linear"],
        "mean_random_selected_target_gain": means["random"],
        "mean_oracle_target_gain": means["oracle"],
        "max_minus_linear_target_gain": means["max"] - means["linear"],
        "max_minus_random_target_gain": means["max"] - means["random"],
        "max_target_optimal_cells": max_optimal,
        "max_vs_linear_wins_ties_losses": max_linear,
        "max_vs_random_wins_ties_losses": max_random,
        "fixtures_with_positive_max_minus_linear": positive_fixtures,
    }
    max_rho_value = metrics["mean_max_target_spearman"]
    gates = {
        "exact_10_fixtures_50_cells_200_actions": (
            len(fixtures) == 10 and cells == 50 and cells * 4 == 200
        ),
        "at_least_30_cells_have_target_gain_spread": target_spread >= 30,
        "oracle_beats_random_by_at_least_0_20": (
            means["oracle"] >= means["random"] + 0.20
        ),
        "at_least_40_cells_have_max_score_spread": max_score_spread >= 40,
        "at_least_30_defined_max_target_correlations": len(max_rhos) >= 30,
        "mean_max_target_spearman_at_least_0_20": (
            max_rho_value is not None and max_rho_value >= 0.20
        ),
        "max_gain_at_least_0_15_above_linear": (
            means["max"] >= means["linear"] + 0.15
        ),
        "max_wins_exceed_losses_vs_linear": max_linear[0] > max_linear[2],
        "max_gain_at_least_0_15_above_random": (
            means["max"] >= means["random"] + 0.15
        ),
        "max_wins_exceed_losses_vs_random": max_random[0] > max_random[2],
        "at_least_7_of_10_fixtures_improve_over_linear": (
            positive_fixtures >= 7
        ),
        "at_least_20_of_50_max_choices_are_target_optimal": (
            max_optimal >= 20
        ),
    }
    return metrics, fixture_metrics, gates


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    fixtures = [_synthetic_fixture(index) for index in range(2)]
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        initial_raw = refresh._complete(
            models.generator,
            [base.initial_messages(fixture.seed_message) for fixture in fixtures],
            max_new_tokens=900,
        )
        refresh._checkpoint_stage(raw_path, raw, "initial", initial_raw)
        initials = [base.parse_initial(value) for value in initial_raw]

        simulator_raw = refresh._complete(
            models.simulator,
            [
                base.simulator_root_messages(
                    fixtures[index].simulator_system,
                    initials[index].roots[0],
                )
                for index in range(2)
            ],
            max_new_tokens=180,
        )
        refresh._checkpoint_stage(raw_path, raw, "root_answers", simulator_raw)
        root_answers = [
            base._clean_text(value, maximum=1_200) for value in simulator_raw
        ]

        need_requests = []
        for index in range(4):
            need_requests.append(
                needs.information_need_messages(
                    fixtures[index % 2].seed_message,
                    initials[index % 2],
                    index % base.ROOT_COUNT,
                    root_answers[index % 2],
                )
            )
        need_raw = refresh._complete(
            models.generator,
            need_requests,
            max_new_tokens=650,
        )
        refresh._checkpoint_stage(raw_path, raw, "need_beliefs", need_raw)
        parsed_needs = [
            needs.parse_information_need_belief(
                response,
                initials[index % 2],
                index % base.ROOT_COUNT,
            )
            for index, response in enumerate(need_raw)
        ]

        checklist_raw = refresh._complete(
            models.checklist_judge,
            [
                immediate_judgment_messages(
                    fixtures[index],
                    initials[index],
                    [root_answers[index]] * base.ROOT_COUNT,
                )
                for index in range(2)
            ],
            max_new_tokens=320,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "checklist_judgments",
            checklist_raw,
        )
        parsed_judgments = [
            base.parse_checklist_judgment(value) for value in checklist_raw
        ]
        usage = refresh.aggregate_usage(models)
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    gates = {
        "exact_10_physical_requests": (
            usage["physical_requests"] == EXPECTED_SERVING_REQUESTS
        ),
        "exact_10_http_attempts": (
            usage["http_attempts"] == EXPECTED_SERVING_REQUESTS
        ),
        "all_stage_parsers_pass": (
            len(initials) == 2
            and len(root_answers) == 2
            and len(parsed_needs) == 4
            and len(parsed_judgments) == 2
        ),
        "all_need_outputs_have_max_score_spread": all(
            len(
                {
                    max(profile)
                    for profile in belief.resolution_probabilities
                }
            )
            > 1
            for belief in parsed_needs
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_10": (
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
            "synthetic_only": True,
            "scientific_endpoint_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "models": {
                "root_and_need_generator": base.GENERATOR_MODEL_ID,
                "simulator": base.SIMULATOR_MODEL_ID,
                "checklist_judge": base.CHECKLIST_JUDGE_MODEL_ID,
            },
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
    if len(fixtures) != 10:
        raise ValueError("fresh confirmation requires ten fixtures")
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "confirmation_ids": list(CONFIRMATION_IDS),
        "development_split_sha256": DEVELOPMENT_SPLIT_SHA256,
    }
    try:
        record_fixtures = [fixtures[index] for index in range(0, 10, 2)]
        initial_raw = refresh._complete(
            models.generator,
            [
                base.initial_messages(fixture.seed_message)
                for fixture in record_fixtures
            ],
            max_new_tokens=900,
        )
        refresh._checkpoint_stage(raw_path, raw, "initial", initial_raw)
        initials = {
            fixture.record_id: base.parse_initial(response)
            for fixture, response in zip(record_fixtures, initial_raw)
        }

        root_requests = []
        for fixture in fixtures:
            initial = initials[fixture.record_id]
            for root in initial.roots:
                root_requests.append(
                    base.simulator_root_messages(
                        fixture.simulator_system,
                        root,
                    )
                )
        root_raw = refresh._complete(
            models.simulator,
            root_requests,
            max_new_tokens=220,
        )
        refresh._checkpoint_stage(raw_path, raw, "root_answers", root_raw)
        root_clean = [
            base._clean_text(value, maximum=1_200) for value in root_raw
        ]
        root_answers = [
            root_clean[index : index + base.ROOT_COUNT]
            for index in range(0, len(root_clean), base.ROOT_COUNT)
        ]

        need_requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index in range(base.ROOT_COUNT):
                need_requests.append(
                    needs.information_need_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        root_answers[fixture_index][root_index],
                    )
                )
        need_raw = refresh._complete(
            models.generator,
            need_requests,
            max_new_tokens=650,
        )
        refresh._checkpoint_stage(raw_path, raw, "need_beliefs", need_raw)
        beliefs_flat = []
        for response_index, response in enumerate(need_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            beliefs_flat.append(
                needs.parse_information_need_belief(
                    response,
                    initials[fixtures[fixture_index].record_id],
                    root_index,
                )
            )
        beliefs = [
            beliefs_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(beliefs_flat), base.ROOT_COUNT)
        ]

        checklist_raw = refresh._complete(
            models.checklist_judge,
            [
                immediate_judgment_messages(
                    fixture,
                    initials[fixture.record_id],
                    root_answers[fixture_index],
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
        metrics, fixture_metrics, scientific_gates = _score_metrics(
            fixtures,
            initials,
            beliefs,
            judgments,
        )
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    accounting_gates = {
        "exact_115_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_115_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_stage_counts_5_50_50_10": (
            len(initial_raw) == 5
            and len(root_raw) == 50
            and len(need_raw) == 50
            and len(checklist_raw) == 10
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_60": (
            usage["adapter_cost_usd"] <= MECHANICS_MAX_COST_USD
        ),
    }
    gates = {**accounting_gates, **scientific_gates}
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "stage": "mechanics",
            "confirmation_ids": list(CONFIRMATION_IDS),
            "development_split_sha256": DEVELOPMENT_SPLIT_SHA256,
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "stage_counts": {
                "initial_root_banks": 5,
                "hidden_world_root_answers": 50,
                "information_need_beliefs": 50,
                "checklist_judgments": 10,
            },
            "max_resolution_frozen_before_fresh_content": True,
            "paired_controls": ["linear_same_belief", "seeded_random"],
            "random_seed": RANDOM_SEED,
            "compiler_target_or_checklist_content": False,
            "checklist_labels_generated_after_all_compiler_checkpoints": True,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "development_confirmation": True,
            "holdout_accessed": False,
            "models": {
                "root_and_need_generator": base.GENERATOR_MODEL_ID,
                "simulator": base.SIMULATOR_MODEL_ID,
                "checklist_judge": base.CHECKLIST_JUDGE_MODEL_ID,
            },
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


def _dry_selected_actions() -> list[int]:
    judgment = base.ChecklistJudgment(
        tuple(
            tuple(int(bit) for bit in value)
            for value in DeterministicFreshChecklist.BITS
        ),
        tuple(
            tuple(int(bit) for bit in value)
            for value in DeterministicFreshChecklist.BITS
        ),
        tuple(
            tuple(int(bit) for bit in value)
            for value in DeterministicFreshChecklist.BITS
        ),
    )
    return needs._oracle_actions([judgment] * 10)


def _dry_models(*, serving: bool) -> ModelBundle:
    selected = (
        [0, 1, 2, 3]
        if serving
        else _dry_selected_actions()
    )
    return ModelBundle(
        generator=DeterministicFreshGenerator(
            "generator",
            selected_actions=selected,
        ),
        simulator=base.DeterministicFixtureModel("simulator"),
        checklist_judge=DeterministicFreshChecklist("checklist_judge"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("serving", "mechanics"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--settings", type=Path)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    verify_manifest(args.manifest)
    fixtures = None
    public_fixture = None
    if args.stage == "mechanics":
        if args.settings is None or args.baseline is None:
            parser.error("--settings and --baseline require mechanics")
        fixtures, public_fixture = build_fresh_fixtures(
            settings_path=args.settings,
            baseline_path=args.baseline,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.07 if args.stage == "serving" else 0.35
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else 50
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 950
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    models = (
        _dry_models(serving=args.stage == "serving")
        if args.dry_run
        else refresh._build_models(config)
    )

    try:
        if args.stage == "serving":
            result = run_serving_gate(models, raw_path=raw_path)
        else:
            assert fixtures is not None
            result = run_mechanics_gate(
                fixtures,
                models,
                raw_path=raw_path,
            )
        result["protocol"]["manifest_sha256"] = MANIFEST_PUBLIC_SHA256
        result["protocol"]["dry_run"] = args.dry_run
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        if public_fixture is not None:
            result["protocol"]["private_fixture_sha256"] = public_fixture[
                "private_fixture_sha256"
            ]
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "usage": (
                exc.usage
                if isinstance(exc, GateExecutionError)
                else refresh.aggregate_usage(models)
            ),
            "raw_exists": raw_path.exists(),
            "dry_run": args.dry_run,
        }
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        base._checkpoint(args.output_dir / "GATE_FAILURE.json", failure)
        raise

    output_name = "SERVING.json" if args.stage == "serving" else "MECHANICS.json"
    base._checkpoint(args.output_dir / output_name, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    if not result["gates"]["all_pass"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
