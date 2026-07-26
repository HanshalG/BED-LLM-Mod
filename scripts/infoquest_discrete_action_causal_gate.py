#!/usr/bin/env python3
"""Test refreshed versus fixed support on a shared InfoQuest action bank."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base


INTERFACE_VERSION = "infoquest-discrete-action-causal-1"
CHOICE_LABELS = ("A", "B", "C", "D")
EXPECTED_SERVING_REQUESTS = 7
EXPECTED_MECHANICS_REQUESTS = 159
SERVING_MAX_COST_USD = 0.10
MECHANICS_MAX_COST_USD = 0.85

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


def candidate_bank(
    initial: base.InitialPolicy,
    asked_root_index: int,
) -> tuple[str, ...]:
    if not 0 <= asked_root_index < base.ROOT_COUNT:
        raise ValueError("asked root index is out of range")
    candidates = tuple(
        root
        for index, root in enumerate(initial.roots)
        if index != asked_root_index
    )
    if len(candidates) != len(CHOICE_LABELS):
        raise ValueError("candidate action bank must contain four roots")
    return candidates


def _choice_request(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
) -> dict[str, Any]:
    candidates = candidate_bank(initial, asked_root_index)
    return {
        "ambiguous_seed_message": seed_message,
        "initial_hypotheses": list(initial.hypotheses),
        "clarification_question": initial.roots[asked_root_index],
        "user_answer": answer,
        "candidate_actions": dict(zip(CHOICE_LABELS, candidates)),
    }


def dynamic_choice_messages(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "STAGE=DYNAMIC_CHOICE. Regenerate eight distinct concrete "
                "latent contexts using the complete question and answer as "
                "binding evidence. New contexts may enter and old contexts "
                "may leave. Then choose exactly one supplied candidate action. "
                "Output only one JSON object with string fields h1..h8 and "
                "choice. choice must be exactly one of A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _choice_request(
                    seed_message,
                    initial,
                    asked_root_index,
                    answer,
                ),
                separators=(",", ":"),
            ),
        },
    ]


def fixed_choice_messages(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "STAGE=FIXED_CHOICE. Keep the supplied eight hypotheses "
                "exactly fixed. Copy h1..h8 verbatim without adding, removing, "
                "rewriting, or reordering any hypothesis. Then choose exactly "
                "one supplied candidate action. Output only one JSON object "
                "with string fields h1..h8 and choice. choice must be exactly "
                "one of A, B, C, or D."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                _choice_request(
                    seed_message,
                    initial,
                    asked_root_index,
                    answer,
                ),
                separators=(",", ":"),
            ),
        },
    ]


def parse_choice_policy(
    response: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    *,
    require_fixed_support: bool,
) -> base.RefreshPolicy:
    expected = {
        *(f"h{index}" for index in range(1, base.SUPPORT_SIZE + 1)),
        "choice",
    }
    value = base._parse_exact_object(response, expected)
    hypotheses = tuple(
        base._clean_text(value[f"h{index}"])
        for index in range(1, base.SUPPORT_SIZE + 1)
    )
    if len({base._normalize(item) for item in hypotheses}) != base.SUPPORT_SIZE:
        raise ValueError("choice hypotheses are not distinct")
    if require_fixed_support and hypotheses != initial.hypotheses:
        raise ValueError("fixed-choice response changed the initial support")
    choice = base._clean_text(value["choice"], maximum=100)
    if choice not in CHOICE_LABELS:
        raise ValueError("choice is not exactly one of A, B, C, or D")
    candidates = candidate_bank(initial, asked_root_index)
    return base.RefreshPolicy(
        hypotheses=hypotheses,
        followup=candidates[CHOICE_LABELS.index(choice)],
    )


def _choice_label(
    policy: base.RefreshPolicy,
    initial: base.InitialPolicy,
    asked_root_index: int,
) -> str:
    candidates = candidate_bank(initial, asked_root_index)
    try:
        return CHOICE_LABELS[candidates.index(policy.followup)]
    except ValueError as exc:  # pragma: no cover - parser guarantees this
        raise ValueError("policy follow-up is outside candidate bank") from exc


def _public_metrics(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    dynamic: Sequence[Sequence[base.RefreshPolicy]],
    fixed: Sequence[Sequence[base.RefreshPolicy]],
    dynamic_answers: Sequence[Sequence[str]],
    fixed_answers: Sequence[Sequence[str]],
    judgments: Sequence[base.ChecklistJudgment],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    metrics, fixture_metrics, gates = refresh._public_metrics(
        fixtures,
        initials,
        root_answers,
        dynamic,
        fixed,
        dynamic_answers,
        fixed_answers,
        judgments,
    )
    diverse_fixtures = 0
    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        dynamic_labels = [
            _choice_label(policy, initial, root_index)
            for root_index, policy in enumerate(dynamic[fixture_index])
        ]
        fixed_labels = [
            _choice_label(policy, initial, root_index)
            for root_index, policy in enumerate(fixed[fixture_index])
        ]
        if len(set(dynamic_labels)) >= 2:
            diverse_fixtures += 1
        fixture_metrics[fixture_index]["dynamic_choice_labels"] = dynamic_labels
        fixture_metrics[fixture_index]["fixed_choice_labels"] = fixed_labels
    metrics["fixtures_with_at_least_2_dynamic_choice_labels"] = (
        diverse_fixtures
    )
    metrics["dynamic_choice_differs_from_fixed_cells"] = metrics[
        "dynamic_followup_differs_from_fixed_cells"
    ]
    gates["at_least_4_fixtures_have_dynamic_choice_diversity"] = (
        diverse_fixtures >= 4
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
                dynamic_choice_messages(
                    seed_message,
                    initial,
                    0,
                    root_answer,
                )
            ],
            max_new_tokens=1_100,
        )
        refresh._checkpoint_stage(raw_path, raw, "dynamic_choice", dynamic_raw)
        dynamic = parse_choice_policy(
            dynamic_raw[0],
            initial,
            0,
            require_fixed_support=False,
        )

        fixed_raw = refresh._complete(
            models.generator,
            [
                fixed_choice_messages(
                    seed_message,
                    initial,
                    0,
                    root_answer,
                )
            ],
            max_new_tokens=1_100,
        )
        refresh._checkpoint_stage(raw_path, raw, "fixed_choice", fixed_raw)
        fixed = parse_choice_policy(
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
                    dynamic.followup,
                ),
                base.simulator_followup_messages(
                    simulator_system,
                    initial.roots[0],
                    root_answer,
                    fixed.followup,
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
                    [dynamic] * base.ROOT_COUNT,
                    [followup_answers[0]] * base.ROOT_COUNT,
                    [fixed.followup] * base.ROOT_COUNT,
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
        "cost_at_most_0_10": (
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
            "choice_labels": list(CHOICE_LABELS),
            "models": {
                "generator_dynamic_fixed": refresh.GENERATOR_MODEL_ID,
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
                    dynamic_choice_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        answer,
                    )
                )
                fixed_requests.append(
                    fixed_choice_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        answer,
                    )
                )

        dynamic_raw = refresh._complete(
            models.generator,
            dynamic_requests,
            max_new_tokens=1_200,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "dynamic_choices",
            dynamic_raw,
        )
        dynamic_flat = []
        for response_index, response in enumerate(dynamic_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            dynamic_flat.append(
                parse_choice_policy(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=False,
                )
            )
        dynamic = [
            dynamic_flat[index : index + base.ROOT_COUNT]
            for index in range(0, len(dynamic_flat), base.ROOT_COUNT)
        ]

        fixed_raw = refresh._complete(
            models.generator,
            fixed_requests,
            max_new_tokens=1_200,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "fixed_choices",
            fixed_raw,
        )
        fixed_flat = []
        for response_index, response in enumerate(fixed_raw):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            initial = initials[fixtures[fixture_index].record_id]
            fixed_flat.append(
                parse_choice_policy(
                    response,
                    initial,
                    root_index,
                    require_fixed_support=True,
                )
            )
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
                        dynamic[fixture_index][root_index].followup,
                    )
                )
                followup_requests.append(
                    base.simulator_followup_messages(
                        *common,
                        fixed[fixture_index][root_index].followup,
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
                    dynamic[fixture_index],
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
            dynamic,
            fixed,
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
            "choice_labels": list(CHOICE_LABELS),
            "candidate_actions_per_cell": len(CHOICE_LABELS),
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "models": {
                "generator_dynamic_fixed": refresh.GENERATOR_MODEL_ID,
                "simulator": refresh.SIMULATOR_MODEL_ID,
                "checklist_judge": refresh.CHECKLIST_JUDGE_MODEL_ID,
            },
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "compute_matched_support_outputs": True,
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
        if all(stage in {"DYNAMIC_CHOICE", "FIXED_CHOICE"} for stage in stages):
            responses = []
            for stage, messages in zip(stages, batch_messages):
                request = json.loads(messages[-1]["content"])
                if stage == "FIXED_CHOICE":
                    hypotheses = request["initial_hypotheses"]
                    choice = "D"
                else:
                    seed = request["ambiguous_seed_message"]
                    root = request["clarification_question"]
                    hypotheses = [
                        (
                            f"{seed} Refreshed context {index} after {root} "
                            f"has goal {index} and constraint {index}."
                        )
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    ]
                    match = re.search(r"detail ([1-5])", root)
                    root_number = int(match.group(1)) if match else 1
                    choice = CHOICE_LABELS[(root_number - 1) % 4]
                value = {
                    **{
                        f"h{index}": hypotheses[index - 1]
                        for index in range(1, base.SUPPORT_SIZE + 1)
                    },
                    "choice": choice,
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
        0.05 if args.stage == "serving" else 0.58
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
