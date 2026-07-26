#!/usr/bin/env python3
"""Evaluate a target-relevant LLM information-need belief on InfoQuest."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import load_config
from scripts import infoquest_cached_answer_ranking_diagnostic as diagnostic
from scripts import infoquest_cached_partition_eig_gate as cached
from scripts import infoquest_discrete_action_causal_gate as discrete
from scripts import infoquest_partition_eig_causal_gate as partition
from scripts import infoquest_refresh_continuation_gate as refresh
from scripts import infoquest_support_causal_link_gate as base
from scripts import infoquest_target_alignment_audit as alignment


INTERFACE_VERSION = "infoquest-information-need-belief-2"
TARGET_AUDIT_SHA256 = (
    "d550cdfe9770b355c59ef999059175b79a5c7e01a8b0c3f82a96d62a98d916cf"
)
NEED_COUNT = 5
EXPECTED_SERVING_REQUESTS = 10
EXPECTED_MECHANICS_REQUESTS = 30
SERVING_MAX_COST_USD = 0.08
MECHANICS_MAX_COST_USD = 0.35

ModelBundle = refresh.ModelBundle
GateExecutionError = refresh.GateExecutionError


@dataclass(frozen=True)
class InformationNeedBelief:
    needs: tuple[str, ...]
    weights: tuple[int, ...]
    resolution_probabilities: tuple[tuple[int, ...], ...]
    scores: tuple[float, ...]
    selected_action_index: int
    selected_question: str


def expected_resolved_mass(
    weights: Sequence[int],
    resolution_probabilities: Sequence[int],
) -> float:
    if len(weights) != NEED_COUNT or len(resolution_probabilities) != NEED_COUNT:
        raise ValueError("information-need score requires five values")
    total = sum(weights)
    if total <= 0:
        raise ValueError("information-need weights have zero total")
    return sum(
        weight * probability
        for weight, probability in zip(weights, resolution_probabilities)
    ) / (100.0 * total)


def information_need_messages(
    seed_message: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
    answer: str,
) -> list[dict[str, str]]:
    candidates = discrete.candidate_bank(
        initial,
        asked_root_index,
    )
    request = {
        "ambiguous_request": seed_message,
        "observed_question": initial.roots[asked_root_index],
        "observed_answer": answer,
        "candidate_questions": {
            key.upper(): question
            for key, question in zip(partition.ACTION_KEYS, candidates)
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "STAGE=INFORMATION_NEED_BELIEF. Infer exactly five distinct, "
                "atomic decision-critical unknown values that remain unresolved "
                "and whose resolution could materially change the quality of "
                "the final response to the user's request. Each need must name "
                "one missing value, not a broad topic, scenario, goal, answer, "
                "or question. Do not infer the hidden truth. Assign each need "
                "an integer importance weight from 1 to 100. For each candidate "
                "A-D, predict as an integer from 0 to 100 the probability that "
                "receiving its answer would resolve each need. Do not choose a "
                "candidate and do not mention any evaluation checklist. Output "
                "only one JSON object with exactly six fields: n is an array of "
                "five need strings, w is an array of five weights, and a, b, c, "
                "and d are arrays of five resolution probabilities."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(request, separators=(",", ":")),
        },
    ]


def _bounded_integer(
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


def parse_information_need_belief(
    response: str,
    initial: base.InitialPolicy,
    asked_root_index: int,
) -> InformationNeedBelief:
    keys = {"n", "w", *partition.ACTION_KEYS}
    value = base._parse_exact_object(response, keys)
    for key in keys:
        if not isinstance(value[key], list) or len(value[key]) != NEED_COUNT:
            raise ValueError(f"{key} must contain exactly five values")
    needs = tuple(base._clean_text(item) for item in value["n"])
    if len({base._normalize(item) for item in needs}) != NEED_COUNT:
        raise ValueError("information needs are not distinct")
    weights = tuple(
        _bounded_integer(
            item,
            minimum=1,
            maximum=100,
            name=f"w{index}",
        )
        for index, item in enumerate(value["w"], start=1)
    )
    profiles = tuple(
        tuple(
            _bounded_integer(
                item,
                minimum=0,
                maximum=100,
                name=f"{action}{index}",
            )
            for index, item in enumerate(value[action], start=1)
        )
        for action in partition.ACTION_KEYS
    )
    scores = tuple(
        expected_resolved_mass(weights, profile) for profile in profiles
    )
    selected = max(range(len(scores)), key=scores.__getitem__)
    candidates = discrete.candidate_bank(
        initial,
        asked_root_index,
    )
    return InformationNeedBelief(
        needs=needs,
        weights=weights,
        resolution_probabilities=profiles,
        scores=scores,
        selected_action_index=selected,
        selected_question=candidates[selected],
    )


class DeterministicNeedGenerator(cached.CachedDeterministicGenerator):
    def __init__(
        self,
        name: str,
        *,
        selected_actions: Sequence[int] | None = None,
    ) -> None:
        super().__init__(name)
        self.selected_actions = list(selected_actions or [])
        self.selection_cursor = 0

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
            for messages in batch_messages:
                request = json.loads(messages[-1]["content"])
                if self.selection_cursor < len(self.selected_actions):
                    selected = self.selected_actions[self.selection_cursor]
                else:
                    digest = hashlib.sha256(
                        request["observed_answer"].encode()
                    ).digest()
                    selected = digest[0] % 4
                self.selection_cursor += 1
                profiles = []
                for action_index in range(4):
                    value = 90 if action_index == selected else 10 + action_index
                    profiles.append([value] * NEED_COUNT)
                response = {
                    "n": [
                        f"Missing decision-critical value {index}"
                        for index in range(1, NEED_COUNT + 1)
                    ],
                    "w": [100, 80, 60, 40, 20],
                    **{
                        action: profiles[action_index]
                        for action_index, action in enumerate(
                            partition.ACTION_KEYS
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


def _reshape(values: Sequence[Any]) -> list[list[Any]]:
    if len(values) != 6 * base.ROOT_COUNT:
        raise ValueError("information-need response count is not thirty")
    return [
        list(values[index : index + base.ROOT_COUNT])
        for index in range(0, len(values), base.ROOT_COUNT)
    ]


def _selected_root_index(
    initial: base.InitialPolicy,
    question: str,
) -> int:
    normalized = base._normalize(question)
    matches = [
        index
        for index, root in enumerate(initial.roots)
        if base._normalize(root) == normalized
    ]
    if len(matches) != 1:
        raise ValueError("selected question does not map to one root")
    return matches[0]


def _target_metrics(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    beliefs: Sequence[Sequence[InformationNeedBelief]],
    dynamic_baseline: Sequence[Sequence[Any]],
    fixed_baseline: Sequence[Sequence[Any]],
    judgments: Sequence[base.ChecklistJudgment],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, bool]]:
    selected_gains: list[int] = []
    dynamic_gains: list[int] = []
    fixed_gains: list[int] = []
    oracle_gains: list[int] = []
    regrets: list[int] = []
    rhos: list[float] = []
    score_spread = 0
    optimal = 0
    wins = ties = losses = 0
    fixture_metrics = []

    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        judgment = judgments[fixture_index]
        fixture_need: list[int] = []
        fixture_fixed: list[int] = []
        for root_index in range(base.ROOT_COUNT):
            candidate_indices = [
                index
                for index in range(base.ROOT_COUNT)
                if index != root_index
            ]
            gains = [
                alignment.additive_gain(
                    judgment.immediate[root_index],
                    judgment.immediate[candidate_index],
                )
                for candidate_index in candidate_indices
            ]
            oracle = max(gains)
            belief = beliefs[fixture_index][root_index]
            need_gain = gains[belief.selected_action_index]
            dynamic_root = _selected_root_index(
                initial,
                dynamic_baseline[fixture_index][root_index].selected_question,
            )
            fixed_root = _selected_root_index(
                initial,
                fixed_baseline[fixture_index][root_index].selected_question,
            )
            dynamic_gain = gains[candidate_indices.index(dynamic_root)]
            fixed_gain = gains[candidate_indices.index(fixed_root)]

            selected_gains.append(need_gain)
            dynamic_gains.append(dynamic_gain)
            fixed_gains.append(fixed_gain)
            oracle_gains.append(oracle)
            regrets.append(oracle - need_gain)
            optimal += need_gain == oracle
            score_spread += max(belief.scores) > min(belief.scores)
            rho = alignment.spearman(belief.scores, gains)
            if rho is not None:
                rhos.append(rho)
            if need_gain > fixed_gain:
                wins += 1
            elif need_gain == fixed_gain:
                ties += 1
            else:
                losses += 1
            fixture_need.append(need_gain)
            fixture_fixed.append(fixed_gain)

        need_mean = sum(fixture_need) / base.ROOT_COUNT
        fixed_mean = sum(fixture_fixed) / base.ROOT_COUNT
        fixture_metrics.append(
            {
                "fixture_id": fixture.fixture_id,
                "mean_information_need_target_gain": need_mean,
                "mean_fixed_target_gain": fixed_mean,
                "information_need_minus_fixed_target_gain": (
                    need_mean - fixed_mean
                ),
            }
        )

    cells = len(selected_gains)
    mean_gain = sum(selected_gains) / cells
    mean_dynamic = sum(dynamic_gains) / cells
    mean_fixed = sum(fixed_gains) / cells
    mean_oracle = sum(oracle_gains) / cells
    mean_rho = sum(rhos) / len(rhos) if rhos else None
    positive_fixtures = sum(
        item["information_need_minus_fixed_target_gain"] > 0
        for item in fixture_metrics
    )
    metrics = {
        "fixtures": len(fixtures),
        "root_world_cells": cells,
        "candidate_actions": cells * 4,
        "cells_with_information_need_score_spread": score_spread,
        "defined_within_cell_target_spearman": len(rhos),
        "mean_information_need_target_spearman": mean_rho,
        "mean_information_need_selected_target_gain": mean_gain,
        "mean_v3_dynamic_selected_target_gain": mean_dynamic,
        "mean_v3_fixed_selected_target_gain": mean_fixed,
        "mean_oracle_target_gain": mean_oracle,
        "mean_information_need_oracle_regret": sum(regrets) / cells,
        "information_need_target_optimal_cells": optimal,
        "information_need_vs_fixed_wins_ties_losses": [wins, ties, losses],
        "fixtures_with_positive_information_need_minus_fixed": (
            positive_fixtures
        ),
    }
    gates = {
        "exact_6_fixtures_30_cells_120_actions": (
            len(fixtures) == 6 and cells == 30 and cells * 4 == 120
        ),
        "at_least_24_cells_have_score_spread": score_spread >= 24,
        "at_least_15_defined_target_correlations": len(rhos) >= 15,
        "mean_target_spearman_at_least_0_20": (
            mean_rho is not None and mean_rho >= 0.20
        ),
        "mean_target_gain_at_least_0_15_above_fixed": (
            mean_gain >= mean_fixed + 0.15
        ),
        "information_need_wins_exceed_losses": wins > losses,
        "at_least_4_of_6_fixtures_improve_over_fixed": positive_fixtures >= 4,
        "at_least_12_of_30_choices_are_target_optimal": optimal >= 12,
    }
    return metrics, fixture_metrics, gates


def run_serving_gate(
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    initials = [
        base.InitialPolicy(
            hypotheses=tuple(f"Unused context {index}" for index in range(8)),
            roots=tuple(
                f"What is synthetic detail {index}?" for index in range(5)
            ),
        )
        for _ in range(EXPECTED_SERVING_REQUESTS)
    ]
    raw: dict[str, Any] = {"interface_version": INTERFACE_VERSION}
    try:
        responses = refresh._complete(
            models.generator,
            [
                information_need_messages(
                    f"Synthetic ambiguous request {index}.",
                    initials[index],
                    index % base.ROOT_COUNT,
                    f"Synthetic answer {index}.",
                )
                for index in range(EXPECTED_SERVING_REQUESTS)
            ],
            max_new_tokens=650,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "information_need_beliefs",
            responses,
        )
        parsed = [
            parse_information_need_belief(
                response,
                initials[index],
                index % base.ROOT_COUNT,
            )
            for index, response in enumerate(responses)
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
        "all_10_outputs_parse": len(parsed) == EXPECTED_SERVING_REQUESTS,
        "all_10_outputs_have_score_spread": all(
            max(item.scores) > min(item.scores) for item in parsed
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_08": (
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
            "information_needs": NEED_COUNT,
            "target_or_checklist_content_in_prompt": False,
            "scientific_endpoint_evaluated": False,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "model": base.GENERATOR_MODEL_ID,
        },
        "gates": gates,
        "usage": usage,
    }


def run_mechanics_gate(
    fixtures: Sequence[base.WorldFixture],
    initials: dict[int, base.InitialPolicy],
    root_answers: Sequence[Sequence[str]],
    dynamic_baseline: Sequence[Sequence[Any]],
    fixed_baseline: Sequence[Sequence[Any]],
    judgments: Sequence[base.ChecklistJudgment],
    fixture_ids: Sequence[str],
    models: ModelBundle,
    *,
    raw_path: Path,
) -> dict[str, Any]:
    if [fixture.fixture_id for fixture in fixtures] != list(fixture_ids):
        raise ValueError("fixture order differs across target artifacts")
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "source_v3_public_sha256": diagnostic.V3_PUBLIC_SHA256,
        "source_v3_raw_sha256": diagnostic.V3_RAW_SHA256,
        "source_target_audit_sha256": TARGET_AUDIT_SHA256,
    }
    try:
        requests = []
        for fixture_index, fixture in enumerate(fixtures):
            initial = initials[fixture.record_id]
            for root_index in range(base.ROOT_COUNT):
                requests.append(
                    information_need_messages(
                        fixture.seed_message,
                        initial,
                        root_index,
                        root_answers[fixture_index][root_index],
                    )
                )
        responses = refresh._complete(
            models.generator,
            requests,
            max_new_tokens=650,
        )
        refresh._checkpoint_stage(
            raw_path,
            raw,
            "information_need_beliefs",
            responses,
        )
        parsed = []
        for response_index, response in enumerate(responses):
            fixture_index = response_index // base.ROOT_COUNT
            root_index = response_index % base.ROOT_COUNT
            parsed.append(
                parse_information_need_belief(
                    response,
                    initials[fixtures[fixture_index].record_id],
                    root_index,
                )
            )
        beliefs = _reshape(parsed)
        usage = refresh.aggregate_usage(models)
        metrics, fixture_metrics, scientific_gates = _target_metrics(
            fixtures,
            initials,
            beliefs,
            dynamic_baseline,
            fixed_baseline,
            judgments,
        )
    except Exception as exc:
        base._checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            refresh.aggregate_usage(models),
        ) from exc

    accounting_gates = {
        "exact_30_physical_requests": (
            usage["physical_requests"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "exact_30_http_attempts": (
            usage["http_attempts"] == EXPECTED_MECHANICS_REQUESTS
        ),
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "cost_at_most_0_35": (
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
            "expected_requests": EXPECTED_MECHANICS_REQUESTS,
            "cached_common_histories": True,
            "source_v3_public_sha256": diagnostic.V3_PUBLIC_SHA256,
            "source_v3_raw_sha256": diagnostic.V3_RAW_SHA256,
            "source_target_audit_sha256": TARGET_AUDIT_SHA256,
            "target_gain_proxy": (
                "new immediate checklist bits from candidate root answer"
            ),
            "information_need_belief_is_regenerated_after_history": True,
            "target_or_checklist_content_in_prompt": False,
            "simulator_calls": 0,
            "checklist_judge_calls": 0,
            "reasoning_requested": False,
            "repairs_or_reissues": 0,
            "development_only_if_passed": True,
            "model": base.GENERATOR_MODEL_ID,
        },
        "metrics": metrics,
        "fixture_metrics": fixture_metrics,
        "gates": gates,
        "usage": usage,
    }


def _oracle_actions(
    judgments: Sequence[base.ChecklistJudgment],
) -> list[int]:
    actions = []
    for judgment in judgments:
        for root_index in range(base.ROOT_COUNT):
            candidates = [
                index
                for index in range(base.ROOT_COUNT)
                if index != root_index
            ]
            gains = [
                alignment.additive_gain(
                    judgment.immediate[root_index],
                    judgment.immediate[candidate],
                )
                for candidate in candidates
            ]
            actions.append(max(range(4), key=gains.__getitem__))
    return actions


def _verify_target_audit(path: Path) -> None:
    if hashlib.sha256(path.read_bytes()).hexdigest() != TARGET_AUDIT_SHA256:
        raise ValueError("target-alignment audit SHA-256 mismatch")
    audit = json.loads(path.read_text())
    gates = audit.get("gates", {})
    required = (
        "exact_6_fixtures_30_cells_120_actions",
        "selected_path_additive_agreement_at_least_0_90",
        "at_least_15_cells_have_target_gain_spread",
        "oracle_has_at_least_0_15_gain_headroom_over_fixed",
        "zero_llm_calls",
    )
    if not all(gates.get(key) is True for key in required):
        raise ValueError("target-alignment opportunity gates are not bound")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("serving", "mechanics"), required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--settings", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--cached-raw", type=Path)
    parser.add_argument("--cached-public", type=Path)
    parser.add_argument("--v3-raw", type=Path)
    parser.add_argument("--v3-public", type=Path)
    parser.add_argument("--diagnostic-raw", type=Path)
    parser.add_argument("--diagnostic-public", type=Path)
    parser.add_argument("--target-audit", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    fixtures, public_fixture = base.build_fixtures(
        settings_path=args.settings,
        baseline_path=args.baseline,
    )
    loaded = None
    if args.stage == "mechanics":
        required_paths = (
            args.cached_raw,
            args.cached_public,
            args.v3_raw,
            args.v3_public,
            args.diagnostic_raw,
            args.diagnostic_public,
            args.target_audit,
        )
        if any(path is None for path in required_paths):
            parser.error("all cached target artifacts are required for mechanics")
        assert all(path is not None for path in required_paths)
        _verify_target_audit(args.target_audit)
        initials, root_answers = cached.load_cached_histories(
            fixtures,
            raw_path=args.cached_raw,
            public_path=args.cached_public,
        )
        dynamic, fixed = diagnostic.load_v3_beliefs(
            fixtures,
            initials,
            raw_path=args.v3_raw,
            public_path=args.v3_public,
        )
        judgments, fixture_ids = alignment.load_diagnostic_judgments(
            raw_path=args.diagnostic_raw,
            public_path=args.diagnostic_public,
        )
        loaded = (
            initials,
            root_answers,
            dynamic,
            fixed,
            judgments,
            fixture_ids,
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = (
        0.06 if args.stage == "serving" else 0.20
    )
    config.openrouter_run_budget_usd = (
        SERVING_MAX_COST_USD
        if args.stage == "serving"
        else MECHANICS_MAX_COST_USD
    )
    config.openrouter_concurrency = (
        EXPECTED_SERVING_REQUESTS
        if args.stage == "serving"
        else EXPECTED_MECHANICS_REQUESTS
    )
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 700
    config.log_path = args.output_dir / "run.log"
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"

    if args.dry_run:
        selected_actions = (
            _oracle_actions(loaded[4])
            if args.stage == "mechanics" and loaded is not None
            else [index % 4 for index in range(EXPECTED_SERVING_REQUESTS)]
        )
        models = ModelBundle(
            generator=DeterministicNeedGenerator(
                "generator",
                selected_actions=selected_actions,
            ),
            simulator=base.DeterministicFixtureModel("simulator"),
            checklist_judge=base.DeterministicFixtureModel("checklist_judge"),
        )
    else:
        models = refresh._build_models(config)

    try:
        if args.stage == "serving":
            result = run_serving_gate(models, raw_path=raw_path)
        else:
            assert loaded is not None
            result = run_mechanics_gate(
                fixtures,
                *loaded,
                models,
                raw_path=raw_path,
            )
        result["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
        result["protocol"]["fixture_sha256"] = public_fixture[
            "private_fixture_sha256"
        ]
        result["protocol"]["dry_run"] = args.dry_run
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
