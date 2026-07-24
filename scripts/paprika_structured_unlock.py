#!/usr/bin/env python3
"""Test whether diagnostic observations unlock Paprika cause-remedy hypotheses."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core import BeliefState
from environments.paprika_customer_service.data import load_paprika_tasks
from environments.paprika_customer_service.env import (
    PaprikaCustomerServiceEnvironment,
    _dedupe,
    _dedupe_indices,
)
from environments.paprika_customer_service.parsing import (
    parse_json_object,
    parse_string_list,
)
from environments.paprika_customer_service.prompts import (
    customer_messages,
    faithfulness_messages,
    faithfulness_repair_messages,
    filtering_messages,
    hypothesis_messages,
    refinement_messages,
)
from environments.paprika_customer_service.types import (
    PaprikaAction,
    PaprikaObservation,
)
from helpers import Config, load_config
from model_factory import build_model_adapter


DEVELOPMENT_INDICES = (92, 157, 199, 250, 398, 514, 556, 574, 584, 597, 602, 618)
SEMANTIC_COVERAGE_THRESHOLD = 0.80


class CoverageBatchError(RuntimeError):
    def __init__(self, message: str, *, row: int, response: str) -> None:
        super().__init__(message)
        self.row = row
        self.response = response


def diagnostic_candidate_messages(
    scenario: str,
    hypotheses: Sequence[str],
    count: int,
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Generate atomic diagnostic questions for customer-service "
                "troubleshooting. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Scenario: {scenario}\n"
                "Current possible cause-and-remedy hypotheses:\n- "
                + "\n- ".join(hypotheses)
                + f"\n\nReturn exactly {count} distinct diagnostic questions as "
                '{"candidates":[{"query":"...","kind":"diagnostic",'
                '"outcomes":["...","...","Not checked / cannot determine"]}]}. '
                "Each question must ask for one customer-observable fact, test, "
                "or inspection result that distinguishes the hypotheses. Do not "
                "suggest, perform, identify, or name a remedy. Do not ask compound "
                "questions. Give 3-5 mutually exclusive outcomes that directly "
                "answer the question, including an uncertainty outcome."
            ),
        },
    ]


def semantic_coverage_messages(
    scenario: str,
    private_solution: str,
    supports: Sequence[tuple[str, Sequence[str]]],
) -> list[dict[str, str]]:
    payload = {
        "scenario": scenario,
        "private_solution": private_solution,
        "supports": [
            {
                "id": support_id,
                "hypotheses": list(hypotheses),
            }
            for support_id, hypotheses in supports
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "You are a strict semantic coverage evaluator. The unknown solution "
                "was hidden from every hypothesis generator. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each support, score the single best hypothesis against the "
                "private solution. A score of 1 requires the same underlying cause "
                "and a remedy that would fix that cause. A merely plausible "
                "alternative fix, shared symptom, generic troubleshooting step, "
                "or cause without a compatible remedy must score below 0.8. "
                "Return exactly "
                '{"supports":[{"id":"...","best_match_score":0.0,'
                '"best_hypothesis_index":0,"reason":"brief"}]}. '
                "Use null for best_hypothesis_index only when the support is empty. "
                "Preserve support IDs and order.\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_coverage_response(
    text: str,
    support_ids: Sequence[str],
    support_sizes: Sequence[int],
) -> list[dict[str, Any]]:
    raw = parse_json_object(text).get("supports")
    if not isinstance(raw, list) or len(raw) != len(support_ids):
        raise ValueError("coverage response must contain one row per support")
    parsed: list[dict[str, Any]] = []
    for expected_id, support_size, row in zip(
        support_ids, support_sizes, raw, strict=True
    ):
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("coverage response changed support IDs or order")
        score = row.get("best_match_score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("best_match_score must be numeric")
        score = float(score)
        if not 0.0 <= score <= 1.0:
            raise ValueError("best_match_score must be in [0, 1]")
        index = row.get("best_hypothesis_index")
        index_valid = True
        reported_index = index
        if support_size == 0:
            if index is not None:
                index_valid = False
                index = None
        elif (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < support_size
        ):
            index_valid = False
            index = None
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("coverage response requires a brief reason")
        parsed.append(
            {
                "id": expected_id,
                "best_match_score": score,
                "best_hypothesis_index": index,
                "reported_best_hypothesis_index": reported_index,
                "best_hypothesis_index_valid": index_valid,
                "reason": reason.strip(),
                "covered": score >= SEMANTIC_COVERAGE_THRESHOLD,
            }
        )
    return parsed


def _parse_consistency(text: str) -> bool:
    value = parse_json_object(text).get("consistent")
    if not isinstance(value, bool):
        raise ValueError("faithfulness response requires boolean consistent")
    return value


def _parse_diagnostic_candidates(
    env: PaprikaCustomerServiceEnvironment,
    text: str,
    scenario: str,
    count: int,
) -> list[PaprikaAction]:
    actions = env._parse_candidates(text, scenario, [], count)
    if any(action.kind != "diagnostic" for action in actions):
        raise ValueError("unlock probe accepts diagnostic questions only")
    forbidden = (
        "replace",
        "reset",
        "restart",
        "repair",
        "reinstall",
        "update",
        "clean",
        "adjust",
        "tighten",
        "refill",
        "clear",
    )
    for action in actions:
        words = set(action.query.casefold().replace("/", " ").split())
        if words.intersection(forbidden):
            raise ValueError("diagnostic candidate contains a remedy verb")
    return actions


def _generate_candidates_many(
    env: PaprikaCustomerServiceEnvironment,
    questioner: Any,
    states: Sequence[BeliefState[str]],
    scenarios: Sequence[str],
    count: int,
    config: Config,
) -> list[list[PaprikaAction]]:
    messages = [
        diagnostic_candidate_messages(scenario, state.hypotheses, count)
        for scenario, state in zip(scenarios, states, strict=True)
    ]
    temperature = float(config.generation_temperature_diverse)
    responses = env._cached_complete_many(
        questioner,
        messages,
        temperature,
        namespace="unlock:diagnostic_candidates",
    )
    return env._parse_many_with_retries(
        questioner,
        messages,
        responses,
        temperature,
        namespace="unlock:diagnostic_candidates",
        parsers=[
            (
                lambda text, scenario=scenario: _parse_diagnostic_candidates(
                    env, text, scenario, count
                )
            )
            for scenario in scenarios
        ],
    )


def _observe_many(
    env: PaprikaCustomerServiceEnvironment,
    evaluator: Any,
    actions: Sequence[PaprikaAction],
    solutions: Sequence[str],
) -> list[PaprikaObservation]:
    temperature = float(env.config.answer_temperature)
    reply_messages = [
        customer_messages(action, solution)
        for action, solution in zip(actions, solutions, strict=True)
    ]
    replies = env._cached_complete_many(
        env.answerer,
        reply_messages,
        temperature,
        namespace="unlock:customer",
    )
    env._simulator_faithfulness_observations += len(actions)
    active = list(range(len(actions)))
    maximum = int(env.config.paprika_structured_max_retries)
    for attempt in range(maximum + 1):
        check_messages = [
            faithfulness_messages(actions[index], solutions[index], replies[index])
            for index in active
        ]
        checks = env._cached_complete_many(
            evaluator,
            check_messages,
            0.0,
            namespace=f"unlock:faithfulness:{attempt}",
        )
        env._simulator_faithfulness_checks += len(active)
        failed = [
            index
            for index, response in zip(active, checks, strict=True)
            if not _parse_consistency(response)
        ]
        if not failed:
            break
        if attempt == 0:
            env._simulator_faithfulness_raw_contradictions += len(failed)
        if attempt >= maximum:
            env._simulator_faithfulness_failures += len(failed)
            raise ValueError(
                f"customer simulator failed faithfulness at rows {failed}"
            )
        repair_messages = [
            faithfulness_repair_messages(reply_messages[index], replies[index])
            for index in failed
        ]
        repaired = env._cached_complete_many(
            env.answerer,
            repair_messages,
            temperature,
            namespace=f"unlock:customer_faithfulness_retry:{attempt + 1}",
        )
        env._simulator_faithfulness_repairs += len(failed)
        for index, response, messages in zip(
            failed, repaired, repair_messages, strict=True
        ):
            replies[index] = response
            reply_messages[index] = messages
        active = failed
    if any("goal reached" in reply.casefold() for reply in replies):
        raise ValueError("diagnostic-only probe unexpectedly reached a terminal remedy")
    observations = []
    for reply in replies:
        observations.append(
            PaprikaObservation(
                reply=reply.strip(),
                mapped_outcome="diagnostic observation",
                mapped_cleanly=True,
                goal_reached=False,
            )
        )
    return observations


def _refresh_supports_many(
    env: PaprikaCustomerServiceEnvironment,
    questioner: Any,
    states: Sequence[BeliefState[str]],
    histories: Sequence[Sequence[tuple[PaprikaAction, PaprikaObservation]]],
    count: int,
    config: Config,
) -> list[list[str]]:
    scenarios = [history[0][0].scenario for history in histories]
    generation_messages = [
        refinement_messages(
            scenario,
            state.hypotheses,
            history,
            count,
        )
        for scenario, state, history in zip(
            scenarios, states, histories, strict=True
        )
    ]
    temperature = float(config.generation_temperature_diverse)
    generated_responses = env._cached_complete_many(
        questioner,
        generation_messages,
        temperature,
        namespace="unlock:hypothesis_refinement",
    )
    refined = env._parse_many_with_retries(
        questioner,
        generation_messages,
        generated_responses,
        temperature,
        namespace="unlock:hypothesis_refinement",
        parsers=[
            (
                lambda text: parse_string_list(
                    text,
                    "refined_hypotheses",
                    minimum=count,
                    maximum=count,
                )
            )
            for _history in histories
        ],
    )
    merged = [
        _dedupe([*state.hypotheses, *new_items])
        for state, new_items in zip(states, refined, strict=True)
    ]
    filter_prompts = [
        filtering_messages(scenario, hypotheses, history)
        for scenario, hypotheses, history in zip(
            scenarios, merged, histories, strict=True
        )
    ]
    filter_responses = env._cached_complete_many(
        questioner,
        filter_prompts,
        float(config.generation_temperature_simple),
        namespace="unlock:hypothesis_filter",
    )
    filters = env._parse_many_with_retries(
        questioner,
        filter_prompts,
        filter_responses,
        float(config.generation_temperature_simple),
        namespace="unlock:hypothesis_filter",
        parsers=[parse_json_object for _history in histories],
    )
    supports = []
    for hypotheses, payload in zip(merged, filters, strict=True):
        indices = payload.get("keep_indices")
        if not isinstance(indices, list):
            raise ValueError("hypothesis filter must return keep_indices")
        kept = _dedupe_indices(indices, len(hypotheses))
        if not kept:
            raise ValueError("hypothesis filter rejected every candidate")
        supports.append([hypotheses[index] for index in kept])
    return supports


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    initial_omissions = sum(not record["initial"]["covered"] for record in records)
    recovered = sum(
        not record["initial"]["covered"]
        and any(candidate["coverage"]["covered"] for candidate in record["candidates"])
        for record in records
    )
    spreads = [
        max(candidate["coverage"]["best_match_score"] for candidate in record["candidates"])
        - min(candidate["coverage"]["best_match_score"] for candidate in record["candidates"])
        for record in records
    ]
    oracle_gains = [
        max(candidate["coverage"]["best_match_score"] for candidate in record["candidates"])
        - record["initial"]["best_match_score"]
        for record in records
    ]
    summary = {
        "num_tasks": len(records),
        "num_candidate_branches": sum(len(record["candidates"]) for record in records),
        "initial_covered": len(records) - initial_omissions,
        "initial_omitted": initial_omissions,
        "initially_omitted_recovered_by_any_candidate": recovered,
        "recovery_fraction_among_initial_omissions": (
            recovered / initial_omissions if initial_omissions else 0.0
        ),
        "mean_oracle_best_match_gain": float(np.mean(oracle_gains)),
        "tasks_with_candidate_score_spread_at_least_0_15": sum(
            spread >= 0.15 for spread in spreads
        ),
        "mean_candidate_score_spread": float(np.mean(spreads)),
    }
    gates = {
        "all_twelve_tasks_completed": len(records) == 12,
        "all_forty_eight_branches_completed": summary["num_candidate_branches"] == 48,
        "initial_support_non_saturated": initial_omissions >= 6,
        "at_least_three_omitted_solutions_recovered": recovered >= 3,
        "mean_oracle_best_match_gain_at_least_0_10": (
            summary["mean_oracle_best_match_gain"] >= 0.10
        ),
        "at_least_four_tasks_have_candidate_spread": (
            summary["tasks_with_candidate_score_spread_at_least_0_15"] >= 4
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {**summary, "gates": gates}


def _build_models(
    config: Config,
    judge_model: str,
) -> tuple[Any, Any, Any]:
    pair = config.model_pairs[0]
    questioner = build_model_adapter(pair.questioner, config)
    answerer = build_model_adapter(pair.answerer, config)
    judge_spec = replace(
        pair.questioner,
        model=judge_model,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    judge = build_model_adapter(judge_spec, config)
    return questioner, answerer, judge


def run_serving_smoke(
    config: Config,
    *,
    judge_model: str,
) -> dict[str, Any]:
    """Exercise ten representative physical requests without running the gate."""
    questioner, answerer, judge = _build_models(config, judge_model)
    env = PaprikaCustomerServiceEnvironment(config, answerer)
    env.set_questioner(questioner)
    tasks = load_paprika_tasks(
        config.paprika_data_path,
        split="train",
        verify_official_hash=bool(config.paprika_verify_official_hash),
    )
    selected = [tasks[index] for index in DEVELOPMENT_INDICES[:2]]
    hypothesis_count = int(config.paprika_num_hypotheses)
    hypothesis_prompts = [
        hypothesis_messages(task.scenario, hypothesis_count)
        for task in selected
    ]
    hypothesis_responses = questioner.chat_complete_messages_batched(
        hypothesis_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    hypotheses = [
        parse_string_list(
            response,
            "hypotheses",
            minimum=hypothesis_count,
            maximum=hypothesis_count,
        )
        for response in hypothesis_responses
    ]
    candidate_count = int(config.paprika_num_candidates)
    candidate_prompts = [
        diagnostic_candidate_messages(task.scenario, support, candidate_count)
        for task, support in zip(selected, hypotheses, strict=True)
    ]
    candidate_responses = questioner.chat_complete_messages_batched(
        candidate_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    candidates = [
        _parse_diagnostic_candidates(
            env,
            response,
            task.scenario,
            candidate_count,
        )
        for task, response in zip(selected, candidate_responses, strict=True)
    ]
    actions = [cell[0] for cell in candidates]
    customer_prompts = [
        customer_messages(action, task.solution)
        for action, task in zip(actions, selected, strict=True)
    ]
    replies = answerer.chat_complete_messages_batched(
        customer_prompts,
        temperature=float(config.answer_temperature),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    if any(not reply.strip() for reply in replies):
        raise ValueError("serving smoke returned an empty customer reply")
    observations = [
        PaprikaObservation(
            reply=reply.strip(),
            mapped_outcome="diagnostic observation",
            mapped_cleanly=True,
            goal_reached=False,
        )
        for reply in replies
    ]
    refresh_count = int(config.paprika_num_refresh_hypotheses)
    refinement_prompts = [
        refinement_messages(
            task.scenario,
            support,
            [(action, observation)],
            refresh_count,
        )
        for task, support, action, observation in zip(
            selected, hypotheses, actions, observations, strict=True
        )
    ]
    refinement_responses = questioner.chat_complete_messages_batched(
        refinement_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    refined = [
        parse_string_list(
            response,
            "refined_hypotheses",
            minimum=refresh_count,
            maximum=refresh_count,
        )
        for response in refinement_responses
    ]
    coverage_prompts = [
        semantic_coverage_messages(
            task.scenario,
            task.solution,
            [("initial", support), ("refreshed", new_support)],
        )
        for task, support, new_support in zip(
            selected, hypotheses, refined, strict=True
        )
    ]
    coverage_responses = judge.chat_complete_messages_batched(
        coverage_prompts,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    coverage = [
        parse_coverage_response(
            response,
            ["initial", "refreshed"],
            [len(support), len(new_support)],
        )
        for response, support, new_support in zip(
            coverage_responses, hypotheses, refined, strict=True
        )
    ]
    usages = {
        "questioner": questioner.usage_snapshot(),
        "answerer": answerer.usage_snapshot(),
        "judge": judge.usage_snapshot(),
    }
    physical_requests = sum(
        int(usage["adapter_requests"]) for usage in usages.values()
    )
    return {
        "schema_version": 1,
        "status": "passed" if physical_requests == 10 else "failed",
        "physical_requests": physical_requests,
        "expected_physical_requests": 10,
        "task_indices": list(DEVELOPMENT_INDICES[:2]),
        "stage_counts": {
            "initial_hypotheses": 2,
            "diagnostic_candidates": 2,
            "customer_replies": 2,
            "refinements": 2,
            "coverage_judgments": 2,
        },
        "parsed_support_sizes": [
            {
                "initial": len(support),
                "refreshed": len(new_support),
            }
            for support, new_support in zip(hypotheses, refined, strict=True)
        ],
        "coverage_parser_rows": coverage,
        "usage": usages,
    }


def run_unlock(
    config: Config,
    *,
    judge_model: str,
) -> dict[str, Any]:
    questioner, answerer, judge = _build_models(config, judge_model)
    env = PaprikaCustomerServiceEnvironment(config, answerer)
    env.set_questioner(questioner)
    tasks = load_paprika_tasks(
        config.paprika_data_path,
        split="train",
        verify_official_hash=bool(config.paprika_verify_official_hash),
    )
    selected = [tasks[index] for index in DEVELOPMENT_INDICES]
    initial_states = [
        env._initial_for_task(task, questioner, config)
        for task in selected
    ]
    candidates_many = _generate_candidates_many(
        env,
        questioner,
        initial_states,
        [task.scenario for task in selected],
        int(config.paprika_num_candidates),
        config,
    )
    flat_actions = [
        action
        for candidates in candidates_many
        for action in candidates
    ]
    flat_tasks = [
        task
        for task, candidates in zip(selected, candidates_many, strict=True)
        for _action in candidates
    ]
    flat_states = [
        state
        for state, candidates in zip(initial_states, candidates_many, strict=True)
        for _action in candidates
    ]
    observations = _observe_many(
        env,
        questioner,
        flat_actions,
        [task.solution for task in flat_tasks],
    )
    histories = [
        [(action, observation)]
        for action, observation in zip(flat_actions, observations, strict=True)
    ]
    refreshed = _refresh_supports_many(
        env,
        questioner,
        flat_states,
        histories,
        int(config.paprika_num_refresh_hypotheses),
        config,
    )
    records: list[dict[str, Any]] = []
    offset = 0
    coverage_messages = []
    support_metadata: list[tuple[list[str], list[int]]] = []
    for task, initial_state, candidates in zip(
        selected, initial_states, candidates_many, strict=True
    ):
        branch_supports = refreshed[offset : offset + len(candidates)]
        ids = ["initial", *[f"candidate_{index}" for index in range(len(candidates))]]
        supports = [list(initial_state.hypotheses), *branch_supports]
        coverage_messages.append(
            semantic_coverage_messages(
                task.scenario,
                task.solution,
                list(zip(ids, supports, strict=True)),
            )
        )
        support_metadata.append((ids, [len(support) for support in supports]))
        offset += len(candidates)
    coverage_responses = judge.chat_complete_messages_batched(
        coverage_messages,
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    coverage_rows = []
    for row, (response, (ids, sizes)) in enumerate(
        zip(coverage_responses, support_metadata, strict=True)
    ):
        try:
            coverage_rows.append(
                parse_coverage_response(response, ids, sizes)
            )
        except ValueError as exc:
            raise CoverageBatchError(
                str(exc),
                row=row,
                response=response,
            ) from exc
    offset = 0
    for task, state, candidates, rows in zip(
        selected, initial_states, candidates_many, coverage_rows, strict=True
    ):
        candidate_records = []
        for index, action in enumerate(candidates):
            observation = observations[offset + index]
            candidate_records.append(
                {
                    "index": index,
                    "query": action.query,
                    "outcomes": list(action.outcomes),
                    "private_solution_conditioned_reply_measurement_only": observation.reply,
                    "refreshed_support": refreshed[offset + index],
                    "coverage": rows[index + 1],
                }
            )
        records.append(
            {
                "task_index": DEVELOPMENT_INDICES[len(records)],
                "task_id": task.task_id,
                "scenario": task.scenario,
                "private_solution_measurement_only": task.solution,
                "initial_support": list(state.hypotheses),
                "initial": rows[0],
                "candidates": candidate_records,
            }
        )
        offset += len(candidates)
    summary = summarize(records)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "development_gate_failed",
        "protocol": {
            "split": "train",
            "selection_seed": 24287,
            "task_indices": list(DEVELOPMENT_INDICES),
            "semantic_coverage_threshold": SEMANTIC_COVERAGE_THRESHOLD,
            "target_hidden_from_generation": True,
            "target_used_for_measurement_only": True,
            "judge_model": judge_model,
        },
        "summary": summary,
        "records": records,
        "usage": {
            "questioner": questioner.usage_snapshot(),
            "answerer": answerer.usage_snapshot(),
            "judge": judge.usage_snapshot(),
        },
        "simulator_faithfulness": env._simulator_faithfulness_metrics(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development"),
        default="development",
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        if args.stage == "serving_smoke":
            payload = run_serving_smoke(
                config,
                judge_model=args.judge_model,
            )
            filename = "SERVING_SMOKE.json"
        else:
            payload = run_unlock(config, judge_model=args.judge_model)
            filename = "DEVELOPMENT.json"
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
            "failing_coverage_row": getattr(exc, "row", None),
            "raw_failing_coverage_response": getattr(exc, "response", None),
        }
        filename = (
            "SERVING_SMOKE_FAILURE.json"
            if args.stage == "serving_smoke"
            else "DEVELOPMENT_FAILURE.json"
        )
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    report = (
        {"status": payload["status"], **payload["summary"]}
        if "summary" in payload
        else payload
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
