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
    filtering_messages,
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
        if support_size == 0:
            if index is not None:
                raise ValueError("empty support requires a null best index")
        elif (
            isinstance(index, bool)
            or not isinstance(index, int)
            or not 0 <= index < support_size
        ):
            raise ValueError("best_hypothesis_index is outside its support")
        reason = row.get("reason")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("coverage response requires a brief reason")
        parsed.append(
            {
                "id": expected_id,
                "best_match_score": score,
                "best_hypothesis_index": index,
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
    check_messages = [
        faithfulness_messages(action, solution, reply)
        for action, solution, reply in zip(actions, solutions, replies, strict=True)
    ]
    checks = env._cached_complete_many(
        evaluator,
        check_messages,
        0.0,
        namespace="unlock:faithfulness",
    )
    consistent = [
        _parse_consistency(text)
        for text in checks
    ]
    if not all(consistent):
        failed = [index for index, value in enumerate(consistent) if not value]
        raise ValueError(f"customer simulator failed faithfulness at rows {failed}")
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


def run_unlock(
    config: Config,
    *,
    judge_model: str,
) -> dict[str, Any]:
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
    coverage_rows = [
        parse_coverage_response(response, ids, sizes)
        for response, (ids, sizes) in zip(
            coverage_responses, support_metadata, strict=True
        )
    ]
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
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--judge-model", default="openai/gpt-5.4-mini")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload = run_unlock(config, judge_model=args.judge_model)
        filename = "DEVELOPMENT.json"
    except Exception as exc:
        payload = {
            "schema_version": 1,
            "status": "failed_closed",
            "error": f"{type(exc).__name__}: {exc}",
        }
        filename = "DEVELOPMENT_FAILURE.json"
        (args.output_dir / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / filename).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
