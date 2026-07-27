#!/usr/bin/env python3
"""Run robust shared comparative planning on HotpotQA directional unlocks."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.hotpot_causal_belief_smoke import (
    GateExecutionError,
    HYPOTHESIS_COUNT,
    MODEL_ID,
    RANDOM_CONTROL_SEED,
    ROOT_CONTEXT_INDICES,
    _answer_and_enabling_titles,
    _candidate_titles,
    _normalized_state,
    _usage_snapshot,
    load_task,
    token_f1,
)
from scripts.hotpot_future_uplift_confirmation import (
    SPLIT_HASHES,
    metadata_splits,
    qualification,
    selected_rows,
)
from scripts.hotpot_directional_unlock_audit import sha256_file


INTERFACE_VERSION = "hotpot-shared-comparative-v2-1"
ROOT_COUNT = 4
DEVELOPMENT_TASKS = 5
EXPECTED_CALLS_PER_TASK = 10
SERVING_COST_CAP = 0.25
DEVELOPMENT_COST_CAP = 1.50


def _canonical_text(value: str) -> str:
    return " ".join(value.split())


def parse_hypothesis_lines(text: str) -> list[str]:
    if text != text.strip():
        raise ValueError("hypothesis response must be canonical text")
    lines = text.splitlines()
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError("hypothesis response has wrong line count")
    values = []
    for index, line in enumerate(lines, start=1):
        prefix = f"H{index}|"
        if not line.startswith(prefix):
            raise ValueError("hypothesis response has invalid prefix order")
        value = _canonical_text(line[len(prefix) :])
        if not value:
            raise ValueError("hypothesis must be nonempty")
        values.append(value)
    normalized = {_canonical_text(value).casefold() for value in values}
    if len(normalized) != HYPOTHESIS_COUNT:
        raise ValueError("hypotheses must be distinct")
    return values


def _parse_permutation(
    value: str,
    *,
    prefix: str,
    count: int,
) -> list[int]:
    fields = value.split("|")
    expected = {f"{prefix}{index + 1}" for index in range(count)}
    if len(fields) != count + 1 or fields[0] != "ORDER":
        raise ValueError("order line has invalid shape")
    if set(fields[1:]) != expected:
        raise ValueError("order line is not a complete permutation")
    return [int(field[len(prefix) :]) - 1 for field in fields[1:]]


def parse_myopic_order(text: str) -> list[int]:
    if text != text.strip() or "\n" in text:
        raise ValueError("myopic order must be one canonical line")
    return _parse_permutation(text, prefix="R", count=ROOT_COUNT)


def parse_plan(text: str, *, candidate_count: int) -> dict[str, Any]:
    if text != text.strip():
        raise ValueError("plan response must be canonical text")
    lines = text.splitlines()
    if len(lines) != ROOT_COUNT + 1:
        raise ValueError("plan response has wrong line count")
    order = _parse_permutation(lines[0], prefix="R", count=ROOT_COUNT)
    followups: dict[int, int] = {}
    for line in lines[1:]:
        fields = line.split("|")
        if (
            len(fields) != 2
            or fields[0] not in {f"R{index + 1}" for index in range(ROOT_COUNT)}
            or fields[1]
            not in {f"T{index + 1}" for index in range(candidate_count)}
        ):
            raise ValueError("plan followup line is invalid")
        root_index = int(fields[0][1:]) - 1
        if root_index in followups:
            raise ValueError("plan repeats a root followup")
        followups[root_index] = int(fields[1][1:]) - 1
    if set(followups) != set(range(ROOT_COUNT)):
        raise ValueError("plan lacks a root followup")
    return {"root_order": order, "followup_indices": followups}


def parse_final_lines(text: str) -> dict[str, Any]:
    if text != text.strip():
        raise ValueError("final response must be canonical text")
    lines = text.splitlines()
    if len(lines) != 2 or not lines[0].startswith("ANSWER|"):
        raise ValueError("final response has invalid shape")
    answer = _canonical_text(lines[0][len("ANSWER|") :])
    fields = lines[1].split("|")
    if (
        not answer
        or len(fields) != 2
        or fields[0] != "CONFIDENCE"
        or not fields[1].isdigit()
        or not 0 <= int(fields[1]) <= 100
    ):
        raise ValueError("final response has invalid values")
    return {"answer": answer, "confidence": int(fields[1])}


def hypothesis_messages(
    question: str,
    titles: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "candidate_titles": [
            {"id": f"C{index + 1}", "title": title}
            for index, title in enumerate(titles)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Maintain an open-world belief state for a two-document question. "
                "Generate eight distinct plausible answer and support-chain "
                "hypotheses from the question and candidate article titles. Return "
                "exactly eight lines H1|... through H8|... in numeric order and no "
                "other text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def refresh_line_messages(
    question: str,
    initial_hypotheses: Sequence[str],
    root_title: str,
    root_paragraph: str,
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "prior_hypotheses": list(initial_hypotheses),
        "revealed_article": {
            "title": root_title,
            "paragraph": root_paragraph,
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Refresh an open-world belief state after one article is revealed. "
                "Return eight distinct unresolved hypotheses about the answer and "
                "which evidence or article is still needed. Preserve supported "
                "ideas and revise contradicted ones. Make the state sufficient for "
                "another agent that cannot see the question or paragraph. Return "
                "exactly eight lines H1|... through H8|... in numeric order and no "
                "other text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def myopic_rank_messages(
    initial_hypotheses: Sequence[str],
    root_titles: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "unresolved_hypotheses": list(initial_hypotheses),
        "root_titles": [
            {"root_id": f"R{index + 1}", "title": title}
            for index, title in enumerate(root_titles)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Rank four candidate first articles using only the supplied current "
                "belief state and titles. Prefer the article most likely to resolve "
                "the answer uncertainty immediately. Return exactly one line "
                "ORDER|R1|R2|R3|R4 containing every root ID once from best to worst; "
                "the shown order is syntax only. Return no other text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def plan_rank_messages(
    *,
    states: Sequence[Sequence[str]],
    all_titles: Sequence[str],
    root_context_indices: Sequence[int],
) -> list[dict[str, str]]:
    roots = []
    for root_index, (state, context_index) in enumerate(
        zip(states, root_context_indices, strict=True)
    ):
        candidates = _candidate_titles(all_titles, context_index)
        roots.append(
            {
                "root_id": f"R{root_index + 1}",
                "revealed_root_title": all_titles[context_index],
                "unresolved_hypotheses": list(state),
                "candidate_titles": [
                    {"title_id": f"T{index + 1}", "title": title}
                    for index, title in enumerate(candidates)
                ],
            }
        )
    payload = {"root_plans": roots}
    return [
        {
            "role": "system",
            "content": (
                "Compare four two-step retrieval plans using only each root's "
                "revealed article title, supplied post-reveal belief state, and "
                "candidate next-article titles. "
                "For every root choose the one next title most likely to resolve its "
                "missing evidence. Then rank the complete two-article paths by the "
                "resulting ability to answer: account for evidence already resolved "
                "by the revealed root as represented in its post-reveal state, as "
                "well as evidence its best continuation can resolve. Do not rank on "
                "continuation quality alone. Treat root states independently and do "
                "not infer an original question. Return exactly five lines: first one "
                "ORDER line containing R1..R4 once from best to worst, then exactly "
                "R1|Tn, R2|Tn, R3|Tn, R4|Tn in numeric root order. Return no other "
                "text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def final_line_messages(
    question: str,
    evidence: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {"question": question, "revealed_articles": list(evidence)}
    return [
        {
            "role": "system",
            "content": (
                "Answer using only the supplied articles. Return exactly two lines: "
                "ANSWER|your concise answer and CONFIDENCE|n where n is an integer "
                "0 through 100. Return no other text."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def _build_model(config: Config) -> Any:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Hotpot V2 config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _build_task(
    row: dict[str, Any],
    roots: Sequence[int],
) -> dict[str, Any]:
    titles = [str(value) for value in row["context"]["title"]]
    paragraphs = [
        " ".join(str(sentence) for sentence in sentences)
        for sentences in row["context"]["sentences"]
    ]
    return {
        "row": row,
        "titles": titles,
        "paragraphs": paragraphs,
        "roots": [int(value) for value in roots],
    }


def serving_tasks(validation_path: Path) -> list[dict[str, Any]]:
    row = load_task(validation_path)
    return [_build_task(row, ROOT_CONTEXT_INDICES)]


def development_tasks(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], int]:
    _metadata, splits = metadata_splits(paths)
    cohort = selected_rows(paths, splits["development"])
    qualified = [
        (row, qualification(row))
        for row in cohort
        if qualification(row)["qualifies"]
    ]
    if len(qualified) != DEVELOPMENT_TASKS:
        raise ValueError(
            f"development has {len(qualified)} qualifying rows, expected "
            f"{DEVELOPMENT_TASKS}"
        )
    return [
        _build_task(row, diagnostic["root_context_indices"])
        for row, diagnostic in qualified
    ], len(cohort)


def _rank_scores(order: Sequence[int]) -> list[int]:
    scores = [0] * len(order)
    for position, root_index in enumerate(order):
        scores[root_index] = len(order) - position
    return scores


def _pairwise_accuracy(
    order: Sequence[int],
    values: Sequence[int],
) -> tuple[int, int]:
    scores = _rank_scores(order)
    correct = comparable = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                continue
            comparable += 1
            score_delta = scores[left] - scores[right]
            value_delta = values[left] - values[right]
            correct += int((score_delta > 0) == (value_delta > 0))
    return correct, comparable


def _sign_flip_p(differences: Sequence[int]) -> float:
    nonzero = [value for value in differences if value != 0]
    if not nonzero:
        return 1.0
    observed = sum(nonzero)
    return sum(
        sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(nonzero)
        )
        >= observed
        for mask in range(1 << len(nonzero))
    ) / (1 << len(nonzero))


def _selection(
    root_index: int,
    plan: dict[str, Any],
) -> dict[str, int]:
    return {
        "root_index": root_index,
        "followup_candidate_index": int(
            plan["followup_indices"][root_index]
        ),
    }


def _coverage(
    task: dict[str, Any],
    selection: dict[str, int],
    support_titles: set[str],
) -> tuple[int, list[str]]:
    context_index = task["roots"][selection["root_index"]]
    candidates = _candidate_titles(task["titles"], context_index)
    chosen = [
        task["titles"][context_index],
        candidates[selection["followup_candidate_index"]],
    ]
    return len(set(chosen) & support_titles), chosen


def run_tasks(
    config: Config,
    *,
    tasks: Sequence[dict[str, Any]],
    stage: str,
    raw_path: Path,
    cohort_rows_materialized: int,
    model_adapter: Any | None = None,
    plan_message_builder: Any = plan_rank_messages,
    plan_response_parser: Any = parse_plan,
    interface_version: str = INTERFACE_VERSION,
) -> dict[str, Any]:
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "stage": stage,
        "task_ids": [str(task["row"]["id"]) for task in tasks],
    }
    try:
        initial_raw = model.chat_complete_messages_batched(
            [
                hypothesis_messages(
                    str(task["row"]["question"]), task["titles"]
                )
                for task in tasks
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        _checkpoint(raw_path, raw)
        initials = [parse_hypothesis_lines(text) for text in initial_raw]

        refresh_keys = [
            (task_index, root_index)
            for task_index in range(len(tasks))
            for root_index in range(ROOT_COUNT)
        ]
        refresh_raw = model.chat_complete_messages_batched(
            [
                refresh_line_messages(
                    str(tasks[task_index]["row"]["question"]),
                    initials[task_index],
                    tasks[task_index]["titles"][
                        tasks[task_index]["roots"][root_index]
                    ],
                    tasks[task_index]["paragraphs"][
                        tasks[task_index]["roots"][root_index]
                    ],
                )
                for task_index, root_index in refresh_keys
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["refreshes"] = refresh_raw
        _checkpoint(raw_path, raw)
        parsed_refreshes = [
            parse_hypothesis_lines(text) for text in refresh_raw
        ]
        refreshes = [
            parsed_refreshes[index * ROOT_COUNT : (index + 1) * ROOT_COUNT]
            for index in range(len(tasks))
        ]

        myopic_raw = model.chat_complete_messages_batched(
            [
                myopic_rank_messages(
                    initial,
                    [task["titles"][index] for index in task["roots"]],
                )
                for task, initial in zip(tasks, initials, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["myopic"] = myopic_raw
        _checkpoint(raw_path, raw)
        myopic_orders = [parse_myopic_order(text) for text in myopic_raw]

        plan_states = {
            "aligned": refreshes,
            "fixed": [
                [list(initial) for _ in range(ROOT_COUNT)]
                for initial in initials
            ],
            "shuffled": [
                [
                    task_refreshes[(root_index + 1) % ROOT_COUNT]
                    for root_index in range(ROOT_COUNT)
                ]
                for task_refreshes in refreshes
            ],
        }
        plans: dict[str, list[dict[str, Any]]] = {}
        for name, states_by_task in plan_states.items():
            responses = model.chat_complete_messages_batched(
                [
                    plan_message_builder(
                        states=states,
                        all_titles=task["titles"],
                        root_context_indices=task["roots"],
                    )
                    for task, states in zip(
                        tasks, states_by_task, strict=True
                    )
                ],
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            raw[name] = responses
            _checkpoint(raw_path, raw)
            plans[name] = [
                plan_response_parser(
                    text, candidate_count=len(task["titles"]) - 1
                )
                for text, task in zip(responses, tasks, strict=True)
            ]

        frozen_selections = []
        final_payloads = []
        for task_index, task in enumerate(tasks):
            aligned = plans["aligned"][task_index]
            selections = {
                "nonmyopic": _selection(
                    aligned["root_order"][0], aligned
                ),
                "myopic": _selection(
                    myopic_orders[task_index][0], aligned
                ),
                "fixed": _selection(
                    plans["fixed"][task_index]["root_order"][0],
                    plans["fixed"][task_index],
                ),
                "shuffled": _selection(
                    plans["shuffled"][task_index]["root_order"][0],
                    plans["shuffled"][task_index],
                ),
                "random": _selection(
                    random.Random(
                        RANDOM_CONTROL_SEED + task_index
                    ).randrange(ROOT_COUNT),
                    aligned,
                ),
            }
            frozen_selections.append(selections)
            nonmyopic = selections["nonmyopic"]
            context_index = task["roots"][nonmyopic["root_index"]]
            candidates = _candidate_titles(task["titles"], context_index)
            selected_titles = [
                task["titles"][context_index],
                candidates[nonmyopic["followup_candidate_index"]],
            ]
            final_payloads.append(
                [
                    {
                        "title": title,
                        "paragraph": task["paragraphs"][
                            task["titles"].index(title)
                        ],
                    }
                    for title in selected_titles
                ]
            )
        raw["frozen_selections"] = frozen_selections
        _checkpoint(raw_path, raw)

        final_raw = model.chat_complete_messages_batched(
            [
                final_line_messages(
                    str(task["row"]["question"]), evidence
                )
                for task, evidence in zip(
                    tasks, final_payloads, strict=True
                )
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["final"] = final_raw
        _checkpoint(raw_path, raw)
        finals = [parse_final_lines(text) for text in final_raw]
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    records = []
    refresh_changed = 0
    distinct_refresh_tasks = 0
    aligned_fixed_changes = 0
    aligned_shuffled_changes = 0
    pair_totals = {
        "nonmyopic": [0, 0],
        "myopic": [0, 0],
        "fixed": [0, 0],
        "shuffled": [0, 0],
    }
    for task_index, task in enumerate(tasks):
        initial_state = _normalized_state(initials[task_index])
        states = [
            _normalized_state(state) for state in refreshes[task_index]
        ]
        refresh_changed += sum(state != initial_state for state in states)
        distinct_refresh_tasks += int(len(set(states)) == ROOT_COUNT)
        aligned_fixed_changes += int(
            plans["aligned"][task_index]["root_order"]
            != plans["fixed"][task_index]["root_order"]
        )
        aligned_shuffled_changes += int(
            plans["aligned"][task_index]["root_order"]
            != plans["shuffled"][task_index]["root_order"]
        )
        support_titles = {
            str(value)
            for value in task["row"]["supporting_facts"]["title"]
        }
        policy_rows = {}
        for name, selection in frozen_selections[task_index].items():
            coverage, titles = _coverage(
                task, selection, support_titles
            )
            answer_title, enabling_title = _answer_and_enabling_titles(
                task["row"]
            )
            policy_rows[name] = {
                **selection,
                "support_coverage": coverage,
                "root_role": (
                    "enabling"
                    if titles[0] == enabling_title
                    else "answer"
                    if titles[0] == answer_title
                    else "distractor"
                ),
                "followup_role": (
                    "answer"
                    if titles[1] == answer_title
                    else "enabling"
                    if titles[1] == enabling_title
                    else "distractor"
                ),
            }
        root_values = []
        aligned_plan = plans["aligned"][task_index]
        for root_index in range(ROOT_COUNT):
            coverage, _titles = _coverage(
                task,
                _selection(root_index, aligned_plan),
                support_titles,
            )
            root_values.append(coverage)
        orders = {
            "nonmyopic": aligned_plan["root_order"],
            "myopic": myopic_orders[task_index],
            "fixed": plans["fixed"][task_index]["root_order"],
            "shuffled": plans["shuffled"][task_index]["root_order"],
        }
        for name, order in orders.items():
            correct, comparable = _pairwise_accuracy(order, root_values)
            pair_totals[name][0] += correct
            pair_totals[name][1] += comparable
        records.append(
            {
                "task_id": str(task["row"]["id"]),
                "myopic_order": myopic_orders[task_index],
                "aligned_order": aligned_plan["root_order"],
                "fixed_order": plans["fixed"][task_index]["root_order"],
                "shuffled_order": plans["shuffled"][task_index][
                    "root_order"
                ],
                "root_values": root_values,
                "policies": policy_rows,
                "final_answer_token_f1": token_f1(
                    finals[task_index]["answer"],
                    str(task["row"]["answer"]),
                ),
            }
        )

    def policy_total(name: str) -> int:
        return sum(
            row["policies"][name]["support_coverage"] for row in records
        )

    def comparison(name: str) -> dict[str, Any]:
        differences = [
            row["policies"]["nonmyopic"]["support_coverage"]
            - row["policies"][name]["support_coverage"]
            for row in records
        ]
        return {
            "gain": sum(differences),
            "wins": sum(value > 0 for value in differences),
            "ties": sum(value == 0 for value in differences),
            "losses": sum(value < 0 for value in differences),
            "exact_one_sided_sign_flip_p": _sign_flip_p(differences),
        }

    comparisons = {
        name: comparison(name)
        for name in ("myopic", "fixed", "shuffled", "random")
    }
    accuracies = {
        name: correct / comparable if comparable else 0.0
        for name, (correct, comparable) in pair_totals.items()
    }
    generator = usage["generator"]
    expected_requests = len(tasks) * EXPECTED_CALLS_PER_TASK
    gates = {
        "all_tasks_complete": len(records) == len(tasks),
        "exact_logical_request_count": usage["physical_requests"]
        == expected_requests,
        "http_attempts_match_requests_plus_retries": int(
            generator.get("http_attempts", -1)
        )
        == expected_requests + int(generator.get("retry_count", 0)),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": True,
        "all_refreshes_change": refresh_changed == len(tasks) * ROOT_COUNT,
        "all_tasks_have_four_distinct_refreshes": distinct_refresh_tasks
        == len(tasks),
    }
    if stage == "development":
        myopic_comparison = comparisons["myopic"]
        gates.update(
            {
                "aligned_differs_from_fixed_at_least_3": (
                    aligned_fixed_changes >= 3
                ),
                "aligned_differs_from_shuffled_at_least_3": (
                    aligned_shuffled_changes >= 3
                ),
                "nonmyopic_changes_at_least_4_myopic_roots": sum(
                    row["policies"]["nonmyopic"]["root_index"]
                    != row["policies"]["myopic"]["root_index"]
                    for row in records
                )
                >= 4,
                "nonmyopic_selects_enabling_at_least_4": sum(
                    row["policies"]["nonmyopic"]["root_role"]
                    == "enabling"
                    for row in records
                )
                >= 4,
                "nonmyopic_covers_at_least_9_supports": (
                    policy_total("nonmyopic") >= 9
                ),
                "nonmyopic_gains_at_least_3_vs_myopic": (
                    myopic_comparison["gain"] >= 3
                ),
                "nonmyopic_wins_at_least_4_vs_myopic": (
                    myopic_comparison["wins"] >= 4
                ),
                "nonmyopic_zero_losses_vs_myopic": (
                    myopic_comparison["losses"] == 0
                ),
                "nonmyopic_sign_flip_p_at_most_0_0625": (
                    myopic_comparison["exact_one_sided_sign_flip_p"]
                    <= 0.0625
                ),
                "nonmyopic_gains_at_least_2_vs_fixed": (
                    comparisons["fixed"]["gain"] >= 2
                ),
                "nonmyopic_gains_at_least_2_vs_shuffled": (
                    comparisons["shuffled"]["gain"] >= 2
                ),
                "nonmyopic_gains_at_least_2_vs_random": (
                    comparisons["random"]["gain"] >= 2
                ),
                "nonmyopic_root_accuracy_at_least_0_65": (
                    accuracies["nonmyopic"] >= 0.65
                ),
                "nonmyopic_root_accuracy_gain_at_least_0_10": (
                    accuracies["nonmyopic"]
                    >= accuracies["myopic"] + 0.10
                ),
                "mean_final_answer_f1_at_least_0_50": (
                    sum(row["final_answer_token_f1"] for row in records)
                    / len(records)
                    >= 0.50
                ),
                "cost_at_most_1_50": usage["adapter_cost_usd"]
                <= DEVELOPMENT_COST_CAP,
            }
        )
    else:
        gates["cost_at_most_0_25"] = (
            usage["adapter_cost_usd"] <= SERVING_COST_CAP
        )
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "stage": stage,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "task_ids": [record["task_id"] for record in records],
            "development_split_sha256": (
                SPLIT_HASHES["development"]
                if stage == "development"
                else None
            ),
            "cohort_rows_materialized": cohort_rows_materialized,
            "selected_qualifying_count": len(records),
            "expected_calls_per_task": EXPECTED_CALLS_PER_TASK,
            "myopic_uses_aligned_receding_followup": True,
            "endpoint_hidden_until_all_model_outputs_frozen": True,
            "semantic_reissues_allowed": False,
            "bounded_transport_retries_configured": True,
        },
        "metrics": {
            "policy_support_totals": {
                name: policy_total(name)
                for name in (
                    "nonmyopic",
                    "myopic",
                    "fixed",
                    "shuffled",
                    "random",
                )
            },
            "comparisons": comparisons,
            "root_pairwise_accuracy": accuracies,
            "root_changes_vs_myopic": sum(
                row["policies"]["nonmyopic"]["root_index"]
                != row["policies"]["myopic"]["root_index"]
                for row in records
            ),
            "nonmyopic_enabling_roots": sum(
                row["policies"]["nonmyopic"]["root_role"] == "enabling"
                for row in records
            ),
            "refreshes_changed": refresh_changed,
            "tasks_with_four_distinct_refreshes": distinct_refresh_tasks,
            "aligned_fixed_order_changes": aligned_fixed_changes,
            "aligned_shuffled_order_changes": aligned_shuffled_changes,
            "mean_final_answer_token_f1": sum(
                row["final_answer_token_f1"] for row in records
            )
            / len(records),
        },
        "records": records,
        "gates": gates,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=("serving", "development"), required=True
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--train-shard", type=Path, action="append")
    parser.add_argument("--validation-data", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_max_output_tokens = min(
        config.openrouter_max_output_tokens, 2048
    )
    config.openrouter_concurrency = min(config.openrouter_concurrency, 64)
    config.log_path = args.output_dir / "run.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    if args.stage == "serving":
        if args.validation_data is None:
            parser.error("serving requires --validation-data")
        tasks = serving_tasks(args.validation_data)
        cohort_rows = 1
        config.openrouter_projected_cost_usd = 0.15
        config.openrouter_run_budget_usd = SERVING_COST_CAP
    else:
        if not args.train_shard or len(args.train_shard) != 2:
            parser.error("development requires exactly two --train-shard values")
        tasks, cohort_rows = development_tasks(tuple(args.train_shard))
        config.openrouter_projected_cost_usd = 0.75
        config.openrouter_run_budget_usd = DEVELOPMENT_COST_CAP
    try:
        payload = run_tasks(
            config,
            tasks=tasks,
            stage=args.stage,
            raw_path=raw_path,
            cohort_rows_materialized=cohort_rows,
        )
        payload["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
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
            failure["private_raw_sha256"] = sha256_file(raw_path)
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / args.stage.upper()
    output_path = output_path.with_suffix(".json")
    output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output_path),
                "metrics": payload["metrics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
