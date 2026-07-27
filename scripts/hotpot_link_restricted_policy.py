#!/usr/bin/env python3
"""Develop Hotpot non-myopic planning on paragraph-link actions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.hotpot_causal_belief_smoke import (
    GateExecutionError,
    RANDOM_CONTROL_SEED,
    _answer_and_enabling_titles,
    _normalized_state,
    _usage_snapshot,
    token_f1,
)
from scripts.hotpot_directional_unlock_audit import (
    mentioned_context_titles,
    sha256_file,
)
from scripts.hotpot_future_uplift_confirmation import (
    qualification,
    selected_rows,
)
from scripts.hotpot_link_restricted_manifest import (
    DEVELOPMENT_SELECTION_COUNT,
    _source_splits,
)
from scripts.hotpot_shared_comparative_v2 import (
    ROOT_COUNT,
    _build_model,
    final_line_messages,
    hypothesis_messages,
    myopic_rank_messages,
    parse_final_lines,
    parse_hypothesis_lines,
    refresh_line_messages,
)
from scripts.hotpot_shared_comparative_v4 import (
    myopic_rank_row_messages,
    parse_myopic_rank_rows,
)


INTERFACE_VERSION = "hotpot-link-restricted-policy-1"
EXPECTED_CALLS_PER_TASK = 10
SERVING_COST_CAP = 0.25
DEVELOPMENT_COST_CAP = 3.0


def _build_task(row: dict[str, Any], roots: Sequence[int]) -> dict[str, Any]:
    titles = [str(value) for value in row["context"]["title"]]
    paragraphs = [
        " ".join(str(sentence) for sentence in sentences)
        for sentences in row["context"]["sentences"]
    ]
    candidate_titles = [
        mentioned_context_titles(
            paragraph_text=paragraphs[context_index],
            context_titles=titles,
            root_title=titles[context_index],
        )
        for context_index in roots
    ]
    return {
        "row": row,
        "titles": titles,
        "paragraphs": paragraphs,
        "roots": [int(value) for value in roots],
        "candidate_titles": candidate_titles,
    }


def load_tasks(
    paths: Sequence[Path],
    *,
    split_name: str,
    count: int,
) -> tuple[list[dict[str, Any]], int]:
    _metadata, splits = _source_splits(paths)
    cohort = selected_rows(paths, splits[split_name])
    qualified = []
    for row in cohort:
        try:
            diagnostic = qualification(row)
        except ValueError:
            continue
        if diagnostic["qualifies"]:
            qualified.append((row, diagnostic))
    if len(qualified) < count:
        raise ValueError(
            f"{split_name} has {len(qualified)} qualifying rows, needs {count}"
        )
    tasks = [
        _build_task(row, diagnostic["root_context_indices"])
        for row, diagnostic in qualified[:count]
    ]
    for task in tasks:
        values = oracle_root_values(task)
        answer_title, enabling_title = _answer_and_enabling_titles(task["row"])
        enabling_root = next(
            index
            for index, context_index in enumerate(task["roots"])
            if task["titles"][context_index] == enabling_title
        )
        if values[enabling_root] != 2 or values.count(2) != 1:
            raise ValueError("link-restricted task lacks unique enabling optimum")
        if answer_title == enabling_title:
            raise AssertionError("support roles must be distinct")
    return tasks, len(cohort)


def link_plan_messages(
    *,
    states: Sequence[Sequence[str]],
    task: dict[str, Any],
) -> list[dict[str, str]]:
    roots = []
    for root_index, state in enumerate(states):
        context_index = task["roots"][root_index]
        actions = [{"action_id": "S", "title": "STOP"}]
        actions.extend(
            {
                "action_id": f"T{index + 1}",
                "title": title,
            }
            for index, title in enumerate(task["candidate_titles"][root_index])
        )
        roots.append(
            {
                "root_id": f"R{root_index + 1}",
                "revealed_root_title": task["titles"][context_index],
                "post_reveal_belief_state": list(state),
                "available_next_actions": actions,
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "Compare four complete retrieval paths. Each root has already "
                "revealed one article, a post-reveal belief state, and only the "
                "next actions actually linked from that article. For every root, "
                "choose the action most likely to complete the evidence needed "
                "to answer; use S only when no displayed title helps. Rank the "
                "complete paths using evidence resolved by both the root and "
                "follow-up, not continuation quality alone. Treat roots "
                "independently. Return exactly four lines in numeric root order: "
                "R1|A|k, R2|A|k, R3|A|k, R4|A|k. A must be one displayed action "
                "ID for that root and k must use ranks 1 through 4 exactly once, "
                "with 1 best. Use literal | delimiters and return no other text."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {"root_paths": roots}, separators=(",", ":")
            ),
        },
    ]


def parse_link_plan(
    text: str,
    *,
    candidate_counts: Sequence[int],
) -> dict[str, Any]:
    if text != text.strip():
        raise ValueError("link plan must be canonical text")
    lines = text.splitlines()
    if len(lines) != ROOT_COUNT:
        raise ValueError("link plan has wrong line count")
    actions: dict[int, int | None] = {}
    ranks: dict[int, int] = {}
    for root_index, (line, candidate_count) in enumerate(
        zip(lines, candidate_counts, strict=True)
    ):
        fields = line.split("|")
        allowed = {"S"} | {
            f"T{index + 1}" for index in range(candidate_count)
        }
        if (
            len(fields) != 3
            or fields[0] != f"R{root_index + 1}"
            or fields[1] not in allowed
            or not fields[2].isdigit()
            or not 1 <= int(fields[2]) <= ROOT_COUNT
        ):
            raise ValueError("link plan row is invalid")
        actions[root_index] = (
            None if fields[1] == "S" else int(fields[1][1:]) - 1
        )
        ranks[root_index] = int(fields[2])
    if set(ranks.values()) != set(range(1, ROOT_COUNT + 1)):
        raise ValueError("link plan ranks are not a permutation")
    return {
        "root_order": sorted(range(ROOT_COUNT), key=ranks.__getitem__),
        "followup_indices": actions,
    }


def _selection(root_index: int, plan: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_index": int(root_index),
        "followup_candidate_index": plan["followup_indices"][root_index],
    }


def selected_titles(
    task: dict[str, Any],
    selection: dict[str, Any],
) -> list[str]:
    root_index = selection["root_index"]
    context_index = task["roots"][root_index]
    titles = [task["titles"][context_index]]
    followup = selection["followup_candidate_index"]
    if followup is not None:
        titles.append(task["candidate_titles"][root_index][followup])
    return titles


def support_coverage(
    task: dict[str, Any],
    selection: dict[str, Any],
) -> int:
    support_titles = {
        str(value) for value in task["row"]["supporting_facts"]["title"]
    }
    return len(set(selected_titles(task, selection)) & support_titles)


def oracle_root_values(task: dict[str, Any]) -> list[int]:
    values = []
    for root_index in range(ROOT_COUNT):
        actions: list[int | None] = [None]
        actions.extend(range(len(task["candidate_titles"][root_index])))
        values.append(
            max(
                support_coverage(
                    task,
                    {
                        "root_index": root_index,
                        "followup_candidate_index": action,
                    },
                )
                for action in actions
            )
        )
    return values


def _pairwise_accuracy(
    order: Sequence[int], values: Sequence[int]
) -> tuple[int, int]:
    rank = {root: position for position, root in enumerate(order)}
    correct = comparable = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                continue
            comparable += 1
            correct += int(
                (rank[left] < rank[right]) == (values[left] > values[right])
            )
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


def run_tasks(
    config: Config,
    *,
    tasks: Sequence[dict[str, Any]],
    stage: str,
    raw_path: Path,
    cohort_rows_materialized: int,
    model_adapter: Any | None = None,
) -> dict[str, Any]:
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "stage": stage,
        "task_ids": [str(task["row"]["id"]) for task in tasks],
    }
    try:
        initial_raw = model.chat_complete_messages_batched(
            [
                hypothesis_messages(str(task["row"]["question"]), task["titles"])
                for task in tasks
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
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
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        parsed_refreshes = [
            parse_hypothesis_lines(text) for text in refresh_raw
        ]
        refreshes = [
            parsed_refreshes[index * ROOT_COUNT : (index + 1) * ROOT_COUNT]
            for index in range(len(tasks))
        ]

        myopic_raw = model.chat_complete_messages_batched(
            [
                myopic_rank_row_messages(
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
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        myopic_orders = [parse_myopic_rank_rows(text) for text in myopic_raw]

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
                    link_plan_messages(states=states, task=task)
                    for task, states in zip(
                        tasks, states_by_task, strict=True
                    )
                ],
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            raw[name] = responses
            raw_path.write_text(
                json.dumps(raw, indent=2) + "\n", encoding="utf-8"
            )
            plans[name] = [
                parse_link_plan(
                    text,
                    candidate_counts=[
                        len(values) for values in task["candidate_titles"]
                    ],
                )
                for text, task in zip(responses, tasks, strict=True)
            ]

        frozen = []
        final_evidence = []
        for task_index, task in enumerate(tasks):
            aligned = plans["aligned"][task_index]
            selections = {
                "nonmyopic": _selection(aligned["root_order"][0], aligned),
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
            frozen.append(selections)
            final_evidence.append(
                [
                    {
                        "title": title,
                        "paragraph": task["paragraphs"][
                            task["titles"].index(title)
                        ],
                    }
                    for title in selected_titles(
                        task, selections["nonmyopic"]
                    )
                ]
            )
        raw["frozen_selections"] = frozen
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")

        final_raw = model.chat_complete_messages_batched(
            [
                final_line_messages(str(task["row"]["question"]), evidence)
                for task, evidence in zip(tasks, final_evidence, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["final"] = final_raw
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        finals = [parse_final_lines(text) for text in final_raw]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raw_path.write_text(json.dumps(raw, indent=2) + "\n", encoding="utf-8")
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}", _usage_snapshot(model)
        ) from exc

    records = []
    refresh_changed = distinct_refresh_tasks = 0
    aligned_fixed_changes = aligned_shuffled_changes = 0
    pair_totals = {
        name: [0, 0]
        for name in ("nonmyopic", "myopic", "fixed", "shuffled")
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
        answer_title, enabling_title = _answer_and_enabling_titles(task["row"])
        policy_rows = {}
        for name, selection in frozen[task_index].items():
            root_title = task["titles"][
                task["roots"][selection["root_index"]]
            ]
            policy_rows[name] = {
                **selection,
                "support_coverage": support_coverage(task, selection),
                "root_role": (
                    "enabling"
                    if root_title == enabling_title
                    else "answer"
                    if root_title == answer_title
                    else "distractor"
                ),
            }
        values = oracle_root_values(task)
        orders = {
            "nonmyopic": plans["aligned"][task_index]["root_order"],
            "myopic": myopic_orders[task_index],
            "fixed": plans["fixed"][task_index]["root_order"],
            "shuffled": plans["shuffled"][task_index]["root_order"],
        }
        for name, order in orders.items():
            correct, comparable = _pairwise_accuracy(order, values)
            pair_totals[name][0] += correct
            pair_totals[name][1] += comparable
        records.append(
            {
                "task_id": str(task["row"]["id"]),
                "oracle_root_values": values,
                "myopic_order": myopic_orders[task_index],
                "aligned_order": plans["aligned"][task_index]["root_order"],
                "fixed_order": plans["fixed"][task_index]["root_order"],
                "shuffled_order": plans["shuffled"][task_index]["root_order"],
                "policies": policy_rows,
                "final_answer_token_f1": token_f1(
                    finals[task_index]["answer"], str(task["row"]["answer"])
                ),
            }
        )

    def total(name: str) -> int:
        return sum(row["policies"][name]["support_coverage"] for row in records)

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
    expected = len(tasks) * EXPECTED_CALLS_PER_TASK
    generator = usage["generator"]
    gates = {
        "all_tasks_complete": len(records) == len(tasks),
        "exact_logical_request_count": usage["physical_requests"] == expected,
        "http_attempts_match_requests_plus_retries": int(
            generator.get("http_attempts", -1)
        )
        == expected + int(generator.get("retry_count", 0)),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": int(generator.get("forced_exits", -1)) == 0,
        "all_responses_parsed_without_repair": True,
        "all_refreshes_change": refresh_changed == len(tasks) * ROOT_COUNT,
        "all_tasks_have_four_distinct_refreshes": distinct_refresh_tasks
        == len(tasks),
    }
    if stage == "development":
        myopic = comparisons["myopic"]
        fixed = comparisons["fixed"]
        gates.update(
            {
                "aligned_differs_fixed_at_least_5": aligned_fixed_changes >= 5,
                "aligned_differs_shuffled_at_least_5": (
                    aligned_shuffled_changes >= 5
                ),
                "root_changes_vs_myopic_at_least_4": sum(
                    row["policies"]["nonmyopic"]["root_index"]
                    != row["policies"]["myopic"]["root_index"]
                    for row in records
                )
                >= 4,
                "nonmyopic_enabling_roots_at_least_12": sum(
                    row["policies"]["nonmyopic"]["root_role"] == "enabling"
                    for row in records
                )
                >= 12,
                "enabling_root_gain_vs_myopic_at_least_3": (
                    sum(
                        row["policies"]["nonmyopic"]["root_role"]
                        == "enabling"
                        for row in records
                    )
                    >= sum(
                        row["policies"]["myopic"]["root_role"] == "enabling"
                        for row in records
                    )
                    + 3
                ),
                "support_coverage_at_least_32": total("nonmyopic") >= 32,
                "gain_vs_myopic_at_least_3": myopic["gain"] >= 3,
                "wins_minus_losses_vs_myopic_at_least_3": (
                    myopic["wins"] - myopic["losses"] >= 3
                ),
                "losses_vs_myopic_at_most_2": myopic["losses"] <= 2,
                "sign_flip_p_vs_myopic_at_most_0_05": (
                    myopic["exact_one_sided_sign_flip_p"] <= 0.05
                ),
                "gain_vs_fixed_at_least_2": fixed["gain"] >= 2,
                "wins_minus_losses_vs_fixed_at_least_2": (
                    fixed["wins"] - fixed["losses"] >= 2
                ),
                "losses_vs_fixed_at_most_3": fixed["losses"] <= 3,
                "gain_vs_shuffled_at_least_4": (
                    comparisons["shuffled"]["gain"] >= 4
                ),
                "gain_vs_random_at_least_8": (
                    comparisons["random"]["gain"] >= 8
                ),
                "root_accuracy_at_least_0_70": accuracies["nonmyopic"]
                >= 0.70,
                "root_accuracy_gain_vs_myopic_at_least_0_05": (
                    accuracies["nonmyopic"] >= accuracies["myopic"] + 0.05
                ),
                "root_accuracy_gain_vs_fixed_at_least_0_05": (
                    accuracies["nonmyopic"] >= accuracies["fixed"] + 0.05
                ),
                "mean_final_answer_f1_at_least_0_50": (
                    sum(row["final_answer_token_f1"] for row in records)
                    / len(records)
                    >= 0.50
                ),
                "cost_at_most_3": usage["adapter_cost_usd"]
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
            "interface_version": INTERFACE_VERSION,
            "stage": stage,
            "task_ids": [row["task_id"] for row in records],
            "cohort_rows_materialized": cohort_rows_materialized,
            "selected_qualifying_count": len(records),
            "expected_calls_per_task": EXPECTED_CALLS_PER_TASK,
            "followup_actions_are_paragraph_links": True,
            "myopic_uses_aligned_receding_followup": True,
            "endpoint_hidden_until_outputs_frozen": True,
            "confirmation_endpoints_opened": False,
        },
        "metrics": {
            "policy_support_totals": {
                name: total(name)
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
            "enabling_root_counts": {
                name: sum(
                    row["policies"][name]["root_role"] == "enabling"
                    for row in records
                )
                for name in (
                    "nonmyopic",
                    "myopic",
                    "fixed",
                    "shuffled",
                    "random",
                )
            },
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
    parser.add_argument(
        "--train-shard", type=Path, action="append", required=True
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    if len(args.train_shard) != 2:
        parser.error("exactly two --train-shard values are required")
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_concurrency = min(config.openrouter_concurrency, 64)
    config.openrouter_max_output_tokens = min(
        config.openrouter_max_output_tokens, 2048
    )
    config.log_path = args.output_dir / "run.log"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    raw_path = private_dir / "RAW_RESPONSES.json"
    if args.stage == "serving":
        tasks, cohort_rows = load_tasks(
            args.train_shard, split_name="mechanics", count=1
        )
        config.openrouter_projected_cost_usd = 0.15
        config.openrouter_run_budget_usd = SERVING_COST_CAP
    else:
        tasks, cohort_rows = load_tasks(
            args.train_shard,
            split_name="development",
            count=DEVELOPMENT_SELECTION_COUNT,
        )
        config.openrouter_projected_cost_usd = 1.5
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
    output_path = (args.output_dir / args.stage.upper()).with_suffix(".json")
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
