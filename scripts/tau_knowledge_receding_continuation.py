#!/usr/bin/env python3
"""Evaluate focused receding continuation selection on tau-Knowledge trees."""

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

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.tau_knowledge_first_link_scorer import (
    DOCUMENT_EXCERPT_CHARS,
    MAX_SCORER_INPUT_CHARS,
    _selected_index,
    pairwise_ranking_points,
    parse_scores,
    scorer_messages,
)
from scripts.tau_knowledge_retrieval_opportunity import (
    BM25Corpus,
    FIRST_QUERY_COUNT,
    FOLLOWUP_QUERY_COUNT,
    FORMAL_IDS,
    GateExecutionError,
    HYPOTHESIS_COUNT,
    OPENING_PATTERN,
    SCHEMA_VERSION,
    SEARCH_TOP_K,
    TAU_COMMIT,
    _build_model,
    _checkpoint,
    _git_head,
    _serialize_results,
    _usage_snapshot,
    analyze_record,
)
from scripts.tau_knowledge_retrieval_opportunity_v2 import (
    OPPORTUNITY_IDS as V2_OPPORTUNITY_IDS,
    PREVIOUSLY_INSPECTED_IDS,
    SMOKE_IDS as V2_SMOKE_IDS,
    followup_messages,
    initial_messages,
    opening_messages,
    parse_followup,
    parse_initial,
    parse_opening,
)


INTERFACE_VERSION = 1
FRESH_SELECTION_SEED = 24337
RANDOM_CONTROL_SEED = 24338
SMOKE_IDS = ("task_018", "task_008")
FRESH_CONFIRMATION_IDS = (
    "task_025",
    "task_021",
    "task_028",
    "task_015",
    "task_016",
    "task_062",
    "task_048",
    "task_081",
    "task_029",
    "task_022",
    "task_019",
    "task_027",
    "task_012",
    "task_089",
    "task_023",
    "task_020",
    "task_007",
    "task_005",
    "task_017",
    "task_004",
)
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_IDS) * FIRST_QUERY_COUNT,
    "development": len(FORMAL_IDS) * FIRST_QUERY_COUNT,
    "confirmation": len(FRESH_CONFIRMATION_IDS)
    * (1 + 1 + FIRST_QUERY_COUNT + 1 + 1 + FIRST_QUERY_COUNT),
}


def _load_tau_data(
    tau_root: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    root = Path(tau_root)
    if _git_head(root) != TAU_COMMIT:
        raise ValueError("tau-Knowledge checkout commit does not match")
    domain = root / "data" / "tau2" / "domains" / "banking_knowledge"
    document_paths = sorted((domain / "documents").glob("*.json"))
    task_paths = sorted((domain / "tasks").glob("task_*.json"))
    if len(document_paths) != 698 or len(task_paths) != 97:
        raise ValueError("tau-Knowledge corpus shape does not match")
    documents = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in document_paths
    ]
    tasks = [
        json.loads(path.read_text(encoding="utf-8")) for path in task_paths
    ]
    if any(
        set(document) != {"id", "title", "content"} for document in documents
    ):
        raise ValueError("tau-Knowledge document schema does not match")
    return documents, tasks


def load_fresh_confirmation(
    tau_root: str | Path,
) -> tuple[list[dict[str, str]], list[dict[str, Any]]]:
    documents, tasks = _load_tau_data(tau_root)
    used = set(
        V2_SMOKE_IDS
        + V2_OPPORTUNITY_IDS
        + PREVIOUSLY_INSPECTED_IDS
    )
    eligible = sorted(
        task["id"]
        for task in tasks
        if not OPENING_PATTERN.search(task["user_scenario"]["instructions"])
        and task["id"] not in used
    )
    random.Random(FRESH_SELECTION_SEED).shuffle(eligible)
    if eligible != list(FRESH_CONFIRMATION_IDS):
        raise ValueError("fresh receding confirmation split does not reproduce")
    by_id = {task["id"]: task for task in tasks}
    selected = [by_id[task_id] for task_id in FRESH_CONFIRMATION_IDS]
    if any(not task.get("required_documents") for task in selected):
        raise ValueError("fresh confirmation task lacks endpoint")
    return documents, selected


def load_public_records(
    path: str | Path,
    *,
    stage: str,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]] | None,
    list[dict[str, Any]] | None,
]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    records = payload.get("records")
    if not isinstance(records, list):
        raise ValueError("public artifact lacks records")
    if stage == "serving_smoke":
        if tuple(record.get("task_id") for record in records[:2]) != SMOKE_IDS:
            raise ValueError("serving smoke artifact does not match")
        return records[:2], None, None
    if stage == "development":
        if tuple(record.get("task_id") for record in records) != FORMAL_IDS:
            raise ValueError("development artifact does not match")
        myopic = payload.get("myopic_scores")
        nonmyopic = payload.get("nonmyopic_scores")
        if not isinstance(myopic, list) or not isinstance(nonmyopic, list):
            raise ValueError("development artifact lacks frozen root scores")
        return records, myopic, nonmyopic
    raise ValueError("public records are only used for smoke or development")


def _clean_excerpt(value: str) -> str:
    return " ".join(value.split())[:DOCUMENT_EXCERPT_CHARS]


def compact_continuation_input(
    record: dict[str, Any],
    root_index: int,
) -> dict[str, Any]:
    branch = record["first_branches"][root_index]
    document_refs: dict[str, str] = {}
    catalog: list[dict[str, str]] = []

    def add_results(results: Sequence[dict[str, Any]]) -> list[str]:
        refs = []
        for result in results:
            document_id = str(result["id"])
            if document_id not in document_refs:
                ref = f"D{len(document_refs) + 1}"
                document_refs[document_id] = ref
                catalog.append(
                    {
                        "ref": ref,
                        "title": str(result["title"]),
                        "excerpt": _clean_excerpt(str(result["content"])),
                    }
                )
            refs.append(document_refs[document_id])
        return refs

    payload = {
        "customer_opening": record["opening"],
        "initial_information_needs": record[
            "initial_information_need_hypotheses"
        ],
        "realized_first_query": branch["query"],
        "first_result_refs": add_results(branch["first_results"]),
        "refreshed_information_needs": branch[
            "refreshed_information_need_hypotheses"
        ],
        "candidate_followups": [
            {
                "followup": index,
                "query": followup["query"],
                "result_refs": add_results(followup["results"]),
            }
            for index, followup in enumerate(branch["followups"], start=1)
        ],
        "document_catalog": catalog,
    }
    encoded = json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
    if len(encoded) > MAX_SCORER_INPUT_CHARS:
        raise ValueError("focused continuation input exceeds character cap")
    return payload


def _continuation_schema() -> dict[str, str]:
    schema: dict[str, str] = {}
    for index in range(1, FOLLOWUP_QUERY_COUNT + 1):
        schema[f"followup_{index}_score"] = (
            "canonical decimal digit string from 0 through 100"
        )
        schema[f"followup_{index}_rationale"] = "one short sentence"
    return schema


def continuation_messages(
    record: dict[str, Any],
    root_index: int,
) -> list[dict[str, str]]:
    payload = compact_continuation_input(record, root_index)
    return [
        {
            "role": "system",
            "content": (
                "Choose one next search after a realized banking-policy search. "
                "Ground every score in the customer request and supplied documents. "
                "Required-document labels and evaluation answers are hidden. Return "
                "one flat strict JSON object."
            ),
        },
        {
            "role": "user",
            "content": (
                "Score each candidate followup by the total distinct useful policy "
                "coverage in the union of the already acquired first results and "
                "that followup's results. Reward missing prerequisites, exceptions, "
                "eligibility rules, calculations, and procedures that resolve the "
                "customer's inferred needs. Treat generated information needs as "
                "fallible hypotheses. Do not reward document count, verbosity, "
                "surface relevance, or coverage duplicated by the first results. "
                "Use the full 0-100 range when warranted. Return exactly these keys: "
                + json.dumps(
                    _continuation_schema(),
                    ensure_ascii=True,
                    separators=(",", ":"),
                )
                + ". Retrieval data: "
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_continuation_scores(text: str) -> dict[str, Any]:
    payload = _parse_json_object(text)
    if set(payload) != set(_continuation_schema()):
        raise ValueError("continuation response has unexpected keys")
    scores = []
    rationales = []
    for index in range(1, FOLLOWUP_QUERY_COUNT + 1):
        score = payload[f"followup_{index}_score"]
        if (
            not isinstance(score, str)
            or not score.isdigit()
            or (len(score) > 1 and score.startswith("0"))
            or not 0 <= int(score) <= 100
        ):
            raise ValueError("continuation score is not canonical")
        rationale = payload[f"followup_{index}_rationale"]
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("continuation rationale must be nonempty")
        scores.append(int(score))
        rationales.append(" ".join(rationale.split()))
    return {"scores": scores, "rationales": rationales}


def _generate_fresh_trees(
    model: Any,
    corpus: BM25Corpus,
    tasks: Sequence[dict[str, Any]],
    config: Config,
    raw: dict[str, Any],
    raw_checkpoint_path: Path | None,
) -> list[dict[str, Any]]:
    opening_raw = model.chat_complete_messages_batched(
        [
            opening_messages(task["user_scenario"]["instructions"])
            for task in tasks
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["openings"] = opening_raw
    _checkpoint(raw_checkpoint_path, stage="confirmation", raw=raw)
    openings = [parse_opening(text) for text in opening_raw]

    initial_raw = model.chat_complete_messages_batched(
        [initial_messages(opening) for opening in openings],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial"] = initial_raw
    _checkpoint(raw_checkpoint_path, stage="confirmation", raw=raw)
    initial = [parse_initial(text) for text in initial_raw]
    first_results = [
        [corpus.search(query) for query in queries]
        for _hypotheses, queries in initial
    ]
    followup_keys = [
        (case_index, root_index)
        for case_index in range(len(tasks))
        for root_index in range(FIRST_QUERY_COUNT)
    ]
    followup_raw = model.chat_complete_messages_batched(
        [
            followup_messages(
                openings[case_index],
                initial[case_index][1][root_index],
                first_results[case_index][root_index],
            )
            for case_index, root_index in followup_keys
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["followups"] = followup_raw
    _checkpoint(raw_checkpoint_path, stage="confirmation", raw=raw)
    parsed_followups = [
        parse_followup(
            text,
            first_query=initial[case_index][1][root_index],
        )
        for (case_index, root_index), text in zip(
            followup_keys, followup_raw, strict=True
        )
    ]
    lookup = dict(zip(followup_keys, parsed_followups, strict=True))
    records = []
    for case_index, task in enumerate(tasks):
        hypotheses, queries = initial[case_index]
        branches = []
        for root_index, query in enumerate(queries):
            refreshed, next_queries = lookup[case_index, root_index]
            branches.append(
                {
                    "query": query,
                    "first_results": _serialize_results(
                        first_results[case_index][root_index]
                    ),
                    "refreshed_information_need_hypotheses": refreshed,
                    "followups": [
                        {
                            "query": next_query,
                            "results": _serialize_results(
                                corpus.search(next_query)
                            ),
                        }
                        for next_query in next_queries
                    ],
                }
            )
        records.append(
            {
                "task_id": task["id"],
                "opening": openings[case_index],
                "required_documents": list(task["required_documents"]),
                "initial_information_need_hypotheses": hypotheses,
                "first_branches": branches,
            }
        )
    return records


def _policy_diagnostic(
    record: dict[str, Any],
    myopic: dict[str, Any],
    nonmyopic: dict[str, Any],
    continuation: Sequence[dict[str, Any]],
    *,
    case_index: int,
) -> dict[str, Any]:
    endpoint = analyze_record(record)
    pair_values = endpoint["pair_counts"]
    myopic_root = _selected_index(myopic["scores"])
    nonmyopic_root = _selected_index(nonmyopic["scores"])
    focused_indices = [
        _selected_index(score["scores"]) for score in continuation
    ]
    joint_index = nonmyopic["best_followup_indices"][nonmyopic_root]
    rng = random.Random(RANDOM_CONTROL_SEED + case_index)
    random_root = rng.randrange(FIRST_QUERY_COUNT)
    random_followup = rng.randrange(FOLLOWUP_QUERY_COUNT)
    myopic_value = pair_values[myopic_root][focused_indices[myopic_root]]
    nonmyopic_value = pair_values[nonmyopic_root][
        focused_indices[nonmyopic_root]
    ]
    joint_value = pair_values[nonmyopic_root][joint_index]
    return {
        "task_id": record["task_id"],
        "myopic_root_index": myopic_root,
        "nonmyopic_root_index": nonmyopic_root,
        "focused_followup_indices": focused_indices,
        "myopic_receding_value": myopic_value,
        "nonmyopic_receding_value": nonmyopic_value,
        "nonmyopic_joint_value": joint_value,
        "random_strategy_root_index": random_root,
        "random_strategy_followup_index": random_followup,
        "random_strategy_value": pair_values[random_root][random_followup],
        "nonmyopic_advantage_over_myopic": nonmyopic_value - myopic_value,
        "nonmyopic_advantage_over_joint": nonmyopic_value - joint_value,
        "nonmyopic_advantage_over_random": nonmyopic_value
        - pair_values[random_root][random_followup],
        "selected_root_oracle_tail": max(pair_values[nonmyopic_root]),
        "selected_root_continuation_loss": max(pair_values[nonmyopic_root])
        - nonmyopic_value,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    continuation_scores: Sequence[Sequence[dict[str, Any]]],
    usage: dict[str, Any],
    *,
    stage: str,
    myopic_scores: Sequence[dict[str, Any]] | None,
    nonmyopic_scores: Sequence[dict[str, Any]] | None,
) -> dict[str, Any]:
    root_rows = []
    for record, case_scores in zip(
        records, continuation_scores, strict=True
    ):
        pair_values = analyze_record(record)["pair_counts"]
        for root_index, (values, score) in enumerate(
            zip(pair_values, case_scores, strict=True)
        ):
            points, comparable = pairwise_ranking_points(
                score["scores"], values
            )
            selected = _selected_index(score["scores"])
            oracle = max(values)
            root_rows.append(
                {
                    "task_id": record["task_id"],
                    "root_index": root_index,
                    "pair_values": values,
                    "scores": score["scores"],
                    "selected_followup_index": selected,
                    "selected_value": values[selected],
                    "oracle_value": oracle,
                    "regret": oracle - values[selected],
                    "pairwise_points": points,
                    "pairwise_comparable_count": comparable,
                }
            )
    comparable = sum(row["pairwise_comparable_count"] for row in root_rows)
    points = sum(row["pairwise_points"] for row in root_rows)
    accuracy = points / comparable if comparable else 0.0
    hit_count = sum(row["regret"] == 0 for row in root_rows)
    total_regret = sum(row["regret"] for row in root_rows)
    expected_cases = (
        len(SMOKE_IDS)
        if stage == "serving_smoke"
        else len(FORMAL_IDS)
        if stage == "development"
        else len(FRESH_CONFIRMATION_IDS)
    )
    gates = {
        "all_cases_complete": len(records) == expected_cases,
        "all_focused_scores_complete": len(root_rows)
        == expected_cases * FIRST_QUERY_COUNT,
        "exact_physical_request_count": int(usage["physical_requests"])
        == EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
    }
    if stage == "serving_smoke":
        gates.update(
            {
                "scores_vary_on_at_least_8_of_10_roots": sum(
                    len(set(row["scores"])) >= 2 for row in root_rows
                )
                >= 8,
                "focused_pairwise_accuracy_at_least_0_55": accuracy >= 0.55,
                "focused_optimal_followup_at_least_7_of_10": hit_count >= 7,
            }
        )

    policy_rows = []
    root_ranking_summary: dict[str, Any] = {}
    if myopic_scores is not None and nonmyopic_scores is not None:
        policy_rows = [
            _policy_diagnostic(
                record,
                myopic,
                nonmyopic,
                scores,
                case_index=index,
            )
            for index, (record, myopic, nonmyopic, scores) in enumerate(
                zip(
                    records,
                    myopic_scores,
                    nonmyopic_scores,
                    continuation_scores,
                    strict=True,
                )
            )
        ]
        myopic_advantages = [
            row["nonmyopic_advantage_over_myopic"] for row in policy_rows
        ]
        random_advantages = [
            row["nonmyopic_advantage_over_random"] for row in policy_rows
        ]
        joint_improvement = sum(
            row["nonmyopic_advantage_over_joint"] for row in policy_rows
        )
        selected_loss = sum(
            row["selected_root_continuation_loss"] for row in policy_rows
        )
        if stage == "development":
            gates.update(
                {
                    "comparable_followup_pairs_at_least_200": comparable >= 200,
                    "focused_pairwise_accuracy_at_least_0_60": accuracy >= 0.60,
                    "focused_optimal_followup_rate_at_least_0_72": (
                        hit_count / len(root_rows) >= 0.72
                    ),
                    "focused_total_regret_at_most_28": total_regret <= 28,
                    "selected_root_continuation_loss_at_most_5": (
                        selected_loss <= 5
                    ),
                    "end_to_end_wins_over_myopic_at_least_3": sum(
                        value > 0 for value in myopic_advantages
                    )
                    >= 3,
                    "end_to_end_losses_to_myopic_at_most_2": sum(
                        value < 0 for value in myopic_advantages
                    )
                    <= 2,
                    "end_to_end_total_advantage_over_myopic_at_least_3": (
                        sum(myopic_advantages) >= 3
                    ),
                    "focused_improvement_over_joint_at_least_5": (
                        joint_improvement >= 5
                    ),
                }
            )
        elif stage == "confirmation":
            root_rows_scored = [
                analyze_record(record) for record in records
            ]
            myopic_root_points = 0.0
            nonmyopic_root_points = 0.0
            root_comparable = 0
            for endpoint, myopic, nonmyopic in zip(
                root_rows_scored,
                myopic_scores,
                nonmyopic_scores,
                strict=True,
            ):
                values = [max(row) for row in endpoint["pair_counts"]]
                my_points, count = pairwise_ranking_points(
                    myopic["scores"], values
                )
                non_points, non_count = pairwise_ranking_points(
                    nonmyopic["scores"], values
                )
                if count != non_count:
                    raise ValueError("root comparison counts do not align")
                myopic_root_points += my_points
                nonmyopic_root_points += non_points
                root_comparable += count
            myopic_root_accuracy = (
                myopic_root_points / root_comparable
                if root_comparable
                else 0.0
            )
            nonmyopic_root_accuracy = (
                nonmyopic_root_points / root_comparable
                if root_comparable
                else 0.0
            )
            root_ranking_summary = {
                "root_pairwise_comparable_count": root_comparable,
                "myopic_root_pairwise_accuracy": myopic_root_accuracy,
                "nonmyopic_root_pairwise_accuracy": nonmyopic_root_accuracy,
                "root_pairwise_accuracy_gain": (
                    nonmyopic_root_accuracy - myopic_root_accuracy
                ),
            }
            gates.update(
                {
                    "root_comparable_pairs_at_least_50": root_comparable >= 50,
                    "nonmyopic_root_accuracy_at_least_0_60": (
                        nonmyopic_root_accuracy >= 0.60
                    ),
                    "root_accuracy_gain_at_least_0_05": (
                        nonmyopic_root_accuracy - myopic_root_accuracy >= 0.05
                    ),
                    "continuation_comparable_pairs_at_least_200": (
                        comparable >= 200
                    ),
                    "focused_pairwise_accuracy_at_least_0_60": accuracy >= 0.60,
                    "focused_optimal_followup_rate_at_least_0_70": (
                        hit_count / len(root_rows) >= 0.70
                    ),
                    "focused_mean_regret_at_most_0_30": (
                        total_regret / len(root_rows) <= 0.30
                    ),
                    "selected_root_continuation_loss_at_most_5": (
                        selected_loss <= 5
                    ),
                    "end_to_end_wins_over_myopic_at_least_4": sum(
                        value > 0 for value in myopic_advantages
                    )
                    >= 4,
                    "end_to_end_losses_to_myopic_at_most_2": sum(
                        value < 0 for value in myopic_advantages
                    )
                    <= 2,
                    "end_to_end_total_advantage_over_myopic_at_least_4": (
                        sum(myopic_advantages) >= 4
                    ),
                    "end_to_end_wins_over_random_at_least_6": sum(
                        value > 0 for value in random_advantages
                    )
                    >= 6,
                    "end_to_end_losses_to_random_at_most_4": sum(
                        value < 0 for value in random_advantages
                    )
                    <= 4,
                    "end_to_end_total_advantage_over_random_at_least_5": (
                        sum(random_advantages) >= 5
                    ),
                    "focused_improvement_over_joint_at_least_4": (
                        joint_improvement >= 4
                    ),
                }
            )
    gates["all_pass"] = all(gates.values())
    summary: dict[str, Any] = {
        "num_cases": len(records),
        "root_diagnostics": root_rows,
        "focused_pairwise_accuracy": accuracy,
        "focused_pairwise_comparable_count": comparable,
        "focused_optimal_followup_count": hit_count,
        "focused_optimal_followup_rate": (
            hit_count / len(root_rows) if root_rows else 0.0
        ),
        "focused_total_regret": total_regret,
        "focused_mean_regret": (
            total_regret / len(root_rows) if root_rows else 0.0
        ),
        "policy_diagnostics": policy_rows,
        "gates": gates,
    }
    if policy_rows:
        for name, key in (
            ("myopic", "nonmyopic_advantage_over_myopic"),
            ("joint", "nonmyopic_advantage_over_joint"),
            ("random", "nonmyopic_advantage_over_random"),
        ):
            values = [row[key] for row in policy_rows]
            summary[f"end_to_end_{name}_win_count"] = sum(
                value > 0 for value in values
            )
            summary[f"end_to_end_{name}_loss_count"] = sum(
                value < 0 for value in values
            )
            summary[f"end_to_end_{name}_tie_count"] = sum(
                value == 0 for value in values
            )
            summary[f"end_to_end_total_advantage_over_{name}"] = sum(values)
        summary["selected_root_continuation_loss_total"] = sum(
            row["selected_root_continuation_loss"] for row in policy_rows
        )
    summary.update(root_ranking_summary)
    if stage == "confirmation" and policy_rows:
        summary["fresh_selection_seed"] = FRESH_SELECTION_SEED
        summary["random_control_seed"] = RANDOM_CONTROL_SEED
    return summary


def run_gate(
    config: Config,
    *,
    stage: str,
    tau_root: str | Path,
    input_artifact: str | Path | None,
    raw_checkpoint_path: Path | None,
) -> dict[str, Any]:
    model = _build_model(config)
    raw: dict[str, Any] = {}
    try:
        if stage in {"serving_smoke", "development"}:
            if input_artifact is None:
                raise ValueError(f"{stage} requires --input-artifact")
            records, myopic_scores, nonmyopic_scores = load_public_records(
                input_artifact, stage=stage
            )
        elif stage == "confirmation":
            documents, tasks = load_fresh_confirmation(tau_root)
            records = _generate_fresh_trees(
                model,
                BM25Corpus(documents),
                tasks,
                config,
                raw,
                raw_checkpoint_path,
            )
            myopic_raw = model.chat_complete_messages_batched(
                [
                    scorer_messages(record, include_followups=False)
                    for record in records
                ],
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            raw["myopic_scores"] = myopic_raw
            _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
            myopic_scores = [
                parse_scores(text, include_followups=False)
                for text in myopic_raw
            ]
            nonmyopic_raw = model.chat_complete_messages_batched(
                [
                    scorer_messages(record, include_followups=True)
                    for record in records
                ],
                temperature=0.0,
                block_size=config.batched_block_size,
                max_new_tokens=config.openrouter_max_output_tokens,
            )
            raw["nonmyopic_scores"] = nonmyopic_raw
            _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
            nonmyopic_scores = [
                parse_scores(text, include_followups=True)
                for text in nonmyopic_raw
            ]
        else:
            raise ValueError("unknown receding continuation stage")

        focused_keys = [
            (case_index, root_index)
            for case_index in range(len(records))
            for root_index in range(FIRST_QUERY_COUNT)
        ]
        focused_raw = model.chat_complete_messages_batched(
            [
                continuation_messages(records[case_index], root_index)
                for case_index, root_index in focused_keys
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["focused_continuations"] = focused_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        parsed = [
            parse_continuation_scores(text) for text in focused_raw
        ]
        continuation_scores = [
            parsed[
                case_index
                * FIRST_QUERY_COUNT : (case_index + 1) * FIRST_QUERY_COUNT
            ]
            for case_index in range(len(records))
        ]
        usage = _usage_snapshot(model)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    summary = summarize(
        records,
        continuation_scores,
        usage,
        stage=stage,
        myopic_scores=myopic_scores,
        nonmyopic_scores=nonmyopic_scores,
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "interface_version": INTERFACE_VERSION,
            "source_repository": "https://github.com/sierra-research/tau2-bench",
            "source_commit": TAU_COMMIT,
            "task_ids": [record["task_id"] for record in records],
            "fresh_selection_seed": (
                FRESH_SELECTION_SEED if stage == "confirmation" else None
            ),
            "random_control_seed": RANDOM_CONTROL_SEED,
            "first_query_count": FIRST_QUERY_COUNT,
            "followup_query_count": FOLLOWUP_QUERY_COUNT,
            "search_top_k": SEARCH_TOP_K,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "focused_one_root_per_call": True,
            "required_documents_hidden_from_model": True,
            "reasoning_disabled": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "myopic_scores": myopic_scores,
        "nonmyopic_scores": nonmyopic_scores,
        "continuation_scores": continuation_scores,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--tau-root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "development", "confirmation"),
        required=True,
    )
    parser.add_argument("--input-artifact", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.10
        config.openrouter_run_budget_usd = 0.50
    elif args.stage == "development":
        config.openrouter_projected_cost_usd = 1.00
        config.openrouter_run_budget_usd = 3.00
    else:
        config.openrouter_projected_cost_usd = 3.50
        config.openrouter_run_budget_usd = 8.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = {
        "serving_smoke": "SERVING_SMOKE.json",
        "development": "DEVELOPMENT.json",
        "confirmation": "CONFIRMATION.json",
    }[args.stage]
    try:
        payload = run_gate(
            config,
            stage=args.stage,
            tau_root=args.tau_root,
            input_artifact=args.input_artifact,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        (args.output_dir / f"{args.stage.upper()}_FAILURE.json").write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output_path = args.output_dir / output_name
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
    main()
