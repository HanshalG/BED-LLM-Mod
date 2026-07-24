#!/usr/bin/env python3
"""Gate path-dependent answer beliefs under adaptive MuSiQue retrieval."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.musique_answer_belief_bridge_gate import (
    BELIEF_SIZE,
    FORMAL_IDS as MENU_FORMAL_IDS,
    RESERVE_IDS as MENU_RESERVE_IDS,
    SMOKE_IDS as MENU_SMOKE_IDS,
    _belief_schema,
    _build_models,
    _checkpoint,
    _normalized,
    _sha256,
    _usage_snapshot,
    equivalence_messages,
    gold_doc_ids,
    parse_belief_response,
    parse_equivalence,
    truth_probability,
)
from scripts.musique_chain_transition_gate import (
    DATA_SHA256,
    FORMAL_IDS as CHAIN_FORMAL_IDS,
    PREVIOUS_IDS as CHAIN_PREVIOUS_IDS,
    SMOKE_IDS as CHAIN_SMOKE_IDS,
    is_eligible_row,
)


SCHEMA_VERSION = 1
SELECTION_SEED = 24328
FIRST_ACTION_COUNT = 6
DIRECT_QUERY_COUNT = 3
BRIDGE_QUERY_COUNT = 3
SECOND_ACTION_COUNT = 4
SMOKE_IDS = (
    "2hop__95773_51329",
    "2hop__94507_654855",
)
FORMAL_IDS = (
    "2hop__564984_139312",
    "2hop__56307_604644",
    "2hop__64175_90973",
    "2hop__446324_620302",
    "2hop__559076_55984",
    "2hop__826864_17335",
)
RESERVE_IDS = (
    "2hop__142699_67465",
    "2hop__51760_122023",
    "2hop__40970_90367",
    "2hop__2299_38663",
    "2hop__106864_80460",
    "2hop__153573_109006",
)
EXPECTED_REQUESTS = {
    "serving_smoke": len(SMOKE_IDS)
    * (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    ),
    "opportunity": len(FORMAL_IDS)
    * (
        1
        + FIRST_ACTION_COUNT
        + FIRST_ACTION_COUNT * SECOND_ACTION_COUNT
        + 1
        + 1
    ),
}


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def selected_ids(stage: str) -> tuple[str, ...]:
    if stage == "serving_smoke":
        return SMOKE_IDS
    if stage == "opportunity":
        return FORMAL_IDS
    raise ValueError("stage must be serving_smoke or opportunity")


def load_selected_rows(data_path: str | Path, stage: str) -> list[dict[str, Any]]:
    path = Path(data_path)
    if _sha256(path) != DATA_SHA256:
        raise ValueError("MuSiQue data hash does not match the frozen dev artifact")
    prior_ids = set(
        CHAIN_PREVIOUS_IDS
        + CHAIN_SMOKE_IDS
        + CHAIN_FORMAL_IDS
        + MENU_SMOKE_IDS
        + MENU_FORMAL_IDS
        + MENU_RESERVE_IDS
    )
    frozen_all = SMOKE_IDS + FORMAL_IDS + RESERVE_IDS
    rows_by_id: dict[str, dict[str, Any]] = {}
    eligible_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if is_eligible_row(row) and row_id not in prior_ids:
                eligible_ids.append(row_id)
            if row_id in frozen_all:
                rows_by_id[row_id] = row
    reproduced = tuple(
        str(value)
        for value in np.random.default_rng(SELECTION_SEED).choice(
            sorted(eligible_ids),
            size=len(frozen_all),
            replace=False,
        )
    )
    if reproduced != frozen_all:
        raise ValueError("frozen target-blind MuSiQue selection does not reproduce")
    wanted = selected_ids(stage)
    missing = [row_id for row_id in wanted if row_id not in rows_by_id]
    if missing:
        raise ValueError(f"frozen MuSiQue rows are missing: {missing}")
    return [rows_by_id[row_id] for row_id in wanted]


def document_map(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    row_seed = int.from_bytes(
        hashlib.sha256(
            f"{SELECTION_SEED}:{row['id']}".encode("utf-8")
        ).digest()[:8],
        "big",
    )
    order = np.random.default_rng(row_seed).permutation(len(row["paragraphs"]))
    return {
        f"d{doc_index + 1:02d}": row["paragraphs"][int(paragraph_index)]
        for doc_index, paragraph_index in enumerate(order)
    }


def _tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


class BM25Index:
    """Small deterministic BM25 index over one sealed MuSiQue context."""

    def __init__(self, documents: dict[str, dict[str, Any]]) -> None:
        self.doc_ids = list(documents)
        self.tokens = [
            _tokens(
                str(documents[doc_id]["title"])
                + " "
                + str(documents[doc_id]["paragraph_text"])
            )
            for doc_id in self.doc_ids
        ]
        self.term_counts = [Counter(tokens) for tokens in self.tokens]
        self.average_length = float(np.mean([len(tokens) for tokens in self.tokens]))
        document_frequency: Counter[str] = Counter()
        for tokens in self.tokens:
            document_frequency.update(set(tokens))
        count = len(self.tokens)
        self.idf = {
            token: math.log(
                1.0 + (count - frequency + 0.5) / (frequency + 0.5)
            )
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        query_tokens = _tokens(query)
        values: list[float] = []
        for tokens, counts in zip(
            self.tokens, self.term_counts, strict=True
        ):
            score = 0.0
            length_scale = 1.0 - 0.75 + 0.75 * (
                len(tokens) / self.average_length
            )
            for token in query_tokens:
                frequency = counts[token]
                if frequency == 0:
                    continue
                score += self.idf.get(token, 0.0) * (
                    frequency * 2.5
                    / (frequency + 1.5 * length_scale)
                )
            values.append(score)
        return values

    def retrieve(self, query: str, *, exclude: Sequence[str] = ()) -> str:
        excluded = set(exclude)
        ranked = sorted(
            range(len(self.doc_ids)),
            key=lambda index: (-self.scores(query)[index], index),
        )
        for index in ranked:
            if self.doc_ids[index] not in excluded:
                return self.doc_ids[index]
        raise ValueError("retrieval excluded every document")


def retrieval_messages(
    row: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    *,
    opened_doc_ids: Sequence[str] = (),
    previous_belief: Sequence[dict[str, Any]] | None = None,
    request_queries: bool = False,
) -> list[dict[str, str]]:
    payload: dict[str, Any] = {
        "question": row["question"],
        "opened_search_results_in_order": [
            {
                "title": str(documents[doc_id]["title"]),
                "text": str(documents[doc_id]["paragraph_text"]),
            }
            for doc_id in opened_doc_ids
        ],
    }
    if previous_belief is not None:
        payload["previous_answer_belief_for_context"] = list(previous_belief)
    schema: dict[str, Any] = {"belief": _belief_schema()}
    if request_queries and not opened_doc_ids:
        schema["direct_queries"] = [
            f"direct search query {index + 1}"
            for index in range(DIRECT_QUERY_COUNT)
        ]
        schema["bridge_queries"] = [
            f"bridge search query {index + 1}"
            for index in range(BRIDGE_QUERY_COUNT)
        ]
    elif request_queries:
        schema["next_queries"] = [
            f"observation-conditioned search query {index + 1}"
            for index in range(SECOND_ACTION_COUNT)
        ]
    if not opened_doc_ids:
        query_instruction = (
            f" Also propose exactly {DIRECT_QUERY_COUNT} diverse direct queries "
            "intended to retrieve a passage that answers the final question in "
            "one search, and exactly "
            f"{BRIDGE_QUERY_COUNT} diverse bridge queries intended to retrieve "
            "an intermediate entity or relation needed to formulate a later "
            "search. Do not assume document titles are visible."
        )
    elif request_queries:
        query_instruction = (
            f" Also propose exactly {SECOND_ACTION_COUNT} diverse follow-up "
            "queries that use concrete information newly visible in the search "
            "result to retrieve missing evidence for the final answer. Do not "
            "repeat an earlier query."
        )
    else:
        query_instruction = ""
    return [
        {
            "role": "system",
            "content": (
                "Maintain a target-blind probabilistic belief over final answers "
                "while searching a sealed corpus. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Build the answer belief afresh from only the question and opened "
                f"search results. Return exactly {BELIEF_SIZE} distinct concrete "
                "final-answer strings and probabilities summing to 1. Do not use "
                "outside factual memory and do not treat an intermediate bridge "
                "entity as the final answer."
                f"{query_instruction} Return exactly this schema: "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_query_response(
    text: str,
    *,
    stage: str,
) -> tuple[list[dict[str, Any]], list[str]]:
    belief, _unused = parse_belief_response(
        text,
        available_doc_ids=(),
    )
    payload = _parse_json_object(text)
    if stage == "initial":
        direct = payload.get("direct_queries")
        bridge = payload.get("bridge_queries")
        if (
            not isinstance(direct, list)
            or len(direct) != DIRECT_QUERY_COUNT
            or not isinstance(bridge, list)
            or len(bridge) != BRIDGE_QUERY_COUNT
        ):
            raise ValueError("initial query groups have the wrong size")
        queries = [*direct, *bridge]
    elif stage == "followup":
        queries = payload.get("next_queries")
        if not isinstance(queries, list) or len(queries) != SECOND_ACTION_COUNT:
            raise ValueError("next_queries has the wrong size")
    else:
        raise ValueError("query stage must be initial or followup")
    if (
        any(not isinstance(query, str) or not _normalized(query) for query in queries)
        or len({_normalized(query) for query in queries}) != len(queries)
    ):
        raise ValueError("queries must be distinct nonempty strings")
    return belief, [" ".join(query.split()) for query in queries]


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    one = [
        float(branch["truth_probability"])
        for branch in record["first_branches"]
    ]
    pair_values = [
        [
            float(second["truth_probability"])
            for second in branch["second_branches"]
        ]
        for branch in record["first_branches"]
    ]
    greedy_first = max(
        range(len(one)), key=lambda index: (one[index], -index)
    )
    oracle_first, oracle_second = max(
        (
            (first_index, second_index)
            for first_index in range(FIRST_ACTION_COUNT)
            for second_index in range(SECOND_ACTION_COUNT)
        ),
        key=lambda pair: (
            pair_values[pair[0]][pair[1]],
            -pair[0],
            -pair[1],
        ),
    )
    greedy_continuation = max(pair_values[greedy_first])
    gold_root, gold_second = record["gold_doc_ids"]
    root_candidates = [
        index
        for index, branch in enumerate(record["first_branches"])
        if branch["retrieved_doc_id"] == gold_root
    ]
    gold_pairs = [
        (first_index, second_index)
        for first_index in root_candidates
        for second_index, second in enumerate(
            record["first_branches"][first_index]["second_branches"]
        )
        if second["retrieved_doc_id"] == gold_second
    ]
    oracle_is_gold = (oracle_first, oracle_second) in gold_pairs
    replay_gap = abs(
        float(record["replay_truth_probability"])
        - pair_values[
            int(record["replay_indices"][0])
        ][int(record["replay_indices"][1])]
    )
    return {
        "distinct_first_retrieval_count": len(
            {
                branch["retrieved_doc_id"]
                for branch in record["first_branches"]
            }
        ),
        "gold_root_retrieved": bool(root_candidates),
        "gold_second_retrieved_after_gold_root": bool(gold_pairs),
        "gold_pair_is_oracle": oracle_is_gold,
        "greedy_first_index": greedy_first,
        "oracle_first_index": oracle_first,
        "oracle_second_index": oracle_second,
        "oracle_first_differs_from_greedy": oracle_first != greedy_first,
        "best_one_step_truth_probability": max(one),
        "oracle_pair_truth_probability": pair_values[oracle_first][oracle_second],
        "greedy_continuation_truth_probability": greedy_continuation,
        "pair_gain_over_best_one_step": (
            pair_values[oracle_first][oracle_second] - max(one)
        ),
        "nonmyopic_probability_gap": (
            pair_values[oracle_first][oracle_second] - greedy_continuation
        ),
        "replay_truth_probability_gap": replay_gap,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    diagnostics = [analyze_record(record) for record in records]
    base_gates = {
        "all_cases_complete": len(records) == len(selected_ids(stage)),
        "exact_physical_request_count": int(usage["physical_requests"])
        == EXPECTED_REQUESTS[stage],
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
        "all_replay_gaps_finite": all(
            math.isfinite(row["replay_truth_probability_gap"])
            for row in diagnostics
        ),
    }
    if stage == "serving_smoke":
        gates = {
            **base_gates,
            "all_beliefs_size_8": all(
                record["all_belief_sizes_valid"] for record in records
            ),
            "at_least_two_distinct_first_retrievals_each": all(
                row["distinct_first_retrieval_count"] >= 2
                for row in diagnostics
            ),
        }
        gates["all_pass"] = all(gates.values())
        return {
            "num_cases": len(records),
            "case_diagnostics": diagnostics,
            "gates": gates,
        }

    root_count = sum(row["gold_root_retrieved"] for row in diagnostics)
    continuation_count = sum(
        row["gold_second_retrieved_after_gold_root"] for row in diagnostics
    )
    gold_oracle_count = sum(row["gold_pair_is_oracle"] for row in diagnostics)
    first_differs_count = sum(
        row["oracle_first_differs_from_greedy"] for row in diagnostics
    )
    pair_gain_count = sum(
        row["pair_gain_over_best_one_step"] >= 0.10 for row in diagnostics
    )
    gap_count = sum(
        row["nonmyopic_probability_gap"] >= 0.10 for row in diagnostics
    )
    mean_initial = float(
        np.mean([record["initial_truth_probability"] for record in records])
    )
    mean_distinct = float(
        np.mean(
            [row["distinct_first_retrieval_count"] for row in diagnostics]
        )
    )
    mean_gain = float(
        np.mean([row["pair_gain_over_best_one_step"] for row in diagnostics])
    )
    mean_gap = float(
        np.mean([row["nonmyopic_probability_gap"] for row in diagnostics])
    )
    replay_gaps = [
        row["replay_truth_probability_gap"] for row in diagnostics
    ]
    summary = {
        "num_cases": len(records),
        "mean_initial_truth_probability": mean_initial,
        "mean_distinct_first_retrieval_count": mean_distinct,
        "gold_root_retrieval_count": root_count,
        "gold_continuation_retrieval_count": continuation_count,
        "gold_pair_is_oracle_count": gold_oracle_count,
        "oracle_first_differs_from_greedy_count": first_differs_count,
        "pair_gain_at_least_0_10_count": pair_gain_count,
        "nonmyopic_gap_at_least_0_10_count": gap_count,
        "mean_pair_gain_over_best_one_step": mean_gain,
        "mean_nonmyopic_probability_gap": mean_gap,
        "mean_replay_truth_probability_gap": float(np.mean(replay_gaps)),
        "max_replay_truth_probability_gap": max(replay_gaps),
        "case_diagnostics": [
            {"row_id": record["row_id"], **diagnostic}
            for record, diagnostic in zip(records, diagnostics, strict=True)
        ],
    }
    gates = {
        **base_gates,
        "initial_belief_not_saturated": mean_initial <= 0.25,
        "mean_distinct_first_retrievals_at_least_3": mean_distinct >= 3.0,
        "gold_root_retrieval_count_at_least_4": root_count >= 4,
        "gold_continuation_retrieval_count_at_least_4": continuation_count >= 4,
        "gold_pair_is_oracle_count_at_least_2": gold_oracle_count >= 2,
        "oracle_first_differs_count_at_least_2": first_differs_count >= 2,
        "pair_gain_count_at_least_3": pair_gain_count >= 3,
        "nonmyopic_gap_count_at_least_2": gap_count >= 2,
        "mean_pair_gain_at_least_0_10": mean_gain >= 0.10,
        "mean_nonmyopic_gap_at_least_0_07": mean_gap >= 0.07,
        "mean_replay_gap_at_most_0_10": float(np.mean(replay_gaps)) <= 0.10,
        "max_replay_gap_at_most_0_25": max(replay_gaps) <= 0.25,
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_selected_rows(data_path, stage)
    if len(config.model_pairs) != 1:
        raise ValueError("adaptive retrieval gate requires one model pair")
    generator, judge = _build_models(config)
    raw: dict[str, Any] = {}
    try:
        documents = [document_map(row) for row in rows]
        indices = [BM25Index(doc_map) for doc_map in documents]
        initial_raw = generator.chat_complete_messages_batched(
            [
                retrieval_messages(row, doc_map, request_queries=True)
                for row, doc_map in zip(rows, documents, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        initial = [
            parse_query_response(text, stage="initial")
            for text in initial_raw
        ]

        first_keys: list[tuple[int, int]] = []
        first_queries: dict[tuple[int, int], str] = {}
        first_doc_ids: dict[tuple[int, int], str] = {}
        first_messages = []
        for row_index, (belief, queries) in enumerate(initial):
            shuffled = list(queries)
            np.random.default_rng(SELECTION_SEED * 1000 + row_index).shuffle(
                shuffled
            )
            initial[row_index] = (belief, shuffled)
            for first_index, query in enumerate(shuffled):
                key = (row_index, first_index)
                doc_id = indices[row_index].retrieve(query)
                first_keys.append(key)
                first_queries[key] = query
                first_doc_ids[key] = doc_id
                first_messages.append(
                    retrieval_messages(
                        rows[row_index],
                        documents[row_index],
                        opened_doc_ids=(doc_id,),
                        previous_belief=belief,
                        request_queries=True,
                    )
                )
        first_raw = generator.chat_complete_messages_batched(
            first_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["first_branches"] = first_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        first_parsed = [
            parse_query_response(text, stage="followup")
            for text in first_raw
        ]
        first_lookup: dict[
            tuple[int, int], tuple[list[dict[str, Any]], list[str]]
        ] = {}
        for key, (belief, queries) in zip(
            first_keys, first_parsed, strict=True
        ):
            row_index, first_index = key
            shuffled = list(queries)
            np.random.default_rng(
                SELECTION_SEED * 1_000_000
                + row_index * 100
                + first_index
            ).shuffle(shuffled)
            first_lookup[key] = (belief, shuffled)

        second_keys: list[tuple[int, int, int]] = []
        second_queries: dict[tuple[int, int, int], str] = {}
        second_doc_ids: dict[tuple[int, int, int], str] = {}
        second_messages = []
        for row_index, first_index in first_keys:
            first_belief, queries = first_lookup[row_index, first_index]
            first_doc_id = first_doc_ids[row_index, first_index]
            for second_index, query in enumerate(queries):
                key = (row_index, first_index, second_index)
                second_doc_id = indices[row_index].retrieve(
                    query, exclude=(first_doc_id,)
                )
                second_keys.append(key)
                second_queries[key] = query
                second_doc_ids[key] = second_doc_id
                second_messages.append(
                    retrieval_messages(
                        rows[row_index],
                        documents[row_index],
                        opened_doc_ids=(first_doc_id, second_doc_id),
                        previous_belief=first_belief,
                    )
                )
        second_raw = generator.chat_complete_messages_batched(
            second_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["second_branches"] = second_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        second_parsed = [
            parse_belief_response(
                text,
                available_doc_ids=(),
            )[0]
            for text in second_raw
        ]
        second_lookup = {
            key: belief
            for key, belief in zip(second_keys, second_parsed, strict=True)
        }

        replay_keys = [
            (row_index, 0, 0) for row_index in range(len(rows))
        ]
        replay_messages = []
        for row_index, first_index, second_index in replay_keys:
            first_belief, _queries = first_lookup[row_index, first_index]
            replay_messages.append(
                retrieval_messages(
                    rows[row_index],
                    documents[row_index],
                    opened_doc_ids=(
                        first_doc_ids[row_index, first_index],
                        second_doc_ids[
                            row_index, first_index, second_index
                        ],
                    ),
                    previous_belief=first_belief,
                )
            )
        replay_raw = generator.chat_complete_messages_batched(
            replay_messages,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["replays"] = replay_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        replay_beliefs = [
            parse_belief_response(text, available_doc_ids=())[0]
            for text in replay_raw
        ]

        states_by_row: list[list[tuple[str, Sequence[dict[str, Any]]]]] = []
        for row_index, (initial_belief, _queries) in enumerate(initial):
            states: list[tuple[str, Sequence[dict[str, Any]]]] = [
                ("initial", initial_belief)
            ]
            for first_index in range(FIRST_ACTION_COUNT):
                first_belief, _next_queries = first_lookup[
                    row_index, first_index
                ]
                states.append((f"a{first_index}", first_belief))
                for second_index in range(SECOND_ACTION_COUNT):
                    states.append(
                        (
                            f"a{first_index}>b{second_index}",
                            second_lookup[
                                row_index, first_index, second_index
                            ],
                        )
                    )
            states.append(("replay:a0>b0", replay_beliefs[row_index]))
            states_by_row.append(states)
        judge_raw = judge.chat_complete_messages_batched(
            [
                equivalence_messages(str(row["answer"]), states)
                for row, states in zip(rows, states_by_row, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["equivalence"] = judge_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        equivalences = [
            parse_equivalence(text, states)
            for text, states in zip(judge_raw, states_by_row, strict=True)
        ]

        records = []
        for row_index, row in enumerate(rows):
            initial_belief, _queries = initial[row_index]
            equivalence = equivalences[row_index]
            first_branches = []
            for first_index in range(FIRST_ACTION_COUNT):
                first_belief, _next_queries = first_lookup[
                    row_index, first_index
                ]
                first_branches.append(
                    {
                        "query": first_queries[row_index, first_index],
                        "retrieved_doc_id": first_doc_ids[
                            row_index, first_index
                        ],
                        "truth_probability": truth_probability(
                            first_belief, equivalence[f"a{first_index}"]
                        ),
                        "second_branches": [
                            {
                                "query": second_queries[
                                    row_index, first_index, second_index
                                ],
                                "retrieved_doc_id": second_doc_ids[
                                    row_index, first_index, second_index
                                ],
                                "truth_probability": truth_probability(
                                    second_lookup[
                                        row_index, first_index, second_index
                                    ],
                                    equivalence[
                                        f"a{first_index}>b{second_index}"
                                    ],
                                ),
                            }
                            for second_index in range(SECOND_ACTION_COUNT)
                        ],
                    }
                )
            records.append(
                {
                    "row_id": row["id"],
                    "gold_doc_ids": list(
                        gold_doc_ids(row, documents[row_index])
                    ),
                    "initial_truth_probability": truth_probability(
                        initial_belief, equivalence["initial"]
                    ),
                    "first_branches": first_branches,
                    "replay_indices": [0, 0],
                    "replay_truth_probability": truth_probability(
                        replay_beliefs[row_index],
                        equivalence["replay:a0>b0"],
                    ),
                    "all_belief_sizes_valid": all(
                        len(belief) == BELIEF_SIZE
                        for _state_id, belief in states_by_row[row_index]
                    ),
                }
            )
        usage = _usage_snapshot(generator, judge)
    except Exception as exc:
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(generator, judge),
        ) from exc

    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "dataset": "MuSiQue answerable dev v1.0",
            "dataset_sha256": DATA_SHA256,
            "selection_seed": SELECTION_SEED,
            "row_ids": list(selected_ids(stage)),
            "reserve_ids": list(RESERVE_IDS),
            "belief_size": BELIEF_SIZE,
            "first_action_count": FIRST_ACTION_COUNT,
            "second_action_count": SECOND_ACTION_COUNT,
            "retrieval": "deterministic BM25 over sealed 20-document context",
            "document_titles_hidden_before_retrieval": True,
            "document_ids_shuffled_per_row": True,
            "gold_hidden_from_generator": True,
            "truth_used_only_by_post_generation_equivalence_judge": True,
            "reasoning_disabled": True,
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "raw_responses_private_and_untracked": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--data-path",
        type=Path,
        default=Path("external/musique/data/musique_ans_v1.0_dev.jsonl"),
    )
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "opportunity"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.50
        config.openrouter_run_budget_usd = 1.50
    else:
        config.openrouter_projected_cost_usd = 1.50
        config.openrouter_run_budget_usd = 4.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json"
        if args.stage == "serving_smoke"
        else "OPPORTUNITY.json"
    )
    try:
        payload = run_gate(
            config,
            data_path=args.data_path,
            stage=args.stage,
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
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                **payload["summary"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
