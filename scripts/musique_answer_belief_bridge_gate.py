#!/usr/bin/env python3
"""Gate non-myopic answer-belief regeneration on fresh MuSiQue questions."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence
import unicodedata

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import _parse_json_object
from scripts.musique_chain_transition_gate import (
    DATA_SHA256,
    FORMAL_IDS as CHAIN_FORMAL_IDS,
    PREVIOUS_IDS as CHAIN_PREVIOUS_IDS,
    SMOKE_IDS as CHAIN_SMOKE_IDS,
    is_eligible_row,
)


SCHEMA_VERSION = 1
SELECTION_SEED = 24327
BELIEF_SIZE = 8
FIRST_ACTION_COUNT = 6
SECOND_ACTION_COUNT = 4
SMOKE_IDS = (
    "2hop__30390_92972",
    "2hop__609313_20273",
)
FORMAL_IDS = (
    "2hop__77263_84616",
    "2hop__73615_42173",
    "2hop__61714_42553",
    "2hop__510679_57816",
    "2hop__804417_126089",
    "2hop__684287_78303",
)
RESERVE_IDS = (
    "2hop__390947_232243",
    "2hop__47071_162399",
    "2hop__471509_136043",
    "2hop__161151_50883",
    "2hop__286093_361551",
    "2hop__15368_17873",
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
TRUTH_PROBABILITY_FLOOR = 1e-4
TRUTH_COVERAGE_THRESHOLD = 0.05


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _normalized(value: str) -> str:
    ascii_value = (
        unicodedata.normalize("NFKD", value)
        .encode("ascii", "ignore")
        .decode("ascii")
        .casefold()
    )
    return " ".join(re.findall(r"[a-z0-9]+", ascii_value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
        CHAIN_PREVIOUS_IDS + CHAIN_SMOKE_IDS + CHAIN_FORMAL_IDS
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


def gold_doc_ids(
    row: dict[str, Any],
    documents: dict[str, dict[str, Any]],
) -> tuple[str, str]:
    doc_id_by_idx = {
        int(paragraph["idx"]): doc_id
        for doc_id, paragraph in documents.items()
    }
    return tuple(
        doc_id_by_idx[int(step["paragraph_support_idx"])]
        for step in row["question_decomposition"]
    )  # type: ignore[return-value]


def _belief_schema() -> list[dict[str, Any]]:
    return [
        {"answer": f"possible answer {index + 1}", "probability": 0.125}
        for index in range(BELIEF_SIZE)
    ]


def _document_catalog(documents: dict[str, dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {"doc_id": doc_id, "title": str(paragraph["title"])}
        for doc_id, paragraph in documents.items()
    ]


def belief_messages(
    row: dict[str, Any],
    documents: dict[str, dict[str, Any]],
    *,
    opened_doc_ids: Sequence[str] = (),
    previous_belief: Sequence[dict[str, Any]] | None = None,
    request_actions: int = 0,
) -> list[dict[str, str]]:
    visible_documents = [
        {
            "doc_id": doc_id,
            "title": str(documents[doc_id]["title"]),
            "text": str(documents[doc_id]["paragraph_text"]),
        }
        for doc_id in opened_doc_ids
    ]
    payload: dict[str, Any] = {
        "question": row["question"],
        "available_document_titles": _document_catalog(documents),
        "opened_documents_in_order": visible_documents,
    }
    if previous_belief is not None:
        payload["previous_answer_belief_for_context"] = list(previous_belief)
    response_schema: dict[str, Any] = {"belief": _belief_schema()}
    if request_actions == FIRST_ACTION_COUNT:
        response_schema["first_documents"] = [
            f"d{index + 1:02d}" for index in range(FIRST_ACTION_COUNT)
        ]
    elif request_actions == SECOND_ACTION_COUNT:
        response_schema["next_documents"] = [
            f"d{index + 1:02d}" for index in range(SECOND_ACTION_COUNT)
        ]
    action_instruction = ""
    if request_actions:
        stage = "first" if not opened_doc_ids else "next"
        action_instruction = (
            f" Also choose exactly {request_actions} pairwise-distinct {stage} "
            "document IDs from the available catalog, excluding already opened "
            "documents. Prefer documents that can reveal a missing bridge needed "
            "to identify the answer, not documents that merely resemble the final "
            "answer type."
        )
    return [
        {
            "role": "system",
            "content": (
                "Maintain a target-blind probabilistic belief over final answers "
                "using only the visible question, titles, and opened documents. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Build the answer belief afresh from all visible evidence. Return "
                f"exactly {BELIEF_SIZE} distinct concrete final-answer strings and "
                "probabilities that sum to 1. Do not use outside factual memory, do "
                "not name a hidden benchmark answer, and do not collapse bridge "
                "entities into final answers unless visible evidence supports it."
                f"{action_instruction} Return exactly this schema: "
                + json.dumps(response_schema, ensure_ascii=True, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_belief_response(
    text: str,
    *,
    available_doc_ids: Sequence[str],
    opened_doc_ids: Sequence[str] = (),
    action_key: str | None = None,
    action_count: int = 0,
) -> tuple[list[dict[str, Any]], list[str]]:
    payload = _parse_json_object(text)
    rows = payload.get("belief")
    if not isinstance(rows, list) or len(rows) != BELIEF_SIZE:
        raise ValueError(f"belief must contain exactly {BELIEF_SIZE} rows")
    belief: list[dict[str, Any]] = []
    seen_answers: set[str] = set()
    total = 0.0
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("belief rows must be objects")
        answer = row.get("answer")
        probability = row.get("probability")
        if not isinstance(answer, str) or not _normalized(answer):
            raise ValueError("belief answer must be a nonempty string")
        normalized = _normalized(answer)
        if normalized in seen_answers:
            raise ValueError("belief answers must be distinct")
        seen_answers.add(normalized)
        if (
            isinstance(probability, bool)
            or not isinstance(probability, (int, float))
            or not math.isfinite(float(probability))
            or float(probability) < 0.0
        ):
            raise ValueError("belief probabilities must be finite and nonnegative")
        total += float(probability)
        belief.append({"answer": " ".join(answer.split()), "probability": float(probability)})
    if not 0.90 <= total <= 1.10:
        raise ValueError("belief probabilities must sum within [0.90, 1.10]")
    for row in belief:
        row["probability"] = float(row["probability"]) / total

    if action_key is None:
        return belief, []
    actions = payload.get(action_key)
    if not isinstance(actions, list) or len(actions) != action_count:
        raise ValueError(f"{action_key} must contain exactly {action_count} IDs")
    available = set(available_doc_ids)
    opened = set(opened_doc_ids)
    if (
        any(not isinstance(action, str) for action in actions)
        or len(set(actions)) != action_count
        or any(action not in available or action in opened for action in actions)
    ):
        raise ValueError(f"{action_key} contains invalid document IDs")
    return belief, list(actions)


def equivalence_messages(
    truth: str,
    states: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> list[dict[str, str]]:
    payload = {
        "reference_answer": truth,
        "states": [
            {
                "state_id": state_id,
                "candidate_answers": [
                    {"index": index, "answer": row["answer"]}
                    for index, row in enumerate(belief)
                ],
            }
            for state_id, belief in states
        ],
    }
    schema = {
        "matches": [
            {"state_id": state_id, "matching_indices": []}
            for state_id, _belief in states
        ]
    }
    return [
        {
            "role": "system",
            "content": (
                "Judge semantic equivalence to a reference answer. Return strict "
                "JSON only and do not infer new answers."
            ),
        },
        {
            "role": "user",
            "content": (
                "For every state, list exactly the zero-based indices whose candidate "
                "answer expresses the same answer as the reference. Accept harmless "
                "paraphrases, aliases, formatting, and equivalent dates or numbers. "
                "Reject related entities, bridge entities, broader categories, and "
                "answers that require adding missing facts. Preserve every state_id "
                "and its order. Return "
                + json.dumps(schema, ensure_ascii=True, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_equivalence(
    text: str,
    states: Sequence[tuple[str, Sequence[dict[str, Any]]]],
) -> dict[str, list[int]]:
    rows = _parse_json_object(text).get("matches")
    if not isinstance(rows, list) or len(rows) != len(states):
        raise ValueError("equivalence response has the wrong state count")
    parsed: dict[str, list[int]] = {}
    for row, (expected_id, belief) in zip(rows, states, strict=True):
        if not isinstance(row, dict) or row.get("state_id") != expected_id:
            raise ValueError("equivalence state IDs or order changed")
        indices = row.get("matching_indices")
        if (
            not isinstance(indices, list)
            or any(
                isinstance(index, bool)
                or not isinstance(index, int)
                or index < 0
                or index >= len(belief)
                for index in indices
            )
            or len(indices) != len(set(indices))
        ):
            raise ValueError("equivalence indices are invalid")
        parsed[expected_id] = list(indices)
    return parsed


def truth_probability(
    belief: Sequence[dict[str, Any]],
    matching_indices: Sequence[int],
) -> float:
    return float(
        max(
            TRUTH_PROBABILITY_FLOOR,
            sum(float(belief[index]["probability"]) for index in matching_indices),
        )
    )


def analyze_record(record: dict[str, Any]) -> dict[str, Any]:
    one = {
        branch["first_doc_id"]: float(branch["truth_probability"])
        for branch in record["first_branches"]
    }
    pairs = {
        f"{branch['first_doc_id']}>{second['second_doc_id']}": float(
            second["truth_probability"]
        )
        for branch in record["first_branches"]
        for second in branch["second_branches"]
    }
    first_order = [branch["first_doc_id"] for branch in record["first_branches"]]
    pair_order = list(pairs)
    greedy = max(first_order, key=lambda key: (one[key], -first_order.index(key)))
    oracle_pair = max(
        pair_order, key=lambda key: (pairs[key], -pair_order.index(key))
    )
    greedy_continuation = max(
        value for key, value in pairs.items() if key.startswith(f"{greedy}>")
    )
    gold_root, gold_second = record["gold_doc_ids"]
    gold_pair_key = f"{gold_root}>{gold_second}"
    replay_gap = abs(
        float(record["replay_truth_probability"])
        - float(pairs[record["replay_pair"]])
    )
    return {
        "greedy_first_doc_id": greedy,
        "oracle_pair": oracle_pair,
        "gold_pair": gold_pair_key,
        "gold_root_is_candidate": gold_root in one,
        "gold_second_proposed_after_gold_root": gold_pair_key in pairs,
        "oracle_pair_is_gold_pair": oracle_pair == gold_pair_key,
        "oracle_first_differs_from_greedy": oracle_pair.split(">", 1)[0] != greedy,
        "gold_root_differs_from_greedy": gold_root != greedy,
        "best_one_step_truth_probability": max(one.values()),
        "oracle_pair_truth_probability": pairs[oracle_pair],
        "greedy_continuation_truth_probability": greedy_continuation,
        "pair_gain_over_best_one_step": pairs[oracle_pair] - max(one.values()),
        "nonmyopic_probability_gap": pairs[oracle_pair] - greedy_continuation,
        "gold_pair_truth_probability": pairs.get(
            gold_pair_key, TRUTH_PROBABILITY_FLOOR
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
    expected = len(selected_ids(stage))
    base_gates = {
        "all_cases_complete": len(records) == expected,
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
            "all_gold_blind_beliefs_size_8": all(
                record["all_belief_sizes_valid"] for record in records
            ),
        }
        gates["all_pass"] = all(gates.values())
        return {
            "num_cases": len(records),
            "case_diagnostics": diagnostics,
            "gates": gates,
        }

    root_count = sum(row["gold_root_is_candidate"] for row in diagnostics)
    continuation_count = sum(
        row["gold_second_proposed_after_gold_root"] for row in diagnostics
    )
    gold_oracle_count = sum(row["oracle_pair_is_gold_pair"] for row in diagnostics)
    first_differs_count = sum(
        row["oracle_first_differs_from_greedy"] for row in diagnostics
    )
    gold_differs_count = sum(
        row["gold_root_differs_from_greedy"] for row in diagnostics
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
        "gold_root_candidate_count": root_count,
        "gold_continuation_candidate_count": continuation_count,
        "gold_pair_is_oracle_count": gold_oracle_count,
        "oracle_first_differs_from_greedy_count": first_differs_count,
        "gold_root_differs_from_greedy_count": gold_differs_count,
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
        "gold_root_candidate_count_at_least_4": root_count >= 4,
        "gold_continuation_candidate_count_at_least_4": continuation_count >= 4,
        "gold_pair_is_oracle_count_at_least_2": gold_oracle_count >= 2,
        "oracle_first_differs_count_at_least_2": first_differs_count >= 2,
        "gold_root_differs_count_at_least_2": gold_differs_count >= 2,
        "pair_gain_count_at_least_3": pair_gain_count >= 3,
        "nonmyopic_gap_count_at_least_2": gap_count >= 2,
        "mean_pair_gain_at_least_0_08": mean_gain >= 0.08,
        "mean_nonmyopic_gap_at_least_0_05": mean_gap >= 0.05,
        "mean_replay_gap_at_most_0_10": float(np.mean(replay_gaps)) <= 0.10,
        "max_replay_gap_at_most_0_25": max(replay_gaps) <= 0.25,
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def _build_models(config: Config) -> tuple[Any, Any]:
    generator_spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    judge_spec = replace(
        config.model_pairs[0].answerer,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    return (
        build_model_adapter(generator_spec, config),
        build_model_adapter(judge_spec, config),
    )


def _usage_snapshot(generator: Any, judge: Any) -> dict[str, Any]:
    generator_usage = generator.usage_snapshot()
    judge_usage = judge.usage_snapshot()
    return {
        "physical_requests": int(generator_usage["adapter_requests"])
        + int(judge_usage["adapter_requests"]),
        "reasoning_tokens": int(generator_usage["adapter_reasoning_tokens"])
        + int(judge_usage["adapter_reasoning_tokens"]),
        "adapter_cost_usd": float(generator_usage["adapter_cost_usd"])
        + float(judge_usage["adapter_cost_usd"]),
        "generator": generator_usage,
        "judge": judge_usage,
    }


def _checkpoint(path: Path | None, *, stage: str, raw: dict[str, Any]) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {"schema_version": SCHEMA_VERSION, "stage": stage, **raw},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def run_gate(
    config: Config,
    *,
    data_path: str | Path,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    rows = load_selected_rows(data_path, stage)
    if len(config.model_pairs) != 1:
        raise ValueError("MuSiQue answer-belief gate requires one model pair")
    generator, judge = _build_models(config)
    raw: dict[str, Any] = {}
    try:
        documents = [document_map(row) for row in rows]
        initial_raw = generator.chat_complete_messages_batched(
            [
                belief_messages(
                    row,
                    doc_map,
                    request_actions=FIRST_ACTION_COUNT,
                )
                for row, doc_map in zip(rows, documents, strict=True)
            ],
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["initial"] = initial_raw
        _checkpoint(raw_checkpoint_path, stage=stage, raw=raw)
        initial = [
            parse_belief_response(
                text,
                available_doc_ids=list(doc_map),
                action_key="first_documents",
                action_count=FIRST_ACTION_COUNT,
            )
            for text, doc_map in zip(initial_raw, documents, strict=True)
        ]

        first_keys: list[tuple[int, str]] = []
        first_messages = []
        for row_index, ((belief, actions), row, doc_map) in enumerate(
            zip(initial, rows, documents, strict=True)
        ):
            shuffled = list(actions)
            np.random.default_rng(SELECTION_SEED * 1000 + row_index).shuffle(
                shuffled
            )
            initial[row_index] = (belief, shuffled)
            for first_doc_id in shuffled:
                first_keys.append((row_index, first_doc_id))
                first_messages.append(
                    belief_messages(
                        row,
                        doc_map,
                        opened_doc_ids=(first_doc_id,),
                        previous_belief=belief,
                        request_actions=SECOND_ACTION_COUNT,
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
            parse_belief_response(
                text,
                available_doc_ids=list(documents[row_index]),
                opened_doc_ids=(first_doc_id,),
                action_key="next_documents",
                action_count=SECOND_ACTION_COUNT,
            )
            for text, (row_index, first_doc_id) in zip(
                first_raw, first_keys, strict=True
            )
        ]

        second_keys: list[tuple[int, str, str]] = []
        second_messages = []
        first_lookup: dict[tuple[int, str], tuple[list[dict[str, Any]], list[str]]] = {}
        for key, parsed in zip(first_keys, first_parsed, strict=True):
            row_index, first_doc_id = key
            belief, next_actions = parsed
            shuffled = list(next_actions)
            np.random.default_rng(
                SELECTION_SEED * 1_000_000
                + row_index * 100
                + list(initial[row_index][1]).index(first_doc_id)
            ).shuffle(shuffled)
            first_lookup[key] = (belief, shuffled)
            for second_doc_id in shuffled:
                second_keys.append((row_index, first_doc_id, second_doc_id))
                second_messages.append(
                    belief_messages(
                        rows[row_index],
                        documents[row_index],
                        opened_doc_ids=(first_doc_id, second_doc_id),
                        previous_belief=belief,
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
                available_doc_ids=list(documents[row_index]),
                opened_doc_ids=(first_doc_id, second_doc_id),
            )[0]
            for text, (row_index, first_doc_id, second_doc_id) in zip(
                second_raw, second_keys, strict=True
            )
        ]
        second_lookup = {
            key: belief
            for key, belief in zip(second_keys, second_parsed, strict=True)
        }

        replay_keys: list[tuple[int, str, str]] = []
        replay_messages = []
        for row_index, (_belief, first_actions) in enumerate(initial):
            first_doc_id = first_actions[0]
            first_belief, second_actions = first_lookup[row_index, first_doc_id]
            second_doc_id = second_actions[0]
            replay_keys.append((row_index, first_doc_id, second_doc_id))
            replay_messages.append(
                belief_messages(
                    rows[row_index],
                    documents[row_index],
                    opened_doc_ids=(first_doc_id, second_doc_id),
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
            parse_belief_response(
                text,
                available_doc_ids=list(documents[row_index]),
                opened_doc_ids=(first_doc_id, second_doc_id),
            )[0]
            for text, (row_index, first_doc_id, second_doc_id) in zip(
                replay_raw, replay_keys, strict=True
            )
        ]

        states_by_row: list[list[tuple[str, Sequence[dict[str, Any]]]]] = []
        for row_index, (initial_belief, first_actions) in enumerate(initial):
            states: list[tuple[str, Sequence[dict[str, Any]]]] = [
                ("initial", initial_belief)
            ]
            for first_doc_id in first_actions:
                first_belief, second_actions = first_lookup[
                    row_index, first_doc_id
                ]
                states.append((first_doc_id, first_belief))
                for second_doc_id in second_actions:
                    states.append(
                        (
                            f"{first_doc_id}>{second_doc_id}",
                            second_lookup[
                                row_index, first_doc_id, second_doc_id
                            ],
                        )
                    )
            replay_key = replay_keys[row_index]
            states.append(
                (
                    f"replay:{replay_key[1]}>{replay_key[2]}",
                    replay_beliefs[row_index],
                )
            )
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
            initial_belief, first_actions = initial[row_index]
            equivalence = equivalences[row_index]
            first_branches = []
            for first_doc_id in first_actions:
                first_belief, second_actions = first_lookup[
                    row_index, first_doc_id
                ]
                first_branches.append(
                    {
                        "first_doc_id": first_doc_id,
                        "truth_probability": truth_probability(
                            first_belief, equivalence[first_doc_id]
                        ),
                        "second_branches": [
                            {
                                "second_doc_id": second_doc_id,
                                "truth_probability": truth_probability(
                                    second_lookup[
                                        row_index, first_doc_id, second_doc_id
                                    ],
                                    equivalence[
                                        f"{first_doc_id}>{second_doc_id}"
                                    ],
                                ),
                            }
                            for second_doc_id in second_actions
                        ],
                    }
                )
            replay_key = replay_keys[row_index]
            replay_state_id = (
                f"replay:{replay_key[1]}>{replay_key[2]}"
            )
            records.append(
                {
                    "row_id": row["id"],
                    "gold_doc_ids": list(gold_doc_ids(row, documents[row_index])),
                    "initial_truth_probability": truth_probability(
                        initial_belief, equivalence["initial"]
                    ),
                    "first_branches": first_branches,
                    "replay_pair": f"{replay_key[1]}>{replay_key[2]}",
                    "replay_truth_probability": truth_probability(
                        replay_beliefs[row_index],
                        equivalence[replay_state_id],
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
            "expected_physical_requests": EXPECTED_REQUESTS[stage],
            "gold_hidden_from_generator": True,
            "truth_used_only_by_post_generation_equivalence_judge": True,
            "document_ids_shuffled_per_row": True,
            "reasoning_disabled": True,
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
