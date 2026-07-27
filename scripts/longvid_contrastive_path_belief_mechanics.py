#!/usr/bin/env python3
"""Rank LongVid paths from LLM-regenerated semantic belief trajectories."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Callable, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.bright_biology_unlock_audit import BM25Corpus, tokenize
from scripts.longvid_bridge_path_opportunity_audit import (
    CAPTION_SHA256,
    QA_SHA256,
    load_selected_captions,
    root_queries,
    sha256_file,
    stream_json_array,
)
from scripts.longvid_four_hop_support_smoke import MODEL_ID
from scripts.longvid_four_hop_tradeoff_opportunity import load_selected_qa


INTERFACE_VERSION = "longvid-contrastive-path-belief-mechanics-1"
LAYOUT_SEED = 270741
RANDOM_CONTROL_SEED = 270742
SUPPORT_SIZE = 6
PATH_DEPTH = 4
EXPECTED_REQUESTS = 44
MAX_COST_USD = 0.75
PROJECTED_COST_USD = 0.40
OBSERVATION_CHAR_CAP = 2_000
MAX_NEW_TOKENS = 1_100
MIN_CHANGED_REFRESHES = 28
MIN_VALID_ANCHOR_REFRESHES = 32
MIN_RANKABLE_TASKS = 3
MIN_FINAL_ACCURACY = 0.75
MIN_POLICY_CHANGES = 2
MIN_CORRECT_POLICY_CHANGES = 1
MIN_FINAL_COVERAGE_GAIN = 2

STRUCTURAL_AUDIT_SHA256 = (
    "3cb882b1facbef2ababe2f8be68529423773c362a1eb4402b68d0bb240cc3ce0"
)
SUPPORT_SMOKE_SHA256 = (
    "0e594dbdaa748160e9006830992cca000ccfc7badec8cdde3b6d7ca56d412239"
)
EXCLUDED_PRIOR_ROWS = (955, 1802, 540, 479, 1332, 1068)
TASK_LAYOUT = (
    (2156, (1, 0)),
    (2062, (2, 0)),
    (1689, (11, 12)),
    (1648, (0, 10)),
)
TASK_LAYOUT_HASH = (
    "abf7091e39ac9116ac33e7e3b8fe88c41d81c2a602bc1e3493c2f1d0edf6242d"
)


class StructuredChatModel(Protocol):
    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class MechanicsExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def layout_hash() -> str:
    return layout_hash_for(TASK_LAYOUT)


def layout_hash_for(
    task_layout: Sequence[tuple[int, tuple[int, int]]],
) -> str:
    text = "\n".join(
        f"{row_index}:{roots[0]},{roots[1]}"
        for row_index, roots in task_layout
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )


def _normalize(value: str) -> str:
    return " ".join(tokenize(value))


def support_response_format() -> dict[str, Any]:
    properties: dict[str, Any] = {}
    for index in range(1, SUPPORT_SIZE + 1):
        properties[f"hypothesis_{index}"] = {
            "type": "string",
            "minLength": 8,
            "maxLength": 360,
        }
        properties[f"weight_{index}"] = {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
        }
        properties[f"anchor_{index}"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 80,
        }
        properties[f"query_{index}"] = {
            "type": "string",
            "minLength": 2,
            "maxLength": 180,
        }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "longvid_semantic_belief_support",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def rank_response_format() -> dict[str, Any]:
    properties = {
        "choice": {"type": "string", "enum": ["A", "B"]},
        "confidence": {
            "type": "integer",
            "minimum": 51,
            "maximum": 100,
        },
        "unresolved_need": {
            "type": "string",
            "minLength": 3,
            "maxLength": 280,
        },
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "longvid_contrastive_path_rank",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def parse_support(text: str) -> list[dict[str, Any]]:
    payload = json.loads(text)
    expected = {
        f"{field}_{index}"
        for index in range(1, SUPPORT_SIZE + 1)
        for field in ("hypothesis", "weight", "anchor", "query")
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("support response fields do not match the schema")
    support = []
    for index in range(1, SUPPORT_SIZE + 1):
        hypothesis = payload[f"hypothesis_{index}"].strip()
        weight = payload[f"weight_{index}"]
        anchor = payload[f"anchor_{index}"].strip()
        query = payload[f"query_{index}"].strip()
        if not hypothesis or not anchor or not query:
            raise ValueError("support strings must be nonempty")
        if isinstance(weight, bool) or not isinstance(weight, int):
            raise ValueError("support weights must be integers")
        support.append(
            {
                "hypothesis": hypothesis,
                "weight": weight,
                "anchor": anchor,
                "query": query,
            }
        )
    if len({_normalize(row["hypothesis"]) for row in support}) != SUPPORT_SIZE:
        raise ValueError("support hypotheses must be distinct")
    if len({_normalize(row["query"]) for row in support}) != SUPPORT_SIZE:
        raise ValueError("support queries must be distinct")
    if len({row["weight"] for row in support}) < 2:
        raise ValueError("support must express nonuniform confidence")
    return support


def parse_rank(text: str) -> dict[str, Any]:
    payload = json.loads(text)
    if not isinstance(payload, dict) or set(payload) != {
        "choice",
        "confidence",
        "unresolved_need",
    }:
        raise ValueError("rank response fields do not match the schema")
    if payload["choice"] not in {"A", "B"}:
        raise ValueError("rank choice is invalid")
    confidence = payload["confidence"]
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, int)
        or not 51 <= confidence <= 100
    ):
        raise ValueError("rank confidence is invalid")
    unresolved = payload["unresolved_need"].strip()
    if not unresolved:
        raise ValueError("rank unresolved need is empty")
    return {
        "choice": payload["choice"],
        "confidence": confidence,
        "unresolved_need": unresolved,
    }


def initial_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Maintain six distinct semantic hypotheses about the unseen "
                "multi-clip evidence chain needed to answer a long-video "
                "question. Give each a confidence weight and a concise next "
                "search. Do not answer the question and do not reason aloud. "
                "Return only the schema-conforming object. Every anchor must "
                "be exactly QUESTION."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n\n"
                "Generate a diverse support over plausible four-link evidence "
                "chains. Each query should seek a different first link."
            ),
        },
    ]


def _support_for_prompt(support: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis": row["hypothesis"],
            "weight": row["weight"],
            "anchor": row["anchor"],
            "query": row["query"],
        }
        for row in support
    ]


def refresh_messages(
    question: str,
    previous_query: str,
    observation: str,
    previous_support: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Regenerate six semantic hypotheses after one new video-caption "
                "observation. Preserve alternatives, revise confidence, and seek "
                "the next missing evidence link. Do not answer the question and "
                "do not reason aloud. Return only the schema-conforming object. "
                "Each anchor must copy a meaningful token from OBSERVATION that "
                "is absent from QUESTION and PREVIOUS_QUERY, and each next query "
                "must contain its anchor."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n"
                f"PREVIOUS_QUERY:\n{previous_query}\n"
                f"OBSERVATION:\n{observation[:OBSERVATION_CHAR_CAP]}\n"
                "PREVIOUS_SUPPORT:\n"
                + _canonical(_support_for_prompt(previous_support))
            ),
        },
    ]


def _trajectory_for_rank(
    path: dict[str, Any],
    *,
    final: bool,
) -> dict[str, Any]:
    limit = PATH_DEPTH if final else 1
    return {
        "root_index": path["root_index"],
        "queries": path["queries"][:limit],
        "belief_supports": [
            _support_for_prompt(support)
            for support in path["supports"][1 : limit + 1]
        ],
    }


def rank_messages(
    question: str,
    row_index: int,
    candidates: Sequence[dict[str, Any]],
    *,
    final: bool,
) -> list[dict[str, str]]:
    stage = "FINAL_FOUR_STEP" if final else "IMMEDIATE_ONE_STEP"
    return [
        {
            "role": "system",
            "content": (
                "Compare two blinded retrieval trajectories using only their "
                "LLM-regenerated semantic belief supports and search queries. "
                "Choose the trajectory more likely to expose the complete "
                "multi-clip evidence chain needed by the question. Do not answer "
                "the question, infer hidden clip IDs, or reason aloud. Return only "
                "the schema-conforming object."
            ),
        },
        {
            "role": "user",
            "content": "\n".join(
                [
                    f"ROW_ID={row_index}",
                    f"STAGE={stage}",
                    f"QUESTION={question}",
                    "TRAJECTORY_A="
                    + _canonical(
                        _trajectory_for_rank(candidates[0], final=final)
                    ),
                    "TRAJECTORY_B="
                    + _canonical(
                        _trajectory_for_rank(candidates[1], final=final)
                    ),
                ]
            ),
        },
    ]


def valid_anchor_count(
    support: Sequence[dict[str, Any]],
    *,
    question: str,
    previous_query: str,
    observation: str,
) -> int:
    observation_terms = set(tokenize(observation))
    excluded = set(tokenize(question)).union(tokenize(previous_query))
    count = 0
    for row in support:
        anchor_terms = tokenize(row["anchor"])
        query_terms = set(tokenize(row["query"]))
        if (
            len(anchor_terms) == 1
            and anchor_terms[0] in observation_terms
            and anchor_terms[0] not in excluded
            and anchor_terms[0] in query_terms
        ):
            count += 1
    return count


def support_signature(support: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(
        _canonical(_support_for_prompt(support)).encode("utf-8")
    ).hexdigest()


def _select_query(support: Sequence[dict[str, Any]]) -> str:
    index = max(
        range(len(support)),
        key=lambda value: (support[value]["weight"], -value),
    )
    return str(support[index]["query"])


def _visible_rows(
    qa_path: Path,
    selected_ids: set[int],
) -> dict[int, dict[str, str]]:
    rows: dict[int, dict[str, str]] = {}
    for index, raw in stream_json_array(qa_path):
        if index not in selected_ids:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"QA row {index} is not an object")
        question = str(raw.get("question", "")).strip()
        video_id = str(raw.get("vid", "")).strip()
        category = str(raw.get("category", "")).strip()
        if not question or not video_id or not category:
            raise ValueError(f"QA row {index} has empty visible metadata")
        if str(raw.get("hop_level", "")) != "4-Hop":
            raise ValueError(f"QA row {index} is not four-hop")
        rows[index] = {
            "question": question,
            "video_id": video_id,
            "category": category,
        }
    if set(rows) != selected_ids:
        raise ValueError("visible QA load missed selected rows")
    return rows


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _usage(model: StructuredChatModel) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot.get("adapter_requests", 0)),
        "http_attempts": int(snapshot.get("http_attempts", 0)),
        "retry_count": int(snapshot.get("retry_count", 0)),
        "reasoning_tokens": int(snapshot.get("adapter_reasoning_tokens", 0)),
        "forced_exits": int(snapshot.get("forced_exits", 0)),
        "prompt_tokens": int(snapshot.get("adapter_prompt_tokens", 0)),
        "completion_tokens": int(snapshot.get("adapter_completion_tokens", 0)),
        "adapter_cost_usd": float(snapshot.get("adapter_cost_usd", 0.0)),
        "model": snapshot,
    }


def analyze_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    rng = random.Random(RANDOM_CONTROL_SEED)
    final_correct = 0
    immediate_correct = 0
    rankable = 0
    policy_changes = 0
    correct_policy_changes = 0
    final_total = 0
    immediate_total = 0
    random_total = 0
    task_metrics = []
    for record in records:
        coverages = [
            int(candidate["coverage_count"])
            for candidate in record["candidates"]
        ]
        final_choice = 0 if record["final_rank"]["choice"] == "A" else 1
        immediate_choice = (
            0 if record["immediate_rank"]["choice"] == "A" else 1
        )
        random_choice = rng.randrange(2)
        is_rankable = coverages[0] != coverages[1]
        best = max(coverages)
        if is_rankable:
            rankable += 1
            final_correct += int(coverages[final_choice] == best)
            immediate_correct += int(coverages[immediate_choice] == best)
        changed = final_choice != immediate_choice
        policy_changes += int(changed)
        correct_policy_changes += int(
            changed and coverages[final_choice] > coverages[immediate_choice]
        )
        final_total += coverages[final_choice]
        immediate_total += coverages[immediate_choice]
        random_total += coverages[random_choice]
        task_metrics.append(
            {
                "row_index": record["row_index"],
                "category": record["category"],
                "root_indices": [
                    candidate["root_index"]
                    for candidate in record["candidates"]
                ],
                "coverage_counts": coverages,
                "immediate_choice": immediate_choice,
                "final_choice": final_choice,
                "random_choice": random_choice,
                "immediate_confidence": record["immediate_rank"]["confidence"],
                "final_confidence": record["final_rank"]["confidence"],
                "rankable": is_rankable,
                "policy_changed": changed,
            }
        )
    final_accuracy = final_correct / rankable if rankable else 0.0
    immediate_accuracy = immediate_correct / rankable if rankable else 0.0
    return {
        "task_metrics": task_metrics,
        "summary": {
            "rankable_task_count": rankable,
            "final_pairwise_correct": final_correct,
            "immediate_pairwise_correct": immediate_correct,
            "final_pairwise_accuracy": final_accuracy,
            "immediate_pairwise_accuracy": immediate_accuracy,
            "policy_change_count": policy_changes,
            "correct_policy_change_count": correct_policy_changes,
            "final_selected_coverage_total": final_total,
            "immediate_selected_coverage_total": immediate_total,
            "random_selected_coverage_total": random_total,
            "final_coverage_gain_over_immediate": (
                final_total - immediate_total
            ),
        },
    }


class DeterministicFixtureModel:
    """Exercise the complete protocol without model or endpoint leakage."""

    GREEDY_ROOT = {2156: 0, 2062: 0, 1689: 11, 1648: 0}
    ORACLE_ROOT = {2156: 1, 2062: 2, 1689: 12, 1648: 10}

    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        *,
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        name = response_format["json_schema"]["name"]
        responses = []
        for offset, messages in enumerate(batch_messages):
            user = messages[-1]["content"]
            if name == "longvid_contrastive_path_rank":
                row = int(re.search(r"ROW_ID=(\d+)", user).group(1))
                final = "STAGE=FINAL_FOUR_STEP" in user
                roots = [
                    int(value)
                    for value in re.findall(r'"root_index":(\d+)', user)
                ]
                target = (
                    self.ORACLE_ROOT[row] if final else self.GREEDY_ROOT[row]
                )
                choice = "A" if roots[0] == target else "B"
                responses.append(
                    json.dumps(
                        {
                            "choice": choice,
                            "confidence": 80,
                            "unresolved_need": "fixture missing evidence link",
                        },
                        separators=(",", ":"),
                    )
                )
                continue
            question_match = re.search(
                r"QUESTION:\n(.+?)(?:\nPREVIOUS_QUERY:|\n\nGenerate)",
                user,
                flags=re.DOTALL,
            )
            query_match = re.search(
                r"PREVIOUS_QUERY:\n(.+?)\nOBSERVATION:",
                user,
                flags=re.DOTALL,
            )
            observation_match = re.search(
                r"OBSERVATION:\n(.+?)\nPREVIOUS_SUPPORT:",
                user,
                flags=re.DOTALL,
            )
            if question_match and query_match and observation_match:
                excluded = set(tokenize(question_match.group(1))).union(
                    tokenize(query_match.group(1))
                )
                anchors = [
                    token
                    for token in tokenize(observation_match.group(1))
                    if token not in excluded
                ]
                anchor = anchors[0] if anchors else "caption"
            else:
                anchor = "QUESTION"
            suffix = self.requests + offset
            payload = {}
            for index in range(1, SUPPORT_SIZE + 1):
                payload[f"hypothesis_{index}"] = (
                    f"Fixture evidence chain {suffix} hypothesis {index}"
                )
                payload[f"weight_{index}"] = 10 + index + (suffix % 3)
                payload[f"anchor_{index}"] = anchor
                payload[f"query_{index}"] = (
                    f"{anchor} fixture continuation {suffix} {index}"
                )
            responses.append(json.dumps(payload, separators=(",", ":")))
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_prompt_tokens": 0,
            "adapter_completion_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def _complete_structured(
    model: StructuredChatModel,
    messages: list[list[dict[str, str]]],
    response_format: dict[str, Any],
    *,
    block_size: int,
) -> list[str]:
    return model.chat_complete_messages_batched_structured(
        messages,
        temperature=0.0,
        block_size=block_size,
        response_format=response_format,
        max_new_tokens=MAX_NEW_TOKENS,
    )


def run_mechanics(
    config: Config,
    *,
    qa_path: Path,
    caption_path: Path,
    raw_path: Path,
    model: StructuredChatModel,
    task_layout: Sequence[tuple[int, tuple[int, int]]] = TASK_LAYOUT,
    task_layout_hash: str = TASK_LAYOUT_HASH,
    excluded_prior_rows: Sequence[int] = EXCLUDED_PRIOR_ROWS,
    response_format_name: str = "chat_strict_flat_json_schema",
    interface_version: str = INTERFACE_VERSION,
    layout_seed: int = LAYOUT_SEED,
    support_parser: Callable[[str], list[dict[str, Any]]] = parse_support,
    rank_parser: Callable[[str], dict[str, Any]] = parse_rank,
) -> dict[str, Any]:
    if sha256_file(qa_path) != QA_SHA256:
        raise ValueError("QA SHA-256 does not match frozen source")
    if sha256_file(caption_path) != CAPTION_SHA256:
        raise ValueError("caption SHA-256 does not match frozen source")
    if layout_hash_for(task_layout) != task_layout_hash:
        raise AssertionError("task layout hash does not reproduce")
    if set(excluded_prior_rows) & {row for row, _ in task_layout}:
        raise AssertionError("task layout reuses a prior mechanics row")

    root = Path(__file__).resolve().parents[1]
    bindings = {
        "structural_audit": (
            root
            / "results/nonmyopic/longvid_four_hop_tradeoff_opportunity/AUDIT.json"
        ),
        "support_smoke": (
            root
            / "results/nonmyopic/longvid_four_hop_support_smoke/"
            "longvid-four-hop-support-smoke-v2-20260727T145208Z/SMOKE.json"
        ),
    }
    expected_bindings = {
        "structural_audit": STRUCTURAL_AUDIT_SHA256,
        "support_smoke": SUPPORT_SMOKE_SHA256,
    }
    for name, path in bindings.items():
        if sha256_file(path) != expected_bindings[name]:
            raise ValueError(f"{name} artifact hash does not match")

    selected_ids = {row_index for row_index, _ in task_layout}
    visible_rows = _visible_rows(qa_path, selected_ids)
    video_ids = {row["video_id"] for row in visible_rows.values()}
    captions = load_selected_captions(caption_path, video_ids)
    corpora = {
        row_index: BM25Corpus(captions[row["video_id"]])
        for row_index, row in visible_rows.items()
    }
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "task_layout": task_layout,
        "task_layout_hash": task_layout_hash,
        "excluded_prior_rows": excluded_prior_rows,
        "endpoint_loaded": False,
        "visible_rows": visible_rows,
        "responses": {
            "initial": None,
            "refreshes": [],
            "immediate_ranks": None,
            "final_ranks": None,
        },
    }
    try:
        initial_responses = _complete_structured(
            model,
            [
                initial_messages(visible_rows[row_index]["question"])
                for row_index, _ in task_layout
            ],
            support_response_format(),
            block_size=config.openrouter_concurrency,
        )
        raw["responses"]["initial"] = initial_responses
        _checkpoint(raw_path, raw)
        initial_supports = [
            support_parser(response) for response in initial_responses
        ]
        if any(
            {row["anchor"] for row in support} != {"QUESTION"}
            for support in initial_supports
        ):
            raise ValueError("initial support anchors must all be QUESTION")

        paths: list[dict[str, Any]] = []
        for task_offset, (row_index, root_indices) in enumerate(task_layout):
            row = visible_rows[row_index]
            corpus = corpora[row_index]
            roots = root_queries(row["question"], corpus)
            for candidate_index, root_index in enumerate(root_indices):
                if root_index >= len(roots):
                    raise ValueError(
                        f"row {row_index} root {root_index} is unavailable"
                    )
                query = roots[root_index]
                result = corpus.search(query, top_k=1)
                if not result:
                    raise ValueError(f"row {row_index} root retrieves no caption")
                paths.append(
                    {
                        "task_offset": task_offset,
                        "row_index": row_index,
                        "candidate_index": candidate_index,
                        "root_index": root_index,
                        "question": row["question"],
                        "previous_query": query,
                        "current_document": result[0],
                        "retrieved_ids": [result[0]["id"]],
                        "queries": [query],
                        "supports": [initial_supports[task_offset]],
                        "support_signatures": [
                            support_signature(initial_supports[task_offset])
                        ],
                        "valid_anchor_counts": [],
                    }
                )

        for depth in range(PATH_DEPTH):
            requests = [
                refresh_messages(
                    path["question"],
                    path["previous_query"],
                    path["current_document"]["raw_source"],
                    path["supports"][-1],
                )
                for path in paths
            ]
            responses = _complete_structured(
                model,
                requests,
                support_response_format(),
                block_size=config.openrouter_concurrency,
            )
            raw["responses"]["refreshes"].append(responses)
            _checkpoint(raw_path, raw)
            supports = [support_parser(response) for response in responses]
            for path, support in zip(paths, supports):
                anchor_count = valid_anchor_count(
                    support,
                    question=path["question"],
                    previous_query=path["previous_query"],
                    observation=path["current_document"]["raw_source"],
                )
                path["valid_anchor_counts"].append(anchor_count)
                path["supports"].append(support)
                path["support_signatures"].append(
                    support_signature(support)
                )
                if depth < PATH_DEPTH - 1:
                    next_query = _select_query(support)
                    result = corpora[path["row_index"]].search(
                        next_query,
                        top_k=1,
                        excluded_ids=set(path["retrieved_ids"]),
                    )
                    if not result:
                        raise ValueError(
                            f"row {path['row_index']} path has no next caption"
                        )
                    path["previous_query"] = next_query
                    path["current_document"] = result[0]
                    path["retrieved_ids"].append(result[0]["id"])
                    path["queries"].append(next_query)

        grouped_paths = [
            [
                path for path in paths if path["row_index"] == row_index
            ]
            for row_index, _ in task_layout
        ]
        immediate_responses = _complete_structured(
            model,
            [
                rank_messages(
                    visible_rows[row_index]["question"],
                    row_index,
                    candidates,
                    final=False,
                )
                for (row_index, _), candidates in zip(
                    task_layout, grouped_paths, strict=True
                )
            ],
            rank_response_format(),
            block_size=config.openrouter_concurrency,
        )
        raw["responses"]["immediate_ranks"] = immediate_responses
        _checkpoint(raw_path, raw)
        immediate_ranks = [
            rank_parser(response) for response in immediate_responses
        ]

        final_responses = _complete_structured(
            model,
            [
                rank_messages(
                    visible_rows[row_index]["question"],
                    row_index,
                    candidates,
                    final=True,
                )
                for (row_index, _), candidates in zip(
                    task_layout, grouped_paths, strict=True
                )
            ],
            rank_response_format(),
            block_size=config.openrouter_concurrency,
        )
        raw["responses"]["final_ranks"] = final_responses
        _checkpoint(raw_path, raw)
        final_ranks = [rank_parser(response) for response in final_responses]

        records = []
        changed_refreshes = 0
        valid_anchor_refreshes = 0
        complete_paths = 0
        for task_offset, ((row_index, _), candidates) in enumerate(
            zip(task_layout, grouped_paths, strict=True)
        ):
            public_candidates = []
            for path in candidates:
                changed_refreshes += sum(
                    left != right
                    for left, right in zip(
                        path["support_signatures"],
                        path["support_signatures"][1:],
                    )
                )
                valid_anchor_refreshes += sum(
                    count >= 4 for count in path["valid_anchor_counts"]
                )
                complete_paths += int(
                    len(path["retrieved_ids"]) == PATH_DEPTH
                    and len(set(path["retrieved_ids"])) == PATH_DEPTH
                )
                public_candidates.append(
                    {
                        "root_index": path["root_index"],
                        "retrieved_ids": path["retrieved_ids"],
                        "queries": path["queries"],
                        "support_signatures": path["support_signatures"],
                        "valid_anchor_counts": path["valid_anchor_counts"],
                    }
                )
            records.append(
                {
                    "row_index": row_index,
                    "category": visible_rows[row_index]["category"],
                    "candidates": public_candidates,
                    "immediate_rank": immediate_ranks[task_offset],
                    "final_rank": final_ranks[task_offset],
                }
            )

        raw["policy_before_endpoint"] = records
        raw["endpoint_loaded"] = False
        _checkpoint(raw_path, raw)

        endpoint_rows = load_selected_qa(qa_path, selected_ids)
        raw["endpoint_loaded"] = True
        for record in records:
            evidence = {
                str(int(value))
                for value in endpoint_rows[record["row_index"]][
                    "evidence_slices"
                ]
            }
            for candidate in record["candidates"]:
                candidate["coverage_count"] = len(
                    evidence & set(candidate["retrieved_ids"])
                )
        raw["records_with_endpoint"] = records
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    analysis = analyze_records(records)
    summary = analysis["summary"]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_structured_outputs_parse": True,
        "endpoint_loaded_after_policy_freeze": raw["endpoint_loaded"],
        "all_eight_paths_complete_and_unique": complete_paths == 8,
        "at_least_28_of_32_supports_change": (
            changed_refreshes >= MIN_CHANGED_REFRESHES
        ),
        "all_32_refreshes_have_four_valid_anchors": (
            valid_anchor_refreshes >= MIN_VALID_ANCHOR_REFRESHES
        ),
        "rankable_tasks_at_least_3": (
            summary["rankable_task_count"] >= MIN_RANKABLE_TASKS
        ),
        "final_pairwise_accuracy_at_least_0_75": (
            summary["final_pairwise_accuracy"] >= MIN_FINAL_ACCURACY
        ),
        "final_pairwise_accuracy_beats_immediate": (
            summary["final_pairwise_accuracy"]
            > summary["immediate_pairwise_accuracy"]
        ),
        "at_least_two_policy_changes": (
            summary["policy_change_count"] >= MIN_POLICY_CHANGES
        ),
        "at_least_one_correct_policy_change": (
            summary["correct_policy_change_count"]
            >= MIN_CORRECT_POLICY_CHANGES
        ),
        "final_coverage_gain_at_least_2": (
            summary["final_coverage_gain_over_immediate"]
            >= MIN_FINAL_COVERAGE_GAIN
        ),
        "final_selected_coverage_beats_random": (
            summary["final_selected_coverage_total"]
            > summary["random_selected_coverage_total"]
        ),
        "cost_at_most_0_75": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "layout_seed": layout_seed,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "task_layout_hash": task_layout_hash,
            "excluded_prior_rows": list(excluded_prior_rows),
            "expected_requests": EXPECTED_REQUESTS,
            "response_format": response_format_name,
            "repairs_reissues_or_retries": 0,
            "max_cost_usd": MAX_COST_USD,
            "endpoint_loaded_after_policy_freeze": True,
        },
        "diagnostics": {
            "changed_refresh_count": changed_refreshes,
            "valid_anchor_refresh_count": valid_anchor_refreshes,
            "complete_unique_path_count": complete_paths,
        },
        **analysis,
        "gates": gates,
        "usage": usage,
    }


def _nonthinking_spec(spec: Any) -> Any:
    return replace(
        spec,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )


def _build_model(config: Config) -> StructuredChatModel:
    spec = _nonthinking_spec(config.model_pairs[0].questioner)
    if spec.model != MODEL_ID:
        raise ValueError("LongVid contrastive config selects the wrong model")
    return build_model_adapter(spec, config)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--qa-path", type=Path, required=True)
    parser.add_argument("--caption-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = PROJECTED_COST_USD
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 8
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = MAX_NEW_TOKENS
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: StructuredChatModel | None = None
    try:
        model = (
            DeterministicFixtureModel()
            if args.dry_run
            else _build_model(config)
        )
        payload = run_mechanics(
            config,
            qa_path=args.qa_path,
            caption_path=args.caption_path,
            raw_path=raw_path,
            model=model,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, MechanicsExecutionError):
            failure["usage"] = exc.usage
        elif model is not None:
            failure["usage"] = _usage(model)
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        _checkpoint(args.output_dir / "MECHANICS_FAILURE.json", failure)
        raise
    _checkpoint(args.output_dir / "MECHANICS.json", payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "summary": payload["summary"],
                "diagnostics": payload["diagnostics"],
                "gates": payload["gates"],
                "usage": payload["usage"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
