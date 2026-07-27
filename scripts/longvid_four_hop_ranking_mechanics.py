#!/usr/bin/env python3
"""Test LLM support-entropy ranking on disclosed LongVid four-hop root pairs."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sys
from typing import Any, Protocol, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.bright_biology_unlock_audit import BM25Corpus, tokenize
from scripts.longvid_bridge_path_opportunity_audit import (
    CAPTION_SHA256,
    QA_SHA256,
    _id_hash,
    load_selected_captions,
    root_queries,
    sha256_file,
    stream_json_array,
)
from scripts.longvid_four_hop_support_smoke import (
    MODEL_ID,
    SUPPORT_SIZE,
    parse_support,
    support_signature,
)
from scripts.longvid_four_hop_tradeoff_opportunity import load_selected_qa


INTERFACE_VERSION = "longvid-four-hop-ranking-mechanics-1"
SEED = 270739
RANDOM_CONTROL_SEED = 270740
EXPECTED_REQUESTS = 54
MAX_COST_USD = 0.75
PROJECTED_COST_USD = 0.35
OBSERVATION_CHAR_CAP = 2_000
MIN_VALID_ANCHORS_PER_REFRESH = 4
MIN_CHANGED_REFRESHES = 44
MIN_RANKABLE_TASKS = 4
MIN_SCORE_RANGE_TASKS = 4
MIN_FINAL_PAIRWISE_ACCURACY = 0.75
MIN_FINAL_SPEARMAN = 0.25

STRUCTURAL_AUDIT_SHA256 = (
    "3cb882b1facbef2ababe2f8be68529423773c362a1eb4402b68d0bb240cc3ce0"
)
STRUCTURAL_CONFIRMATION_SHA256 = (
    "895e448c047ca7afa924393da0a4f637a21735c8a770336fa17db0a72fdd4081"
)
SUPPORT_SMOKE_SHA256 = (
    "0e594dbdaa748160e9006830992cca000ccfc7badec8cdde3b6d7ca56d412239"
)

TASK_LAYOUT = (
    (955, (5, 0)),
    (1802, (2, 1)),
    (540, (0, 9)),
    (479, (5, 4)),
    (1332, (2, 3)),
    (1068, (3, 13)),
)
TASK_LAYOUT_HASH = (
    "6beaa5bccbe31aac2eedb44cf73ea2c3a6f11846c88012d325bb3c0247fbd3a3"
)


class ChatModel(Protocol):
    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]: ...

    def usage_snapshot(self) -> dict[str, Any]: ...


class MechanicsExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


def _canonical(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")


def _normalize(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.casefold()))


def layout_hash() -> str:
    text = "\n".join(
        f"{row_index}:{roots[0]},{roots[1]}"
        for row_index, roots in TASK_LAYOUT
    )
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def support_entropy(support: Sequence[dict[str, Any]]) -> float:
    weights = np.asarray([float(row["weight"]) for row in support], dtype=float)
    probabilities = weights / weights.sum()
    return float(-np.sum(probabilities * np.log(probabilities)))


def initial_messages(question: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You maintain a semantic belief support over possible multi-clip "
                "evidence chains for video retrieval. Do not reason aloud. Return "
                "exactly six lines and no other text. Each line must be "
                "Hn|weight|anchor|search_query|hypothesis. Labels are H1 through "
                "H6 in order. weight is an integer 1..100 and at least two weights "
                "must differ. anchor must be QUESTION. search_query is concise, "
                "hypothesis states a distinct plausible chain of unseen evidence. "
                "Do not use the pipe character inside fields."
            ),
        },
        {
            "role": "user",
            "content": (
                "Generate six distinct weighted evidence-chain hypotheses and six "
                f"distinct first searches for this question:\n{question}"
            ),
        },
    ]


def _format_support(support: Sequence[dict[str, Any]]) -> str:
    return "\n".join(
        (
            f"{row['label']}|{row['weight']}|{row['anchor']}|"
            f"{row['search_query']}|{row['hypothesis']}"
        )
        for row in support
    )


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
                "You update a semantic belief support after one retrieved video "
                "caption. Do not reason aloud. Return exactly six lines and no "
                "other text. Each line must be "
                "Hn|weight|anchor|search_query|hypothesis. Labels are H1 through "
                "H6 in order. weight is an integer 1..100 and at least two weights "
                "must differ. For every line, anchor must be one exact alphanumeric "
                "token copied from OBSERVATION, absent from QUESTION and "
                "PREVIOUS_QUERY. Each search_query must use its anchor and seek "
                "the next missing evidence link. Hypotheses must be distinct and "
                "updated from the observation. Do not use pipes inside fields."
            ),
        },
        {
            "role": "user",
            "content": (
                f"QUESTION:\n{question}\n"
                f"PREVIOUS_QUERY:\n{previous_query}\n"
                f"OBSERVATION:\n{observation[:OBSERVATION_CHAR_CAP]}\n"
                f"PREVIOUS_SUPPORT:\n{_format_support(previous_support)}"
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
    return sum(
        _normalize(row["anchor"]) in observation_terms
        and _normalize(row["anchor"]) not in excluded
        and _normalize(row["anchor"]) in set(tokenize(row["search_query"]))
        for row in support
    )


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


def _select_query(support: Sequence[dict[str, Any]]) -> str:
    index = max(
        range(len(support)),
        key=lambda value: (support[value]["weight"], -value),
    )
    return str(support[index]["search_query"])


def _average_ranks(values: Sequence[float]) -> np.ndarray:
    order = sorted(range(len(values)), key=lambda index: (values[index], index))
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average = (start + end - 1) / 2.0 + 1.0
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def _spearman(values: Sequence[float], targets: Sequence[float]) -> float:
    if len(values) != len(targets) or len(values) < 2:
        return 0.0
    value_ranks = _average_ranks(values)
    target_ranks = _average_ranks(targets)
    if np.std(value_ranks) == 0.0 or np.std(target_ranks) == 0.0:
        return 0.0
    statistic = float(np.corrcoef(value_ranks, target_ranks)[0, 1])
    return statistic if math.isfinite(statistic) else 0.0


def analyze_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    flattened = [
        candidate
        for record in records
        for candidate in record["candidates"]
    ]
    final_scores = [float(candidate["final_score"]) for candidate in flattened]
    immediate_scores = [
        float(candidate["immediate_score"]) for candidate in flattened
    ]
    coverage = [float(candidate["coverage_count"]) for candidate in flattened]
    final_rho = _spearman(final_scores, coverage)
    immediate_rho = _spearman(immediate_scores, coverage)

    random_generator = random.Random(RANDOM_CONTROL_SEED)
    task_metrics = []
    final_correct = 0
    immediate_correct = 0
    rankable_count = 0
    score_range_count = 0
    final_total = 0
    immediate_total = 0
    random_total = 0
    for record in records:
        candidates = record["candidates"]
        final_choice = max(
            range(2),
            key=lambda index: (candidates[index]["final_score"], -index),
        )
        immediate_choice = max(
            range(2),
            key=lambda index: (candidates[index]["immediate_score"], -index),
        )
        random_choice = random_generator.randrange(2)
        coverage_pair = [int(candidate["coverage_count"]) for candidate in candidates]
        rankable = coverage_pair[0] != coverage_pair[1]
        if rankable:
            rankable_count += 1
            best_coverage = max(coverage_pair)
            final_correct += int(
                coverage_pair[final_choice] == best_coverage
            )
            immediate_correct += int(
                coverage_pair[immediate_choice] == best_coverage
            )
        score_range = abs(
            float(candidates[0]["final_score"])
            - float(candidates[1]["final_score"])
        )
        score_range_count += int(score_range >= 0.02)
        final_total += coverage_pair[final_choice]
        immediate_total += coverage_pair[immediate_choice]
        random_total += coverage_pair[random_choice]
        task_metrics.append(
            {
                "row_index": record["row_index"],
                "category": record["category"],
                "root_indices": [
                    int(candidate["root_index"]) for candidate in candidates
                ],
                "coverage_counts": coverage_pair,
                "immediate_scores": [
                    float(candidate["immediate_score"])
                    for candidate in candidates
                ],
                "final_scores": [
                    float(candidate["final_score"])
                    for candidate in candidates
                ],
                "rankable": rankable,
                "final_choice": final_choice,
                "immediate_choice": immediate_choice,
                "random_choice": random_choice,
                "score_range": score_range,
            }
        )
    final_accuracy = (
        final_correct / rankable_count if rankable_count else 0.0
    )
    immediate_accuracy = (
        immediate_correct / rankable_count if rankable_count else 0.0
    )
    return {
        "task_metrics": task_metrics,
        "summary": {
            "rankable_task_count": rankable_count,
            "final_score_range_task_count": score_range_count,
            "final_pairwise_correct": final_correct,
            "immediate_pairwise_correct": immediate_correct,
            "final_pairwise_accuracy": final_accuracy,
            "immediate_pairwise_accuracy": immediate_accuracy,
            "final_score_spearman": final_rho,
            "immediate_score_spearman": immediate_rho,
            "final_selected_coverage_total": final_total,
            "immediate_selected_coverage_total": immediate_total,
            "random_selected_coverage_total": random_total,
        },
    }


def _usage(model: ChatModel) -> dict[str, Any]:
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


class DeterministicFixtureModel:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses = []
        for offset, messages in enumerate(batch_messages):
            user = messages[-1]["content"]
            question_match = re.search(
                r"QUESTION:\n(.+?)\nPREVIOUS_QUERY:",
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
            lines = [
                (
                    f"H{index}|{10 + index + (suffix % 3)}|{anchor}|"
                    f"{anchor} fixture continuation {suffix} {index}|"
                    f"Distinct fixture evidence chain {suffix} number {index}"
                )
                for index in range(1, SUPPORT_SIZE + 1)
            ]
            responses.append("\n".join(lines))
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


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_mechanics(
    config: Config,
    *,
    qa_path: Path,
    caption_path: Path,
    raw_path: Path,
    model: ChatModel,
) -> dict[str, Any]:
    if sha256_file(qa_path) != QA_SHA256:
        raise ValueError("QA SHA-256 does not match frozen source")
    if sha256_file(caption_path) != CAPTION_SHA256:
        raise ValueError("caption SHA-256 does not match frozen source")
    if layout_hash() != TASK_LAYOUT_HASH:
        raise AssertionError("task layout hash does not reproduce")
    root = Path(__file__).resolve().parents[1]
    bindings = {
        "structural_audit": (
            root
            / "results/nonmyopic/longvid_four_hop_tradeoff_opportunity/AUDIT.json"
        ),
        "structural_confirmation": (
            root
            / "results/nonmyopic/longvid_four_hop_tradeoff_confirmation_v2/"
            "CONFIRMATION.json"
        ),
        "support_smoke": (
            root
            / "results/nonmyopic/longvid_four_hop_support_smoke/"
            "longvid-four-hop-support-smoke-v2-20260727T145208Z/SMOKE.json"
        ),
    }
    expected_bindings = {
        "structural_audit": STRUCTURAL_AUDIT_SHA256,
        "structural_confirmation": STRUCTURAL_CONFIRMATION_SHA256,
        "support_smoke": SUPPORT_SMOKE_SHA256,
    }
    for name, path in bindings.items():
        if sha256_file(path) != expected_bindings[name]:
            raise ValueError(f"{name} artifact hash does not match")

    selected_ids = {row_index for row_index, _ in TASK_LAYOUT}
    visible_rows = _visible_rows(qa_path, selected_ids)
    video_ids = {row["video_id"] for row in visible_rows.values()}
    captions = load_selected_captions(caption_path, video_ids)
    corpora = {
        row_index: BM25Corpus(captions[row["video_id"]])
        for row_index, row in visible_rows.items()
    }
    raw: dict[str, Any] = {
        "interface_version": INTERFACE_VERSION,
        "task_layout": TASK_LAYOUT,
        "task_layout_hash": TASK_LAYOUT_HASH,
        "endpoint_loaded": False,
        "visible_rows": visible_rows,
        "responses": {"initial": None, "depths": []},
    }
    try:
        initial_responses = model.chat_complete_messages_batched(
            [
                initial_messages(visible_rows[row_index]["question"])
                for row_index, _ in TASK_LAYOUT
            ],
            temperature=0.0,
            block_size=config.openrouter_concurrency,
            max_new_tokens=900,
        )
        raw["responses"]["initial"] = initial_responses
        _checkpoint(raw_path, raw)
        initial_supports = [
            parse_support(response) for response in initial_responses
        ]

        paths: list[dict[str, Any]] = []
        for task_offset, (row_index, root_indices) in enumerate(TASK_LAYOUT):
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
                        "initial_support": initial_supports[task_offset],
                        "current_support": initial_supports[task_offset],
                        "initial_entropy": support_entropy(
                            initial_supports[task_offset]
                        ),
                        "entropies": [],
                        "support_signatures": [
                            support_signature(initial_supports[task_offset])
                        ],
                        "valid_anchor_counts": [],
                    }
                )

        for depth in range(4):
            requests = [
                refresh_messages(
                    path["question"],
                    path["previous_query"],
                    path["current_document"]["raw_source"],
                    path["current_support"],
                )
                for path in paths
            ]
            responses = model.chat_complete_messages_batched(
                requests,
                temperature=0.0,
                block_size=config.openrouter_concurrency,
                max_new_tokens=900,
            )
            raw["responses"]["depths"].append(responses)
            _checkpoint(raw_path, raw)
            supports = [parse_support(response) for response in responses]
            for path, support in zip(paths, supports):
                anchor_count = valid_anchor_count(
                    support,
                    question=path["question"],
                    previous_query=path["previous_query"],
                    observation=path["current_document"]["raw_source"],
                )
                path["valid_anchor_counts"].append(anchor_count)
                path["entropies"].append(support_entropy(support))
                path["support_signatures"].append(support_signature(support))
                path["current_support"] = support
                if depth < 3:
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

        policy_records = []
        changed_refreshes = 0
        valid_refreshes = 0
        complete_paths = 0
        for row_index, _ in TASK_LAYOUT:
            candidates = []
            for path in [
                value for value in paths if value["row_index"] == row_index
            ]:
                changed_refreshes += sum(
                    left != right
                    for left, right in zip(
                        path["support_signatures"],
                        path["support_signatures"][1:],
                    )
                )
                valid_refreshes += sum(
                    count >= MIN_VALID_ANCHORS_PER_REFRESH
                    for count in path["valid_anchor_counts"]
                )
                complete_paths += int(
                    len(path["retrieved_ids"]) == 4
                    and len(set(path["retrieved_ids"])) == 4
                )
                candidates.append(
                    {
                        "root_index": path["root_index"],
                        "retrieved_ids": path["retrieved_ids"],
                        "initial_entropy": path["initial_entropy"],
                        "entropies": path["entropies"],
                        "immediate_score": (
                            path["initial_entropy"] - path["entropies"][0]
                        ),
                        "final_score": (
                            path["initial_entropy"] - path["entropies"][-1]
                        ),
                        "valid_anchor_counts": path["valid_anchor_counts"],
                        "support_signatures": path["support_signatures"],
                    }
                )
            policy_records.append(
                {
                    "row_index": row_index,
                    "category": visible_rows[row_index]["category"],
                    "candidates": candidates,
                }
            )
        raw["policy_records_before_endpoint"] = policy_records
        raw["endpoint_loaded"] = False
        _checkpoint(raw_path, raw)

        endpoint_rows = load_selected_qa(qa_path, selected_ids)
        raw["endpoint_loaded"] = True
        for record in policy_records:
            evidence = {
                str(int(value))
                for value in endpoint_rows[record["row_index"]]["evidence_slices"]
            }
            for candidate in record["candidates"]:
                candidate["coverage_count"] = len(
                    evidence & set(candidate["retrieved_ids"])
                )
        raw["records_with_endpoint"] = policy_records
        _checkpoint(raw_path, raw)
        usage = _usage(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise MechanicsExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage(model),
        ) from exc

    analysis = analyze_records(policy_records)
    summary = analysis["summary"]
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_supports_parse": True,
        "endpoint_loaded_only_after_policy_freeze": raw["endpoint_loaded"],
        "all_twelve_paths_complete_and_unique": complete_paths == 12,
        "at_least_44_of_48_supports_change": (
            changed_refreshes >= MIN_CHANGED_REFRESHES
        ),
        "all_48_refreshes_have_four_valid_anchors": valid_refreshes == 48,
        "rankable_tasks_at_least_4": (
            summary["rankable_task_count"] >= MIN_RANKABLE_TASKS
        ),
        "final_score_range_tasks_at_least_4": (
            summary["final_score_range_task_count"] >= MIN_SCORE_RANGE_TASKS
        ),
        "final_pairwise_accuracy_at_least_0_75": (
            summary["final_pairwise_accuracy"]
            >= MIN_FINAL_PAIRWISE_ACCURACY
        ),
        "final_pairwise_accuracy_beats_immediate": (
            summary["final_pairwise_accuracy"]
            > summary["immediate_pairwise_accuracy"]
        ),
        "final_spearman_at_least_0_25": (
            summary["final_score_spearman"] >= MIN_FINAL_SPEARMAN
        ),
        "final_spearman_beats_immediate": (
            summary["final_score_spearman"]
            > summary["immediate_score_spearman"]
        ),
        "final_selected_coverage_at_least_immediate": (
            summary["final_selected_coverage_total"]
            >= summary["immediate_selected_coverage_total"]
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
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "reasoning_requested": False,
            "seed": SEED,
            "random_control_seed": RANDOM_CONTROL_SEED,
            "task_layout_hash": TASK_LAYOUT_HASH,
            "expected_requests": EXPECTED_REQUESTS,
            "repairs_or_reissues": 0,
            "max_cost_usd": MAX_COST_USD,
            "endpoint_loaded_after_policy_freeze": True,
        },
        "diagnostics": {
            "changed_refresh_count": changed_refreshes,
            "valid_anchor_refresh_count": valid_refreshes,
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


def _build_model(config: Config) -> ChatModel:
    spec = _nonthinking_spec(config.model_pairs[0].questioner)
    if spec.model != MODEL_ID:
        raise ValueError("LongVid ranking config selects the wrong model")
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
    config.openrouter_concurrency = 12
    config.openrouter_max_retries = 0
    config.openrouter_max_output_tokens = 900
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    model: ChatModel | None = None
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
