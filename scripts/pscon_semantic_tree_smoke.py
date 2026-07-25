#!/usr/bin/env python3
"""Run a truth-preserving PSCon semantic-tree development smoke."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import subprocess
import sys
from typing import Any, Protocol, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.atd_code_first_link_audit import spearman


INTERFACE_VERSION = "pscon-semantic-tree-smoke-1"
SOURCE_REPOSITORY = "https://github.com/JieZouIR/PSCon"
SOURCE_COMMIT = "42eabef33bdc7207841290fdbf4309e1a8d960f9"
CONVERSATION_SHA256 = (
    "219c54ebd94bceca302c3c10b9e9b3b3c0beeda40fa4480c57c7241151bfd49d"
)
KNOWLEDGE_GRAPH_SHA256 = (
    "d7b9dacc8c83aacaa174bb9955ec4240531076f1ff155bc4ccd07d4e53a4293b"
)
GENERATOR_MODEL_ID = "openai/gpt-5.4-mini"
RESPONDER_MODEL_ID = "openai/gpt-5.4"
CONVERSATION_ID = 64_937
EXPECTED_TARGET_ID = "B0CVRZZ1DD"
SEED = 24_385
TREE_ROOT_COUNT = 5
ROOT_OPTION_COUNT = 3
FOLLOWUPS_PER_BRANCH = 2
TREE_GENERATOR_REQUESTS = TREE_ROOT_COUNT + (
    TREE_ROOT_COUNT * ROOT_OPTION_COUNT * FOLLOWUPS_PER_BRANCH
)
WIDTH_ROOT_COUNT = TREE_GENERATOR_REQUESTS
GENERATOR_REQUESTS = TREE_GENERATOR_REQUESTS + WIDTH_ROOT_COUNT
RESPONDER_REQUESTS = (2 * TREE_ROOT_COUNT) + 2
EXPECTED_REQUESTS = GENERATOR_REQUESTS + RESPONDER_REQUESTS
MAX_COST_USD = 0.75

PLANNER_FIELDS = ("title",)
RESPONDER_FIELDS = (
    "title",
    "Brand",
    "Brand Name",
    "Screen Size",
    "Standing screen display size",
    "Display Technology",
    "Resolution",
    "Refresh Rate",
    "Special Feature",
    "Special Features",
    "Date First Available",
    "price",
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


class SmokeExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Product:
    product_id: str
    attributes: dict[str, tuple[str, ...]]

    @property
    def title(self) -> str:
        return self.attributes["title"][0]

    def planner_record(self, index: int) -> dict[str, Any]:
        return {
            "candidate_index": index,
            "title": self.title,
        }

    def responder_record(self) -> dict[str, Any]:
        return {
            key: list(self.attributes[key])
            for key in RESPONDER_FIELDS
            if key in self.attributes
        }


@dataclass(frozen=True)
class SemanticQuery:
    question: str
    options: tuple[str, ...]
    product_ids: tuple[str, ...]
    assignments: tuple[int, ...]

    def label_for(self, product_id: str) -> int:
        return self.assignments[self.product_ids.index(product_id)]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_source(source_root: Path) -> tuple[Path, Path]:
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError(f"PSCon commit is {commit}, expected {SOURCE_COMMIT}")
    conversation_path = source_root / "dataset" / "conversation_en.json"
    graph_path = source_root / "dataset" / "knowledgeGraph_en.txt"
    if _sha256(conversation_path) != CONVERSATION_SHA256:
        raise ValueError("PSCon English conversation hash does not match")
    if _sha256(graph_path) != KNOWLEDGE_GRAPH_SHA256:
        raise ValueError("PSCon English knowledge graph hash does not match")
    return conversation_path, graph_path


def _conversation_row(path: Path) -> dict[str, Any]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    matches = [row for row in rows if row.get("conv_id") == CONVERSATION_ID]
    if len(matches) != 1:
        raise ValueError("frozen PSCon conversation is missing or duplicated")
    return matches[0]


def _visible_task(row: dict[str, Any]) -> tuple[list[str], list[str]]:
    messages = row["conversation"]
    recommendations = [
        message for message in messages if message.get("action") == "Recommend"
    ]
    if not recommendations:
        raise ValueError("frozen conversation has no recommendation pool")
    final_recommendation = recommendations[-1]
    support_ids = list(
        dict.fromkeys(
            item["product_id"] for item in final_recommendation["search_results"]
        )
    )
    final_msg_id = final_recommendation["msg_id"]
    user_history = [
        message["content"].strip()
        for message in messages
        if message.get("role") == "user"
        and message.get("msg_id", 0) < final_msg_id
        and message.get("content", "").strip()
    ]
    return user_history, support_ids


def _target_id(row: dict[str, Any]) -> str:
    liked: list[str] = []
    for message in row["conversation"]:
        for rating in message.get("user_rating", []):
            if rating.get("product_rate", "").rstrip().endswith("(liked)"):
                liked.append(rating["product_id"])
    if liked != [EXPECTED_TARGET_ID]:
        raise ValueError(f"frozen target changed: {liked}")
    return liked[0]


def _load_products(graph_path: Path, support_ids: Sequence[str]) -> list[Product]:
    wanted = set(support_ids)
    records: dict[str, dict[str, list[str]]] = {
        product_id: {} for product_id in support_ids
    }
    with graph_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if len(row) < 3 or row[0] not in wanted:
                continue
            key = row[1].strip()
            value = ", ".join(row[2:]).strip()
            if key and value:
                records[row[0]].setdefault(key, []).append(value)
    products = [
        Product(
            product_id=product_id,
            attributes={
                key: tuple(values) for key, values in records[product_id].items()
            },
        )
        for product_id in support_ids
        if records[product_id].get("title")
    ]
    if len(products) != 20:
        raise ValueError(f"expected 20 title-bearing products, got {len(products)}")
    return products


def parse_semantic_query(
    text: str,
    products: Sequence[Product],
    *,
    expected_options: int | None,
) -> SemanticQuery:
    value = json.loads(text)
    if not isinstance(value, dict) or set(value) != {
        "question",
        "options",
        "assignments",
    }:
        raise ValueError("query must have exactly question/options/assignments")
    question = value["question"]
    options = value["options"]
    assignments = value["assignments"]
    if (
        not isinstance(question, str)
        or "\n" in question
        or not question.endswith("?")
        or not 8 <= len(question) <= 240
    ):
        raise ValueError("query question must be one bounded line ending in ?")
    if (
        not isinstance(options, list)
        or not all(
            isinstance(option, str)
            and option.strip() == option
            and 1 <= len(option) <= 100
            for option in options
        )
        or len(set(options)) != len(options)
    ):
        raise ValueError("query options are malformed or duplicated")
    if expected_options is not None and len(options) != expected_options:
        raise ValueError(f"query must have exactly {expected_options} options")
    if expected_options is None and not 1 <= len(options) <= ROOT_OPTION_COUNT:
        raise ValueError("followup query must have one to three options")
    if not isinstance(assignments, list) or len(assignments) != len(products):
        raise ValueError("query must assign every supplied product exactly once")
    if not all(type(label) is int and 1 <= label <= len(options) for label in assignments):
        raise ValueError("query assignments must be canonical option indices")
    if len(products) >= len(options) and set(assignments) != set(
        range(1, len(options) + 1)
    ):
        raise ValueError("every option must be used when support is large enough")
    return SemanticQuery(
        question=question,
        options=tuple(options),
        product_ids=tuple(product.product_id for product in products),
        assignments=tuple(assignments),
    )


def parse_answer(text: str, option_count: int) -> int:
    value = text.strip()
    if value not in {str(index) for index in range(1, option_count + 1)}:
        raise ValueError("responder answer is not a canonical option index")
    return int(value)


def entropy(labels: Sequence[int]) -> float:
    if not labels:
        return 0.0
    counts = {label: labels.count(label) for label in set(labels)}
    total = len(labels)
    return -sum(
        (count / total) * math.log(count / total) for count in counts.values()
    )


def query_eig(query: SemanticQuery, product_ids: Sequence[str]) -> float:
    return entropy([query.label_for(product_id) for product_id in product_ids])


def _query_messages(
    *,
    user_history: Sequence[str],
    products: Sequence[Product],
    query_index: int,
    history: dict[str, str] | None = None,
) -> list[dict[str, str]]:
    request: dict[str, Any] = {
        "user_history": list(user_history),
        "query_index": query_index,
        "candidate_products": [
            product.planner_record(index)
            for index, product in enumerate(products, start=1)
        ],
    }
    if history is not None:
        request["previous_clarification"] = history
    option_count = min(ROOT_OPTION_COUNT, len(products))
    return [
        {
            "role": "system",
            "content": (
                "Design one useful multiple-choice clarification question for a "
                "shopping assistant. Use only the supplied unstructured product "
                "titles and conversation. The options must be mutually exclusive, "
                "collectively exhaustive for these candidates, concise, and "
                "understandable to a user. Do not mention product IDs, model numbers, "
                "candidate indices, or ask which listed product they want. For a "
                "followup, ask a new question conditioned on the previous answer. "
                f"Return exactly {option_count} options and one integer assignment "
                "per candidate in input order. Output only strict JSON with exactly "
                'these keys: {"question":"...?","options":["...",...],'
                '"assignments":[1,...]}.'
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _answer_messages(
    *,
    product: Product,
    query: SemanticQuery,
) -> list[dict[str, str]]:
    request = {
        "hidden_preferred_product": product.responder_record(),
        "question": query.question,
        "numbered_options": [
            {"index": index, "option": option}
            for index, option in enumerate(query.options, start=1)
        ],
    }
    return [
        {
            "role": "system",
            "content": (
                "Act as an independent closed-book customer simulator. The hidden "
                "preferred product metadata is the entire truth. Select the single "
                "numbered option best supported by that metadata. Do not infer from "
                "candidate order or use outside knowledge. Output only the integer "
                "option index and no prose."
            ),
        },
        {"role": "user", "content": json.dumps(request, separators=(",", ":"))},
    ]


def _generate_queries(
    model: ChatModel,
    config: Config,
    messages: list[list[dict[str, str]]],
    product_groups: Sequence[Sequence[Product]],
    *,
    expected_options: int | None,
    temperature: float = 0.7,
) -> tuple[list[SemanticQuery], list[str]]:
    raw = model.chat_complete_messages_batched(
        messages,
        temperature=temperature,
        block_size=config.openrouter_concurrency,
        max_new_tokens=1200,
    )
    if len(raw) != len(messages):
        raise ValueError("generator response count changed")
    parsed = [
        parse_semantic_query(
            response,
            products,
            expected_options=expected_options,
        )
        for response, products in zip(raw, product_groups, strict=True)
    ]
    return parsed, raw


def _answer_queries(
    model: ChatModel,
    config: Config,
    product: Product,
    queries: Sequence[SemanticQuery],
) -> tuple[list[int], list[str]]:
    messages = [_answer_messages(product=product, query=query) for query in queries]
    raw = model.chat_complete_messages_batched(
        messages,
        temperature=0.0,
        block_size=config.openrouter_concurrency,
        max_new_tokens=8,
    )
    if len(raw) != len(queries):
        raise ValueError("responder response count changed")
    return [
        parse_answer(response, len(query.options))
        for response, query in zip(raw, queries, strict=True)
    ], raw


def _argmax(values: Sequence[float]) -> int:
    return max(range(len(values)), key=lambda index: (values[index], -index))


def _branch_ids(query: SemanticQuery, answer: int) -> list[str]:
    return [
        product_id
        for product_id, assignment in zip(
            query.product_ids, query.assignments, strict=True
        )
        if assignment == answer
    ]


def _target_mass(product_ids: Sequence[str], target_id: str) -> float:
    return (1.0 / len(product_ids)) if target_id in product_ids else 0.0


def _tree_scores(
    roots: Sequence[SemanticQuery],
    followups: Sequence[Sequence[Sequence[SemanticQuery]]],
    support_ids: Sequence[str],
) -> tuple[list[float], list[float]]:
    immediate: list[float] = []
    depth2: list[float] = []
    for root, root_followups in zip(roots, followups, strict=True):
        root_eig = query_eig(root, support_ids)
        expected_tail = 0.0
        for branch_index, branch_queries in enumerate(root_followups, start=1):
            branch = _branch_ids(root, branch_index)
            probability = len(branch) / len(support_ids)
            expected_tail += probability * max(
                query_eig(query, branch) for query in branch_queries
            )
        immediate.append(root_eig)
        depth2.append(root_eig + expected_tail)
    return immediate, depth2


def _execute_tree_roots(
    *,
    roots: Sequence[SemanticQuery],
    followups: Sequence[Sequence[Sequence[SemanticQuery]]],
    root_answers: Sequence[int],
    followup_answers: Sequence[int],
    target_id: str,
) -> list[dict[str, Any]]:
    records = []
    for index, root in enumerate(roots):
        root_answer = root_answers[index]
        after_root = _branch_ids(root, root_answer)
        branch_queries = followups[index][root_answer - 1]
        followup_eigs = [query_eig(query, after_root) for query in branch_queries]
        followup_index = _argmax(followup_eigs)
        followup = branch_queries[followup_index]
        after_followup = _branch_ids(followup, followup_answers[index])
        root_consistent = root.label_for(target_id) == root_answer
        followup_consistent = (
            target_id in followup.product_ids
            and followup.label_for(target_id) == followup_answers[index]
        )
        records.append(
            {
                "root_index": index,
                "root_answer": root_answer,
                "root_consistent": root_consistent,
                "after_root_count": len(after_root),
                "followup_index": followup_index,
                "followup_answer": followup_answers[index],
                "followup_consistent": followup_consistent,
                "final_count": len(after_followup),
                "target_mass": _target_mass(after_followup, target_id),
            }
        )
    return records


def _snapshot(model: ChatModel) -> dict[str, Any]:
    return model.usage_snapshot()


def aggregate_usage(generator: ChatModel, responder: ChatModel) -> dict[str, Any]:
    snapshots = {
        "generator": _snapshot(generator),
        "responder": _snapshot(responder),
    }
    return {
        "physical_requests": sum(
            int(value.get("adapter_requests", 0)) for value in snapshots.values()
        ),
        "http_attempts": sum(
            int(value.get("http_attempts", 0)) for value in snapshots.values()
        ),
        "retry_count": sum(
            int(value.get("retry_count", 0)) for value in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(value.get("adapter_reasoning_tokens", 0))
            for value in snapshots.values()
        ),
        "forced_exits": sum(
            int(value.get("forced_exits", 0)) for value in snapshots.values()
        ),
        "adapter_cost_usd": sum(
            float(value.get("adapter_cost_usd", 0.0))
            for value in snapshots.values()
        ),
        **snapshots,
    }


def _checkpoint(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    config: Config,
    *,
    source_root: Path,
    raw_path: Path,
    generator: ChatModel,
    responder: ChatModel,
) -> dict[str, Any]:
    conversation_path, graph_path = verify_source(source_root)
    row = _conversation_row(conversation_path)
    user_history, raw_support_ids = _visible_task(row)
    products = _load_products(graph_path, raw_support_ids)
    support_ids = [product.product_id for product in products]
    product_by_id = {product.product_id: product for product in products}
    raw: dict[str, Any] = {
        "conversation_id": CONVERSATION_ID,
        "support_ids": support_ids,
        "target_loaded": False,
    }
    try:
        root_messages = [
            _query_messages(
                user_history=user_history,
                products=products,
                query_index=index,
            )
            for index in range(TREE_ROOT_COUNT)
        ]
        roots, raw["tree_roots"] = _generate_queries(
            generator,
            config,
            root_messages,
            [products] * TREE_ROOT_COUNT,
            expected_options=ROOT_OPTION_COUNT,
        )

        followup_messages: list[list[dict[str, str]]] = []
        followup_groups: list[list[Product]] = []
        followup_layout: list[tuple[int, int]] = []
        for root_index, root in enumerate(roots):
            for answer in range(1, ROOT_OPTION_COUNT + 1):
                branch_ids = _branch_ids(root, answer)
                branch_products = [product_by_id[value] for value in branch_ids]
                for followup_index in range(FOLLOWUPS_PER_BRANCH):
                    followup_messages.append(
                        _query_messages(
                            user_history=user_history,
                            products=branch_products,
                            query_index=followup_index,
                            history={
                                "question": root.question,
                                "answer": root.options[answer - 1],
                            },
                        )
                    )
                    followup_groups.append(branch_products)
                    followup_layout.append((root_index, answer - 1))
        flat_followups, raw["tree_followups"] = _generate_queries(
            generator,
            config,
            followup_messages,
            followup_groups,
            expected_options=None,
        )
        followups: list[list[list[SemanticQuery]]] = [
            [[] for _ in range(ROOT_OPTION_COUNT)] for _ in roots
        ]
        for layout, query in zip(followup_layout, flat_followups, strict=True):
            followups[layout[0]][layout[1]].append(query)

        width_messages = [
            _query_messages(
                user_history=user_history,
                products=products,
                query_index=index + TREE_ROOT_COUNT,
            )
            for index in range(WIDTH_ROOT_COUNT)
        ]
        width_roots, raw["width_roots"] = _generate_queries(
            generator,
            config,
            width_messages,
            [products] * WIDTH_ROOT_COUNT,
            expected_options=ROOT_OPTION_COUNT,
        )
        immediate_scores, depth2_scores = _tree_scores(
            roots, followups, support_ids
        )
        width_scores = [query_eig(query, support_ids) for query in width_roots]
        raw["scores_frozen"] = {
            "tree_immediate": immediate_scores,
            "tree_depth2": depth2_scores,
            "width_immediate": width_scores,
        }
        raw["target_loaded"] = False
        _checkpoint(raw_path, raw)

        # The liked product is first inspected only after every planning score freezes.
        target_id = _target_id(row)
        target = product_by_id[target_id]
        raw["target_loaded"] = True
        root_answers, raw["tree_root_answers"] = _answer_queries(
            responder, config, target, roots
        )
        selected_followups = []
        for index, (root, answer) in enumerate(zip(roots, root_answers, strict=True)):
            branch = _branch_ids(root, answer)
            candidates = followups[index][answer - 1]
            selected_followups.append(
                candidates[
                    _argmax([query_eig(query, branch) for query in candidates])
                ]
            )
        followup_answers, raw["tree_followup_answers"] = _answer_queries(
            responder, config, target, selected_followups
        )
        tree_records = _execute_tree_roots(
            roots=roots,
            followups=followups,
            root_answers=root_answers,
            followup_answers=followup_answers,
            target_id=target_id,
        )

        width_first_index = _argmax(width_scores)
        width_first = width_roots[width_first_index]
        width_first_answers, raw["width_first_answer"] = _answer_queries(
            responder, config, target, [width_first]
        )
        width_after_first = _branch_ids(width_first, width_first_answers[0])
        width_second_scores = [
            query_eig(query, width_after_first) if index != width_first_index else -1.0
            for index, query in enumerate(width_roots)
        ]
        width_second_index = _argmax(width_second_scores)
        width_second = width_roots[width_second_index]
        width_second_answers, raw["width_second_answer"] = _answer_queries(
            responder, config, target, [width_second]
        )
        width_final = [
            product_id
            for product_id in width_after_first
            if width_second.label_for(product_id) == width_second_answers[0]
        ]
        raw["complete"] = True
        _checkpoint(raw_path, raw)
        usage = aggregate_usage(generator, responder)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise SmokeExecutionError(
            f"{type(exc).__name__}: {exc}",
            aggregate_usage(generator, responder),
        ) from exc

    endpoints = [record["target_mass"] for record in tree_records]
    myopic_index = _argmax(immediate_scores)
    nonmyopic_index = _argmax(depth2_scores)
    random_index = random.Random(SEED).randrange(TREE_ROOT_COUNT)
    immediate_rho = spearman(immediate_scores, endpoints)
    depth2_rho = spearman(depth2_scores, endpoints)
    width_endpoint = _target_mass(width_final, target_id)
    root_consistency = statistics.fmean(
        float(record["root_consistent"]) for record in tree_records
    )
    followup_consistency = statistics.fmean(
        float(record["followup_consistent"]) for record in tree_records
    )
    gates = {
        "exact_request_count": usage["physical_requests"] == EXPECTED_REQUESTS,
        "exact_http_attempt_count": usage["http_attempts"] == EXPECTED_REQUESTS,
        "zero_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "support_has_twenty_products_and_target": (
            len(support_ids) == 20 and target_id in support_ids
        ),
        "at_least_four_unique_tree_questions": len(
            {query.question for query in roots}
        )
        >= 4,
        "immediate_score_range_at_least_0_10": (
            max(immediate_scores) - min(immediate_scores) >= 0.10
        ),
        "depth2_score_range_at_least_0_10": (
            max(depth2_scores) - min(depth2_scores) >= 0.10
        ),
        "root_responder_consistency_at_least_0_80": root_consistency >= 0.80,
        "followup_responder_consistency_at_least_0_80": (
            followup_consistency >= 0.80
        ),
        "nonmyopic_root_differs_from_myopic": nonmyopic_index != myopic_index,
        "depth2_rho_positive": depth2_rho is not None and depth2_rho > 0.0,
        "depth2_rho_exceeds_immediate": (
            depth2_rho is not None
            and (immediate_rho is None or depth2_rho > immediate_rho)
        ),
        "nonmyopic_endpoint_exceeds_myopic": (
            endpoints[nonmyopic_index] > endpoints[myopic_index]
        ),
        "nonmyopic_endpoint_exceeds_width": (
            endpoints[nonmyopic_index] > width_endpoint
        ),
        "nonmyopic_endpoint_exceeds_random": (
            endpoints[nonmyopic_index] > endpoints[random_index]
        ),
        "cost_at_most_0_75": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "source_repository": SOURCE_REPOSITORY,
            "source_commit": SOURCE_COMMIT,
            "conversation_sha256": CONVERSATION_SHA256,
            "knowledge_graph_sha256": KNOWLEDGE_GRAPH_SHA256,
            "conversation_id": CONVERSATION_ID,
            "generator_model": GENERATOR_MODEL_ID,
            "responder_model": RESPONDER_MODEL_ID,
            "seed": SEED,
            "support_size": len(support_ids),
            "planner_fields": list(PLANNER_FIELDS),
            "tree_root_count": TREE_ROOT_COUNT,
            "root_option_count": ROOT_OPTION_COUNT,
            "followups_per_branch": FOLLOWUPS_PER_BRANCH,
            "tree_generator_requests": TREE_GENERATOR_REQUESTS,
            "width_generator_requests": WIDTH_ROOT_COUNT,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "scores_frozen_before_target_lookup": True,
            "repairs_or_reissues": 0,
        },
        "metrics": {
            "myopic_root_index": myopic_index,
            "nonmyopic_root_index": nonmyopic_index,
            "random_root_index": random_index,
            "width_first_root_index": width_first_index,
            "width_second_root_index": width_second_index,
            "immediate_score_endpoint_spearman": immediate_rho,
            "depth2_score_endpoint_spearman": depth2_rho,
            "myopic_endpoint": endpoints[myopic_index],
            "nonmyopic_endpoint": endpoints[nonmyopic_index],
            "width_endpoint": width_endpoint,
            "random_endpoint": endpoints[random_index],
            "root_responder_consistency": root_consistency,
            "followup_responder_consistency": followup_consistency,
        },
        "tree_roots": [
            {
                **record,
                "question": roots[index].question,
                "options": list(roots[index].options),
                "immediate_eig": immediate_scores[index],
                "depth2_eig": depth2_scores[index],
            }
            for index, record in enumerate(tree_records)
        ],
        "width_execution": {
            "first_question": width_first.question,
            "first_answer": width_first_answers[0],
            "after_first_count": len(width_after_first),
            "second_question": width_second.question,
            "second_answer": width_second_answers[0],
            "final_count": len(width_final),
            "target_mass": width_endpoint,
        },
        "gates": gates,
        "usage": usage,
    }


class DeterministicFixtureModel:
    def __init__(self, role: str) -> None:
        self.role = role
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        max_new_tokens: int | None = None,
    ) -> list[str]:
        del temperature, block_size, max_new_tokens
        responses: list[str] = []
        for messages in batch_messages:
            request = json.loads(messages[-1]["content"])
            if self.role == "responder":
                responses.append("1")
                continue
            count = len(request["candidate_products"])
            option_count = min(ROOT_OPTION_COUNT, count)
            offset = int(request["query_index"]) % option_count
            assignments = [
                ((index + offset) % option_count) + 1 for index in range(count)
            ]
            responses.append(
                json.dumps(
                    {
                        "question": (
                            f"Which fixture preference applies for query "
                            f"{request['query_index']}?"
                        ),
                        "options": [
                            f"Fixture option {index}"
                            for index in range(1, option_count + 1)
                        ],
                        "assignments": assignments,
                    },
                    separators=(",", ":"),
                )
            )
        self.requests += len(responses)
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.0,
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


def _build_models(config: Config) -> tuple[ChatModel, ChatModel]:
    generator_spec = _nonthinking_spec(config.model_pairs[0].questioner)
    responder_spec = _nonthinking_spec(config.model_pairs[0].answerer)
    if generator_spec.model != GENERATOR_MODEL_ID:
        raise ValueError("PSCon config selects the wrong generator")
    if responder_spec.model != RESPONDER_MODEL_ID:
        raise ValueError("PSCon config selects the wrong responder")
    return (
        build_model_adapter(generator_spec, config),
        build_model_adapter(responder_spec, config),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.20
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    if args.dry_run:
        generator: ChatModel = DeterministicFixtureModel("generator")
        responder: ChatModel = DeterministicFixtureModel("responder")
    else:
        generator, responder = _build_models(config)
    try:
        payload = run_smoke(
            config,
            source_root=args.source_root,
            raw_path=raw_path,
            generator=generator,
            responder=responder,
        )
        payload["protocol"]["private_raw_sha256"] = _sha256(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, SmokeExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = _sha256(raw_path)
        _checkpoint(args.output_dir / "SMOKE_FAILURE.json", failure)
        raise
    output = args.output_dir / "SMOKE.json"
    _checkpoint(output, payload)
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(output),
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
