#!/usr/bin/env python3
"""Run the frozen BrowseComp-Plus LLM-native first-link mechanics gate."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


MODEL_ID = "openai/gpt-5.4"
INTERFACE_VERSION = "browsecomp-plus-semantic-mechanics-2"
SOURCE_SHA256 = (
    "d3192f97b171c7d5d34f8472b52abcff895f3fcb97013d82b41073e1046f9f85"
)
TASK_IDS = ("286", "1058", "747", "787", "854")
HYPOTHESIS_COUNT = 8
ROOT_COUNT = 6
RETRIEVAL_TOP_K = 3
INDEX_TEXT_CHARS = 30_000
OBSERVATION_CHARS = 1_200
QUERY_MAX_CHARS = 400
EXPECTED_REQUESTS = 55
MAX_COST_USD = 0.90
RANDOM_SEED = 24_407
TEMPERATURE = 0.0


class GateExecutionError(RuntimeError):
    def __init__(self, message: str, usage: dict[str, Any]) -> None:
        super().__init__(message)
        self.usage = usage


@dataclass(frozen=True)
class Belief:
    hypotheses: tuple[str, ...]
    weights: tuple[int, ...]

    @property
    def signature(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (normalize_answer(hypothesis), weight)
            for hypothesis, weight in zip(self.hypotheses, self.weights)
        )


@dataclass(frozen=True)
class Strategy:
    root_query: str
    direct_score: int
    future_intent: str


@dataclass
class Branch:
    root_index: int
    strategy: Strategy
    root_documents: list[dict[str, Any]]
    root_belief: Belief | None = None
    adaptive_query: str | None = None
    adaptive_documents: list[dict[str, Any]] | None = None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    unpunctuated = "".join(
        character
        for character in lowered
        if character not in string.punctuation
    )
    return " ".join(
        token
        for token in unpunctuated.split()
        if token not in {"a", "an", "the"}
    )


def normalize_query(value: str) -> str:
    return " ".join(value.lower().split())


def _response_lines(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        raise ValueError("response is empty")
    lines = stripped.splitlines()
    if any(line != line.strip() or not line for line in lines):
        raise ValueError("response contains padded or blank lines")
    return lines


def _canonical_integer(value: str, *, minimum: int, maximum: int) -> int:
    if (
        not value.isdigit()
        or (len(value) > 1 and value.startswith("0"))
    ):
        raise ValueError("value must be a canonical integer")
    parsed = int(value)
    if not minimum <= parsed <= maximum:
        raise ValueError("integer is outside its allowed range")
    return parsed


def _parse_belief(lines: Sequence[str]) -> Belief:
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError("wrong hypothesis line count")
    hypotheses: list[str] = []
    weights: list[int] = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|")
        if len(parts) != 3 or parts[0] != f"H{index:02d}":
            raise ValueError("invalid hypothesis line")
        weights.append(
            _canonical_integer(parts[1], minimum=1, maximum=100)
        )
        hypothesis = " ".join(parts[2].split())
        if not hypothesis:
            raise ValueError("hypothesis is empty")
        hypotheses.append(hypothesis)
    normalized = {normalize_answer(value) for value in hypotheses}
    if "" in normalized or len(normalized) != HYPOTHESIS_COUNT:
        raise ValueError("hypotheses must be normalized-distinct")
    if sum(weights) != 100:
        raise ValueError("hypothesis weights must sum to 100")
    return Belief(tuple(hypotheses), tuple(weights))


def _clean_text_field(value: str, *, maximum: int) -> str:
    cleaned = " ".join(value.split())
    if not cleaned or len(cleaned) > maximum or "|" in cleaned:
        raise ValueError("flat text field is invalid")
    return cleaned


def parse_initial(text: str) -> tuple[Belief, list[Strategy]]:
    lines = _response_lines(text)
    if len(lines) != HYPOTHESIS_COUNT + ROOT_COUNT:
        raise ValueError("initial response has wrong line count")
    belief = _parse_belief(lines[:HYPOTHESIS_COUNT])
    strategies: list[Strategy] = []
    for index, line in enumerate(lines[HYPOTHESIS_COUNT:], start=1):
        parts = line.split("|")
        if len(parts) != 4 or parts[0] != f"S{index:02d}":
            raise ValueError("invalid strategy line")
        strategies.append(
            Strategy(
                direct_score=_canonical_integer(
                    parts[1],
                    minimum=0,
                    maximum=100,
                ),
                root_query=_clean_text_field(
                    parts[2],
                    maximum=QUERY_MAX_CHARS,
                ),
                future_intent=_clean_text_field(parts[3], maximum=300),
            )
        )
    roots = {normalize_query(strategy.root_query) for strategy in strategies}
    if len(roots) != ROOT_COUNT:
        raise ValueError("root queries must be normalized-distinct")
    return belief, strategies


def parse_refresh(text: str) -> tuple[Belief, str]:
    lines = _response_lines(text)
    if len(lines) != HYPOTHESIS_COUNT + 1:
        raise ValueError("refresh response has wrong line count")
    belief = _parse_belief(lines[:HYPOTHESIS_COUNT])
    parts = lines[-1].split("|")
    if len(parts) != 2 or parts[0] != "A01":
        raise ValueError("invalid adaptive query line")
    return belief, _clean_text_field(
        parts[1],
        maximum=QUERY_MAX_CHARS,
    )


def parse_future_scores(text: str) -> list[int]:
    lines = _response_lines(text)
    if len(lines) != ROOT_COUNT:
        raise ValueError("future response has wrong line count")
    scores: list[int] = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|")
        if len(parts) != 2 or parts[0] != f"S{index:02d}":
            raise ValueError("invalid future score line")
        scores.append(
            _canonical_integer(parts[1], minimum=0, maximum=100)
        )
    return scores


def parse_terminal(text: str) -> Belief:
    return _parse_belief(_response_lines(text))


def _belief_payload(belief: Belief) -> list[dict[str, Any]]:
    return [
        {"answer": hypothesis, "weight": weight}
        for hypothesis, weight in zip(belief.hypotheses, belief.weights)
    ]


def _document_payload(
    documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "docid": str(document["docid"]),
            "snippet": " ".join(str(document["text"]).split())[
                :OBSERVATION_CHARS
            ],
        }
        for document in documents
    ]


def _belief_grammar() -> list[str]:
    return [
        f"H{index:02d}|positive integer weight|short candidate final answer"
        for index in range(1, HYPOTHESIS_COUNT + 1)
    ]


def initial_messages(task: Mapping[str, Any]) -> list[dict[str, str]]:
    grammar = _belief_grammar() + [
        (
            f"S{index:02d}|0..100 direct score|distinct first search query|"
            "what useful continuation this root could enable"
        )
        for index in range(1, ROOT_COUNT + 1)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Act as open-world semantic belief and experimental-design "
                "machinery for a hard factual research question. Generate eight "
                "distinct candidate final answers with positive integer weights "
                "summing exactly to 100. Generate six diverse first search "
                "queries. Score each query only for information expected from "
                "that first search, without credit for later searches. Also "
                "state a concise future information intent that this root could "
                "unlock. Do not reveal reasoning. Return only the exact ordered "
                "flat lines requested, with no JSON, markdown, or extra text."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": task["query"],
                    "exact_output_lines": grammar,
                },
                separators=(",", ":"),
            ),
        },
    ]


def refresh_messages(
    task: Mapping[str, Any],
    *,
    initial_belief: Belief,
    strategy: Strategy,
    root_documents: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the open-world answer belief after the first search "
                "observation. Preserve, revise, add, or drop candidate answers "
                "based on the returned text. Then choose one adaptive second "
                "search query targeting the most useful unresolved evidence "
                "made available by this observation. Use positive integer "
                "weights summing exactly to 100. Do not reveal reasoning. Return "
                "only eight ordered H lines and one A01 line, with no JSON, "
                "markdown, or extra text."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": task["query"],
                    "prior_belief": _belief_payload(initial_belief),
                    "first_strategy": {
                        "query": strategy.root_query,
                        "future_intent": strategy.future_intent,
                    },
                    "first_search_results": _document_payload(root_documents),
                    "exact_output_lines": (
                        _belief_grammar() + ["A01|adaptive second query"]
                    ),
                },
                separators=(",", ":"),
            ),
        },
    ]


def future_scorer_messages(
    task: Mapping[str, Any],
    *,
    branches: Sequence[Branch],
    beliefs: Sequence[Belief],
    queries: Sequence[str],
) -> list[dict[str, str]]:
    bundles = []
    for index, (branch, belief, query) in enumerate(
        zip(branches, beliefs, queries),
        start=1,
    ):
        bundles.append(
            {
                "strategy_id": f"S{index:02d}",
                "first_query": branch.strategy.root_query,
                "direct_score": branch.strategy.direct_score,
                "future_intent": branch.strategy.future_intent,
                "belief_after_first_search": _belief_payload(belief),
                "adaptive_second_query": query,
            }
        )
    return [
        {
            "role": "system",
            "content": (
                "Score only the incremental future information value of each "
                "adaptive continuation bundle. High value means the updated "
                "semantic belief identifies unresolved alternatives and the "
                "second query can discriminate or complete the evidence chain. "
                "Do not reward the first search's direct value again. Treat "
                "bundles independently and use a sharp 0-100 range. Return only "
                "six ordered S lines with no explanation."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "question": task["query"],
                    "continuation_bundles": bundles,
                    "exact_output_lines": [
                        f"S{index:02d}|0..100 future-only score"
                        for index in range(1, ROOT_COUNT + 1)
                    ],
                },
                separators=(",", ":"),
            ),
        },
    ]


def terminal_messages(
    task: Mapping[str, Any],
    *,
    root_belief: Belief,
    branch: Branch,
    policy: str,
) -> list[dict[str, str]]:
    assert branch.adaptive_query is not None
    assert branch.adaptive_documents is not None
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the final open-world answer belief after the second "
                "search. Use only the supplied question, prior belief, and "
                "returned documents as observed evidence. Keep eight distinct "
                "short answers with positive integer weights summing exactly "
                "to 100. Return only the eight ordered H lines."
            ),
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "policy": policy,
                    "question": task["query"],
                    "belief_after_first_search": _belief_payload(root_belief),
                    "first_query": branch.strategy.root_query,
                    "first_results": _document_payload(branch.root_documents),
                    "second_query": branch.adaptive_query,
                    "second_results": _document_payload(
                        branch.adaptive_documents
                    ),
                    "exact_output_lines": _belief_grammar(),
                },
                separators=(",", ":"),
            ),
        },
    ]


def _tokenize(value: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) > 1
    ]


class TaskBM25:
    def __init__(self, documents: Sequence[Mapping[str, Any]]) -> None:
        deduped: dict[str, dict[str, Any]] = {}
        for document in documents:
            docid = str(document["docid"])
            deduped[docid] = {
                "docid": docid,
                "text": str(document["text"]),
            }
        self.documents = list(deduped.values())
        self.tokens = [
            _tokenize(document["text"][:INDEX_TEXT_CHARS])
            for document in self.documents
        ]
        self.frequencies = [Counter(tokens) for tokens in self.tokens]
        self.lengths = [len(tokens) for tokens in self.tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths)
            if self.lengths
            else 1.0
        )
        self.document_frequency = Counter(
            token for tokens in self.tokens for token in set(tokens)
        )

    def search(self, query: str) -> list[dict[str, Any]]:
        query_tokens = _tokenize(query)
        count = len(self.documents)
        scores: list[tuple[float, int]] = []
        for index, frequencies in enumerate(self.frequencies):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies[token]
                if not frequency:
                    continue
                document_frequency = self.document_frequency[token]
                inverse = math.log(
                    1.0
                    + (count - document_frequency + 0.5)
                    / (document_frequency + 0.5)
                )
                denominator = frequency + 1.2 * (
                    0.25
                    + 0.75
                    * self.lengths[index]
                    / self.average_length
                )
                score += inverse * frequency * 2.2 / denominator
            scores.append((score, index))
        ranked = sorted(
            scores,
            key=lambda item: (-item[0], item[1]),
        )[:RETRIEVAL_TOP_K]
        return [
            {
                **self.documents[index],
                "score": score,
            }
            for score, index in ranked
        ]


def load_tasks(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("BrowseComp-Plus mechanics source hash changed")
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    if tuple(str(row["query_id"]) for row in rows) != TASK_IDS:
        raise ValueError("BrowseComp-Plus mechanics task order changed")
    return rows


def _task_documents(task: Mapping[str, Any]) -> list[dict[str, Any]]:
    documents: list[dict[str, Any]] = []
    for field in ("gold_docs", "evidence_docs", "negative_docs"):
        documents.extend(task[field])
    random.Random(RANDOM_SEED + int(task["query_id"])).shuffle(documents)
    return documents


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
        raise ValueError("BrowseComp-Plus mechanics model changed")
    return build_model_adapter(spec, config)


def _usage_snapshot(model: Any) -> dict[str, Any]:
    snapshot = model.usage_snapshot()
    return {
        "physical_requests": int(snapshot["adapter_requests"]),
        "http_attempts": int(snapshot.get("http_attempts", -1)),
        "retry_count": int(snapshot.get("retry_count", -1)),
        "reasoning_tokens": int(snapshot["adapter_reasoning_tokens"]),
        "forced_exits": int(snapshot.get("forced_exits", -1)),
        "adapter_cost_usd": float(snapshot["adapter_cost_usd"]),
        "generator": snapshot,
    }


def _checkpoint(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _complete(
    model: Any,
    messages: list[list[dict[str, str]]],
    config: Config,
) -> list[str]:
    return model.chat_complete_messages_batched(
        messages,
        temperature=TEMPERATURE,
        block_size=min(len(messages), config.openrouter_concurrency),
        max_new_tokens=config.openrouter_max_output_tokens,
    )


def _argmax(values: Sequence[int | float]) -> int:
    return max(
        range(len(values)),
        key=lambda index: (values[index], -index),
    )


def run_model_stage(
    config: Config,
    *,
    source_path: Path,
    raw_path: Path,
    model_adapter: Any | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    tasks = load_tasks(source_path)
    retrievers = [TaskBM25(_task_documents(task)) for task in tasks]
    model = model_adapter if model_adapter is not None else _build_model(config)
    raw: dict[str, Any] = {"task_ids": list(TASK_IDS)}
    try:
        initial_responses = _complete(
            model,
            [initial_messages(task) for task in tasks],
            config,
        )
        raw["initial_responses"] = initial_responses
        initials = [parse_initial(response) for response in initial_responses]
        _checkpoint(raw_path, raw)

        branches_by_task: list[list[Branch]] = []
        for retriever, (_, strategies) in zip(retrievers, initials):
            branches_by_task.append(
                [
                    Branch(
                        root_index=index,
                        strategy=strategy,
                        root_documents=retriever.search(
                            strategy.root_query
                        ),
                    )
                    for index, strategy in enumerate(strategies)
                ]
            )

        refresh_specs = [
            (task_index, branch.root_index)
            for task_index, branches in enumerate(branches_by_task)
            for branch in branches
        ]
        refresh_responses = _complete(
            model,
            [
                refresh_messages(
                    tasks[task_index],
                    initial_belief=initials[task_index][0],
                    strategy=branches_by_task[task_index][root_index].strategy,
                    root_documents=branches_by_task[task_index][
                        root_index
                    ].root_documents,
                )
                for task_index, root_index in refresh_specs
            ],
            config,
        )
        raw["refresh_responses"] = refresh_responses
        for response, (task_index, root_index) in zip(
            refresh_responses,
            refresh_specs,
            strict=True,
        ):
            branch = branches_by_task[task_index][root_index]
            branch.root_belief, branch.adaptive_query = parse_refresh(response)
            branch.adaptive_documents = retrievers[task_index].search(
                branch.adaptive_query
            )
        _checkpoint(raw_path, raw)

        scorer_specs: list[tuple[int, str]] = [
            (task_index, variant)
            for task_index in range(len(tasks))
            for variant in ("aligned", "shuffled")
        ]
        scorer_responses = _complete(
            model,
            [
                future_scorer_messages(
                    tasks[task_index],
                    branches=branches_by_task[task_index],
                    beliefs=(
                        [
                            branch.root_belief
                            for branch in branches_by_task[task_index]
                        ]
                        if variant == "aligned"
                        else [
                            branches_by_task[task_index][
                                (index + 1) % ROOT_COUNT
                            ].root_belief
                            for index in range(ROOT_COUNT)
                        ]
                    ),
                    queries=(
                        [
                            branch.adaptive_query
                            for branch in branches_by_task[task_index]
                        ]
                        if variant == "aligned"
                        else [
                            branches_by_task[task_index][
                                (index + 1) % ROOT_COUNT
                            ].adaptive_query
                            for index in range(ROOT_COUNT)
                        ]
                    ),
                )
                for task_index, variant in scorer_specs
            ],
            config,
        )
        raw["scorer_responses"] = scorer_responses
        future_scores: list[dict[str, list[int]]] = [
            {} for _ in tasks
        ]
        for response, (task_index, variant) in zip(
            scorer_responses,
            scorer_specs,
            strict=True,
        ):
            future_scores[task_index][variant] = parse_future_scores(response)
        _checkpoint(raw_path, raw)

        selected_specs: list[tuple[int, str, int]] = []
        for task_index, branches in enumerate(branches_by_task):
            direct = [branch.strategy.direct_score for branch in branches]
            full = [
                direct_score + future_score
                for direct_score, future_score in zip(
                    direct,
                    future_scores[task_index]["aligned"],
                    strict=True,
                )
            ]
            selected_specs.extend(
                [
                    (task_index, "myopic", _argmax(direct)),
                    (task_index, "strategy_d2", _argmax(full)),
                ]
            )
        terminal_responses = _complete(
            model,
            [
                terminal_messages(
                    tasks[task_index],
                    root_belief=branches_by_task[task_index][
                        root_index
                    ].root_belief,
                    branch=branches_by_task[task_index][root_index],
                    policy=policy,
                )
                for task_index, policy, root_index in selected_specs
            ],
            config,
        )
        raw["terminal_responses"] = terminal_responses
        terminals: list[dict[str, Belief]] = [{} for _ in tasks]
        for response, (task_index, policy, _) in zip(
            terminal_responses,
            selected_specs,
            strict=True,
        ):
            terminals[task_index][policy] = parse_terminal(response)
        _checkpoint(raw_path, raw)
        usage = _usage_snapshot(model)
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            _usage_snapshot(model),
        ) from exc

    return {
        "tasks": tasks,
        "initials": initials,
        "branches": branches_by_task,
        "future_scores": future_scores,
        "terminals": terminals,
    }, usage


def _pairwise_accuracy(
    scores: Sequence[int | float],
    values: Sequence[int | float],
) -> tuple[float, int]:
    points = 0.0
    comparable = 0
    for left in range(len(values)):
        for right in range(left + 1, len(values)):
            if values[left] == values[right]:
                continue
            comparable += 1
            score_delta = scores[left] - scores[right]
            value_delta = values[left] - values[right]
            if score_delta == 0:
                points += 0.5
            elif (score_delta > 0) == (value_delta > 0):
                points += 1.0
    return (points / comparable if comparable else 0.5, comparable)


def _gold_mass(belief: Belief, answer: str) -> float:
    normalized = normalize_answer(answer)
    return sum(
        weight / 100.0
        for hypothesis, weight in zip(belief.hypotheses, belief.weights)
        if normalize_answer(hypothesis) == normalized
    )


def analyze_run(
    parsed: dict[str, Any],
    usage: dict[str, Any],
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    direct_points = direct_pairs = 0
    future_points = future_pairs = 0
    full_points = full_pairs = 0
    direct_total_points = direct_total_pairs = 0
    shuffled_points = shuffled_pairs = 0
    changed_beliefs = 0
    adaptive_query_differences = 0
    evidence_gain_branches = 0
    strategy_root_differences = 0
    strategy_wins = 0
    strategy_losses = 0

    for task_index, task in enumerate(parsed["tasks"]):
        evidence = {
            str(document["docid"]) for document in task["evidence_docs"]
        }
        gold = {str(document["docid"]) for document in task["gold_docs"]}
        initial_belief = parsed["initials"][task_index][0]
        branches = parsed["branches"][task_index]
        direct_scores = [
            branch.strategy.direct_score for branch in branches
        ]
        future_scores = parsed["future_scores"][task_index]["aligned"]
        shuffled_future = parsed["future_scores"][task_index]["shuffled"]
        full_scores = [
            direct + future
            for direct, future in zip(
                direct_scores,
                future_scores,
                strict=True,
            )
        ]
        shuffled_full = [
            direct + future
            for direct, future in zip(
                direct_scores,
                shuffled_future,
                strict=True,
            )
        ]
        immediate_values: list[int] = []
        total_values: list[int] = []
        future_values: list[int] = []
        gold_values: list[int] = []
        root_records = []
        for branch in branches:
            assert branch.root_belief is not None
            assert branch.adaptive_query is not None
            assert branch.adaptive_documents is not None
            root_ids = {
                str(document["docid"])
                for document in branch.root_documents
            }
            adaptive_ids = {
                str(document["docid"])
                for document in branch.adaptive_documents
            }
            immediate = len(root_ids & evidence)
            total = len((root_ids | adaptive_ids) & evidence)
            future = total - immediate
            immediate_values.append(immediate)
            total_values.append(total)
            future_values.append(future)
            gold_values.append(len((root_ids | adaptive_ids) & gold))
            changed = branch.root_belief.signature != initial_belief.signature
            query_differs = (
                normalize_query(branch.adaptive_query)
                != normalize_query(branch.strategy.root_query)
            )
            changed_beliefs += changed
            adaptive_query_differences += query_differs
            evidence_gain_branches += future > 0
            root_records.append(
                {
                    "root_index": branch.root_index,
                    "direct_score": branch.strategy.direct_score,
                    "future_score": future_scores[branch.root_index],
                    "shuffled_future_score": shuffled_future[
                        branch.root_index
                    ],
                    "full_score": full_scores[branch.root_index],
                    "immediate_evidence": immediate,
                    "future_evidence_gain": future,
                    "total_evidence": total,
                    "total_gold": gold_values[-1],
                    "belief_changed": changed,
                    "adaptive_query_differs": query_differs,
                }
            )

        direct_accuracy, direct_count = _pairwise_accuracy(
            direct_scores,
            immediate_values,
        )
        future_accuracy, future_count = _pairwise_accuracy(
            future_scores,
            future_values,
        )
        full_accuracy, full_count = _pairwise_accuracy(
            full_scores,
            total_values,
        )
        direct_total_accuracy, direct_total_count = _pairwise_accuracy(
            direct_scores,
            total_values,
        )
        shuffled_accuracy, shuffled_count = _pairwise_accuracy(
            shuffled_full,
            total_values,
        )
        direct_points += direct_accuracy * direct_count
        direct_pairs += direct_count
        future_points += future_accuracy * future_count
        future_pairs += future_count
        full_points += full_accuracy * full_count
        full_pairs += full_count
        direct_total_points += direct_total_accuracy * direct_total_count
        direct_total_pairs += direct_total_count
        shuffled_points += shuffled_accuracy * shuffled_count
        shuffled_pairs += shuffled_count

        myopic_root = _argmax(direct_scores)
        strategy_root = _argmax(full_scores)
        random_root = random.Random(
            RANDOM_SEED + task_index
        ).randrange(ROOT_COUNT)
        strategy_root_differences += strategy_root != myopic_root
        strategy_wins += total_values[strategy_root] > total_values[myopic_root]
        strategy_losses += (
            total_values[strategy_root] < total_values[myopic_root]
        )
        policies = {
            "myopic": myopic_root,
            "strategy_d2": strategy_root,
            "random": random_root,
        }
        records.append(
            {
                "task_id": str(task["query_id"]),
                "roots": root_records,
                "policies": {
                    policy: {
                        "root_index": root_index,
                        "immediate_evidence": immediate_values[root_index],
                        "future_evidence_gain": future_values[root_index],
                        "total_evidence": total_values[root_index],
                        "total_gold": gold_values[root_index],
                        "terminal_gold_mass": (
                            _gold_mass(
                                parsed["terminals"][task_index][policy],
                                str(task["answer"]),
                            )
                            if policy in {"myopic", "strategy_d2"}
                            else None
                        ),
                    }
                    for policy, root_index in policies.items()
                },
            }
        )

    def pooled(points: float, pairs: int) -> float:
        return points / pairs if pairs else 0.5

    direct_immediate_accuracy = pooled(direct_points, direct_pairs)
    aligned_future_accuracy = pooled(future_points, future_pairs)
    aligned_full_accuracy = pooled(full_points, full_pairs)
    direct_total_accuracy = pooled(direct_total_points, direct_total_pairs)
    shuffled_full_accuracy = pooled(shuffled_points, shuffled_pairs)
    gates = {
        "exact_55_physical_requests": (
            usage["physical_requests"] == EXPECTED_REQUESTS
        ),
        "exact_55_http_attempts": (
            usage["http_attempts"] == EXPECTED_REQUESTS
        ),
        "zero_model_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "at_least_27_root_beliefs_change": changed_beliefs >= 27,
        "at_least_24_adaptive_queries_differ": (
            adaptive_query_differences >= 24
        ),
        "at_least_10_branches_gain_evidence": evidence_gain_branches >= 10,
        "direct_immediate_has_at_least_30_pairs": direct_pairs >= 30,
        "future_gain_has_at_least_25_pairs": future_pairs >= 25,
        "full_total_has_at_least_30_pairs": full_pairs >= 30,
        "direct_ranks_immediate_at_least_point_55": (
            direct_immediate_accuracy >= 0.55
        ),
        "future_ranks_gain_at_least_point_55": (
            aligned_future_accuracy >= 0.55
        ),
        "full_ranks_total_at_least_point_58": (
            aligned_full_accuracy >= 0.58
        ),
        "full_beats_direct_total_by_point_05": (
            aligned_full_accuracy - direct_total_accuracy >= 0.05
        ),
        "strategy_root_differs_at_least_2_tasks": (
            strategy_root_differences >= 2
        ),
        "strategy_beats_myopic_at_least_once": strategy_wins >= 1,
        "strategy_loses_to_myopic_at_most_once": strategy_losses <= 1,
        "cost_at_most_point_90": (
            usage["adapter_cost_usd"] <= MAX_COST_USD
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "model": MODEL_ID,
            "task_ids": list(TASK_IDS),
            "hypothesis_count": HYPOTHESIS_COUNT,
            "root_count": ROOT_COUNT,
            "retrieval_top_k": RETRIEVAL_TOP_K,
            "query_max_chars": QUERY_MAX_CHARS,
            "expected_requests": EXPECTED_REQUESTS,
            "reasoning_requested": False,
            "repairs_or_retries": 0,
            "max_cost_usd": MAX_COST_USD,
            "source_sha256": SOURCE_SHA256,
        },
        "summary": {
            "gates": gates,
            "changed_root_belief_count": changed_beliefs,
            "adaptive_query_difference_count": adaptive_query_differences,
            "evidence_gain_branch_count": evidence_gain_branches,
            "direct_immediate_pair_accuracy": direct_immediate_accuracy,
            "direct_immediate_comparable_pairs": direct_pairs,
            "aligned_future_gain_pair_accuracy": aligned_future_accuracy,
            "aligned_future_gain_comparable_pairs": future_pairs,
            "aligned_full_total_pair_accuracy": aligned_full_accuracy,
            "aligned_full_total_comparable_pairs": full_pairs,
            "direct_total_pair_accuracy": direct_total_accuracy,
            "shuffled_full_total_pair_accuracy": shuffled_full_accuracy,
            "shuffled_full_total_comparable_pairs": shuffled_pairs,
            "strategy_root_difference_count": strategy_root_differences,
            "strategy_vs_myopic_evidence_wins": strategy_wins,
            "strategy_vs_myopic_evidence_losses": strategy_losses,
            "policy_mean_total_evidence": {
                policy: sum(
                    record["policies"][policy]["total_evidence"]
                    for record in records
                )
                / len(records)
                for policy in ("myopic", "strategy_d2", "random")
            },
            "policy_mean_terminal_gold_mass": {
                policy: sum(
                    record["policies"][policy]["terminal_gold_mass"]
                    for record in records
                )
                / len(records)
                for policy in ("myopic", "strategy_d2")
            },
        },
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--source-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.55
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = min(config.openrouter_concurrency, 24)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    try:
        parsed, usage = run_model_stage(
            config,
            source_path=args.source_path,
            raw_path=raw_path,
        )
        result = analyze_run(parsed, usage)
        result["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": INTERFACE_VERSION,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        output = args.output_dir / "MECHANICS_FAILURE.json"
        output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    output = args.output_dir / "RESULT.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
