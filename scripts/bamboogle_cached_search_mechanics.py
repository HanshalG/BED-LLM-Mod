#!/usr/bin/env python3
"""Run the frozen Bamboogle cached-search semantic-belief mechanics gate."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import hashlib
import json
import math
from pathlib import Path
import random
import re
import string
import sys
import time
from typing import Any, Callable, Protocol, Sequence
from urllib.parse import urlencode
from urllib.request import Request, urlopen

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.bamboogle_semantic_bed_manifest import SOURCE_SHA256, sha256_file


MODEL_ID = "openai/gpt-5.4"
INTERFACE_VERSION = "bamboogle-cached-search-mechanics-1"
MECHANICS_IDS = (
    "test_87",
    "test_110",
    "test_72",
    "test_69",
    "test_61",
)
HYPOTHESIS_COUNT = 8
ROOT_COUNT = 4
EXPECTED_MODEL_REQUESTS = len(MECHANICS_IDS) * (1 + ROOT_COUNT * 3)
EXPECTED_RETRIEVAL_ACTIONS = len(MECHANICS_IDS) * ROOT_COUNT * 3
TEMPERATURE = 0.0
MAX_NEW_TOKENS = 512
MAX_COST_USD = 0.75
RANDOM_POLICY_SEED = 24_406
MAX_EXTRACT_CHARS = 1_200
WIKIPEDIA_ENDPOINT = "https://en.wikipedia.org/w/api.php"
WIKIPEDIA_USER_AGENT = (
    "BED-LLM-Mod-BamboogleMechanics/1.0 "
    "(research cache; contact repository owner)"
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

    def chat_complete_messages_batched_structured(
        self,
        batch_messages: list[list[dict[str, str]]],
        temperature: float,
        block_size: int,
        response_format: dict[str, Any],
        max_new_tokens: int | None = None,
    ) -> list[str]: ...


class Retriever(Protocol):
    logical_actions: int
    physical_requests: int
    transport_retries: int
    cache_hits: int

    def retrieve(self, query: str) -> list[dict[str, str]]: ...

    def cache_sha256(self) -> str: ...


class GateExecutionError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        usage: dict[str, Any],
        retrieval: dict[str, Any],
    ) -> None:
        super().__init__(message)
        self.usage = usage
        self.retrieval = retrieval


@dataclass(frozen=True)
class MechanicsCodec:
    initial_messages: Callable[[str], list[dict[str, str]]]
    parse_initial: Callable[[str], tuple["Belief", list[str], list[str]]]
    refresh_messages: Callable[
        [str, "Belief", str, Sequence[dict[str, str]]],
        list[dict[str, str]],
    ]
    parse_refresh: Callable[[str], tuple["Belief", str]]
    terminal_messages: Callable[
        [
            str,
            "Belief",
            str,
            Sequence[dict[str, str]],
            str,
            Sequence[dict[str, str]],
        ],
        list[dict[str, str]],
    ]
    parse_terminal: Callable[[str], "Belief"]
    response_format_name: str
    initial_response_format: Callable[[], dict[str, Any]] | None = None
    refresh_response_format: Callable[[], dict[str, Any]] | None = None
    terminal_response_format: Callable[[], dict[str, Any]] | None = None


@dataclass(frozen=True)
class Belief:
    hypotheses: tuple[str, ...]
    weights: tuple[int, ...]

    @property
    def probabilities(self) -> tuple[float, ...]:
        return tuple(weight / 100.0 for weight in self.weights)

    @property
    def entropy_nats(self) -> float:
        return -sum(
            probability * math.log(probability)
            for probability in self.probabilities
            if probability > 0.0
        )

    @property
    def signature(self) -> tuple[tuple[str, int], ...]:
        return tuple(
            (normalize_answer(hypothesis), weight)
            for hypothesis, weight in zip(self.hypotheses, self.weights)
        )


@dataclass
class Branch:
    task_id: str
    root_index: int
    root_query: str
    fixed_query: str
    root_documents: list[dict[str, str]]
    root_belief: Belief | None = None
    adaptive_query: str | None = None
    adaptive_documents: list[dict[str, str]] | None = None
    fixed_documents: list[dict[str, str]] | None = None
    adaptive_belief: Belief | None = None
    fixed_belief: Belief | None = None


def _strict_object(text: str) -> dict[str, Any]:
    payload = json.loads(text.strip())
    if not isinstance(payload, dict):
        raise ValueError("response must be one JSON object")
    return payload


def _hypothesis_key(index: int) -> str:
    return f"hypothesis_{index}"


def _weight_key(index: int) -> str:
    return f"weight_{index}"


def _belief_keys() -> set[str]:
    return {
        key
        for index in range(1, HYPOTHESIS_COUNT + 1)
        for key in (_hypothesis_key(index), _weight_key(index))
    }


def _integer_weight(value: Any) -> int:
    if isinstance(value, bool):
        raise ValueError("belief weight must not be Boolean")
    if isinstance(value, int):
        weight = value
    elif (
        isinstance(value, str)
        and value.isdigit()
        and (len(value) == 1 or not value.startswith("0"))
    ):
        weight = int(value)
    else:
        raise ValueError("belief weight must be an integer")
    if not 1 <= weight <= 100:
        raise ValueError("belief weight is outside 1..100")
    return weight


def _clean_query(value: Any) -> str:
    if not isinstance(value, str):
        raise ValueError("search query must be a string")
    query = " ".join(value.split())
    if not query or len(query) > 200:
        raise ValueError("search query must contain 1..200 characters")
    return query


def parse_belief_payload(payload: dict[str, Any]) -> Belief:
    if set(payload) != _belief_keys():
        raise ValueError("belief response has unexpected keys")
    hypotheses: list[str] = []
    weights: list[int] = []
    for index in range(1, HYPOTHESIS_COUNT + 1):
        hypothesis = payload[_hypothesis_key(index)]
        if not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError("hypothesis must be a nonempty string")
        hypotheses.append(" ".join(hypothesis.split()))
        weights.append(_integer_weight(payload[_weight_key(index)]))
    normalized = [normalize_answer(value) for value in hypotheses]
    if "" in normalized or len(set(normalized)) != HYPOTHESIS_COUNT:
        raise ValueError("hypotheses must be unique after normalization")
    if sum(weights) != 100:
        raise ValueError("belief weights must sum to exactly 100")
    return Belief(tuple(hypotheses), tuple(weights))


def parse_initial(text: str) -> tuple[Belief, list[str], list[str]]:
    payload = _strict_object(text)
    query_keys = {
        f"{prefix}_{index}_query"
        for prefix in ("root", "fixed")
        for index in range(1, ROOT_COUNT + 1)
    }
    if set(payload) != _belief_keys() | query_keys:
        raise ValueError("initial response has unexpected keys")
    belief = parse_belief_payload(
        {key: payload[key] for key in _belief_keys()}
    )
    roots = [
        _clean_query(payload[f"root_{index}_query"])
        for index in range(1, ROOT_COUNT + 1)
    ]
    fixed = [
        _clean_query(payload[f"fixed_{index}_query"])
        for index in range(1, ROOT_COUNT + 1)
    ]
    all_queries = [normalize_query(value) for value in roots + fixed]
    if len(set(all_queries)) != ROOT_COUNT * 2:
        raise ValueError("all initial root and fixed queries must be distinct")
    return belief, roots, fixed


def parse_refresh(text: str) -> tuple[Belief, str]:
    payload = _strict_object(text)
    if set(payload) != _belief_keys() | {"adaptive_query"}:
        raise ValueError("root refresh response has unexpected keys")
    belief = parse_belief_payload(
        {key: payload[key] for key in _belief_keys()}
    )
    return belief, _clean_query(payload["adaptive_query"])


def parse_terminal(text: str) -> Belief:
    return parse_belief_payload(_strict_object(text))


def normalize_answer(value: str) -> str:
    lowered = value.lower()
    without_punctuation = "".join(
        character for character in lowered if character not in string.punctuation
    )
    without_articles = re.sub(r"\b(a|an|the)\b", " ", without_punctuation)
    return " ".join(without_articles.split())


def normalize_query(value: str) -> str:
    return " ".join(value.lower().split())


def answer_matches(prediction: str, golden_answers: Sequence[str]) -> bool:
    normalized_prediction = normalize_answer(prediction)
    return normalized_prediction in {
        normalize_answer(answer) for answer in golden_answers
    }


def gold_mass(belief: Belief, golden_answers: Sequence[str]) -> float:
    return sum(
        weight / 100.0
        for hypothesis, weight in zip(belief.hypotheses, belief.weights)
        if answer_matches(hypothesis, golden_answers)
    )


def top_answer_correct(
    belief: Belief,
    golden_answers: Sequence[str],
) -> bool:
    index = max(
        range(HYPOTHESIS_COUNT),
        key=lambda item: (belief.weights[item], -item),
    )
    return answer_matches(belief.hypotheses[index], golden_answers)


def _belief_output_schema() -> dict[str, str]:
    schema: dict[str, str] = {}
    for index in range(1, HYPOTHESIS_COUNT + 1):
        schema[_hypothesis_key(index)] = (
            "one distinct short candidate final answer"
        )
        schema[_weight_key(index)] = (
            "integer 1..100; all eight weights must sum to 100"
        )
    return schema


def _json_schema_format(
    name: str,
    properties: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": properties,
                "required": list(properties),
                "additionalProperties": False,
            },
        },
    }


def _belief_schema_properties() -> dict[str, dict[str, Any]]:
    properties: dict[str, dict[str, Any]] = {}
    for index in range(1, HYPOTHESIS_COUNT + 1):
        properties[_hypothesis_key(index)] = {
            "type": "string",
            "minLength": 1,
        }
        properties[_weight_key(index)] = {
            "type": "integer",
            "minimum": 1,
            "maximum": 100,
        }
    return properties


def initial_response_format() -> dict[str, Any]:
    properties = _belief_schema_properties()
    for index in range(1, ROOT_COUNT + 1):
        properties[f"root_{index}_query"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
        }
        properties[f"fixed_{index}_query"] = {
            "type": "string",
            "minLength": 1,
            "maxLength": 200,
        }
    return _json_schema_format("bamboogle_initial_belief", properties)


def refresh_response_format() -> dict[str, Any]:
    properties = _belief_schema_properties()
    properties["adaptive_query"] = {
        "type": "string",
        "minLength": 1,
        "maxLength": 200,
    }
    return _json_schema_format("bamboogle_root_refresh", properties)


def terminal_response_format() -> dict[str, Any]:
    return _json_schema_format(
        "bamboogle_terminal_belief",
        _belief_schema_properties(),
    )


def _belief_payload(belief: Belief) -> list[dict[str, Any]]:
    return [
        {
            "hypothesis": hypothesis,
            "weight": weight,
        }
        for hypothesis, weight in zip(belief.hypotheses, belief.weights)
    ]


def initial_messages(question: str) -> list[dict[str, str]]:
    schema = _belief_output_schema()
    for index in range(1, ROOT_COUNT + 1):
        schema[f"root_{index}_query"] = (
            "distinct concise Wikipedia query to make first"
        )
        schema[f"fixed_{index}_query"] = (
            "distinct second Wikipedia query precommitted before seeing results"
        )
    payload = {
        "question": question,
        "required_output": schema,
    }
    return [
        {
            "role": "system",
            "content": (
                "Maintain an open-world categorical belief over short final answers "
                "to a two-hop factual question. Emit eight distinct candidate "
                "answers with integer weights summing exactly to 100. Also design "
                "four diverse first Wikipedia search queries and, for each first "
                "query, a second query fixed now before any result is seen. All "
                "eight query strings must be distinct. Queries should expose "
                "different intermediate entities or relations, not merely rephrase "
                "the full question. Return exactly the requested flat JSON object "
                "with no markdown, explanation, citations, or reasoning."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def refresh_messages(
    question: str,
    initial_belief: Belief,
    root_query: str,
    root_documents: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "prior_belief": _belief_payload(initial_belief),
        "root_search": {
            "query": root_query,
            "results": list(root_documents),
        },
        "required_output": {
            **_belief_output_schema(),
            "adaptive_query": (
                "one concise Wikipedia query chosen after these results"
            ),
        },
    }
    return [
        {
            "role": "system",
            "content": (
                "Update the categorical answer belief using only the question, "
                "prior belief, and returned Wikipedia extracts. Regenerate eight "
                "distinct short candidate final answers with integer weights "
                "summing exactly to 100. Then choose the most useful second "
                "Wikipedia query conditional on what the root results actually "
                "revealed. The query should target unresolved evidence. Return "
                "exactly the requested flat JSON object with no markdown, "
                "explanation, citations, or reasoning."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def terminal_messages(
    question: str,
    root_belief: Belief,
    root_query: str,
    root_documents: Sequence[dict[str, str]],
    second_query: str,
    second_documents: Sequence[dict[str, str]],
) -> list[dict[str, str]]:
    payload = {
        "question": question,
        "prior_belief_after_root": _belief_payload(root_belief),
        "evidence": [
            {
                "query": root_query,
                "results": list(root_documents),
            },
            {
                "query": second_query,
                "results": list(second_documents),
            },
        ],
        "required_output": _belief_output_schema(),
    }
    return [
        {
            "role": "system",
            "content": (
                "Regenerate the categorical belief over short final answers after "
                "both Wikipedia searches. Use the supplied extracts as evidence, "
                "retain genuine alternatives when uncertainty remains, and assign "
                "eight integer weights summing exactly to 100. Return exactly the "
                "requested flat JSON object with no markdown, explanation, "
                "citations, or reasoning."
            ),
        },
        {"role": "user", "content": json.dumps(payload, separators=(",", ":"))},
    ]


def _normalize_extract(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    return " ".join(value.split())[:MAX_EXTRACT_CHARS]


def parse_wikipedia_response(payload: dict[str, Any]) -> list[dict[str, str]]:
    query = payload.get("query")
    if not isinstance(query, dict):
        raise ValueError("Wikipedia response has no query object")
    pages = query.get("pages")
    if not isinstance(pages, list) or not pages:
        raise ValueError("Wikipedia response has no pages")
    ordered = sorted(
        pages,
        key=lambda page: (
            int(page.get("index", 10**9))
            if isinstance(page, dict)
            else 10**9
        ),
    )
    documents: list[dict[str, str]] = []
    for page in ordered[:3]:
        if not isinstance(page, dict):
            raise ValueError("Wikipedia page is not an object")
        title = page.get("title")
        if not isinstance(title, str) or not title.strip():
            raise ValueError("Wikipedia page has no title")
        documents.append(
            {
                "title": " ".join(title.split()),
                "extract": _normalize_extract(page.get("extract", "")),
            }
        )
    if not documents:
        raise ValueError("Wikipedia response produced no documents")
    return documents


class CachedWikipediaRetriever:
    def __init__(
        self,
        cache_dir: Path,
        *,
        timeout_seconds: float = 30.0,
        max_transport_retries: int = 2,
    ) -> None:
        self.cache_dir = cache_dir
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.max_transport_retries = max_transport_retries
        self.logical_actions = 0
        self.physical_requests = 0
        self.transport_retries = 0
        self.cache_hits = 0

    def _cache_path(self, query: str) -> Path:
        digest = hashlib.sha256(
            normalize_query(query).encode("utf-8")
        ).hexdigest()
        return self.cache_dir / f"{digest}.json"

    def retrieve(self, query: str) -> list[dict[str, str]]:
        cleaned_query = _clean_query(query)
        self.logical_actions += 1
        cache_path = self._cache_path(cleaned_query)
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("normalized_query") != normalize_query(cleaned_query):
                raise ValueError("Wikipedia cache query mismatch")
            documents = cached.get("documents")
            if not isinstance(documents, list) or not documents:
                raise ValueError("Wikipedia cache has no parsed documents")
            self.cache_hits += 1
            return [
                {
                    "title": str(document["title"]),
                    "extract": str(document["extract"]),
                }
                for document in documents
            ]

        parameters = {
            "action": "query",
            "format": "json",
            "formatversion": "2",
            "generator": "search",
            "gsrsearch": cleaned_query,
            "gsrlimit": "3",
            "prop": "extracts",
            "exintro": "1",
            "explaintext": "1",
            "redirects": "1",
            "utf8": "1",
        }
        url = f"{WIKIPEDIA_ENDPOINT}?{urlencode(parameters)}"
        last_error: Exception | None = None
        for attempt in range(self.max_transport_retries + 1):
            try:
                self.physical_requests += 1
                request = Request(
                    url,
                    headers={"User-Agent": WIKIPEDIA_USER_AGENT},
                )
                with urlopen(request, timeout=self.timeout_seconds) as response:
                    response_bytes = response.read()
                payload = json.loads(response_bytes)
                if not isinstance(payload, dict):
                    raise ValueError("Wikipedia response is not an object")
                documents = parse_wikipedia_response(payload)
                cache_record = {
                    "interface_version": INTERFACE_VERSION,
                    "query": cleaned_query,
                    "normalized_query": normalize_query(cleaned_query),
                    "request_url": url,
                    "response_sha256": hashlib.sha256(
                        response_bytes
                    ).hexdigest(),
                    "raw_response": payload,
                    "documents": documents,
                }
                cache_path.write_text(
                    json.dumps(cache_record, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8",
                )
                return documents
            except Exception as exc:
                last_error = exc
                if attempt >= self.max_transport_retries:
                    break
                self.transport_retries += 1
                time.sleep(1.0 * (attempt + 1))
        assert last_error is not None
        raise RuntimeError(
            f"Wikipedia retrieval failed for frozen query: {last_error}"
        ) from last_error

    def cache_sha256(self) -> str:
        digest = hashlib.sha256()
        for path in sorted(self.cache_dir.glob("*.json")):
            digest.update(path.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(path.read_bytes())
            digest.update(b"\0")
        return digest.hexdigest()


def _load_selected_rows(path: Path) -> list[dict[str, Any]]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("Bamboogle source hash mismatch")
    wanted = set(MECHANICS_IDS)
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        task_id = str(row.get("id"))
        if task_id in wanted:
            selected[task_id] = row
    if set(selected) != wanted:
        raise ValueError("Bamboogle mechanics IDs do not reproduce")
    return [selected[task_id] for task_id in MECHANICS_IDS]


def load_visible_tasks(path: Path) -> list[dict[str, str]]:
    return [
        {"id": str(row["id"]), "question": str(row["question"])}
        for row in _load_selected_rows(path)
    ]


def load_gold_answers(path: Path) -> dict[str, list[str]]:
    return {
        str(row["id"]): [str(answer) for answer in row["golden_answers"]]
        for row in _load_selected_rows(path)
    }


def _usage_snapshot(model: ChatModel) -> dict[str, Any]:
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


def _retrieval_snapshot(retriever: Retriever) -> dict[str, Any]:
    return {
        "logical_actions": int(retriever.logical_actions),
        "physical_requests": int(retriever.physical_requests),
        "transport_retries": int(retriever.transport_retries),
        "cache_hits": int(retriever.cache_hits),
        "cache_sha256": retriever.cache_sha256(),
    }


def _build_model(config: Config) -> ChatModel:
    spec = replace(
        config.model_pairs[0].questioner,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    if spec.model != MODEL_ID:
        raise ValueError("Bamboogle mechanics config selects the wrong model")
    return build_model_adapter(spec, config)


def _checkpoint(path: Path, raw: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(raw, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _required_belief(value: Belief | None) -> Belief:
    if value is None:
        raise ValueError("branch belief is missing")
    return value


def _required_documents(
    value: list[dict[str, str]] | None,
) -> list[dict[str, str]]:
    if value is None:
        raise ValueError("branch documents are missing")
    return value


def _required_query(value: str | None) -> str:
    if value is None:
        raise ValueError("branch query is missing")
    return value


def _select_lowest_entropy(values: Sequence[float]) -> int:
    return min(range(len(values)), key=lambda index: (values[index], index))


def _complete_batch(
    model: ChatModel,
    messages: list[list[dict[str, str]]],
    *,
    response_format: dict[str, Any] | None,
) -> list[str]:
    kwargs = {
        "temperature": TEMPERATURE,
        "block_size": len(messages),
        "max_new_tokens": MAX_NEW_TOKENS,
    }
    if response_format is None:
        return model.chat_complete_messages_batched(messages, **kwargs)
    return model.chat_complete_messages_batched_structured(
        messages,
        response_format=response_format,
        **kwargs,
    )


def mechanics_codec(*, structured_outputs: bool = False) -> MechanicsCodec:
    return MechanicsCodec(
        initial_messages=initial_messages,
        parse_initial=parse_initial,
        refresh_messages=refresh_messages,
        parse_refresh=parse_refresh,
        terminal_messages=terminal_messages,
        parse_terminal=parse_terminal,
        response_format_name=(
            "chat_strict_json_schema"
            if structured_outputs
            else "prompt_only_json"
        ),
        initial_response_format=(
            initial_response_format if structured_outputs else None
        ),
        refresh_response_format=(
            refresh_response_format if structured_outputs else None
        ),
        terminal_response_format=(
            terminal_response_format if structured_outputs else None
        ),
    )


def summarize_run(
    *,
    tasks: Sequence[dict[str, str]],
    initial_beliefs: Sequence[Belief],
    branches_by_task: Sequence[Sequence[Branch]],
    gold_answers: dict[str, list[str]],
    usage: dict[str, Any],
    retrieval: dict[str, Any],
    interface_version: str = INTERFACE_VERSION,
    structured_outputs: bool = False,
    response_format_name: str | None = None,
) -> dict[str, Any]:
    if len(tasks) != len(initial_beliefs) or len(tasks) != len(branches_by_task):
        raise ValueError("task, initial, and branch groups differ in length")
    random_generator = random.Random(RANDOM_POLICY_SEED)
    records: list[dict[str, Any]] = []
    adaptive_query_difference_count = 0
    top_page_difference_count = 0
    changed_root_belief_count = 0
    action_diversity_task_count = 0
    adaptive_myopic_win_count = 0
    adaptive_myopic_loss_count = 0
    adaptive_fixed_win_count = 0
    adaptive_fixed_loss_count = 0
    adaptive_root_difference_count = 0

    for task, initial_belief, branches in zip(
        tasks,
        initial_beliefs,
        branches_by_task,
    ):
        if len(branches) != ROOT_COUNT:
            raise ValueError("wrong number of roots for a task")
        answers = gold_answers[task["id"]]
        root_entropies: list[float] = []
        adaptive_entropies: list[float] = []
        fixed_entropies: list[float] = []
        adaptive_gold_masses: list[float] = []
        fixed_gold_masses: list[float] = []
        adaptive_top_correct: list[bool] = []
        fixed_top_correct: list[bool] = []
        root_records: list[dict[str, Any]] = []

        for branch in branches:
            root_belief = _required_belief(branch.root_belief)
            adaptive_belief = _required_belief(branch.adaptive_belief)
            fixed_belief = _required_belief(branch.fixed_belief)
            adaptive_documents = _required_documents(
                branch.adaptive_documents
            )
            fixed_documents = _required_documents(branch.fixed_documents)
            adaptive_query = _required_query(branch.adaptive_query)
            changed_root_belief_count += (
                root_belief.signature != initial_belief.signature
            )
            query_differs = (
                normalize_query(adaptive_query)
                != normalize_query(branch.fixed_query)
            )
            adaptive_query_difference_count += query_differs
            adaptive_top = normalize_answer(
                adaptive_documents[0]["title"]
            )
            fixed_top = normalize_answer(fixed_documents[0]["title"])
            top_differs = adaptive_top != fixed_top
            top_page_difference_count += top_differs

            root_entropies.append(root_belief.entropy_nats)
            adaptive_entropies.append(adaptive_belief.entropy_nats)
            fixed_entropies.append(fixed_belief.entropy_nats)
            adaptive_mass = gold_mass(adaptive_belief, answers)
            fixed_mass = gold_mass(fixed_belief, answers)
            adaptive_gold_masses.append(adaptive_mass)
            fixed_gold_masses.append(fixed_mass)
            adaptive_top_correct.append(
                top_answer_correct(adaptive_belief, answers)
            )
            fixed_top_correct.append(top_answer_correct(fixed_belief, answers))
            root_records.append(
                {
                    "root_index": branch.root_index,
                    "root_entropy_nats": root_belief.entropy_nats,
                    "adaptive_terminal_entropy_nats": (
                        adaptive_belief.entropy_nats
                    ),
                    "fixed_terminal_entropy_nats": fixed_belief.entropy_nats,
                    "adaptive_gold_mass": adaptive_mass,
                    "fixed_gold_mass": fixed_mass,
                    "adaptive_top_correct": adaptive_top_correct[-1],
                    "fixed_top_correct": fixed_top_correct[-1],
                    "root_belief_changed": (
                        root_belief.signature != initial_belief.signature
                    ),
                    "adaptive_query_differs_from_fixed": query_differs,
                    "adaptive_top_page_differs_from_fixed": top_differs,
                    "root_result_count": len(branch.root_documents),
                    "adaptive_result_count": len(adaptive_documents),
                    "fixed_result_count": len(fixed_documents),
                }
            )

        myopic_index = _select_lowest_entropy(root_entropies)
        adaptive_index = _select_lowest_entropy(adaptive_entropies)
        fixed_index = _select_lowest_entropy(fixed_entropies)
        random_index = random_generator.randrange(ROOT_COUNT)
        adaptive_root_difference_count += adaptive_index != myopic_index
        policies = {
            "myopic": {
                "root_index": myopic_index,
                "terminal_entropy_nats": adaptive_entropies[myopic_index],
                "gold_mass": adaptive_gold_masses[myopic_index],
                "top_correct": adaptive_top_correct[myopic_index],
            },
            "adaptive_d2": {
                "root_index": adaptive_index,
                "terminal_entropy_nats": adaptive_entropies[adaptive_index],
                "gold_mass": adaptive_gold_masses[adaptive_index],
                "top_correct": adaptive_top_correct[adaptive_index],
            },
            "fixed_d2": {
                "root_index": fixed_index,
                "terminal_entropy_nats": fixed_entropies[fixed_index],
                "gold_mass": fixed_gold_masses[fixed_index],
                "top_correct": fixed_top_correct[fixed_index],
            },
            "random": {
                "root_index": random_index,
                "terminal_entropy_nats": adaptive_entropies[random_index],
                "gold_mass": adaptive_gold_masses[random_index],
                "top_correct": adaptive_top_correct[random_index],
            },
        }
        adaptive_mass = float(policies["adaptive_d2"]["gold_mass"])
        myopic_mass = float(policies["myopic"]["gold_mass"])
        fixed_mass = float(policies["fixed_d2"]["gold_mass"])
        adaptive_myopic_win_count += adaptive_mass > myopic_mass
        adaptive_myopic_loss_count += adaptive_mass < myopic_mass
        adaptive_fixed_win_count += adaptive_mass > fixed_mass
        adaptive_fixed_loss_count += adaptive_mass < fixed_mass
        mass_range = max(adaptive_gold_masses) - min(adaptive_gold_masses)
        action_diversity_task_count += mass_range >= 0.25
        records.append(
            {
                "task_id": task["id"],
                "initial_entropy_nats": initial_belief.entropy_nats,
                "adaptive_terminal_entropy_range_nats": (
                    max(adaptive_entropies) - min(adaptive_entropies)
                ),
                "adaptive_terminal_gold_mass_range": mass_range,
                "policies": policies,
                "roots": root_records,
            }
        )

    gates = {
        "exact_65_physical_model_requests": (
            usage["physical_requests"] == EXPECTED_MODEL_REQUESTS
        ),
        "exact_65_http_model_attempts": (
            usage["http_attempts"] == EXPECTED_MODEL_REQUESTS
        ),
        "zero_model_retries": usage["retry_count"] == 0,
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "zero_forced_exits": usage["forced_exits"] == 0,
        "all_responses_parsed_without_repair": True,
        "exact_60_logical_retrieval_actions": (
            retrieval["logical_actions"] == EXPECTED_RETRIEVAL_ACTIONS
        ),
        "all_retrieval_actions_nonempty": all(
            root[field] > 0
            for record in records
            for root in record["roots"]
            for field in (
                "root_result_count",
                "adaptive_result_count",
                "fixed_result_count",
            )
        ),
        "all_20_root_beliefs_changed": (
            changed_root_belief_count == len(MECHANICS_IDS) * ROOT_COUNT
        ),
        "at_least_15_adaptive_queries_differ": (
            adaptive_query_difference_count >= 15
        ),
        "at_least_10_adaptive_top_pages_differ": (
            top_page_difference_count >= 10
        ),
        "at_least_3_tasks_have_gold_mass_range_point_25": (
            action_diversity_task_count >= 3
        ),
        "adaptive_d2_root_differs_at_least_2_tasks": (
            adaptive_root_difference_count >= 2
        ),
        "adaptive_d2_beats_myopic_at_least_once": (
            adaptive_myopic_win_count >= 1
        ),
        "adaptive_d2_loses_to_myopic_at_most_once": (
            adaptive_myopic_loss_count <= 1
        ),
        "adaptive_d2_beats_fixed_at_least_once": (
            adaptive_fixed_win_count >= 1
        ),
        "adaptive_d2_loses_to_fixed_at_most_once": (
            adaptive_fixed_loss_count <= 1
        ),
        "cost_at_most_0_75": usage["adapter_cost_usd"] <= MAX_COST_USD,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": interface_version,
            "source_sha256": SOURCE_SHA256,
            "task_ids": list(MECHANICS_IDS),
            "model": MODEL_ID,
            "reasoning_requested": False,
            "temperature": TEMPERATURE,
            "hypothesis_count": HYPOTHESIS_COUNT,
            "root_count": ROOT_COUNT,
            "expected_model_requests": EXPECTED_MODEL_REQUESTS,
            "expected_retrieval_actions": EXPECTED_RETRIEVAL_ACTIONS,
            "retrieval_top_k": 3,
            "max_extract_chars": MAX_EXTRACT_CHARS,
            "random_policy_seed": RANDOM_POLICY_SEED,
            "gold_hidden_until_checkpoint": True,
            "scientific_retries_or_repairs": 0,
            "response_format": response_format_name or (
                "chat_strict_json_schema" if structured_outputs else
                "prompt_only_json"
            ),
            "max_cost_usd": MAX_COST_USD,
        },
        "summary": {
            "gates": gates,
            "changed_root_belief_count": changed_root_belief_count,
            "adaptive_query_difference_count": (
                adaptive_query_difference_count
            ),
            "adaptive_top_page_difference_count": top_page_difference_count,
            "action_diversity_task_count": action_diversity_task_count,
            "adaptive_root_difference_count": (
                adaptive_root_difference_count
            ),
            "adaptive_vs_myopic_gold_mass_wins": (
                adaptive_myopic_win_count
            ),
            "adaptive_vs_myopic_gold_mass_losses": (
                adaptive_myopic_loss_count
            ),
            "adaptive_vs_fixed_gold_mass_wins": adaptive_fixed_win_count,
            "adaptive_vs_fixed_gold_mass_losses": adaptive_fixed_loss_count,
            "policy_top_correct_counts": {
                policy: sum(
                    bool(record["policies"][policy]["top_correct"])
                    for record in records
                )
                for policy in ("myopic", "adaptive_d2", "fixed_d2", "random")
            },
            "policy_mean_gold_mass": {
                policy: sum(
                    float(record["policies"][policy]["gold_mass"])
                    for record in records
                )
                / len(records)
                for policy in ("myopic", "adaptive_d2", "fixed_d2", "random")
            },
        },
        "records": records,
        "usage": usage,
        "retrieval": retrieval,
    }


def run_mechanics(
    config: Config,
    *,
    data_path: Path,
    raw_path: Path,
    model_adapter: ChatModel | None = None,
    retriever: Retriever | None = None,
    cache_dir: Path | None = None,
    interface_version: str = INTERFACE_VERSION,
    structured_outputs: bool = False,
    codec: MechanicsCodec | None = None,
) -> dict[str, Any]:
    tasks = load_visible_tasks(data_path)
    if codec is not None and structured_outputs:
        raise ValueError("custom codec cannot use structured_outputs")
    active_codec = codec or mechanics_codec(
        structured_outputs=structured_outputs
    )
    model = model_adapter if model_adapter is not None else _build_model(config)
    if retriever is None:
        if cache_dir is None:
            raise ValueError("cache_dir is required without a test retriever")
        retriever = CachedWikipediaRetriever(cache_dir)
    raw: dict[str, Any] = {
        "interface_version": interface_version,
        "task_ids": list(MECHANICS_IDS),
    }
    try:
        initial_responses = _complete_batch(
            model,
            [
                active_codec.initial_messages(task["question"])
                for task in tasks
            ],
            response_format=(
                active_codec.initial_response_format()
                if active_codec.initial_response_format is not None
                else None
            ),
        )
        if len(initial_responses) != len(tasks):
            raise ValueError("wrong number of initial responses")
        raw["initial_responses"] = initial_responses
        parsed_initials = [
            active_codec.parse_initial(value) for value in initial_responses
        ]
        initial_beliefs = [value[0] for value in parsed_initials]
        branches_by_task: list[list[Branch]] = []
        for task, (_, root_queries, fixed_queries) in zip(
            tasks,
            parsed_initials,
        ):
            branches: list[Branch] = []
            for root_index, (root_query, fixed_query) in enumerate(
                zip(root_queries, fixed_queries)
            ):
                branches.append(
                    Branch(
                        task_id=task["id"],
                        root_index=root_index,
                        root_query=root_query,
                        fixed_query=fixed_query,
                        root_documents=retriever.retrieve(root_query),
                    )
                )
            branches_by_task.append(branches)
        raw["root_retrievals"] = [
            {
                "task_id": branch.task_id,
                "root_index": branch.root_index,
                "root_query": branch.root_query,
                "fixed_query": branch.fixed_query,
                "root_documents": branch.root_documents,
            }
            for branches in branches_by_task
            for branch in branches
        ]
        _checkpoint(raw_path, raw)

        flat_branches = [
            branch
            for branches in branches_by_task
            for branch in branches
        ]
        refresh_responses = _complete_batch(
            model,
            [
                active_codec.refresh_messages(
                    task["question"],
                    initial_belief,
                    branch.root_query,
                    branch.root_documents,
                )
                for task, initial_belief, branches in zip(
                    tasks,
                    initial_beliefs,
                    branches_by_task,
                )
                for branch in branches
            ],
            response_format=(
                active_codec.refresh_response_format()
                if active_codec.refresh_response_format is not None
                else None
            ),
        )
        if len(refresh_responses) != len(flat_branches):
            raise ValueError("wrong number of root refresh responses")
        raw["refresh_responses"] = refresh_responses
        for branch, response in zip(flat_branches, refresh_responses):
            (
                branch.root_belief,
                branch.adaptive_query,
            ) = active_codec.parse_refresh(response)
            branch.adaptive_documents = retriever.retrieve(
                _required_query(branch.adaptive_query)
            )
            branch.fixed_documents = retriever.retrieve(branch.fixed_query)
        raw["second_retrievals"] = [
            {
                "task_id": branch.task_id,
                "root_index": branch.root_index,
                "adaptive_query": branch.adaptive_query,
                "adaptive_documents": branch.adaptive_documents,
                "fixed_query": branch.fixed_query,
                "fixed_documents": branch.fixed_documents,
            }
            for branch in flat_branches
        ]
        _checkpoint(raw_path, raw)

        adaptive_terminal_responses = _complete_batch(
            model,
            [
                active_codec.terminal_messages(
                    task["question"],
                    _required_belief(branch.root_belief),
                    branch.root_query,
                    branch.root_documents,
                    _required_query(branch.adaptive_query),
                    _required_documents(branch.adaptive_documents),
                )
                for task, branches in zip(tasks, branches_by_task)
                for branch in branches
            ],
            response_format=(
                active_codec.terminal_response_format()
                if active_codec.terminal_response_format is not None
                else None
            ),
        )
        if len(adaptive_terminal_responses) != len(flat_branches):
            raise ValueError("wrong number of adaptive terminal responses")
        raw["adaptive_terminal_responses"] = adaptive_terminal_responses
        for branch, response in zip(
            flat_branches,
            adaptive_terminal_responses,
        ):
            branch.adaptive_belief = active_codec.parse_terminal(response)
        _checkpoint(raw_path, raw)

        fixed_terminal_responses = _complete_batch(
            model,
            [
                active_codec.terminal_messages(
                    task["question"],
                    _required_belief(branch.root_belief),
                    branch.root_query,
                    branch.root_documents,
                    branch.fixed_query,
                    _required_documents(branch.fixed_documents),
                )
                for task, branches in zip(tasks, branches_by_task)
                for branch in branches
            ],
            response_format=(
                active_codec.terminal_response_format()
                if active_codec.terminal_response_format is not None
                else None
            ),
        )
        if len(fixed_terminal_responses) != len(flat_branches):
            raise ValueError("wrong number of fixed terminal responses")
        raw["fixed_terminal_responses"] = fixed_terminal_responses
        for branch, response in zip(flat_branches, fixed_terminal_responses):
            branch.fixed_belief = active_codec.parse_terminal(response)
        _checkpoint(raw_path, raw)

        usage = _usage_snapshot(model)
        retrieval_snapshot = _retrieval_snapshot(retriever)
        gold_answers = load_gold_answers(data_path)
        return summarize_run(
            tasks=tasks,
            initial_beliefs=initial_beliefs,
            branches_by_task=branches_by_task,
            gold_answers=gold_answers,
            usage=usage,
            retrieval=retrieval_snapshot,
            interface_version=interface_version,
            structured_outputs=structured_outputs,
            response_format_name=active_codec.response_format_name,
        )
    except Exception as exc:
        _checkpoint(raw_path, raw)
        raise GateExecutionError(
            f"{type(exc).__name__}: {exc}",
            usage=_usage_snapshot(model),
            retrieval=_retrieval_snapshot(retriever),
        ) from exc


def run_cli(
    *,
    interface_version: str = INTERFACE_VERSION,
    structured_outputs: bool = False,
    codec: MechanicsCodec | None = None,
) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-c", "--config", type=Path, required=True)
    parser.add_argument("--data-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--private-cache-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    config = load_config(args.config)
    config.run_id = args.run_id
    config.openrouter_projected_cost_usd = 0.40
    config.openrouter_run_budget_usd = MAX_COST_USD
    config.openrouter_concurrency = 64
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_raw_dir = args.private_raw_dir / args.run_id
    private_cache_dir = args.private_cache_dir / args.run_id
    private_raw_dir.mkdir(parents=True, exist_ok=True)
    private_cache_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_raw_dir / "RAW_RESPONSES.json"

    try:
        result = run_mechanics(
            config,
            data_path=args.data_path,
            raw_path=raw_path,
            cache_dir=private_cache_dir,
            interface_version=interface_version,
            structured_outputs=structured_outputs,
            codec=codec,
        )
        result["protocol"]["private_raw_sha256"] = sha256_file(raw_path)
    except Exception as exc:
        failure: dict[str, Any] = {
            "schema_version": 1,
            "status": "failed_closed",
            "interface_version": interface_version,
            "error": f"{type(exc).__name__}: {exc}",
        }
        if isinstance(exc, GateExecutionError):
            failure["usage"] = exc.usage
            failure["retrieval"] = exc.retrieval
        if raw_path.exists():
            failure["private_raw_sha256"] = sha256_file(raw_path)
        output = args.output_dir / "MECHANICS_FAILURE.json"
        output.write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise

    output = args.output_dir / "MECHANICS.json"
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "output": str(output),
                "summary": result["summary"],
                "usage": result["usage"],
                "retrieval": result["retrieval"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    run_cli()
