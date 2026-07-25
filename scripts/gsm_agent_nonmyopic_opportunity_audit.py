#!/usr/bin/env python3
"""Audit strict two-query retrieval opportunities in GSM-Agent."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import itertools
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "a596464ea79ae0b8b84830d1c78a7d065177b0e8"
SOURCE_SHA256 = (
    "948e1ad488ef5e7bc1ad9d605684441e77cf796ef923639d0c5414ae4fe3778c"
)
SOURCE_PROBLEMS = 7_323
SOURCE_DOCUMENTS = 32_315
TEST_PROBLEMS = 1_073
SELECTION_SEED = 24_363
OPPORTUNITY_SIZE = 500
DEVELOPMENT_SIZE = 100
HOLDOUT_SIZE = 473
SPLIT_HASHES = {
    "all_test": "3245e5f8813d5871e02a5f2ce397104b96e7ae5d270ff1a1ffc7ed15404366cb",
    "opportunity": "852a1ce0796adac76188152a2f17ce31377fadc433a904ba3898c5e3bde764d7",
    "development": "8c9830fa5d114decc160f63e87eddf7aafa8e7ae3cfb3095ac7b779845fc6543",
    "holdout": "79e8b4b40f3c77d09f3ffcd2ada5750d3188f47e94c7b7ab5bd738bed5d4c5a5",
}
PAGE_SIZE = 5
ROOT_QUERY_LIMIT = 24
CONTINUATION_QUERY_LIMIT = 30

STOPWORDS = frozenset(
    """
    a about after again against all am an and any are as at be because been before
    being below between both but by can did do does doing down during each few for
    from further had has have having he her here hers herself him himself his how
    i if in into is it its itself just me more most my myself no nor not now of off
    on once only or other our ours ourselves out over own same she should so some
    such than that the their theirs them themselves then there these they this
    those through to too under until up very was we were what when where which
    while who whom why will with you your yours yourself yourselves many much
    money total altogether difference did does spent spend cost costs paid pay
    """.split()
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_list_hash(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold().replace("_", " "))


def informative_tokens(value: str) -> list[str]:
    return [
        token
        for token in tokens(value)
        if token not in STOPWORDS and (len(token) > 1 or token.isdigit())
    ]


def split_test_ids(test_ids: Iterable[str]) -> dict[str, list[str]]:
    ordered = sorted(str(value) for value in test_ids)
    if len(ordered) != TEST_PROBLEMS or len(set(ordered)) != TEST_PROBLEMS:
        raise ValueError("unexpected GSM-Agent test ID universe")
    random.Random(SELECTION_SEED).shuffle(ordered)
    return {
        "all_test": ordered,
        "opportunity": ordered[:OPPORTUNITY_SIZE],
        "development": ordered[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": ordered[OPPORTUNITY_SIZE + DEVELOPMENT_SIZE :],
    }


@dataclass(frozen=True)
class CorpusDocument:
    document_id: str
    content: str
    metadata: dict[str, Any]

    @property
    def visible_text(self) -> str:
        return " ".join(
            (
                self.document_id,
                self.content,
                json.dumps(self.metadata, ensure_ascii=True, sort_keys=True),
            )
        )


class BM25Index:
    """Small dependency-free BM25 index with deterministic tie breaking."""

    def __init__(self, documents: Sequence[CorpusDocument]) -> None:
        self.documents = list(documents)
        self.document_tokens = [tokens(doc.content) for doc in documents]
        self.lengths = [len(values) for values in self.document_tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        for index, values in enumerate(self.document_tokens):
            for token, frequency in Counter(values).items():
                self.postings[token].append((index, frequency))
        count = len(self.documents)
        self.idf = {
            token: math.log(
                1.0 + (count - len(posting) + 0.5) / (len(posting) + 0.5)
            )
            for token, posting in self.postings.items()
        }
        self._cache: dict[tuple[str, ...], tuple[int, ...]] = {}

    def search(self, query: str, *, limit: int = PAGE_SIZE) -> tuple[int, ...]:
        query_tokens = tuple(tokens(query))
        cache_key = (*query_tokens, f"limit={limit}")
        cached = self._cache.get(cache_key)
        if cached is not None:
            return cached
        scores: dict[int, float] = defaultdict(float)
        k1 = 1.5
        b = 0.75
        average_length = self.average_length or 1.0
        for token, query_frequency in Counter(query_tokens).items():
            idf = self.idf.get(token)
            if idf is None:
                continue
            for index, frequency in self.postings[token]:
                denominator = frequency + k1 * (
                    1.0 - b + b * self.lengths[index] / average_length
                )
                scores[index] += (
                    query_frequency
                    * idf
                    * frequency
                    * (k1 + 1.0)
                    / denominator
                )
        ranked = sorted(scores, key=lambda index: (-scores[index], index))
        if len(ranked) < limit:
            present = set(ranked)
            ranked.extend(
                index
                for index in range(len(self.documents))
                if index not in present
            )
        result = tuple(ranked[:limit])
        self._cache[cache_key] = result
        return result


def _dedupe_limited(values: Iterable[str], limit: int) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(tokens(value))
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
        if len(result) == limit:
            break
    return result


def root_queries(question: str, index: BM25Index) -> list[str]:
    values = informative_tokens(question)
    unique = list(dict.fromkeys(values))
    rare = sorted(
        unique,
        key=lambda token: (-index.idf.get(token, 0.0), unique.index(token)),
    )[:6]

    def candidates() -> Iterable[str]:
        yield question
        for width in (2, 3, 4):
            for start in range(max(0, len(values) - width + 1)):
                yield " ".join(values[start : start + width])
        for left, right in itertools.combinations(rare, 2):
            yield f"{left} {right}"
        for token in rare:
            yield token

    return _dedupe_limited(candidates(), ROOT_QUERY_LIMIT)


def continuation_queries(
    question: str,
    page: Sequence[CorpusDocument],
    index: BM25Index,
) -> list[str]:
    question_tokens = set(tokens(question))
    visible_values: list[str] = []
    for document in page:
        visible_values.extend(informative_tokens(document.visible_text))
    unique = [
        token
        for token in dict.fromkeys(visible_values)
        if token not in question_tokens
    ]
    rare = sorted(
        unique,
        key=lambda token: (-index.idf.get(token, 0.0), unique.index(token)),
    )[:8]

    def candidates() -> Iterable[str]:
        for token in rare:
            yield token
            yield f"{question} {token}"
        for left, right in itertools.combinations(rare[:6], 2):
            yield f"{left} {right}"
        for document in page:
            yield document.document_id
            yield f"{question} {document.document_id}"

    return _dedupe_limited(candidates(), CONTINUATION_QUERY_LIMIT)


def build_corpus(
    data: dict[str, Any],
) -> tuple[list[CorpusDocument], dict[str, dict[str, Any]]]:
    documents: list[CorpusDocument] = []
    seen: set[str] = set()
    entries: dict[str, dict[str, Any]] = {}
    for entry in data["documents"]:
        task_id = str(entry["question_id"])
        entries[task_id] = entry
        for raw in entry["documents"]:
            document_id = str(raw["id"])
            if document_id in seen:
                continue
            seen.add(document_id)
            documents.append(
                CorpusDocument(
                    document_id=document_id,
                    content=str(raw["document"]),
                    metadata=dict(raw.get("metadata") or {}),
                )
            )
    return documents, entries


def _coverage(indices: Iterable[int], oracle_ids: set[str], index: BM25Index) -> int:
    return len(
        {
            index.documents[value].document_id
            for value in indices
            if index.documents[value].document_id in oracle_ids
        }
    )


def analyze_task(
    entry: dict[str, Any],
    index: BM25Index,
) -> dict[str, Any]:
    question = str(entry["question"])
    oracle_ids = {str(value) for value in entry["document_ids"]}
    queries = root_queries(question, index)
    roots: list[dict[str, Any]] = []
    for root_position, query in enumerate(queries):
        first_indices = index.search(query)
        first_page = [index.documents[value] for value in first_indices]
        immediate = _coverage(first_indices, oracle_ids, index)
        best_final = immediate
        best_continuation = None
        for continuation in continuation_queries(question, first_page, index):
            second_indices = index.search(continuation)
            final = _coverage(
                itertools.chain(first_indices, second_indices), oracle_ids, index
            )
            if final > best_final:
                best_final = final
                best_continuation = continuation
        roots.append(
            {
                "root_position": root_position,
                "query": query,
                "immediate_coverage": immediate,
                "best_two_step_coverage": best_final,
                "best_continuation_query": best_continuation,
            }
        )

    greedy = max(
        roots,
        key=lambda row: (
            row["immediate_coverage"],
            -row["root_position"],
        ),
    )
    nonmyopic = max(
        roots,
        key=lambda row: (
            row["best_two_step_coverage"],
            row["immediate_coverage"],
            -row["root_position"],
        ),
    )
    strict = (
        nonmyopic["root_position"] != greedy["root_position"]
        and nonmyopic["immediate_coverage"] < greedy["immediate_coverage"]
        and nonmyopic["best_two_step_coverage"]
        > greedy["best_two_step_coverage"]
        and nonmyopic["best_two_step_coverage"]
        > nonmyopic["immediate_coverage"]
    )
    return {
        "task_id": str(entry["question_id"]),
        "source_split": str(entry.get("source_split", "")),
        "oracle_document_count": len(oracle_ids),
        "root_query_count": len(queries),
        "strict_nonmyopic_opportunity": strict,
        "greedy_root_position": greedy["root_position"],
        "greedy_immediate_coverage": greedy["immediate_coverage"],
        "greedy_best_two_step_coverage": greedy["best_two_step_coverage"],
        "nonmyopic_root_position": nonmyopic["root_position"],
        "nonmyopic_immediate_coverage": nonmyopic["immediate_coverage"],
        "nonmyopic_best_two_step_coverage": nonmyopic[
            "best_two_step_coverage"
        ],
        "immediate_coverage_sacrifice": (
            greedy["immediate_coverage"] - nonmyopic["immediate_coverage"]
        ),
        "two_step_coverage_gain": (
            nonmyopic["best_two_step_coverage"]
            - greedy["best_two_step_coverage"]
        ),
        "nonmyopic_root_query": nonmyopic["query"],
        "nonmyopic_continuation_query": nonmyopic[
            "best_continuation_query"
        ],
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [row for row in rows if row["strict_nonmyopic_opportunity"]]
    strict_four_document = [
        row for row in strict if row["oracle_document_count"] >= 4
    ]
    mean_gain = (
        sum(row["two_step_coverage_gain"] for row in rows) / len(rows)
        if rows
        else 0.0
    )
    gates = {
        "exactly_500_opportunity_tasks": len(rows) == OPPORTUNITY_SIZE,
        "all_tasks_are_official_test_split": all(
            row["source_split"] == "test" for row in rows
        ),
        "at_least_40_strict_nonmyopic_opportunities": len(strict) >= 40,
        "at_least_25_four_document_strict_opportunities": (
            len(strict_four_document) >= 25
        ),
        "mean_two_step_coverage_gain_at_least_point_08": mean_gain >= 0.08,
        "all_strict_rows_have_immediate_sacrifice_and_final_gain": all(
            row["immediate_coverage_sacrifice"] >= 1
            and row["two_step_coverage_gain"] >= 1
            for row in strict
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_tasks": len(rows),
        "strict_nonmyopic_opportunities": len(strict),
        "strict_opportunity_rate": len(strict) / len(rows) if rows else 0.0,
        "strict_four_document_opportunities": len(strict_four_document),
        "mean_two_step_coverage_gain_over_all_tasks": mean_gain,
        "gates": gates,
    }


def run_audit(source_path: Path) -> dict[str, Any]:
    source_hash = sha256_file(source_path)
    if source_hash != SOURCE_SHA256:
        raise ValueError(
            f"source SHA-256 mismatch: expected {SOURCE_SHA256}, got {source_hash}"
        )
    data = json.loads(source_path.read_text(encoding="utf-8"))
    splits = split_test_ids(data["test_problem_ids"])
    observed_hashes = {
        name: ordered_list_hash(values) for name, values in splits.items()
    }
    if observed_hashes != SPLIT_HASHES:
        raise ValueError("frozen GSM-Agent split hashes did not reproduce")
    documents, entries = build_corpus(data)
    if len(entries) != SOURCE_PROBLEMS or len(documents) != SOURCE_DOCUMENTS:
        raise ValueError("unexpected GSM-Agent source counts")
    index = BM25Index(documents)
    rows = [
        analyze_task(entries[task_id], index)
        for task_id in splits["opportunity"]
    ]
    return {
        "status": "passed" if summarize(rows)["gates"]["all_pass"] else "failed",
        "source": {
            "repository": "https://github.com/GuoTianYu2000/GSM-Agent",
            "commit": SOURCE_COMMIT,
            "sha256": source_hash,
            "problem_count": len(entries),
            "document_count": len(documents),
            "test_problem_count": len(splits["all_test"]),
        },
        "selection_seed": SELECTION_SEED,
        "split_sizes": {name: len(values) for name, values in splits.items()},
        "split_hashes": observed_hashes,
        "query_protocol": {
            "page_size": PAGE_SIZE,
            "root_query_limit": ROOT_QUERY_LIMIT,
            "continuation_query_limit": CONTINUATION_QUERY_LIMIT,
            "retrieval": "dependency-free BM25 over official document content",
            "continuation_visibility": "question plus first-page IDs, content, and metadata",
        },
        "summary": summarize(rows),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    artifact = run_audit(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(artifact, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(artifact["summary"], indent=2, sort_keys=True))
    raise SystemExit(0 if artifact["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
