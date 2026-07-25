#!/usr/bin/env python3
"""Test whether MultiHop-RAG has a target-blind non-myopic retrieval gap."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

try:
    from scripts.multihop_rag_semantic_bed_manifest import (
        CORPUS_SHA256,
        EXPECTED_MECHANICS_IDS,
        sha256_file,
    )
except ModuleNotFoundError:
    from multihop_rag_semantic_bed_manifest import (
        CORPUS_SHA256,
        EXPECTED_MECHANICS_IDS,
        sha256_file,
    )


OPEN_MECHANICS_SHA256 = (
    "0db23034c9758823198eb0cb38d17a449c227cb41560d5b509c4d79c6346d6ca"
)
ROOT_RETRIEVAL_SIZE = 3
CONTINUATION_RETRIEVAL_SIZE = 3
STOPWORDS = frozenset(
    """
    a an and are as at be been being but by can could did do does for from
    had has have he her hers him his how i if in into is it its may might
    more most not of on or our she should so than that the their them there
    they this to was were what when where which who why will with would you
    your
    """.split()
)


def _tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) > 1 and token not in STOPWORDS
    ]


def _dedupe(values: Iterable[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = " ".join(value.split())
        if normalized and normalized not in seen and _tokens(normalized):
            seen.add(normalized)
            result.append(normalized)
    return result


class CorpusIndex:
    def __init__(self, corpus: Sequence[dict[str, Any]]) -> None:
        self.corpus = list(corpus)
        texts = [
            " ".join(
                str(document.get(key, ""))
                for key in (
                    "title",
                    "author",
                    "source",
                    "published_at",
                    "category",
                    "body",
                )
            )
            for document in self.corpus
        ]
        self.tokenized = [_tokens(text) for text in texts]
        self.bm25 = BM25Okapi(self.tokenized)
        self.document_frequency = Counter(
            token for tokens in self.tokenized for token in set(tokens)
        )

    def rare_tokens(self, text: str, limit: int) -> list[str]:
        unique = list(dict.fromkeys(_tokens(text)))
        return sorted(
            unique,
            key=lambda token: (
                self.document_frequency[token],
                -len(token),
                token,
            ),
        )[:limit]

    def search(
        self,
        query: str,
        *,
        exclude: Iterable[int] = (),
        limit: int,
    ) -> list[int]:
        scores = self.bm25.get_scores(_tokens(query))
        excluded = set(exclude)
        order = np.lexsort((np.arange(len(scores)), -scores))
        return [
            int(index)
            for index in order
            if int(index) not in excluded
        ][:limit]


def _capitalized_phrases(text: str) -> str:
    phrases = re.findall(
        r"(?<![.!?]\s)(?:\b[A-Z][\w.-]*(?:\s+|$)){1,5}",
        text,
    )
    return " ".join(phrases)


def root_queries(question: str, index: CorpusIndex) -> list[str]:
    clauses = re.split(
        r"[?;]|\b(?:and|while|whereas|compared with|according to)\b",
        question,
        flags=re.IGNORECASE,
    )
    return _dedupe(
        [
            question,
            *(
                clause.strip()
                for clause in clauses
                if len(_tokens(clause)) >= 3
            ),
            _capitalized_phrases(question),
            " ".join(index.rare_tokens(question, 24)),
        ]
    )


def continuation_queries(
    question: str,
    document: dict[str, Any],
    index: CorpusIndex,
) -> list[str]:
    title = str(document.get("title", ""))
    body = str(document.get("body", ""))
    return _dedupe(
        [
            f"{question} {title}",
            f"{question} {' '.join(index.rare_tokens(title + ' ' + body, 12))}",
            f"{title} {' '.join(index.rare_tokens(body, 18))}",
            f"{question} {_capitalized_phrases(title + ' ' + body[:2500])}",
        ]
    )


def _evidence_indices(
    task: dict[str, Any],
    corpus: Sequence[dict[str, Any]],
) -> set[int]:
    evidence_urls = {
        str(item["url"]) for item in task["evidence_list"]
    }
    return {
        index
        for index, document in enumerate(corpus)
        if str(document.get("url")) in evidence_urls
    }


def evaluate_task(
    task: dict[str, Any],
    index: CorpusIndex,
) -> dict[str, Any]:
    evidence = _evidence_indices(task, index.corpus)
    action_results: list[dict[str, Any]] = []
    for root_query in root_queries(str(task["query"]), index):
        root_documents = index.search(
            root_query,
            limit=ROOT_RETRIEVAL_SIZE,
        )
        immediate = len(set(root_documents) & evidence)
        best_total = immediate
        for root_document in root_documents:
            for continuation_query in continuation_queries(
                str(task["query"]),
                index.corpus[root_document],
                index,
            ):
                continuation_documents = index.search(
                    continuation_query,
                    exclude=root_documents,
                    limit=CONTINUATION_RETRIEVAL_SIZE,
                )
                best_total = max(
                    best_total,
                    len(
                        (set(root_documents) | set(continuation_documents))
                        & evidence
                    ),
                )
        action_results.append(
            {
                "immediate_evidence": immediate,
                "best_total_evidence": best_total,
            }
        )
    best_immediate = max(
        result["immediate_evidence"] for result in action_results
    )
    best_total = max(
        result["best_total_evidence"] for result in action_results
    )
    greedy_total = max(
        result["best_total_evidence"]
        for result in action_results
        if result["immediate_evidence"] == best_immediate
    )
    return {
        "task_id": str(task["task_id"]),
        "question_type": str(task["question_type"]),
        "evidence_count": len(evidence),
        "root_action_count": len(action_results),
        "best_immediate_evidence": best_immediate,
        "best_total_evidence": best_total,
        "greedy_root_best_total_evidence": greedy_total,
        "continuation_gain": best_total - best_immediate,
        "strict_root_reversal": best_total > greedy_total,
    }


def build_analysis(
    mechanics_path: Path,
    corpus_path: Path,
) -> dict[str, Any]:
    if sha256_file(mechanics_path) != OPEN_MECHANICS_SHA256:
        raise ValueError("opened MultiHop-RAG mechanics hash changed")
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("MultiHop-RAG corpus hash changed")
    tasks = [
        json.loads(line)
        for line in mechanics_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if tuple(str(task["task_id"]) for task in tasks) != EXPECTED_MECHANICS_IDS:
        raise ValueError("opened MultiHop-RAG mechanics IDs changed")
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    index = CorpusIndex(corpus)
    task_results = [evaluate_task(task, index) for task in tasks]
    reversals = sum(result["strict_root_reversal"] for result in task_results)
    continuation_gains = sum(
        result["continuation_gain"] > 0 for result in task_results
    )
    return {
        "interface_version": "multihop-rag-semantic-bed-mechanics-1",
        "source": {
            "opened_mechanics_sha256": OPEN_MECHANICS_SHA256,
            "corpus_sha256": CORPUS_SHA256,
        },
        "protocol": {
            "root_retrieval_size": ROOT_RETRIEVAL_SIZE,
            "continuation_retrieval_size": CONTINUATION_RETRIEVAL_SIZE,
            "continuation_excludes_root_documents": True,
            "target_blind_query_construction": True,
            "llm_calls": 0,
        },
        "tasks": task_results,
        "summary": {
            "task_count": len(task_results),
            "tasks_with_continuation_gain": continuation_gains,
            "tasks_with_strict_root_reversal": reversals,
            "strict_root_reversal_rate": reversals / len(task_results),
            "opportunity_gate_passed": reversals > 0,
            "decision": (
                "freeze_opportunity_audit"
                if reversals > 0
                else "close_before_opportunity_audit"
            ),
        },
        "content_emitted": False,
        "answers_emitted": False,
        "document_ids_emitted": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mechanics-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analysis = build_analysis(args.mechanics_path, args.corpus_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(analysis, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(analysis["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
