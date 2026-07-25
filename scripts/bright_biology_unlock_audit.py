#!/usr/bin/env python3
"""Audit two-step retrieval opportunity on the frozen BRIGHT biology split."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Callable, Iterable, Sequence

import numpy as np
from rank_bm25 import BM25Okapi


SCHEMA_VERSION = 1
CODE_COMMIT = "d99e8391d967d4c2b3a74732530d2309e2fc92b6"
DATASET_COMMIT = "3066d29c9651a576c8aba4832d249807b181ecae"
EXAMPLES_SHA256 = (
    "6e105c4f09d9a70b8a20ed6a4d0e386823a5545151df41b3f0e64eb5c5987829"
)
DOCUMENTS_SHA256 = (
    "8516d0c233f9c34e9eb6922b56e8a1698e5a6f6e504a9499fcd511cdd5741670"
)
SELECTION_SEED = 24340
OPPORTUNITY_IDS = (
    "98",
    "92",
    "82",
    "39",
    "80",
    "18",
    "45",
    "94",
    "31",
    "52",
    "56",
    "43",
    "50",
    "51",
    "30",
    "29",
    "61",
    "23",
    "60",
    "79",
)
ROOT_TOP_K = 3
FOLLOWUP_TOP_K = 3
DOCUMENT_TERM_COUNT = 16
AGGREGATE_TERM_COUNT = 8
TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "because",
        "before",
        "being",
        "between",
        "could",
        "does",
        "from",
        "have",
        "into",
        "more",
        "most",
        "only",
        "other",
        "should",
        "some",
        "such",
        "than",
        "that",
        "their",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "under",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
        "your",
    }
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def normalized(text: str) -> str:
    return " ".join(tokenize(text))


def root_queries(query: str, reasoning: str) -> list[str]:
    lines = [" ".join(line.split()) for line in query.splitlines() if line.strip()]
    sentences = [
        " ".join(sentence.split())
        for sentence in SENTENCE_PATTERN.split(query)
        if sentence.strip()
    ]
    question_sentences = [
        sentence for sentence in sentences if sentence.rstrip().endswith("?")
    ]
    final_sentence = (
        question_sentences[-1]
        if question_sentences
        else sentences[-1] if sentences else query
    )
    used = {normalized(query), normalized(lines[0] if lines else query)}
    remaining = [
        sentence for sentence in sentences if normalized(sentence) not in used
    ]
    longest_remaining = max(
        remaining,
        key=lambda sentence: (len(tokenize(sentence)), -sentences.index(sentence)),
        default=final_sentence,
    )
    candidates = [
        query,
        lines[0] if lines else query,
        reasoning,
        final_sentence,
        longest_remaining,
    ]
    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            roots.append(" ".join(candidate.split()))
            seen.add(key)
    return roots


class BM25Corpus:
    def __init__(self, documents: Sequence[dict[str, str]]) -> None:
        self.documents = list(documents)
        self.ids = [document["id"] for document in documents]
        if len(self.ids) != len(set(self.ids)):
            raise ValueError("BRIGHT document IDs are not unique")
        self.tokenized = [tokenize(document["content"]) for document in documents]
        self.index = BM25Okapi(self.tokenized)
        self.id_to_index = {
            doc_id: index for index, doc_id in enumerate(self.ids)
        }
        self.doc_lengths = np.asarray(self.index.doc_len, dtype=float)
        self.length_normalizers = self.index.k1 * (
            1.0
            - self.index.b
            + self.index.b * self.doc_lengths / self.index.avgdl
        )
        mutable_postings: dict[str, tuple[list[int], list[float]]] = {}
        for index, frequencies in enumerate(self.index.doc_freqs):
            for token, frequency in frequencies.items():
                indices, values = mutable_postings.setdefault(token, ([], []))
                indices.append(index)
                values.append(float(frequency))
        self.postings = {
            token: (
                np.asarray(indices, dtype=np.int64),
                np.asarray(values, dtype=float),
            )
            for token, (indices, values) in mutable_postings.items()
        }

    def scores(self, query: str) -> np.ndarray:
        """Return scores algebraically identical to rank_bm25's dense loop."""
        scores = np.zeros(len(self.documents), dtype=float)
        for token, multiplicity in Counter(tokenize(query)).items():
            posting = self.postings.get(token)
            if posting is None:
                continue
            indices, frequencies = posting
            contribution = (
                float(self.index.idf.get(token, 0.0))
                * (
                    frequencies
                    * (self.index.k1 + 1.0)
                    / (frequencies + self.length_normalizers[indices])
                )
                * multiplicity
            )
            scores[indices] += contribution
        return scores

    def search(
        self,
        query: str,
        *,
        top_k: int,
        excluded_ids: Iterable[str] = (),
    ) -> list[dict[str, str]]:
        excluded = set(excluded_ids)
        scores = self.scores(query)
        if excluded:
            for doc_id in excluded:
                index = self.id_to_index.get(doc_id)
                if index is not None:
                    scores[index] = -np.inf
        available = int(np.isfinite(scores).sum())
        count = min(top_k, available)
        if count <= 0:
            return []
        candidates = np.argpartition(-scores, count - 1)[:count]
        cutoff = float(np.min(scores[candidates]))
        tied = np.flatnonzero(scores >= cutoff)
        ordered = sorted(tied, key=lambda index: (-scores[index], index))[:count]
        return [self.documents[index] for index in ordered]

    def highest_idf_terms(
        self,
        contents: Sequence[str],
        *,
        excluded_terms: Iterable[str],
        count: int,
    ) -> list[str]:
        excluded = set(excluded_terms)
        terms = {
            token
            for content in contents
            for token in tokenize(content)
            if len(token) >= 3
            and token not in excluded
            and token not in STOPWORDS
            and token in self.index.idf
        }
        return sorted(
            terms,
            key=lambda token: (-float(self.index.idf[token]), token),
        )[:count]


def followup_queries(
    query: str,
    root_documents: Sequence[dict[str, str]],
    corpus: BM25Corpus,
) -> list[str]:
    excluded_terms = set(tokenize(query))
    followups = []
    for document in root_documents:
        terms = corpus.highest_idf_terms(
            [document["content"]],
            excluded_terms=excluded_terms,
            count=DOCUMENT_TERM_COUNT,
        )
        followups.append(" ".join([query, *terms]))
    aggregate_terms = corpus.highest_idf_terms(
        [document["content"] for document in root_documents],
        excluded_terms=excluded_terms,
        count=AGGREGATE_TERM_COUNT,
    )
    followups.append(" ".join([query, *aggregate_terms]))
    return followups


def analyze_example(
    example: dict[str, Any],
    corpus: BM25Corpus,
) -> dict[str, Any]:
    gold_ids = set(example["gold_ids"])
    if not gold_ids:
        raise ValueError(f"BRIGHT example {example['id']} has no gold IDs")
    missing = gold_ids.difference(corpus.ids)
    if missing:
        raise ValueError(
            f"BRIGHT example {example['id']} has missing gold IDs: {sorted(missing)}"
        )
    roots = root_queries(example["query"], example["reasoning"])
    root_records: list[dict[str, Any]] = []
    for root_index, root_query in enumerate(roots):
        root_documents = corpus.search(root_query, top_k=ROOT_TOP_K)
        root_ids = [document["id"] for document in root_documents]
        immediate_utility = len(gold_ids.intersection(root_ids))
        continuation_records = []
        for followup_index, followup_query in enumerate(
            followup_queries(example["query"], root_documents, corpus)
        ):
            followup_documents = corpus.search(
                followup_query,
                top_k=FOLLOWUP_TOP_K,
                excluded_ids=root_ids,
            )
            followup_ids = [document["id"] for document in followup_documents]
            pair_ids = set(root_ids + followup_ids)
            continuation_records.append(
                {
                    "followup_index": followup_index,
                    "query": followup_query,
                    "document_ids": followup_ids,
                    "pair_utility": len(gold_ids.intersection(pair_ids)),
                }
            )
        best_followup = max(
            continuation_records,
            key=lambda record: (
                record["pair_utility"],
                -record["followup_index"],
            ),
        )
        root_records.append(
            {
                "root_index": root_index,
                "query": root_query,
                "document_ids": root_ids,
                "immediate_utility": immediate_utility,
                "best_pair_utility": best_followup["pair_utility"],
                "best_followup_index": best_followup["followup_index"],
                "continuations": continuation_records,
            }
        )

    immediate_root = max(
        root_records,
        key=lambda record: (
            record["immediate_utility"],
            -record["root_index"],
        ),
    )
    oracle_pair = max(
        root_records,
        key=lambda record: (
            record["best_pair_utility"],
            -record["root_index"],
        ),
    )
    distinct_first_results = len(
        {
            record["document_ids"][0]
            for record in root_records
            if record["document_ids"]
        }
    )
    return {
        "id": example["id"],
        "root_count": len(root_records),
        "gold_count": len(gold_ids),
        "distinct_first_results": distinct_first_results,
        "best_first_utility": max(
            record["immediate_utility"] for record in root_records
        ),
        "immediate_root_index": immediate_root["root_index"],
        "immediate_oracle_tail_utility": immediate_root["best_pair_utility"],
        "oracle_root_index": oracle_pair["root_index"],
        "oracle_pair_utility": oracle_pair["best_pair_utility"],
        "pair_gain": oracle_pair["best_pair_utility"]
        - max(record["immediate_utility"] for record in root_records),
        "nonmyopic_gap": oracle_pair["best_pair_utility"]
        - immediate_root["best_pair_utility"],
        "direct_query_utility": root_records[0]["immediate_utility"],
        "roots": root_records,
    }


def summarize_records(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("BRIGHT audit has no records")
    analyzable = sum(record["root_count"] >= 3 for record in records)
    root_diversity = float(
        np.mean([record["distinct_first_results"] for record in records])
    )
    pair_gain_count = sum(record["pair_gain"] >= 1 for record in records)
    mean_pair_gain = float(np.mean([record["pair_gain"] for record in records]))
    root_change_count = sum(
        record["oracle_root_index"] != record["immediate_root_index"]
        for record in records
    )
    gap_count = sum(record["nonmyopic_gap"] >= 1 for record in records)
    mean_gap = float(np.mean([record["nonmyopic_gap"] for record in records]))
    total_gold = sum(record["gold_count"] for record in records)
    direct_gold = sum(record["direct_query_utility"] for record in records)
    direct_coverage = direct_gold / total_gold
    gates = {
        "analyzable_at_least_16": analyzable >= 16,
        "mean_root_diversity_at_least_2_5": root_diversity >= 2.5,
        "pair_gain_count_at_least_10": pair_gain_count >= 10,
        "mean_pair_gain_at_least_0_50": mean_pair_gain >= 0.50,
        "root_change_count_at_least_5": root_change_count >= 5,
        "gap_count_at_least_4": gap_count >= 4,
        "mean_nonmyopic_gap_at_least_0_20": mean_gap >= 0.20,
        "direct_coverage_below_0_60": direct_coverage < 0.60,
    }
    return {
        "record_count": len(records),
        "analyzable_count": analyzable,
        "mean_distinct_first_results": root_diversity,
        "pair_gain_count": pair_gain_count,
        "mean_pair_gain": mean_pair_gain,
        "root_change_count": root_change_count,
        "nonmyopic_gap_count": gap_count,
        "mean_nonmyopic_gap": mean_gap,
        "direct_query_gold_chunks": direct_gold,
        "total_gold_chunks": total_gold,
        "direct_query_coverage": direct_coverage,
        "gates": gates,
        "passed": all(gates.values()),
    }


def load_frozen_data(
    examples_path: Path,
    documents_path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    if _sha256(examples_path) != EXAMPLES_SHA256:
        raise ValueError("BRIGHT examples parquet hash does not match")
    if _sha256(documents_path) != DOCUMENTS_SHA256:
        raise ValueError("BRIGHT documents parquet hash does not match")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover - exercised by the CLI
        raise RuntimeError(
            "pyarrow is required for BRIGHT parquet loading; "
            "run with `uv run --with pyarrow`"
        ) from exc
    examples = pq.read_table(
        examples_path,
        columns=["id", "query", "reasoning", "gold_ids"],
        filters=[("id", "in", list(OPPORTUNITY_IDS))],
    ).to_pylist()
    by_id = {example["id"]: example for example in examples}
    if set(by_id) != set(OPPORTUNITY_IDS):
        raise ValueError("frozen BRIGHT opportunity split does not reproduce")
    selected = [by_id[example_id] for example_id in OPPORTUNITY_IDS]
    documents = pq.read_table(
        documents_path,
        columns=["id", "content"],
    ).to_pylist()
    if len(documents) != 57359:
        raise ValueError("BRIGHT biology corpus shape does not match")
    return selected, documents


def verify_code_checkout(code_root: Path) -> None:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=code_root,
        check=True,
        capture_output=True,
        text=True,
    )
    if result.stdout.strip() != CODE_COMMIT:
        raise ValueError("BRIGHT code checkout commit does not match")


def run_audit(
    *,
    examples_path: Path,
    documents_path: Path,
    code_root: Path,
) -> dict[str, Any]:
    verify_code_checkout(code_root)
    examples, documents = load_frozen_data(examples_path, documents_path)
    corpus = BM25Corpus(documents)
    records = [analyze_example(example, corpus) for example in examples]
    summary = summarize_records(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["passed"] else "failed_gate",
        "protocol": {
            "benchmark": "BRIGHT biology",
            "code_commit": CODE_COMMIT,
            "dataset_commit": DATASET_COMMIT,
            "examples_sha256": EXAMPLES_SHA256,
            "documents_sha256": DOCUMENTS_SHA256,
            "selection_seed": SELECTION_SEED,
            "opportunity_ids": list(OPPORTUNITY_IDS),
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "document_term_count": DOCUMENT_TERM_COUNT,
            "aggregate_term_count": AGGREGATE_TERM_COUNT,
            "api_requests": 0,
        },
        "summary": summary,
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--examples", type=Path, required=True)
    parser.add_argument("--documents", type=Path, required=True)
    parser.add_argument("--code-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = run_audit(
        examples_path=args.examples,
        documents_path=args.documents,
        code_root=args.code_root,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
