#!/usr/bin/env python3
"""Audit observation-enabled two-search opportunity on frozen ScholarGym tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import re
import sqlite3
import sys
from typing import Any, Iterable, Iterator, Sequence


SCHEMA_VERSION = 1
SOURCE_REPO_COMMIT = "cb1c5fc7c796308bef353b18549a92e2739f5b96"
DATASET_REVISION = "be7d917ddac3cd6f2878f81160965f914dab3706"
BENCHMARK_SHA256 = (
    "869f507eacb7f554b8f4e6dc65ea97b01f95140ba5aa86a5119b330c41b9d551"
)
CORPUS_SHA256 = (
    "e1e570321e22bf59784d46a9dd28239350d7ad1ddda213eee2f5730710f239e8"
)
EXPECTED_BENCHMARK_ROWS = 2_536
EXPECTED_CORPUS_ROWS = 570_206
EXPECTED_UNIQUE_GT_IDS = 4_498

OPPORTUNITY_SEED = 270727
DEVELOPMENT_SEED = 270728
HOLDOUT_SEED = 270729
OPPORTUNITY_SIZE = 40
DEVELOPMENT_SIZE = 24
HOLDOUT_SIZE = 24
OPPORTUNITY_POOL_SIZE = 542
DEVELOPMENT_POOL_SIZE = 480
HOLDOUT_POOL_SIZE = 49
OPPORTUNITY_ID_HASH = (
    "02d6cf131f8be972c62d14e7167e1a7e17ef4b23d79e61e7e9f40377ca55f432"
)
DEVELOPMENT_ID_HASH = (
    "dcf14914b3516afe2362915422b99c44beeae16ea4596778dad31f682ac6efdf"
)
HOLDOUT_ID_HASH = (
    "04fd9f531470f94f42d89f56cbc1f676ffa28466e6768ca617a140bbc896f15e"
)
OPPORTUNITY_POOL_ORDER_HASH = (
    "f2cdce46589fc6feddb56f6af2bf24c0a59d9921d92bc1b48918dd98dedb004f"
)
DEVELOPMENT_POOL_ORDER_HASH = (
    "79895a164dc7a0d880a442e68097b834f1a99e2d1ea01fbb6277f5e5b68155ff"
)
HOLDOUT_POOL_ORDER_HASH = (
    "34305e45bb625d0ce8d2f8afc9da030a79d50a1c16f615cc5acba891de2c3916"
)

ROOT_TOP_K = 5
FOLLOWUP_TOP_K = 5
MAX_ROOTS = 16
MAX_FOLLOWUPS = 24
OBSERVATION_CHAR_CAP = 2_000
PER_PAPER_TERMS = 5
AGGREGATE_TERMS = 10

MIN_GT_PAPERS = 2
MIN_ROOTS = 5
MIN_DIVERSE_TASKS = 30
MIN_PAIR_GAIN_TASKS = 15
MIN_MEAN_ORACLE_PAIR_RECALL = 0.15
MIN_MEAN_PAIR_RECALL_GAIN = 0.03
MIN_STRICT_OPPORTUNITIES = 5
MIN_STRICT_TOTAL_GAP = 5
MIN_MEAN_STRICT_NORMALIZED_GAP = 0.15

TOKEN_PATTERN = re.compile(r"[a-z]+")
VERSION_SUFFIX_PATTERN = re.compile(r"v\d+$", re.IGNORECASE)
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_PATTERN = re.compile(r"[,;:]|\s+[—-]\s+")
STOPWORDS = frozenset(
    {
        "about",
        "after",
        "again",
        "also",
        "among",
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
        "paper",
        "papers",
        "research",
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
        "using",
        "very",
        "what",
        "when",
        "where",
        "which",
        "while",
        "with",
        "would",
    }
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def canonical_arxiv_id(value: str) -> str:
    return VERSION_SUFFIX_PATTERN.sub("", str(value).strip())


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(text.casefold())


def normalized(text: str) -> str:
    return " ".join(tokenize(text))


def stream_json_object(path: Path, *, chunk_size: int = 1024 * 1024) -> Iterator[tuple[str, Any]]:
    """Yield entries from a top-level JSON object without loading the file."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        finished = False

        def refill() -> bool:
            nonlocal buffer, position, finished
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if not chunk:
                finished = True
                return False
            buffer += chunk
            return True

        refill()
        while True:
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("empty JSON object")
            if buffer[position] != "{":
                raise ValueError("corpus JSON must be a top-level object")
            position += 1
            break

        expect_entry = True
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace()
                    or (not expect_entry and buffer[position] == ",")
                ):
                    if buffer[position] == ",":
                        expect_entry = True
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unterminated JSON object")

            if buffer[position] == "}":
                position += 1
                break
            if not expect_entry:
                raise ValueError("expected a comma between object entries")

            while True:
                try:
                    key, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if not refill():
                        raise ValueError("invalid JSON object key")
            if not isinstance(key, str):
                raise ValueError("corpus object key must be a string")
            position = end

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("missing colon after object key")
            if buffer[position] != ":":
                raise ValueError("missing colon after object key")
            position += 1

            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if not refill():
                        raise ValueError(f"invalid JSON value for {key}")
            position = end
            expect_entry = False
            yield key, value

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                raise ValueError("trailing content after corpus object")
            if finished or not refill():
                break


def load_benchmark(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw = json.loads(line)
            if not isinstance(raw, dict):
                raise ValueError(f"benchmark row {line_number} is not an object")
            rows.append(raw)
    if len(rows) != EXPECTED_BENCHMARK_ROWS:
        raise ValueError(
            f"benchmark has {len(rows)} rows, expected {EXPECTED_BENCHMARK_ROWS}"
        )
    qids = [str(row["qid"]) for row in rows]
    if len(qids) != len(set(qids)):
        raise ValueError("benchmark qids are not unique")
    return rows


def ground_truth_ids(row: dict[str, Any]) -> tuple[str, ...]:
    papers = row.get("cited_paper", [])
    labels = row.get("gt_label", [])
    if not isinstance(papers, list) or not isinstance(labels, list):
        raise ValueError(f"{row.get('qid')} has invalid ground-truth fields")
    if len(papers) != len(labels):
        raise ValueError(f"{row.get('qid')} ground-truth fields differ in length")
    ids = []
    for paper, label in zip(papers, labels):
        if int(label) != 1:
            continue
        if not isinstance(paper, dict) or not paper.get("arxiv_id"):
            raise ValueError(f"{row.get('qid')} has an invalid positive paper")
        ids.append(canonical_arxiv_id(str(paper["arxiv_id"])))
    return tuple(dict.fromkeys(ids))


def frozen_split_ids(rows: Sequence[dict[str, Any]]) -> dict[str, list[str]]:
    opportunity_pool = [
        str(row["qid"])
        for row in rows
        if str(row["qid"]).startswith("AutoScholarQuery_dev_")
        and len(ground_truth_ids(row)) >= MIN_GT_PAPERS
    ]
    development_pool = [
        str(row["qid"])
        for row in rows
        if str(row["qid"]).startswith("AutoScholarQuery_test_")
        and int(str(row["qid"]).rsplit("_", 1)[1]) >= 100
        and len(ground_truth_ids(row)) >= MIN_GT_PAPERS
    ]
    holdout_pool = [
        str(row["qid"])
        for row in rows
        if str(row["qid"]).startswith("RealScholarQuery_")
        and len(ground_truth_ids(row)) >= MIN_GT_PAPERS
    ]
    specs = {
        "opportunity": (
            opportunity_pool,
            OPPORTUNITY_POOL_SIZE,
            OPPORTUNITY_SEED,
            OPPORTUNITY_SIZE,
            OPPORTUNITY_POOL_ORDER_HASH,
            OPPORTUNITY_ID_HASH,
        ),
        "development": (
            development_pool,
            DEVELOPMENT_POOL_SIZE,
            DEVELOPMENT_SEED,
            DEVELOPMENT_SIZE,
            DEVELOPMENT_POOL_ORDER_HASH,
            DEVELOPMENT_ID_HASH,
        ),
        "holdout": (
            holdout_pool,
            HOLDOUT_POOL_SIZE,
            HOLDOUT_SEED,
            HOLDOUT_SIZE,
            HOLDOUT_POOL_ORDER_HASH,
            HOLDOUT_ID_HASH,
        ),
    }
    selected: dict[str, list[str]] = {}
    for name, (pool, size, seed, take, pool_hash, selected_hash) in specs.items():
        if len(pool) != size:
            raise AssertionError(f"{name} pool has {len(pool)} rows, expected {size}")
        order = list(pool)
        random.Random(seed).shuffle(order)
        if _id_hash(order) != pool_hash:
            raise AssertionError(f"{name} shuffled-pool hash does not reproduce")
        selected[name] = order[:take]
        if _id_hash(selected[name]) != selected_hash:
            raise AssertionError(f"{name} selected-ID hash does not reproduce")
    if set(selected["opportunity"]) & set(selected["development"]):
        raise AssertionError("opportunity and development IDs overlap")
    if set().union(*map(set, selected.values())) & {
        f"AutoScholarQuery_test_{index}" for index in range(100)
    }:
        raise AssertionError("a quarantined ID entered a frozen split")
    return selected


def build_corpus_index(corpus_path: Path, index_path: Path) -> None:
    """Build or validate the frozen FTS5 index."""
    if index_path.exists():
        with sqlite3.connect(index_path) as connection:
            metadata = dict(connection.execute("SELECT key, value FROM metadata"))
            paper_count = connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        if (
            metadata.get("schema_version") == str(SCHEMA_VERSION)
            and metadata.get("corpus_sha256") == CORPUS_SHA256
            and paper_count == EXPECTED_CORPUS_ROWS
        ):
            return
        raise ValueError(f"existing index is incompatible: {index_path}")

    index_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(index_path) as connection:
        connection.executescript(
            """
            PRAGMA journal_mode=OFF;
            PRAGMA synchronous=OFF;
            PRAGMA temp_store=MEMORY;
            PRAGMA cache_size=-262144;
            CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            CREATE TABLE papers (
                rowid INTEGER PRIMARY KEY,
                arxiv_id TEXT NOT NULL UNIQUE,
                title TEXT NOT NULL,
                abstract TEXT NOT NULL,
                date TEXT NOT NULL
            );
            CREATE VIRTUAL TABLE papers_fts USING fts5(
                title,
                abstract,
                content='papers',
                content_rowid='rowid',
                tokenize='unicode61 remove_diacritics 2'
            );
            """
        )
        batch: list[tuple[str, str, str, str]] = []
        seen = 0
        for key, value in stream_json_object(corpus_path):
            if not isinstance(value, dict):
                raise ValueError(f"paper {key} is not an object")
            arxiv_id = canonical_arxiv_id(str(value.get("arxiv_id") or key))
            batch.append(
                (
                    arxiv_id,
                    str(value.get("title", "")),
                    str(value.get("abstract", "")),
                    str(value.get("date", "")),
                )
            )
            seen += 1
            if len(batch) >= 2_000:
                connection.executemany(
                    "INSERT INTO papers(arxiv_id,title,abstract,date) VALUES(?,?,?,?)",
                    batch,
                )
                batch.clear()
        if batch:
            connection.executemany(
                "INSERT INTO papers(arxiv_id,title,abstract,date) VALUES(?,?,?,?)",
                batch,
            )
        if seen != EXPECTED_CORPUS_ROWS:
            raise ValueError(f"corpus has {seen} papers, expected {EXPECTED_CORPUS_ROWS}")
        connection.execute("INSERT INTO papers_fts(papers_fts) VALUES('rebuild')")
        connection.execute(
            "CREATE VIRTUAL TABLE papers_vocab USING fts5vocab(papers_fts, 'row')"
        )
        connection.executemany(
            "INSERT INTO metadata(key,value) VALUES(?,?)",
            [
                ("schema_version", str(SCHEMA_VERSION)),
                ("corpus_sha256", CORPUS_SHA256),
                ("paper_count", str(seen)),
            ],
        )
        connection.commit()


class ScholarCorpus:
    def __init__(self, index_path: Path) -> None:
        self.connection = sqlite3.connect(index_path)
        self.paper_count = int(
            self.connection.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        )

    def close(self) -> None:
        self.connection.close()

    def idf_many(self, terms: Iterable[str]) -> dict[str, float]:
        unique = sorted(set(terms))
        if not unique:
            return {}
        placeholders = ",".join("?" for _ in unique)
        rows = self.connection.execute(
            f"SELECT term, doc FROM papers_vocab WHERE term IN ({placeholders})",
            unique,
        )
        result = {}
        for term, document_frequency in rows:
            result[str(term)] = math.log(
                1.0
                + (self.paper_count - int(document_frequency) + 0.5)
                / (int(document_frequency) + 0.5)
            )
        return result

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
            if len(token) >= 3 and token not in excluded and token not in STOPWORDS
        }
        idfs = self.idf_many(terms)
        return sorted(idfs, key=lambda term: (-idfs[term], term))[:count]

    def search(
        self,
        query: str,
        *,
        top_k: int,
        before_date: str = "",
        excluded_ids: Iterable[str] = (),
    ) -> list[dict[str, str]]:
        tokens = list(dict.fromkeys(tokenize(query)))
        if not tokens:
            return []
        match_query = " OR ".join(f'"{token}"' for token in tokens)
        excluded = sorted(set(excluded_ids))
        clauses = ["papers_fts MATCH ?"]
        parameters: list[Any] = [match_query]
        if before_date:
            clauses.append("(papers.date = '' OR substr(papers.date, 1, 7) <= ?)")
            parameters.append(str(before_date)[:7])
        if excluded:
            placeholders = ",".join("?" for _ in excluded)
            clauses.append(f"papers.arxiv_id NOT IN ({placeholders})")
            parameters.extend(excluded)
        parameters.append(top_k)
        rows = self.connection.execute(
            f"""
            SELECT papers.arxiv_id, papers.title, papers.abstract, papers.date,
                   bm25(papers_fts) AS score
            FROM papers_fts
            JOIN papers ON papers.rowid = papers_fts.rowid
            WHERE {' AND '.join(clauses)}
            ORDER BY score ASC, papers.rowid ASC
            LIMIT ?
            """,
            parameters,
        )
        return [
            {
                "arxiv_id": str(arxiv_id),
                "title": str(title),
                "abstract": str(abstract),
                "date": str(date),
                "score": float(score),
            }
            for arxiv_id, title, abstract, date, score in rows
        ]


def root_queries(query: str, corpus: ScholarCorpus) -> list[str]:
    sentences = [
        " ".join(sentence.split())
        for sentence in SENTENCE_PATTERN.split(query)
        if sentence.strip()
    ]
    clauses = [
        " ".join(clause.split())
        for sentence in sentences
        for clause in CLAUSE_PATTERN.split(sentence)
        if len(tokenize(clause)) >= 4
    ]
    query_tokens = tokenize(query)
    ranked_terms = corpus.highest_idf_terms(
        [query],
        excluded_terms=STOPWORDS,
        count=12,
    )
    rare = set(ranked_terms[:8])
    candidates = [
        query,
        *sentences,
        *clauses,
        " ".join(ranked_terms),
        *ranked_terms[:6],
    ]
    for width in (2, 3, 4):
        for start in range(max(0, len(query_tokens) - width + 1)):
            window = query_tokens[start : start + width]
            if rare.intersection(window):
                candidates.append(" ".join(window))

    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            roots.append(" ".join(candidate.split()))
            seen.add(key)
        if len(roots) >= MAX_ROOTS:
            break
    return roots


def visible_observation(paper: dict[str, str]) -> str:
    return f"{paper['title']}\n{paper['abstract'][:OBSERVATION_CHAR_CAP]}"


def followup_queries(
    initial_query: str,
    root_query: str,
    root_papers: Sequence[dict[str, str]],
    corpus: ScholarCorpus,
) -> list[tuple[str, tuple[str, ...]]]:
    observations = [visible_observation(paper) for paper in root_papers]
    excluded = set(tokenize(initial_query)).union(tokenize(root_query))
    groups = [
        corpus.highest_idf_terms(
            [observation],
            excluded_terms=excluded,
            count=PER_PAPER_TERMS,
        )
        for observation in observations
    ]
    aggregate = corpus.highest_idf_terms(
        observations,
        excluded_terms=excluded,
        count=AGGREGATE_TERMS,
    )
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for terms in [*groups, aggregate]:
        for term in terms:
            candidates.append((f"{initial_query} {term}", (term,)))
            candidates.append((f"{root_query} {term}", (term,)))
        for start in range(max(0, len(terms) - 1)):
            pair = (terms[start], terms[start + 1])
            candidates.append((f"{initial_query} {' '.join(pair)}", pair))
        if terms:
            candidates.append((f"{root_query} {' '.join(terms)}", tuple(terms)))

    followups: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for candidate, observed_terms in candidates:
        key = normalized(candidate)
        valid_terms = tuple(
            term
            for term in observed_terms
            if term not in excluded and term in set(tokenize(" ".join(observations)))
        )
        if key and valid_terms and key not in seen:
            followups.append((" ".join(candidate.split()), valid_terms))
            seen.add(key)
        if len(followups) >= MAX_FOLLOWUPS:
            break
    return followups


def analyze_task(row: dict[str, Any], corpus: ScholarCorpus) -> dict[str, Any]:
    task_id = str(row["qid"])
    query = str(row["query"])
    gt_ids = set(ground_truth_ids(row))
    if len(gt_ids) < MIN_GT_PAPERS:
        raise ValueError(f"{task_id} has fewer than {MIN_GT_PAPERS} GT papers")
    before_date = str(row.get("date", ""))
    roots = root_queries(query, corpus)
    root_records: list[dict[str, Any]] = []
    for root_index, root_query in enumerate(roots):
        first = corpus.search(
            root_query,
            top_k=ROOT_TOP_K,
            before_date=before_date,
        )
        first_ids = {paper["arxiv_id"] for paper in first}
        immediate_count = len(gt_ids & first_ids)
        continuations: list[dict[str, Any]] = []
        for followup_index, (followup, observed_terms) in enumerate(
            followup_queries(query, root_query, first, corpus)
        ):
            second = corpus.search(
                followup,
                top_k=FOLLOWUP_TOP_K,
                before_date=before_date,
                excluded_ids=first_ids,
            )
            second_ids = {paper["arxiv_id"] for paper in second}
            pair_count = len(gt_ids & (first_ids | second_ids))
            continuations.append(
                {
                    "followup_index": followup_index,
                    "pair_count": pair_count,
                    "added_gt_count": len(gt_ids & (second_ids - first_ids)),
                    "observation_term_count": len(observed_terms),
                }
            )
        best = max(
            continuations,
            key=lambda item: (item["pair_count"], -item["followup_index"]),
            default={
                "followup_index": -1,
                "pair_count": immediate_count,
                "added_gt_count": 0,
                "observation_term_count": 0,
            },
        )
        root_records.append(
            {
                "root_index": root_index,
                "top1_id": first[0]["arxiv_id"] if first else None,
                "immediate_count": immediate_count,
                "best_pair_count": int(best["pair_count"]),
                "best_added_gt_count": int(best["added_gt_count"]),
                "best_followup_index": int(best["followup_index"]),
                "best_observation_term_count": int(best["observation_term_count"]),
                "num_followups": len(continuations),
            }
        )

    greedy = max(
        root_records,
        key=lambda item: (
            item["immediate_count"],
            item["best_pair_count"],
            -item["root_index"],
        ),
    )
    oracle = max(
        root_records,
        key=lambda item: (
            item["best_pair_count"],
            item["immediate_count"],
            -item["root_index"],
        ),
    )
    strict = (
        oracle["root_index"] != greedy["root_index"]
        and oracle["immediate_count"] < greedy["immediate_count"]
        and oracle["best_pair_count"] > greedy["best_pair_count"]
        and oracle["best_added_gt_count"] >= 1
        and oracle["best_observation_term_count"] >= 1
    )
    best_immediate = int(greedy["immediate_count"])
    oracle_pair = int(oracle["best_pair_count"])
    gap_count = oracle_pair - int(greedy["best_pair_count"]) if strict else 0
    return {
        "task_id": task_id,
        "num_gt": len(gt_ids),
        "num_roots": len(roots),
        "distinct_root_top1": len(
            {record["top1_id"] for record in root_records if record["top1_id"]}
        ),
        "best_immediate_count": best_immediate,
        "best_immediate_recall": best_immediate / len(gt_ids),
        "oracle_pair_count": oracle_pair,
        "oracle_pair_recall": oracle_pair / len(gt_ids),
        "pair_gain_count": oracle_pair - best_immediate,
        "pair_recall_gain": (oracle_pair - best_immediate) / len(gt_ids),
        "greedy_root_index": int(greedy["root_index"]),
        "greedy_pair_count": int(greedy["best_pair_count"]),
        "oracle_root_index": int(oracle["root_index"]),
        "nonmyopic_gap_count": gap_count,
        "nonmyopic_normalized_gap": gap_count / len(gt_ids),
        "strict_opportunity": strict,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [record for record in records if record["strict_opportunity"]]
    summary = {
        "num_records": len(records),
        "complete_task_count": sum(
            int(
                record["num_gt"] >= MIN_GT_PAPERS
                and record["num_roots"] >= MIN_ROOTS
            )
            for record in records
        ),
        "diverse_root_task_count": sum(
            int(record["distinct_root_top1"] >= 3) for record in records
        ),
        "pair_gain_task_count": sum(
            int(record["pair_gain_count"] >= 1) for record in records
        ),
        "mean_best_immediate_recall": _mean(
            [float(record["best_immediate_recall"]) for record in records]
        ),
        "mean_oracle_pair_recall": _mean(
            [float(record["oracle_pair_recall"]) for record in records]
        ),
        "mean_pair_recall_gain": _mean(
            [float(record["pair_recall_gain"]) for record in records]
        ),
        "strict_opportunity_count": len(strict),
        "strict_total_gap": sum(
            int(record["nonmyopic_gap_count"]) for record in strict
        ),
        "mean_strict_normalized_gap": _mean(
            [float(record["nonmyopic_normalized_gap"]) for record in strict]
        ),
    }
    gates = {
        "all_tasks_complete": summary["complete_task_count"] == OPPORTUNITY_SIZE,
        "diverse_root_tasks_at_least_30": (
            summary["diverse_root_task_count"] >= MIN_DIVERSE_TASKS
        ),
        "pair_gain_tasks_at_least_15": (
            summary["pair_gain_task_count"] >= MIN_PAIR_GAIN_TASKS
        ),
        "mean_oracle_pair_recall_at_least_0_15": (
            summary["mean_oracle_pair_recall"] >= MIN_MEAN_ORACLE_PAIR_RECALL
        ),
        "mean_pair_recall_gain_at_least_0_03": (
            summary["mean_pair_recall_gain"] >= MIN_MEAN_PAIR_RECALL_GAIN
        ),
        "strict_opportunities_at_least_5": (
            summary["strict_opportunity_count"] >= MIN_STRICT_OPPORTUNITIES
        ),
        "strict_total_gap_at_least_5": (
            summary["strict_total_gap"] >= MIN_STRICT_TOTAL_GAP
        ),
        "mean_strict_normalized_gap_at_least_0_15": (
            summary["mean_strict_normalized_gap"]
            >= MIN_MEAN_STRICT_NORMALIZED_GAP
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def verify_gt_coverage(
    rows: Sequence[dict[str, Any]],
    corpus: ScholarCorpus,
) -> None:
    gt_ids = sorted(
        {
            paper_id
            for row in rows
            for paper_id in ground_truth_ids(row)
        }
    )
    if len(gt_ids) != EXPECTED_UNIQUE_GT_IDS:
        raise ValueError(
            f"benchmark has {len(gt_ids)} unique GT IDs, "
            f"expected {EXPECTED_UNIQUE_GT_IDS}"
        )
    missing = 0
    for start in range(0, len(gt_ids), 500):
        batch = gt_ids[start : start + 500]
        placeholders = ",".join("?" for _ in batch)
        found = corpus.connection.execute(
            f"SELECT COUNT(*) FROM papers WHERE arxiv_id IN ({placeholders})",
            batch,
        ).fetchone()[0]
        missing += len(batch) - int(found)
    if missing:
        raise ValueError(f"paper corpus is missing {missing} ground-truth IDs")


def run_audit(
    benchmark_path: Path,
    corpus_path: Path,
    index_path: Path,
) -> dict[str, Any]:
    if sha256_file(benchmark_path) != BENCHMARK_SHA256:
        raise ValueError("benchmark SHA-256 does not match frozen source")
    if sha256_file(corpus_path) != CORPUS_SHA256:
        raise ValueError("corpus SHA-256 does not match frozen source")
    rows = load_benchmark(benchmark_path)
    splits = frozen_split_ids(rows)
    build_corpus_index(corpus_path, index_path)
    by_id = {str(row["qid"]): row for row in rows}
    corpus = ScholarCorpus(index_path)
    try:
        verify_gt_coverage(rows, corpus)
        records = [
            analyze_task(by_id[task_id], corpus)
            for task_id in splits["opportunity"]
        ]
    finally:
        corpus.close()
    summary = summarize(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "gate_passed" if summary["gates"]["all_pass"] else "gate_failed",
        "source": {
            "repository_commit": SOURCE_REPO_COMMIT,
            "dataset_revision": DATASET_REVISION,
            "benchmark_sha256": BENCHMARK_SHA256,
            "corpus_sha256": CORPUS_SHA256,
        },
        "split_hashes": {
            "opportunity": OPPORTUNITY_ID_HASH,
            "development": DEVELOPMENT_ID_HASH,
            "holdout": HOLDOUT_ID_HASH,
        },
        "parameters": {
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "max_roots": MAX_ROOTS,
            "max_followups": MAX_FOLLOWUPS,
            "observation_char_cap": OBSERVATION_CHAR_CAP,
        },
        "summary": summary,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--corpus-path", type=Path, required=True)
    parser.add_argument("--index-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_audit(
        args.benchmark_path,
        args.corpus_path,
        args.index_path,
    )
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"status={payload['status']}")
    print(f"output={args.output_path}")
    return 0 if payload["status"] == "gate_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
