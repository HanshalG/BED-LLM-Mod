#!/usr/bin/env python3
"""Audit two-search hidden-keyword opportunity on frozen DR3-Eval tasks."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bright_biology_unlock_audit import BM25Corpus, STOPWORDS, tokenize


SCHEMA_VERSION = 1
SOURCE_REPO_COMMIT = "86fed3760a8708d48121c4e9eaf0fddc939c6bef"
DATASET_REVISION = "4305f9129529d4510f485af6c997b69e1e85b88d"
QUERY_SHA256 = "52aa6a3ff3ca03f4da962d32a78785d3219aa921e2132d15196960a49afca203"
CONTEXTS_SHA256 = (
    "2011d08d61530690ab7d4980612bd71ea81a9c31eae649df7a11124d6f845a82"
)
SELECTION_SEED = 24691
ELIGIBLE_IDS = tuple(f"{index:03d}" for index in range(13, 51))
MECHANICS_IDS = tuple(f"{index:03d}" for index in range(1, 13))
OPPORTUNITY_SIZE = 20
DEVELOPMENT_SIZE = 8
HOLDOUT_SIZE = 10
OPPORTUNITY_ID_HASH = (
    "c88bed3af2d6cfd5380b1f0907e5a6095304c586441ee942f596b74ef2843cb1"
)
DEVELOPMENT_ID_HASH = (
    "3869d06cb649db66ec7c3182550ee374209cd96111698f7171cf78a3fdb08129"
)
HOLDOUT_ID_HASH = (
    "37f955d04b37582d7acda03e8d235ce11779876b36744d9b0a036b302ebf8185"
)
ALL_ORDER_HASH = (
    "ee884c9d690598bc525b3229855dc0b4984b441218a19aab27051dbfc82fb318"
)

ROOT_TOP_K = 3
FOLLOWUP_TOP_K = 3
MAX_ROOTS = 20
MAX_FOLLOWUPS = 24
OBSERVATION_SEGMENTS = 5
OBSERVATION_CHAR_CAP = 2_000
PER_PAGE_FOLLOWUP_TERMS = 5
AGGREGATE_FOLLOWUP_TERMS = 10

MIN_PAGES = 25
MIN_KEYWORDS = 9
MIN_ROOTS = 5
MIN_DIVERSE_TASKS = 15
MIN_PAIR_GAIN_TASKS = 10
MIN_MEAN_ORACLE_PAIR_COVERAGE = 0.30
MIN_MEAN_PAIR_COVERAGE_GAIN = 0.08
MIN_STRICT_OPPORTUNITIES = 4
MIN_STRICT_TOTAL_GAP = 4
MIN_MEAN_STRICT_NORMALIZED_GAP = 0.10

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_PATTERN = re.compile(r"[,;:]|\s+[—-]\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_hash(ids: Sequence[str]) -> str:
    return hashlib.sha256("\n".join(ids).encode("utf-8")).hexdigest()


def frozen_split_ids() -> dict[str, list[str]]:
    order = list(ELIGIBLE_IDS)
    random.Random(SELECTION_SEED).shuffle(order)
    split = {
        "opportunity": order[:OPPORTUNITY_SIZE],
        "development": order[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": order[OPPORTUNITY_SIZE + DEVELOPMENT_SIZE :],
    }
    expected = {
        "opportunity": OPPORTUNITY_ID_HASH,
        "development": DEVELOPMENT_ID_HASH,
        "holdout": HOLDOUT_ID_HASH,
    }
    for name, ids in split.items():
        if _id_hash(ids) != expected[name]:
            raise AssertionError(f"{name} ID hash does not reproduce")
    if _id_hash(order) != ALL_ORDER_HASH:
        raise AssertionError("complete shuffled-order hash does not reproduce")
    return split


def combined_context_hash(context_root: Path) -> str:
    digest = hashlib.sha256()
    for task_index in range(1, 51):
        path = context_root / f"{task_index:03d}.json"
        payload = path.read_bytes()
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def normalized(text: str) -> str:
    return " ".join(tokenize(text))


def _query_idf(corpus: BM25Corpus, token: str) -> float:
    return float(corpus.index.idf.get(token, -100.0))


def load_queries(query_path: Path, allowed_ids: set[str]) -> dict[str, str]:
    selected: dict[str, str] = {}
    row_count = 0
    with query_path.open(encoding="utf-8") as handle:
        for line in handle:
            row_count += 1
            raw = json.loads(line)
            task_id = str(raw.get("task", "")).zfill(3)
            if task_id in allowed_ids:
                selected[task_id] = str(raw["query"])
    if row_count != 50:
        raise ValueError(f"query file has {row_count} rows, expected 50")
    if set(selected) != allowed_ids:
        raise ValueError("query file does not contain every selected task ID")
    return selected


def load_task_corpus(path: Path, task_id: str) -> tuple[list[dict[str, str]], BM25Corpus]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"{task_id} context is not a list")
    documents: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"{task_id} row {index} is not an object")
        required = ("keyword", "title", "url", "page_body")
        if any(not str(row.get(field, "")).strip() for field in required):
            raise ValueError(f"{task_id} row {index} has an empty required field")
        title = str(row["title"]).strip()
        body = str(row["page_body"]).strip()
        documents.append(
            {
                "id": f"{index:03d}:{row['url']}",
                "content": f"{title}\n{body}",
                "raw_source": body,
                "title": title,
                "keyword": str(row["keyword"]).strip(),
            }
        )
    return documents, BM25Corpus(documents)


def root_queries(query: str, corpus: BM25Corpus) -> list[str]:
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
    ranked_terms = sorted(
        {
            token
            for token in query_tokens
            if len(token) >= 3
            and token not in STOPWORDS
            and token in corpus.index.idf
        },
        key=lambda token: (-_query_idf(corpus, token), token),
    )
    rare = set(ranked_terms[:10])
    candidates = [
        query,
        *sentences,
        *clauses,
        " ".join(ranked_terms[:12]),
        *ranked_terms[:8],
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


def query_observation(
    document: dict[str, str],
    query: str,
    corpus: BM25Corpus,
) -> str:
    """Expose visible title/body snippets without the hidden keyword."""
    segments = [
        " ".join(segment.split())
        for segment in SENTENCE_PATTERN.split(document["raw_source"])
        if segment.strip()
    ]
    query_terms = set(tokenize(query))
    scored: list[tuple[float, int, str]] = []
    for index, segment in enumerate(segments):
        overlap = query_terms.intersection(tokenize(segment))
        score = sum(max(0.0, _query_idf(corpus, term)) for term in overlap)
        scored.append((score, index, segment))
    chosen = sorted(scored, key=lambda row: (-row[0], row[1]))[
        :OBSERVATION_SEGMENTS
    ]
    ordered = [segment for _, _, segment in sorted(chosen, key=lambda row: row[1])]
    visible = f"{document['title']}\n" + "\n".join(ordered)
    return visible[:OBSERVATION_CHAR_CAP]


def followup_queries(
    initial_query: str,
    root_query: str,
    root_documents: Sequence[dict[str, str]],
    corpus: BM25Corpus,
) -> list[str]:
    observations = [
        query_observation(document, root_query, corpus)
        for document in root_documents
    ]
    excluded = set(tokenize(initial_query)).union(tokenize(root_query))
    groups = [
        corpus.highest_idf_terms(
            [observation],
            excluded_terms=excluded,
            count=PER_PAGE_FOLLOWUP_TERMS,
        )
        for observation in observations
    ]
    aggregate = corpus.highest_idf_terms(
        observations,
        excluded_terms=excluded,
        count=AGGREGATE_FOLLOWUP_TERMS,
    )
    candidates: list[str] = []
    for terms in [*groups, aggregate]:
        for term in terms:
            candidates.append(f"{root_query} {term}")
            candidates.append(f"{initial_query} {term}")
        for start in range(max(0, len(terms) - 1)):
            candidates.append(f"{terms[start]} {terms[start + 1]}")
        if terms:
            candidates.append(f"{root_query} {' '.join(terms)}")

    followups: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            followups.append(" ".join(candidate.split()))
            seen.add(key)
        if len(followups) >= MAX_FOLLOWUPS:
            break
    return followups or [root_query]


def _keyword_coverage(
    document_ids: Iterable[str],
    keyword_by_id: dict[str, str],
) -> int:
    return len(
        {
            keyword_by_id[document_id]
            for document_id in set(document_ids)
            if document_id in keyword_by_id
        }
    )


def analyze_task(
    task_id: str,
    query: str,
    documents: Sequence[dict[str, str]],
    corpus: BM25Corpus,
) -> dict[str, Any]:
    keyword_by_id = {
        document["id"]: document["keyword"] for document in documents
    }
    num_keywords = len(set(keyword_by_id.values()))
    roots = root_queries(query, corpus)
    root_records: list[dict[str, Any]] = []
    for root_index, root_query in enumerate(roots):
        first = corpus.search(root_query, top_k=ROOT_TOP_K)
        first_ids = [document["id"] for document in first]
        immediate = _keyword_coverage(first_ids, keyword_by_id)
        continuations: list[dict[str, Any]] = []
        for followup_index, followup in enumerate(
            followup_queries(query, root_query, first, corpus)
        ):
            second = corpus.search(
                followup,
                top_k=FOLLOWUP_TOP_K,
                excluded_ids=first_ids,
            )
            second_ids = [document["id"] for document in second]
            pair = _keyword_coverage(
                [*first_ids, *second_ids],
                keyword_by_id,
            )
            continuations.append(
                {
                    "followup_index": followup_index,
                    "query": followup,
                    "result_ids": second_ids,
                    "pair_keyword_count": pair,
                }
            )
        best_followup_index = max(
            range(len(continuations)),
            key=lambda index: (
                continuations[index]["pair_keyword_count"],
                -index,
            ),
        )
        best_followup = continuations[best_followup_index]
        root_records.append(
            {
                "root_index": root_index,
                "query": root_query,
                "result_ids": first_ids,
                "immediate_keyword_count": immediate,
                "best_followup_index": best_followup_index,
                "oracle_pair_keyword_count": best_followup[
                    "pair_keyword_count"
                ],
                "best_followup": best_followup,
            }
        )

    greedy_root_index = max(
        range(len(root_records)),
        key=lambda index: (
            root_records[index]["immediate_keyword_count"],
            root_records[index]["oracle_pair_keyword_count"],
            -index,
        ),
    )
    oracle_root_index = max(
        range(len(root_records)),
        key=lambda index: (
            root_records[index]["oracle_pair_keyword_count"],
            root_records[index]["immediate_keyword_count"],
            -index,
        ),
    )
    greedy = root_records[greedy_root_index]
    oracle = root_records[oracle_root_index]
    best_immediate = greedy["immediate_keyword_count"]
    oracle_pair = oracle["oracle_pair_keyword_count"]
    greedy_pair = greedy["oracle_pair_keyword_count"]
    strict = (
        oracle_root_index != greedy_root_index
        and oracle["immediate_keyword_count"] < best_immediate
        and oracle_pair > greedy_pair
        and oracle_pair > oracle["immediate_keyword_count"]
    )
    return {
        "task_id": task_id,
        "query_sha256": hashlib.sha256(query.encode("utf-8")).hexdigest(),
        "num_pages": len(documents),
        "num_keywords": num_keywords,
        "num_roots": len(roots),
        "distinct_root_top1": len(
            {
                root["result_ids"][0]
                for root in root_records
                if root["result_ids"]
            }
        ),
        "best_immediate_keyword_count": best_immediate,
        "best_immediate_coverage": best_immediate / num_keywords,
        "oracle_pair_keyword_count": oracle_pair,
        "oracle_pair_coverage": oracle_pair / num_keywords,
        "pair_gain_count": oracle_pair - best_immediate,
        "pair_coverage_gain": (oracle_pair - best_immediate) / num_keywords,
        "greedy_root_index": greedy_root_index,
        "greedy_oracle_tail_keyword_count": greedy_pair,
        "oracle_root_index": oracle_root_index,
        "oracle_root_immediate_keyword_count": oracle[
            "immediate_keyword_count"
        ],
        "nonmyopic_gap_count": oracle_pair - greedy_pair,
        "nonmyopic_normalized_gap": (oracle_pair - greedy_pair) / num_keywords,
        "strict_opportunity": strict,
        "roots": root_records,
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [record for record in records if record["strict_opportunity"]]
    pair_gain = [record for record in records if record["pair_gain_count"] > 0]
    diverse = [
        record for record in records if record["distinct_root_top1"] >= 3
    ]
    mean_pair_coverage = (
        sum(record["oracle_pair_coverage"] for record in records) / len(records)
        if records
        else 0.0
    )
    mean_pair_gain = (
        sum(record["pair_coverage_gain"] for record in records) / len(records)
        if records
        else 0.0
    )
    strict_total_gap = sum(
        record["nonmyopic_gap_count"] for record in strict
    )
    mean_strict_gap = (
        sum(record["nonmyopic_normalized_gap"] for record in strict)
        / len(strict)
        if strict
        else 0.0
    )
    gates = {
        "all_20_opportunity_tasks_complete": len(records) == OPPORTUNITY_SIZE,
        "all_tasks_have_minimum_schema": all(
            record["num_pages"] >= MIN_PAGES
            and record["num_keywords"] >= MIN_KEYWORDS
            and record["num_roots"] >= MIN_ROOTS
            for record in records
        ),
        "diverse_root_tasks_at_least_15": len(diverse) >= MIN_DIVERSE_TASKS,
        "pair_gain_tasks_at_least_10": len(pair_gain) >= MIN_PAIR_GAIN_TASKS,
        "mean_oracle_pair_coverage_at_least_0_30": (
            mean_pair_coverage >= MIN_MEAN_ORACLE_PAIR_COVERAGE
        ),
        "mean_pair_coverage_gain_at_least_0_08": (
            mean_pair_gain >= MIN_MEAN_PAIR_COVERAGE_GAIN
        ),
        "strict_opportunities_at_least_4": (
            len(strict) >= MIN_STRICT_OPPORTUNITIES
        ),
        "strict_total_gap_at_least_4": (
            strict_total_gap >= MIN_STRICT_TOTAL_GAP
        ),
        "mean_strict_normalized_gap_at_least_0_10": (
            mean_strict_gap >= MIN_MEAN_STRICT_NORMALIZED_GAP
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_records": len(records),
        "diverse_root_task_count": len(diverse),
        "pair_gain_task_count": len(pair_gain),
        "strict_opportunity_count": len(strict),
        "strict_task_ids": [record["task_id"] for record in strict],
        "strict_total_gap": strict_total_gap,
        "mean_best_immediate_coverage": (
            sum(record["best_immediate_coverage"] for record in records)
            / len(records)
            if records
            else 0.0
        ),
        "mean_oracle_pair_coverage": mean_pair_coverage,
        "mean_pair_coverage_gain": mean_pair_gain,
        "mean_strict_normalized_gap": mean_strict_gap,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query-file", type=Path, required=True)
    parser.add_argument("--context-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.query_file) != QUERY_SHA256:
        raise ValueError("query file hash does not match frozen release")
    if combined_context_hash(args.context_root) != CONTEXTS_SHA256:
        raise ValueError("context files do not match frozen release")

    opportunity_ids = frozen_split_ids()["opportunity"]
    queries = load_queries(args.query_file, set(opportunity_ids))
    records: list[dict[str, Any]] = []
    for task_id in opportunity_ids:
        documents, corpus = load_task_corpus(
            args.context_root / f"{task_id}.json",
            task_id,
        )
        record = analyze_task(task_id, queries[task_id], documents, corpus)
        records.append(record)
        print(
            f"{task_id}: pages={record['num_pages']} "
            f"keywords={record['num_keywords']} "
            f"strict={record['strict_opportunity']}",
            flush=True,
        )

    summary = summarize(records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "source_repo_commit": SOURCE_REPO_COMMIT,
            "dataset_revision": DATASET_REVISION,
            "query_sha256": QUERY_SHA256,
            "contexts_sha256": CONTEXTS_SHA256,
            "selection_seed": SELECTION_SEED,
            "mechanics_ids": list(MECHANICS_IDS),
            "opportunity_id_hash": OPPORTUNITY_ID_HASH,
            "development_id_hash": DEVELOPMENT_ID_HASH,
            "holdout_id_hash": HOLDOUT_ID_HASH,
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "max_roots": MAX_ROOTS,
            "max_followups": MAX_FOLLOWUPS,
            "api_calls": 0,
            "development_tasks_used": 0,
            "holdout_tasks_used": 0,
        },
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
