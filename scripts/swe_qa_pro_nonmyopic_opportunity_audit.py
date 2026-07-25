#!/usr/bin/env python3
"""Audit two-step code-search opportunity on frozen SWE-QA-Pro rows."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import random
import re
import subprocess
import sys
from typing import Any, Iterable, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bright_biology_unlock_audit import BM25Corpus, STOPWORDS


SCHEMA_VERSION = 1
SOURCE_REPO_COMMIT = "93ac6a4f3af3fe3f86580f62142f47e97e2cc897"
DATASET_REVISION = "596892dac60b6f500f01a7dc2becb9f66593b7b7"
DATASET_SHA256 = (
    "bba4aade95e707d012e11e622a40687bc0efd8cc509ed4a7dd3ba24c6e365737"
)
SELECTION_SEED = 24364
NUM_ROWS = 260
OPPORTUNITY_SIZE = 120
DEVELOPMENT_SIZE = 40
HOLDOUT_SIZE = 100
OPPORTUNITY_INDEX_HASH = (
    "ed46d42df3fd0fd2e208d477503c82e572673b158978a2170bbcd67952099b05"
)
DEVELOPMENT_INDEX_HASH = (
    "aaf9a5e1940298641cdb0ba21921cca59f47120f6189e50f5887ccad80f47663"
)
HOLDOUT_INDEX_HASH = (
    "8b30c97453fc600e9ed0f528bbf64bf1d004151a2e45963c86a768648743c608"
)
ALL_ORDER_HASH = (
    "bf8b3a3c873f181cea4023ecb19257310b56c95522a866ce8a59a7c6b2d404ef"
)

ROOT_TOP_K = 3
FOLLOWUP_TOP_K = 3
MAX_ROOTS = 24
MAX_FOLLOWUPS = 32
PATH_WEIGHT = 6
SOURCE_CHAR_CAP = 120_000
SNIPPET_LINE_RADIUS = 2
SNIPPET_LINE_CAP = 45
PER_FILE_FOLLOWUP_TERMS = 6
AGGREGATE_FOLLOWUP_TERMS = 10
MIN_USABLE_TASKS = 72
MAX_IMMEDIATE_SATURATION_RATE = 0.70
MIN_PAIR_GAIN_TASKS = 18
MIN_STRICT_OPPORTUNITIES = 12
MIN_STRICT_WITH_THREE_EVIDENCE = 8
MIN_STRICT_REPOS = 8
MIN_MEAN_STRICT_NORMALIZED_GAP = 0.15

TEXT_SUFFIXES = frozenset(
    {
        ".c",
        ".cfg",
        ".cpp",
        ".css",
        ".h",
        ".hpp",
        ".ini",
        ".js",
        ".json",
        ".md",
        ".py",
        ".pyi",
        ".rst",
        ".rs",
        ".sh",
        ".toml",
        ".ts",
        ".tsx",
        ".txt",
        ".yaml",
        ".yml",
    }
)
SKIP_PATH_PARTS = frozenset(
    {
        ".git",
        ".tox",
        ".venv",
        "build",
        "dist",
        "node_modules",
        "site-packages",
        "vendor",
        "venv",
    }
)
GENERIC_BASENAMES = frozenset(
    {
        "__init__.py",
        "changelog.md",
        "conftest.py",
        "index.md",
        "license.txt",
        "readme.md",
        "setup.py",
    }
)
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN_PATTERN = re.compile(r"[A-Za-z][A-Za-z0-9]*")
CODE_SPAN_PATTERN = re.compile(r"`([^`\n]{2,160})`")
QUOTED_SPAN_PATTERN = re.compile(r"""["']([^"'\n]{3,120})["']""")
SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _index_hash(indices: Sequence[int]) -> str:
    return hashlib.sha256(
        "\n".join(str(index) for index in indices).encode("utf-8")
    ).hexdigest()


def frozen_split_indices() -> dict[str, list[int]]:
    order = list(range(NUM_ROWS))
    random.Random(SELECTION_SEED).shuffle(order)
    split = {
        "opportunity": order[:OPPORTUNITY_SIZE],
        "development": order[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": order[OPPORTUNITY_SIZE + DEVELOPMENT_SIZE :],
    }
    expected = {
        "opportunity": OPPORTUNITY_INDEX_HASH,
        "development": DEVELOPMENT_INDEX_HASH,
        "holdout": HOLDOUT_INDEX_HASH,
    }
    for name, indices in split.items():
        if _index_hash(indices) != expected[name]:
            raise AssertionError(f"{name} index hash does not reproduce")
    if _index_hash(order) != ALL_ORDER_HASH:
        raise AssertionError("full shuffled order hash does not reproduce")
    return split


def code_tokens(text: str) -> list[str]:
    expanded = CAMEL_BOUNDARY.sub(" ", text.replace("_", " "))
    return TOKEN_PATTERN.findall(expanded.casefold())


def normalized(text: str) -> str:
    return " ".join(code_tokens(text))


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
        errors="replace",
    )
    if result.returncode != 0:
        raise subprocess.CalledProcessError(
            result.returncode,
            result.args,
            output=result.stdout,
            stderr=result.stderr,
        )
    return result.stdout


def repository_documents(
    repo: Path,
    expected_commit: str,
) -> list[dict[str, str]]:
    head = _git(repo, "rev-parse", "HEAD").strip()
    if head != expected_commit:
        raise ValueError(
            f"{repo.name} is at {head}, expected {expected_commit}"
        )
    paths = _git(repo, "ls-files").splitlines()
    documents: list[dict[str, str]] = []
    for relative in paths:
        path = Path(relative)
        if (
            path.suffix.casefold() not in TEXT_SUFFIXES
            or any(part.casefold() in SKIP_PATH_PARTS for part in path.parts)
        ):
            continue
        full_path = repo / path
        try:
            if not full_path.is_file() or full_path.stat().st_size > 512_000:
                continue
            source = full_path.read_text(
                encoding="utf-8", errors="ignore"
            )[:SOURCE_CHAR_CAP]
        except OSError:
            continue
        path_text = " ".join(code_tokens(relative) * PATH_WEIGHT)
        source_text = " ".join(code_tokens(source))
        documents.append(
            {
                "id": relative,
                "content": f"{path_text} {source_text}",
                "raw_source": source,
            }
        )
    if len(documents) < 10:
        raise ValueError(f"{repo.name} exposes too few text files")
    return documents


def gold_evidence_paths(
    answer: str,
    repository_paths: Sequence[str],
) -> list[str]:
    """Recover files explicitly cited by a released reference answer."""
    answer_folded = answer.replace("\\", "/").casefold()
    paths = sorted(set(repository_paths), key=lambda path: (-len(path), path))
    evidence: list[str] = []
    for path in paths:
        normalized_path = path.replace("\\", "/").casefold()
        if normalized_path in answer_folded:
            evidence.append(path)

    basename_counts = Counter(Path(path).name.casefold() for path in paths)
    already = set(evidence)
    for path in paths:
        basename = Path(path).name.casefold()
        stem = Path(path).stem
        if (
            path in already
            or basename_counts[basename] != 1
            or basename in GENERIC_BASENAMES
            or len(stem) < 5
        ):
            continue
        if re.search(
            rf"(?<![A-Za-z0-9_.-]){re.escape(basename)}"
            rf"(?![A-Za-z0-9_.-])",
            answer_folded,
        ):
            evidence.append(path)
    return sorted(set(evidence))


def _query_idf(corpus: BM25Corpus, token: str) -> float:
    return float(corpus.index.idf.get(token, -100.0))


def root_queries(question: str, corpus: BM25Corpus) -> list[str]:
    sentences = [
        " ".join(sentence.split())
        for sentence in SENTENCE_PATTERN.split(question)
        if sentence.strip()
    ]
    code_spans = CODE_SPAN_PATTERN.findall(question)
    quoted_spans = QUOTED_SPAN_PATTERN.findall(question)
    question_tokens = code_tokens(question)
    content_tokens = [
        token
        for token in question_tokens
        if len(token) >= 3
        and token not in STOPWORDS
        and token in corpus.index.idf
    ]
    ranked_tokens = sorted(
        set(content_tokens),
        key=lambda token: (-_query_idf(corpus, token), token),
    )
    candidates: list[str] = [
        question,
        *sentences,
        *code_spans,
        *quoted_spans,
        " ".join(ranked_tokens[:12]),
        *ranked_tokens[:8],
    ]
    rare = set(ranked_tokens[:8])
    for width in (2, 3, 4):
        for start in range(max(0, len(question_tokens) - width + 1)):
            window = question_tokens[start : start + width]
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
    """Return deterministic path plus matching source windows."""
    lines = document["raw_source"].splitlines()
    query_terms = set(code_tokens(query))
    scored: list[tuple[float, int]] = []
    for index, line in enumerate(lines):
        terms = set(code_tokens(line))
        overlap = query_terms.intersection(terms)
        if overlap:
            score = sum(max(0.0, _query_idf(corpus, term)) for term in overlap)
            scored.append((score, index))
    if scored:
        centers = [
            index
            for _, index in sorted(scored, key=lambda row: (-row[0], row[1]))
        ]
    else:
        centers = list(range(min(10, len(lines))))
    selected: set[int] = set()
    for center in centers:
        selected.update(
            range(
                max(0, center - SNIPPET_LINE_RADIUS),
                min(len(lines), center + SNIPPET_LINE_RADIUS + 1),
            )
        )
        if len(selected) >= SNIPPET_LINE_CAP:
            break
    ordered = sorted(selected)[:SNIPPET_LINE_CAP]
    snippets = "\n".join(lines[index] for index in ordered)
    return f"{document['id']}\n{snippets}"


def followup_queries(
    question: str,
    root_query: str,
    root_documents: Sequence[dict[str, str]],
    corpus: BM25Corpus,
) -> list[str]:
    observations = [
        query_observation(document, root_query, corpus)
        for document in root_documents
    ]
    excluded = set(code_tokens(question)).union(code_tokens(root_query))
    term_groups = [
        corpus.highest_idf_terms(
            [observation],
            excluded_terms=excluded,
            count=PER_FILE_FOLLOWUP_TERMS,
        )
        for observation in observations
    ]
    aggregate = corpus.highest_idf_terms(
        observations,
        excluded_terms=excluded,
        count=AGGREGATE_FOLLOWUP_TERMS,
    )
    candidates: list[str] = []
    for terms in [*term_groups, aggregate]:
        for term in terms:
            candidates.append(f"{question} {term}")
            candidates.append(f"{root_query} {term}")
        if terms:
            candidates.append(f"{question} {' '.join(terms)}")

    followups: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            followups.append(candidate)
            seen.add(key)
        if len(followups) >= MAX_FOLLOWUPS:
            break
    if not followups:
        followups.append(question)
    return followups


def _coverage(ids: Iterable[str], evidence: set[str]) -> int:
    return len(set(ids).intersection(evidence))


def analyze_task(
    task: dict[str, Any],
    *,
    row_index: int,
    documents: Sequence[dict[str, str]],
    corpus: BM25Corpus,
) -> dict[str, Any]:
    repository_paths = [document["id"] for document in documents]
    evidence = gold_evidence_paths(task["answer"], repository_paths)
    roots = root_queries(task["question"], corpus)
    task_id = f"{task['repo']}:{row_index}"
    if len(evidence) < 2 or len(roots) < 3:
        return {
            "task_id": task_id,
            "row_index": row_index,
            "repo": task["repo"],
            "commit_id": task["commit_id"],
            "question_sha256": hashlib.sha256(
                task["question"].encode("utf-8")
            ).hexdigest(),
            "num_evidence_files": len(evidence),
            "evidence_files": evidence,
            "num_roots": len(roots),
            "usable": False,
            "reason": (
                "fewer_than_two_evidence_files"
                if len(evidence) < 2
                else "fewer_than_three_roots"
            ),
        }

    evidence_set = set(evidence)
    root_records: list[dict[str, Any]] = []
    for root_index, query in enumerate(roots):
        first = corpus.search(query, top_k=ROOT_TOP_K)
        first_ids = [row["id"] for row in first]
        immediate = _coverage(first_ids, evidence_set)
        continuation_records: list[dict[str, Any]] = []
        for followup_index, followup in enumerate(
            followup_queries(task["question"], query, first, corpus)
        ):
            second = corpus.search(
                followup,
                top_k=FOLLOWUP_TOP_K,
                excluded_ids=first_ids,
            )
            second_ids = [row["id"] for row in second]
            pair_count = _coverage(
                set(first_ids).union(second_ids), evidence_set
            )
            continuation_records.append(
                {
                    "followup_index": followup_index,
                    "query": followup,
                    "result_ids": second_ids,
                    "pair_evidence_count": pair_count,
                }
            )
        best_followup = max(
            range(len(continuation_records)),
            key=lambda index: (
                continuation_records[index]["pair_evidence_count"],
                -index,
            ),
        )
        root_records.append(
            {
                "root_index": root_index,
                "query": query,
                "result_ids": first_ids,
                "immediate_evidence_count": immediate,
                "best_followup_index": best_followup,
                "oracle_pair_evidence_count": continuation_records[
                    best_followup
                ]["pair_evidence_count"],
                "continuations": continuation_records,
            }
        )

    greedy_root = max(
        range(len(root_records)),
        key=lambda index: (
            root_records[index]["immediate_evidence_count"],
            root_records[index]["oracle_pair_evidence_count"],
            -index,
        ),
    )
    oracle_root = max(
        range(len(root_records)),
        key=lambda index: (
            root_records[index]["oracle_pair_evidence_count"],
            root_records[index]["immediate_evidence_count"],
            -index,
        ),
    )
    greedy = root_records[greedy_root]
    oracle = root_records[oracle_root]
    best_immediate = max(
        row["immediate_evidence_count"] for row in root_records
    )
    oracle_pair = oracle["oracle_pair_evidence_count"]
    greedy_pair = greedy["oracle_pair_evidence_count"]
    strict = (
        oracle_root != greedy_root
        and oracle["immediate_evidence_count"]
        < greedy["immediate_evidence_count"]
        and oracle_pair > greedy_pair
        and oracle_pair > oracle["immediate_evidence_count"]
    )
    return {
        "task_id": task_id,
        "row_index": row_index,
        "repo": task["repo"],
        "commit_id": task["commit_id"],
        "question_sha256": hashlib.sha256(
            task["question"].encode("utf-8")
        ).hexdigest(),
        "num_evidence_files": len(evidence),
        "evidence_files": evidence,
        "num_roots": len(roots),
        "usable": True,
        "distinct_root_top1": len(
            {
                row["result_ids"][0]
                for row in root_records
                if row["result_ids"]
            }
        ),
        "best_immediate_evidence_count": best_immediate,
        "best_immediate_coverage": best_immediate / len(evidence),
        "oracle_pair_evidence_count": oracle_pair,
        "oracle_pair_coverage": oracle_pair / len(evidence),
        "pair_gain_count": oracle_pair - best_immediate,
        "greedy_root_index": greedy_root,
        "greedy_oracle_tail_evidence_count": greedy_pair,
        "oracle_root_index": oracle_root,
        "oracle_root_immediate_evidence_count": oracle[
            "immediate_evidence_count"
        ],
        "nonmyopic_gap_count": oracle_pair - greedy_pair,
        "nonmyopic_normalized_gap": (oracle_pair - greedy_pair)
        / len(evidence),
        "strict_opportunity": strict,
        "roots": root_records,
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    usable = [record for record in records if record["usable"]]
    strict = [record for record in usable if record["strict_opportunity"]]
    pair_gain = [
        record for record in usable if record["pair_gain_count"] > 0
    ]
    immediate_saturated = [
        record
        for record in usable
        if record["best_immediate_evidence_count"]
        == record["num_evidence_files"]
    ]
    strict_with_three = [
        record for record in strict if record["num_evidence_files"] >= 3
    ]
    strict_repos = {record["repo"] for record in strict}
    mean_strict_gap = (
        sum(record["nonmyopic_normalized_gap"] for record in strict)
        / len(strict)
        if strict
        else 0.0
    )
    mean_gap = (
        sum(record["nonmyopic_normalized_gap"] for record in usable)
        / len(usable)
        if usable
        else 0.0
    )
    saturation_rate = (
        len(immediate_saturated) / len(usable) if usable else 1.0
    )
    gates = {
        "all_120_opportunity_rows_complete": len(records) == OPPORTUNITY_SIZE,
        "usable_tasks_at_least_72": len(usable) >= MIN_USABLE_TASKS,
        "all_usable_have_at_least_3_roots": all(
            record["num_roots"] >= 3 for record in usable
        ),
        "immediate_saturation_rate_at_most_0_70": (
            saturation_rate <= MAX_IMMEDIATE_SATURATION_RATE
        ),
        "pair_gain_tasks_at_least_18": (
            len(pair_gain) >= MIN_PAIR_GAIN_TASKS
        ),
        "strict_opportunities_at_least_12": (
            len(strict) >= MIN_STRICT_OPPORTUNITIES
        ),
        "strict_with_3_evidence_at_least_8": (
            len(strict_with_three) >= MIN_STRICT_WITH_THREE_EVIDENCE
        ),
        "strict_repositories_at_least_8": (
            len(strict_repos) >= MIN_STRICT_REPOS
        ),
        "mean_strict_normalized_gap_at_least_0_15": (
            mean_strict_gap >= MIN_MEAN_STRICT_NORMALIZED_GAP
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_records": len(records),
        "num_usable": len(usable),
        "num_unusable": len(records) - len(usable),
        "mean_evidence_files_usable": (
            sum(record["num_evidence_files"] for record in usable)
            / len(usable)
            if usable
            else 0.0
        ),
        "mean_distinct_root_top1": (
            sum(record["distinct_root_top1"] for record in usable)
            / len(usable)
            if usable
            else 0.0
        ),
        "immediate_saturation_count": len(immediate_saturated),
        "immediate_saturation_rate": saturation_rate,
        "pair_gain_count": len(pair_gain),
        "strict_opportunity_count": len(strict),
        "strict_opportunity_rate_usable": (
            len(strict) / len(usable) if usable else 0.0
        ),
        "strict_with_3_evidence_count": len(strict_with_three),
        "strict_repository_count": len(strict_repos),
        "strict_repositories": sorted(strict_repos),
        "mean_nonmyopic_normalized_gap_usable": mean_gap,
        "mean_strict_normalized_gap": mean_strict_gap,
        "gates": gates,
    }


def load_opportunity_rows(dataset_path: Path) -> list[tuple[int, dict[str, Any]]]:
    opportunity = set(frozen_split_indices()["opportunity"])
    selected: dict[int, dict[str, Any]] = {}
    row_count = 0
    with dataset_path.open(encoding="utf-8") as handle:
        for row_index, line in enumerate(handle):
            row_count += 1
            if row_index in opportunity:
                selected[row_index] = json.loads(line)
    if row_count != NUM_ROWS:
        raise ValueError(f"dataset has {row_count} rows, expected {NUM_ROWS}")
    return [
        (row_index, selected[row_index])
        for row_index in frozen_split_indices()["opportunity"]
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.dataset) != DATASET_SHA256:
        raise ValueError("dataset hash does not match the frozen release")
    selected = load_opportunity_rows(args.dataset)
    by_repo: dict[str, list[tuple[int, dict[str, Any]]]] = defaultdict(list)
    for row_index, task in selected:
        by_repo[task["repo"]].append((row_index, task))

    records_by_index: dict[int, dict[str, Any]] = {}
    repository_stats: dict[str, Any] = {}
    for repo_name in sorted(by_repo):
        repo_path = args.repo_root / repo_name.rsplit("/", 1)[-1]
        expected_commits = {
            task["commit_id"] for _, task in by_repo[repo_name]
        }
        if len(expected_commits) != 1:
            raise ValueError(f"{repo_name} has multiple benchmark commits")
        expected_commit = next(iter(expected_commits))
        documents = repository_documents(repo_path, expected_commit)
        corpus = BM25Corpus(documents)
        repository_stats[repo_name] = {
            "commit_id": expected_commit,
            "num_documents": len(documents),
        }
        for row_index, task in by_repo[repo_name]:
            records_by_index[row_index] = analyze_task(
                task,
                row_index=row_index,
                documents=documents,
                corpus=corpus,
            )
        print(
            f"{repo_name}: {len(by_repo[repo_name])} tasks, "
            f"{len(documents)} documents",
            flush=True,
        )

    ordered_indices = frozen_split_indices()["opportunity"]
    records = [records_by_index[index] for index in ordered_indices]
    summary = summarize(records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "source_repo_commit": SOURCE_REPO_COMMIT,
            "dataset_revision": DATASET_REVISION,
            "dataset_sha256": DATASET_SHA256,
            "selection_seed": SELECTION_SEED,
            "opportunity_index_hash": OPPORTUNITY_INDEX_HASH,
            "development_index_hash": DEVELOPMENT_INDEX_HASH,
            "holdout_index_hash": HOLDOUT_INDEX_HASH,
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "max_roots": MAX_ROOTS,
            "max_followups": MAX_FOLLOWUPS,
            "path_weight": PATH_WEIGHT,
            "source_char_cap": SOURCE_CHAR_CAP,
            "api_calls": 0,
            "development_rows_used": 0,
            "holdout_rows_used": 0,
        },
        "repository_stats": repository_stats,
        "summary": summary,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
