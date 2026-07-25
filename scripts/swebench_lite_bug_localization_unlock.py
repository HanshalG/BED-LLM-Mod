#!/usr/bin/env python3
"""Audit two-step file-retrieval opportunity on SWE-bench Lite development."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bright_biology_unlock_audit import BM25Corpus


SCHEMA_VERSION = 1
DATASET_REVISION = "69611d31007e1c6731db8bd5b5c3f2d33f5bab6e"
DEV_SHA256 = (
    "24d670403c3c8690a0f0d741bdcb8800322f58a407d6be148ad99d04b5d8bb32"
)
TEST_SHA256 = (
    "f46f2e3f003f2552932393da4b223e1e0456a2c71eba8b73ae58f29646c1278b"
)
SELECTED_IDS = (
    "sqlfluff__sqlfluff-1625",
    "sqlfluff__sqlfluff-2419",
    "sqlfluff__sqlfluff-1733",
    "sqlfluff__sqlfluff-1763",
    "pvlib__pvlib-python-1707",
    "pylint-dev__astroid-1333",
    "pyvista__pyvista-4315",
    "pydicom__pydicom-1413",
    "pydicom__pydicom-1139",
)
ROOT_TOP_K = 3
FOLLOWUP_TOP_K = 3
PATH_WEIGHT = 6
SOURCE_CHAR_CAP = 200_000
FILE_TERM_COUNT = 20
AGGREGATE_TERM_COUNT = 12
CODE_SPAN_PATTERN = re.compile(r"`([^`\n]+)`")
IDENTIFIER_PATTERN = re.compile(
    r"\b(?:[A-Za-z_][A-Za-z0-9_]*Error|"
    r"[A-Za-z_][A-Za-z0-9_]*Exception|"
    r"[A-Za-z_][A-Za-z0-9_]{3,})\b"
)
CAMEL_BOUNDARY = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def code_tokens(text: str) -> list[str]:
    expanded = CAMEL_BOUNDARY.sub(" ", text)
    return TOKEN_PATTERN.findall(expanded.casefold())


def normalized(text: str) -> str:
    return " ".join(code_tokens(text))


def target_file_from_patch(patch: str) -> str:
    files = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path not in files:
                files.append(path)
    if len(files) != 1:
        raise ValueError("SWE-bench Lite development patch is not single-file")
    return files[0]


def has_direct_target_leak(problem: str, target_file: str) -> bool:
    lowered = problem.casefold()
    basename = Path(target_file).name.casefold()
    stem = Path(target_file).stem.casefold()
    return (
        target_file.casefold() in lowered
        or basename in lowered
        or (len(stem) >= 5 and stem in lowered)
    )


def root_queries(problem: str) -> list[str]:
    lines = [" ".join(line.split()) for line in problem.splitlines() if line.strip()]
    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", problem)
        if part.strip()
    ]
    code_spans = CODE_SPAN_PATTERN.findall(problem)
    identifiers = IDENTIFIER_PATTERN.findall(problem)
    candidates = [
        problem,
        lines[0] if lines else problem,
        " ".join(code_spans),
        " ".join(identifiers),
        max(paragraphs, key=lambda part: len(code_tokens(part)), default=problem),
    ]
    roots = []
    seen = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            roots.append(" ".join(candidate.split()))
            seen.add(key)
    return roots


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
        errors="replace",
    )
    return result.stdout


def repository_documents(repo: Path, commit: str) -> list[dict[str, str]]:
    _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    paths = [
        path
        for path in _git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
        if path.endswith(".py")
        and "/.tox/" not in path
        and "/venv/" not in path
        and "/site-packages/" not in path
    ]
    documents = []
    for path in paths:
        source = _git(repo, "show", f"{commit}:{path}")[:SOURCE_CHAR_CAP]
        path_tokens = code_tokens(path)
        source_tokens = code_tokens(source)
        documents.append(
            {
                "id": path,
                "content": " ".join(
                    path_tokens * PATH_WEIGHT + source_tokens
                ),
            }
        )
    if len(documents) < 10:
        raise ValueError("base commit exposes too few Python files")
    return documents


def followup_queries(
    problem: str,
    root_documents: Sequence[dict[str, str]],
    corpus: BM25Corpus,
) -> list[str]:
    excluded = set(code_tokens(problem))
    followups = []
    for document in root_documents:
        terms = corpus.highest_idf_terms(
            [document["content"]],
            excluded_terms=excluded,
            count=FILE_TERM_COUNT,
        )
        followups.append(" ".join([problem, *terms]))
    aggregate = corpus.highest_idf_terms(
        [document["content"] for document in root_documents],
        excluded_terms=excluded,
        count=AGGREGATE_TERM_COUNT,
    )
    followups.append(" ".join([problem, *aggregate]))
    return followups


def analyze_issue(
    issue: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    target_file = target_file_from_patch(issue["patch"])
    if has_direct_target_leak(issue["problem_statement"], target_file):
        raise ValueError("selected development issue directly leaks target file")
    repo = repo_root / issue["repo"].replace("/", "__")
    documents = repository_documents(repo, issue["base_commit"])
    corpus = BM25Corpus(documents)
    if target_file not in corpus.id_to_index:
        raise ValueError("changed file is absent from base-commit Python corpus")
    roots = root_queries(issue["problem_statement"])
    if len(roots) < 3:
        raise ValueError("selected issue has fewer than three root queries")

    root_records = []
    for root_index, query in enumerate(roots):
        first = corpus.search(query, top_k=ROOT_TOP_K)
        first_ids = [row["id"] for row in first]
        immediate = int(target_file in first_ids)
        continuations = []
        for followup_index, followup in enumerate(
            followup_queries(
                issue["problem_statement"],
                first,
                corpus,
            )
        ):
            second = corpus.search(
                followup,
                top_k=FOLLOWUP_TOP_K,
                excluded_ids=first_ids,
            )
            second_ids = [row["id"] for row in second]
            pair_value = int(
                target_file in set(first_ids).union(second_ids)
            )
            continuations.append(
                {
                    "followup_index": followup_index,
                    "result_ids": second_ids,
                    "pair_value": pair_value,
                }
            )
        root_records.append(
            {
                "root_index": root_index,
                "query": query,
                "result_ids": first_ids,
                "immediate_value": immediate,
                "continuations": continuations,
                "oracle_pair_value": max(
                    row["pair_value"] for row in continuations
                ),
            }
        )

    greedy_root = max(
        range(len(root_records)),
        key=lambda index: (
            root_records[index]["immediate_value"],
            -index,
        ),
    )
    oracle_root = max(
        range(len(root_records)),
        key=lambda index: (
            root_records[index]["oracle_pair_value"],
            -index,
        ),
    )
    best_immediate = max(row["immediate_value"] for row in root_records)
    oracle_pair = root_records[oracle_root]["oracle_pair_value"]
    greedy_oracle_tail = root_records[greedy_root]["oracle_pair_value"]
    return {
        "instance_id": issue["instance_id"],
        "repo": issue["repo"],
        "base_commit": issue["base_commit"],
        "target_file": target_file,
        "num_python_files": len(documents),
        "num_roots": len(roots),
        "distinct_root_top1": len(
            {
                row["result_ids"][0]
                for row in root_records
                if row["result_ids"]
            }
        ),
        "best_immediate_value": best_immediate,
        "oracle_pair_value": oracle_pair,
        "pair_gain": oracle_pair - best_immediate,
        "greedy_root_index": greedy_root,
        "oracle_root_index": oracle_root,
        "oracle_first_differs_from_greedy": oracle_root != greedy_root,
        "greedy_oracle_tail_value": greedy_oracle_tail,
        "nonmyopic_gap": oracle_pair - greedy_oracle_tail,
        "roots": root_records,
    }


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    count = len(records)
    mean_distinct = sum(row["distinct_root_top1"] for row in records) / count
    direct_count = sum(row["best_immediate_value"] > 0 for row in records)
    pair_gain_count = sum(row["pair_gain"] > 0 for row in records)
    mean_pair_gain = sum(row["pair_gain"] for row in records) / count
    first_change_count = sum(
        row["oracle_first_differs_from_greedy"] for row in records
    )
    gap_count = sum(row["nonmyopic_gap"] > 0 for row in records)
    mean_gap = sum(row["nonmyopic_gap"] for row in records) / count
    gates = {
        "all_9_issues_complete": count == len(SELECTED_IDS),
        "all_have_at_least_3_roots": all(
            row["num_roots"] >= 3 for row in records
        ),
        "mean_distinct_root_top1_at_least_3": mean_distinct >= 3.0,
        "direct_root_coverage_at_most_6": direct_count <= 6,
        "pair_gain_count_at_least_3": pair_gain_count >= 3,
        "mean_pair_gain_at_least_0_33": mean_pair_gain >= 0.33,
        "oracle_first_differs_count_at_least_2": first_change_count >= 2,
        "nonmyopic_gap_count_at_least_2": gap_count >= 2,
        "mean_nonmyopic_gap_at_least_0_22": mean_gap >= 0.22,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_issues": count,
        "mean_distinct_root_top1": mean_distinct,
        "direct_root_coverage_count": direct_count,
        "pair_gain_count": pair_gain_count,
        "mean_pair_gain": mean_pair_gain,
        "oracle_first_differs_count": first_change_count,
        "nonmyopic_gap_count": gap_count,
        "mean_nonmyopic_gap": mean_gap,
        "gates": gates,
    }


def load_selected_dev(parquet_path: Path) -> list[dict[str, Any]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow is required to read the pinned SWE-bench parquet"
        ) from exc
    table = pq.read_table(parquet_path)
    by_id = {row["instance_id"]: row for row in table.to_pylist()}
    selected = [by_id[instance_id] for instance_id in SELECTED_IDS]
    if [row["instance_id"] for row in selected] != list(SELECTED_IDS):
        raise ValueError("selected development order does not reproduce")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dev-parquet", type=Path, required=True)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.dev_parquet) != DEV_SHA256:
        raise ValueError("development parquet hash does not match")
    if sha256_file(args.test_parquet) != TEST_SHA256:
        raise ValueError("test parquet hash does not match")
    issues = load_selected_dev(args.dev_parquet)
    records = [
        analyze_issue(issue, repo_root=args.repo_root) for issue in issues
    ]
    summary = summarize(records)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "dataset_revision": DATASET_REVISION,
            "dev_sha256": DEV_SHA256,
            "test_sha256": TEST_SHA256,
            "selected_ids": list(SELECTED_IDS),
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "path_weight": PATH_WEIGHT,
            "source_char_cap": SOURCE_CHAR_CAP,
            "api_calls": 0,
            "test_patch_columns_read": False,
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
