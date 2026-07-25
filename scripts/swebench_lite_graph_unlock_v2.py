#!/usr/bin/env python3
"""Audit graph-conditioned file-retrieval opportunity on SWE-bench Lite test."""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import re
import subprocess
from typing import Any, Iterable, Sequence


SCHEMA_VERSION = 1
DATASET_REVISION = "69611d31007e1c6731db8bd5b5c3f2d33f5bab6e"
TEST_SHA256 = (
    "f46f2e3f003f2552932393da4b223e1e0456a2c71eba8b73ae58f29646c1278b"
)
SELECTION_SEED = 24_397
EXPECTED_REPOSITORIES = 12
OPPORTUNITY_PER_REPOSITORY = 2
DEVELOPMENT_PER_REPOSITORY = 1
EXPECTED_OPPORTUNITY = 24
EXPECTED_DEVELOPMENT = 12
EXPECTED_HOLDOUT = 264
ROOT_TOP_K = 3
FOLLOWUP_TOP_K = 3
PATH_WEIGHT = 6
SOURCE_CHAR_CAP = 200_000
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


def hash_ids(ids: Sequence[str]) -> str:
    payload = json.dumps(list(ids), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("ascii")).hexdigest()


def code_tokens(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(CAMEL_BOUNDARY.sub(" ", text).casefold())


def normalized(text: str) -> str:
    return " ".join(code_tokens(text))


def root_queries(problem: str) -> list[str]:
    lines = [" ".join(line.split()) for line in problem.splitlines() if line.strip()]
    paragraphs = [
        " ".join(part.split())
        for part in re.split(r"\n\s*\n", problem)
        if part.strip()
    ]
    candidates = [
        problem,
        lines[0] if lines else problem,
        " ".join(CODE_SPAN_PATTERN.findall(problem)),
        " ".join(IDENTIFIER_PATTERN.findall(problem)),
        max(
            paragraphs,
            key=lambda part: len(code_tokens(part)),
            default=problem,
        ),
    ]
    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            roots.append(" ".join(candidate.split()))
            seen.add(key)
    return roots


def frozen_split(
    metadata_rows: Sequence[dict[str, str]],
) -> dict[str, list[str]]:
    grouped: dict[str, list[str]] = defaultdict(list)
    for row in metadata_rows:
        grouped[str(row["repo"])].append(str(row["instance_id"]))
    if len(grouped) != EXPECTED_REPOSITORIES:
        raise ValueError(
            f"expected {EXPECTED_REPOSITORIES} repositories, got {len(grouped)}"
        )
    rng = random.Random(SELECTION_SEED)
    opportunity: list[str] = []
    development: list[str] = []
    holdout: list[str] = []
    for repo in sorted(grouped):
        ids = sorted(grouped[repo])
        if len(ids) < 3:
            raise ValueError(f"repository {repo} has fewer than three rows")
        rng.shuffle(ids)
        opportunity.extend(ids[:OPPORTUNITY_PER_REPOSITORY])
        development.extend(
            ids[
                OPPORTUNITY_PER_REPOSITORY:
                OPPORTUNITY_PER_REPOSITORY + DEVELOPMENT_PER_REPOSITORY
            ]
        )
        holdout.extend(
            ids[
                OPPORTUNITY_PER_REPOSITORY + DEVELOPMENT_PER_REPOSITORY:
            ]
        )
    split = {
        "opportunity": opportunity,
        "development": development,
        "holdout": holdout,
    }
    expected = {
        "opportunity": EXPECTED_OPPORTUNITY,
        "development": EXPECTED_DEVELOPMENT,
        "holdout": EXPECTED_HOLDOUT,
    }
    if {name: len(ids) for name, ids in split.items()} != expected:
        raise ValueError("metadata split sizes do not reproduce")
    flattened = [item for ids in split.values() for item in ids]
    if len(flattened) != len(set(flattened)) or len(flattened) != len(metadata_rows):
        raise ValueError("metadata split is not disjoint and exhaustive")
    return split


def target_files_from_patch(patch: str) -> list[str]:
    files: list[str] = []
    for line in patch.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
            if path not in files:
                files.append(path)
    return files


def eligible_target_file(patch: str) -> tuple[str | None, str | None]:
    files = target_files_from_patch(patch)
    production_python = [
        path
        for path in files
        if path.endswith(".py")
        and not path.startswith("test/")
        and not path.startswith("tests/")
        and "/test/" not in path
        and "/tests/" not in path
    ]
    if len(files) != 1 or len(production_python) != 1:
        return None, "not_single_production_python_file"
    return production_python[0], None


def has_direct_target_leak(problem: str, target_file: str) -> bool:
    lowered = problem.casefold()
    basename = Path(target_file).name.casefold()
    stem = Path(target_file).stem.casefold()
    return (
        target_file.casefold() in lowered
        or basename in lowered
        or (len(stem) >= 5 and stem in lowered)
    )


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


def repository_sources(repo: Path, commit: str) -> dict[str, str]:
    _git(repo, "cat-file", "-e", f"{commit}^{{commit}}")
    paths = [
        path
        for path in _git(repo, "ls-tree", "-r", "--name-only", commit).splitlines()
        if path.endswith(".py")
        and "/.tox/" not in path
        and "/venv/" not in path
        and "/site-packages/" not in path
    ]
    sources = {
        path: _git(repo, "show", f"{commit}:{path}")[:SOURCE_CHAR_CAP]
        for path in paths
    }
    if len(sources) < 10:
        raise ValueError("base commit exposes too few Python files")
    return sources


def module_aliases(path: str) -> set[str]:
    parts = list(Path(path).with_suffix("").parts)
    if "src" in parts:
        parts = parts[parts.index("src") + 1:]
    if parts and parts[-1] == "__init__":
        parts = parts[:-1]
    aliases = {
        ".".join(parts[index:])
        for index in range(len(parts))
        if len(parts) - index >= 2
    }
    if parts:
        aliases.add(".".join(parts))
    return {alias for alias in aliases if alias}


def _resolve_import(
    name: str,
    aliases_to_paths: dict[str, set[str]],
) -> set[str]:
    direct = aliases_to_paths.get(name)
    if direct:
        return set(direct)
    return {
        path
        for alias, paths in aliases_to_paths.items()
        if alias.endswith(f".{name}") or name.endswith(f".{alias}")
        for path in paths
    }


def build_import_graph(
    source_by_path: dict[str, str],
) -> tuple[dict[str, set[str]], dict[str, set[str]]]:
    aliases_to_paths: dict[str, set[str]] = defaultdict(set)
    for path in source_by_path:
        for alias in module_aliases(path):
            aliases_to_paths[alias].add(path)

    outgoing: dict[str, set[str]] = {
        path: set() for path in source_by_path
    }
    for path, source in source_by_path.items():
        try:
            tree = ast.parse(source)
        except (SyntaxError, ValueError):
            continue
        current_aliases = sorted(module_aliases(path), key=len, reverse=True)
        package = current_aliases[0].split(".")[:-1] if current_aliases else []
        names: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names.extend(alias.name for alias in node.names)
            elif isinstance(node, ast.ImportFrom):
                base = node.module or ""
                if node.level:
                    prefix = package[:max(0, len(package) - node.level + 1)]
                    base = ".".join(prefix + ([base] if base else []))
                if base:
                    names.append(base)
                names.extend(
                    f"{base}.{alias.name}" if base else alias.name
                    for alias in node.names
                )
        for name in names:
            outgoing[path].update(
                _resolve_import(name, aliases_to_paths).difference({path})
            )

    incoming: dict[str, set[str]] = {
        path: set() for path in source_by_path
    }
    for source, targets in outgoing.items():
        for target in targets:
            incoming[target].add(source)
    return outgoing, incoming


def graph_followup_candidates(
    problem: str,
    source_by_path: dict[str, str],
    outgoing: dict[str, set[str]],
    incoming: dict[str, set[str]],
    first_ids: Sequence[str],
    *,
    top_k: int = FOLLOWUP_TOP_K,
) -> list[str]:
    issue_tokens = Counter(
        token for token in code_tokens(problem) if len(token) >= 3
    )
    first = [path for path in first_ids if path in source_by_path]
    scores: dict[str, float] = {}
    for candidate in source_by_path:
        if candidate in first:
            continue
        imported_by_roots = sum(candidate in outgoing[root] for root in first)
        imports_roots = sum(candidate in incoming[root] for root in first)
        same_directory = sum(
            Path(candidate).parent == Path(root).parent for root in first
        )
        if not (imported_by_roots or imports_roots or same_directory):
            continue
        candidate_tokens = set(code_tokens(candidate))
        path_overlap = sum(
            issue_tokens[token] for token in candidate_tokens
        )
        scores[candidate] = (
            5.0 * imported_by_roots
            + 3.0 * imports_roots
            + float(same_directory)
            + 0.25 * path_overlap
        )
    return [
        path
        for path, _ in sorted(
            scores.items(),
            key=lambda item: (-item[1], item[0]),
        )[:top_k]
    ]


def analyze_issue(
    issue: dict[str, Any],
    *,
    repo_root: Path,
) -> dict[str, Any]:
    target_file, exclusion = eligible_target_file(str(issue["patch"]))
    base = {
        "instance_id": str(issue["instance_id"]),
        "repo": str(issue["repo"]),
        "base_commit": str(issue["base_commit"]),
    }
    if exclusion:
        return {**base, "eligible": False, "exclusion_reason": exclusion}
    assert target_file is not None
    if has_direct_target_leak(str(issue["problem_statement"]), target_file):
        return {
            **base,
            "eligible": False,
            "exclusion_reason": "direct_target_leak",
        }

    repo = repo_root / str(issue["repo"]).replace("/", "__")
    source_by_path = repository_sources(repo, str(issue["base_commit"]))
    if target_file not in source_by_path:
        return {
            **base,
            "eligible": False,
            "exclusion_reason": "target_absent_from_python_corpus",
        }

    from scripts.bright_biology_unlock_audit import BM25Corpus

    documents = [
        {
            "id": path,
            "content": " ".join(
                code_tokens(path) * PATH_WEIGHT + code_tokens(source)
            ),
        }
        for path, source in source_by_path.items()
    ]
    corpus = BM25Corpus(documents)
    outgoing, incoming = build_import_graph(source_by_path)
    roots = root_queries(str(issue["problem_statement"]))
    if len(roots) < 3:
        return {
            **base,
            "eligible": False,
            "exclusion_reason": "fewer_than_three_roots",
        }

    root_records: list[dict[str, Any]] = []
    for root_index, query in enumerate(roots):
        first_ids = [
            row["id"] for row in corpus.search(query, top_k=ROOT_TOP_K)
        ]
        followup_ids = graph_followup_candidates(
            str(issue["problem_statement"]),
            source_by_path,
            outgoing,
            incoming,
            first_ids,
        )
        immediate = int(target_file in first_ids)
        pair_value = int(target_file in set(first_ids).union(followup_ids))
        root_records.append(
            {
                "root_index": root_index,
                "query": query,
                "first_result_ids": first_ids,
                "followup_result_ids": followup_ids,
                "immediate_value": immediate,
                "pair_value": pair_value,
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
            root_records[index]["pair_value"],
            -index,
        ),
    )
    best_immediate = root_records[greedy_root]["immediate_value"]
    oracle_pair = root_records[oracle_root]["pair_value"]
    greedy_tail = root_records[greedy_root]["pair_value"]
    return {
        **base,
        "eligible": True,
        "exclusion_reason": None,
        "target_file": target_file,
        "num_python_files": len(source_by_path),
        "num_roots": len(roots),
        "best_immediate_value": best_immediate,
        "oracle_pair_value": oracle_pair,
        "pair_gain": oracle_pair - best_immediate,
        "greedy_root_index": greedy_root,
        "oracle_root_index": oracle_root,
        "greedy_oracle_tail_value": greedy_tail,
        "nonmyopic_gap": oracle_pair - greedy_tail,
        "roots": root_records,
    }


def summarize(
    records: Sequence[dict[str, Any]],
    split: dict[str, list[str]],
) -> dict[str, Any]:
    eligible = [record for record in records if record["eligible"]]
    count = len(eligible)
    pair_gain_count = sum(record["pair_gain"] > 0 for record in eligible)
    gap_records = [
        record for record in eligible if record["nonmyopic_gap"] > 0
    ]
    gap_count = len(gap_records)
    direct_count = sum(
        record["best_immediate_value"] > 0 for record in eligible
    )
    pair_threshold = max(4, math.ceil(0.15 * count))
    gap_threshold = max(4, math.ceil(0.15 * count))
    strict_repositories = len({record["repo"] for record in gap_records})
    mean_pair_gain = (
        sum(record["pair_gain"] for record in eligible) / count
        if count else 0.0
    )
    mean_gap = (
        sum(record["nonmyopic_gap"] for record in eligible) / count
        if count else 0.0
    )
    gates = {
        "split_sizes_exact": {
            name: len(ids) for name, ids in split.items()
        }
        == {
            "opportunity": EXPECTED_OPPORTUNITY,
            "development": EXPECTED_DEVELOPMENT,
            "holdout": EXPECTED_HOLDOUT,
        },
        "all_24_opportunity_rows_complete": len(records) == EXPECTED_OPPORTUNITY,
        "eligible_count_at_least_12": count >= 12,
        "all_eligible_have_at_least_3_roots": all(
            record["num_roots"] >= 3 for record in eligible
        ),
        "direct_coverage_at_most_80_percent": (
            direct_count <= math.floor(0.8 * count) if count else False
        ),
        "pair_gain_count_pass": pair_gain_count >= pair_threshold,
        "nonmyopic_gap_count_pass": gap_count >= gap_threshold,
        "strict_gaps_span_at_least_3_repositories": strict_repositories >= 3,
        "mean_pair_gain_at_least_0_15": mean_pair_gain >= 0.15,
        "mean_nonmyopic_gap_at_least_0_15": mean_gap >= 0.15,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "opportunity_count": len(records),
        "eligible_count": count,
        "excluded_count": len(records) - count,
        "exclusion_reasons": dict(
            Counter(
                record["exclusion_reason"]
                for record in records
                if not record["eligible"]
            )
        ),
        "direct_root_coverage_count": direct_count,
        "pair_gain_count": pair_gain_count,
        "pair_gain_threshold": pair_threshold,
        "nonmyopic_gap_count": gap_count,
        "nonmyopic_gap_threshold": gap_threshold,
        "strict_gap_repository_count": strict_repositories,
        "mean_pair_gain": mean_pair_gain,
        "mean_nonmyopic_gap": mean_gap,
        "gates": gates,
    }


def _load_metadata(test_parquet: Path) -> list[dict[str, str]]:
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required to read SWE-bench") from exc
    return pq.read_table(
        test_parquet,
        columns=["instance_id", "repo", "base_commit"],
    ).to_pylist()


def _load_opportunity_rows(
    test_parquet: Path,
    opportunity_ids: Sequence[str],
) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    table = pq.read_table(
        test_parquet,
        columns=[
            "instance_id",
            "repo",
            "base_commit",
            "problem_statement",
            "patch",
        ],
        filters=[("instance_id", "in", list(opportunity_ids))],
    )
    by_id = {str(row["instance_id"]): row for row in table.to_pylist()}
    if set(by_id) != set(opportunity_ids):
        raise ValueError("opportunity rows do not reproduce")
    return [by_id[instance_id] for instance_id in opportunity_ids]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--test-parquet", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if sha256_file(args.test_parquet) != TEST_SHA256:
        raise ValueError("test parquet hash does not match")
    metadata = _load_metadata(args.test_parquet)
    split = frozen_split(metadata)
    opportunity_rows = _load_opportunity_rows(
        args.test_parquet,
        split["opportunity"],
    )
    records = [
        analyze_issue(issue, repo_root=args.repo_root)
        for issue in opportunity_rows
    ]
    summary = summarize(records, split)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "dataset_revision": DATASET_REVISION,
            "test_sha256": TEST_SHA256,
            "selection_seed": SELECTION_SEED,
            "split_sizes": {
                name: len(ids) for name, ids in split.items()
            },
            "split_id_sha256": {
                name: hash_ids(ids) for name, ids in split.items()
            },
            "opportunity_ids": split["opportunity"],
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "path_weight": PATH_WEIGHT,
            "source_char_cap": SOURCE_CHAR_CAP,
            "api_calls": 0,
            "development_and_holdout_problem_patch_columns_read": False,
        },
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
        )
    )


if __name__ == "__main__":
    main()
