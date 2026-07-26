#!/usr/bin/env python3
"""Audit Mini-Interact for the latent-world structure required by BED."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable


SOURCE_REPOSITORY = "https://github.com/bird-bench/BIRD-Interact"
SOURCE_COMMIT = "451fe2c3518ee1cf908d8139e2913483bd519381"
DATASET_REPOSITORY = "https://huggingface.co/datasets/birdsql/mini-interact"
DATASET_REVISION = "10253f235dbc6092a97e9c0acd30dba6203d8e59"
EXPECTED_DATA_SHA256 = (
    "32eea6778d69f4a052622a9981ee95cc5dddbff62d0d65ff6e06bd5bec757087"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _histogram(values: Iterable[int]) -> dict[str, int]:
    return {
        str(value): count
        for value, count in sorted(Counter(values).items())
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    critical_counts: list[int] = []
    knowledge_counts: list[int] = []
    critical_types: Counter[str] = Counter()
    term_snippets: dict[tuple[str, str], set[str]] = defaultdict(set)
    task_term_snippets: dict[tuple[str, str], set[str]] = defaultdict(set)
    repeated_term_pairs: Counter[tuple[str, str]] = Counter()
    critical_total = 0
    masked_critical = 0

    for row in rows:
        database = str(row.get("selected_database", ""))
        critical = (
            row.get("user_query_ambiguity", {})
            .get("critical_ambiguity", [])
        )
        knowledge = row.get("knowledge_ambiguity", [])
        critical_counts.append(len(critical))
        knowledge_counts.append(len(knowledge))
        critical_total += len(critical)
        for ambiguity in critical:
            critical_types[str(ambiguity.get("type", ""))] += 1
            masked_critical += ambiguity.get("is_mask") is True
            term = " ".join(
                str(ambiguity.get("term", "")).lower().split()
            )
            key = (database, term)
            repeated_term_pairs[key] += 1
            term_snippets[key].add(str(ambiguity.get("sql_snippet", "")))
            task_term_snippets[
                (str(row.get("instance_id", "")), term)
            ].add(str(ambiguity.get("sql_snippet", "")))

    repeated_pairs = {
        key for key, count in repeated_term_pairs.items() if count > 1
    }
    competing_pairs = {
        key for key, snippets in term_snippets.items() if len(snippets) > 1
    }
    all_keys = sorted({key for row in rows for key in row})
    return {
        "rows": len(rows),
        "distinct_instances": len(
            {str(row.get("instance_id", "")) for row in rows}
        ),
        "distinct_databases": len(
            {str(row.get("selected_database", "")) for row in rows}
        ),
        "released_fields": all_keys,
        "missing_clear_query_rows": sum(not row.get("query") for row in rows),
        "empty_solution_sql_rows": sum(not row.get("sol_sql") for row in rows),
        "empty_test_case_rows": sum(not row.get("test_cases") for row in rows),
        "follow_up_rows": sum(bool(row.get("follow_up")) for row in rows),
        "critical_ambiguities": {
            "total": critical_total,
            "count_histogram": _histogram(critical_counts),
            "type_histogram": dict(sorted(critical_types.items())),
            "masked": masked_critical,
            "tasks_with_at_least_three": sum(
                count >= 3 for count in critical_counts
            ),
        },
        "knowledge_ambiguities": {
            "count_histogram": _histogram(knowledge_counts),
            "tasks_with_at_least_two": sum(
                count >= 2 for count in knowledge_counts
            ),
        },
        "released_alternative_interpretations": {
            "same_database_repeated_term_pairs": len(repeated_pairs),
            "same_database_term_pairs_with_distinct_sql_snippets": len(
                competing_pairs
            ),
            "task_term_pairs_with_distinct_sql_snippets": sum(
                len(snippets) > 1
                for snippets in task_term_snippets.values()
            ),
            "explicit_world_prior_field_present": any(
                key in row
                for row in rows
                for key in (
                    "worlds",
                    "hypotheses",
                    "latent_worlds",
                    "world_prior",
                    "hypothesis_prior",
                )
            ),
        },
    }


def build_audit(data_path: Path) -> dict[str, Any]:
    data_sha256 = sha256_file(data_path)
    if data_sha256 != EXPECTED_DATA_SHA256:
        raise ValueError(
            f"unexpected Mini-Interact SHA256: {data_sha256}"
        )
    rows = load_jsonl(data_path)
    return {
        "source": {
            "repository": SOURCE_REPOSITORY,
            "commit": SOURCE_COMMIT,
            "dataset": DATASET_REPOSITORY,
            "dataset_revision": DATASET_REVISION,
            "data_sha256": data_sha256,
        },
        "summary": summarize_rows(rows),
        "decision": {
            "direct_bed_replay_authorized": False,
            "reason": (
                "The release provides one annotated interpretation per "
                "ambiguity term, no alternative latent-world prior, and no "
                "world-conditioned response table."
            ),
            "construction_route": (
                "A separate preregistered environment may generate competing "
                "SQL-intent worlds and use the released simulator/database "
                "machinery, but it must not be reported as direct benchmark "
                "replay."
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit = build_audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
