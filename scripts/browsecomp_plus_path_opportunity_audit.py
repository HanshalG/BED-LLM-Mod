#!/usr/bin/env python3
"""Audit observation-enabled evidence paths in frozen BrowseComp-Plus tasks."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence

from scripts.browsecomp_plus_semantic_bed_manifest import (
    PARQUET_SHA256,
    evidence_bin,
    parse_qrels,
    sha256_file,
)
from scripts.extract_browsecomp_plus_open_mechanics import CANARY


MANIFEST_SHA256 = (
    "fd3d0e05f7110974f3312711e24f3d98ac19470447dcd1fccc6dbbf05221a862"
)
TRAJECTORY_SHA256 = (
    "74e8e2b24d0ff250ec86a26c392a55a9a5d34ac9911d253b2e35e017341cf944"
)
EXPECTED_TASKS = 120
MIN_USABLE_TRAJECTORIES = 110
MIN_TASKS_WITH_EVIDENCE = 45
MIN_BRIDGE_EVIDENCE_TASKS = 20
MIN_BRIDGE_GOLD_TASKS = 8
MIN_DELAYED_EVIDENCE_TASKS = 20
MIN_BRIDGE_AFTER_MISS_TASKS = 10
MIN_BRIDGE_TASKS_BY_BIN = {"low": 3, "mid": 6, "high": 5}
STOPWORDS = {
    "about",
    "according",
    "after",
    "also",
    "answer",
    "article",
    "based",
    "before",
    "being",
    "between",
    "biography",
    "could",
    "during",
    "each",
    "following",
    "from",
    "full",
    "give",
    "have",
    "interview",
    "into",
    "made",
    "name",
    "please",
    "recorded",
    "referred",
    "report",
    "search",
    "should",
    "site",
    "supply",
    "their",
    "there",
    "these",
    "those",
    "under",
    "using",
    "what",
    "when",
    "where",
    "which",
    "while",
    "wikipedia",
    "with",
    "would",
}


def content_tokens(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", value.lower())
        if len(token) >= 4
        and not token.isdigit()
        and token not in STOPWORDS
    }


def decrypt_string(ciphertext: str) -> str:
    encrypted = base64.b64decode(ciphertext)
    key = hashlib.sha256(CANARY.encode("utf-8")).digest()
    expanded = (
        key * (len(encrypted) // len(key))
        + key[: len(encrypted) % len(key)]
    )
    return bytes(a ^ b for a, b in zip(encrypted, expanded)).decode("utf-8")


def analyze_task(
    *,
    question: str,
    events: Sequence[tuple[str, Sequence[dict[str, Any]]]],
    evidence_doc_ids: set[str],
    gold_doc_ids: set[str],
) -> dict[str, Any]:
    question_tokens = content_tokens(question)
    prior_observation_tokens: set[str] = set()
    seen_doc_ids: set[str] = set()
    bridge_evidence_hits = 0
    bridge_gold_hits = 0
    first_evidence_search_index: int | None = None
    bridge_after_miss = False
    previous_new_evidence_count = 0

    for search_index, (query, documents) in enumerate(events, start=1):
        query_tokens = content_tokens(query)
        enabled_tokens = (
            query_tokens - question_tokens
        ) & prior_observation_tokens
        returned_ids = {
            str(document["docid"])
            for document in documents
            if isinstance(document, dict) and document.get("docid") is not None
        }
        new_ids = returned_ids - seen_doc_ids
        new_evidence = new_ids & evidence_doc_ids
        new_gold = new_ids & gold_doc_ids
        if new_evidence and first_evidence_search_index is None:
            first_evidence_search_index = search_index
        is_bridge = len(enabled_tokens) >= 2
        if is_bridge and new_evidence:
            bridge_evidence_hits += 1
            if previous_new_evidence_count == 0:
                bridge_after_miss = True
        if is_bridge and new_gold:
            bridge_gold_hits += 1

        for document in documents:
            if isinstance(document, dict):
                prior_observation_tokens |= content_tokens(
                    str(document.get("snippet", ""))
                )
        seen_doc_ids |= returned_ids
        previous_new_evidence_count = len(new_evidence)

    return {
        "search_count": len(events),
        "evidence_seen_count": len(seen_doc_ids & evidence_doc_ids),
        "gold_seen_count": len(seen_doc_ids & gold_doc_ids),
        "first_evidence_search_index": first_evidence_search_index,
        "bridge_evidence_hit_count": bridge_evidence_hits,
        "bridge_gold_hit_count": bridge_gold_hits,
        "has_bridge_evidence": bridge_evidence_hits > 0,
        "has_bridge_gold": bridge_gold_hits > 0,
        "has_delayed_evidence": (
            first_evidence_search_index is not None
            and first_evidence_search_index >= 3
        ),
        "has_bridge_after_miss": bridge_after_miss,
    }


def load_questions(
    parquet_dir: Path,
    task_ids: Sequence[str],
) -> dict[str, str]:
    try:
        import pyarrow.parquet as parquet
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the official shards") from exc

    rows: list[dict[str, Any]] = []
    for name, expected_hash in PARQUET_SHA256.items():
        path = parquet_dir / name
        if sha256_file(path) != expected_hash:
            raise ValueError(f"{name} hash changed")
        table = parquet.read_table(
            path,
            columns=["query_id", "query"],
            filters=[("query_id", "in", list(task_ids))],
        )
        rows.extend(table.to_pylist())
    questions = {
        str(row["query_id"]): decrypt_string(str(row["query"]))
        for row in rows
    }
    if set(questions) != set(task_ids):
        raise ValueError("opportunity questions did not reproduce")
    return questions


def load_trajectories(
    path: Path,
    task_ids: Sequence[str],
) -> dict[str, dict[str, Any]]:
    if sha256_file(path) != TRAJECTORY_SHA256:
        raise ValueError("released trajectory hash changed")
    wanted = set(task_ids)
    selected: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        query_id = str(row.get("query_id"))
        if query_id in wanted:
            if query_id in selected:
                raise ValueError("duplicate opportunity trajectory")
            selected[query_id] = row
    if set(selected) != wanted:
        raise ValueError("opportunity trajectories did not reproduce")
    return selected


def trajectory_events(row: dict[str, Any]) -> list[tuple[str, list[dict[str, Any]]]]:
    events: list[tuple[str, list[dict[str, Any]]]] = []
    for item in row.get("result", []):
        if (
            item.get("type") != "tool_call"
            or item.get("tool_name") != "search"
        ):
            continue
        arguments = json.loads(item["arguments"])
        query = arguments.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("trajectory search has no query")
        decrypted = json.loads(decrypt_string(item["output"]))
        if not isinstance(decrypted, list) or not all(
            isinstance(document, dict) and "docid" in document
            for document in decrypted
        ):
            raise ValueError("trajectory search output is invalid")
        events.append((query, decrypted))
    return events


def run_audit(
    *,
    manifest_path: Path,
    parquet_dir: Path,
    qrel_dir: Path,
    trajectory_path: Path,
) -> dict[str, Any]:
    if sha256_file(manifest_path) != MANIFEST_SHA256:
        raise ValueError("BrowseComp-Plus manifest hash changed")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    task_ids = manifest["splits"]["opportunity"]["task_ids"]
    if len(task_ids) != EXPECTED_TASKS:
        raise ValueError("opportunity split size changed")
    evidence = parse_qrels(qrel_dir / "qrel_evidence.txt")
    gold = parse_qrels(qrel_dir / "qrel_golds.txt")
    questions = load_questions(parquet_dir, task_ids)
    trajectories = load_trajectories(trajectory_path, task_ids)

    records: list[dict[str, Any]] = []
    for query_id in task_ids:
        row = trajectories[query_id]
        events = trajectory_events(row)
        metrics = analyze_task(
            question=questions[query_id],
            events=events,
            evidence_doc_ids=evidence[query_id],
            gold_doc_ids=gold[query_id],
        )
        records.append(
            {
                "query_id": query_id,
                "status": str(row.get("status")),
                "evidence_bin": evidence_bin(len(evidence[query_id])),
                **metrics,
            }
        )

    usable_count = sum(
        record["status"] == "completed" and record["search_count"] >= 2
        for record in records
    )
    evidence_task_count = sum(
        record["evidence_seen_count"] > 0 for record in records
    )
    bridge_task_count = sum(
        record["has_bridge_evidence"] for record in records
    )
    bridge_gold_task_count = sum(
        record["has_bridge_gold"] for record in records
    )
    delayed_task_count = sum(
        record["has_delayed_evidence"] for record in records
    )
    bridge_after_miss_count = sum(
        record["has_bridge_after_miss"] for record in records
    )
    bridge_by_bin = {
        name: sum(
            record["evidence_bin"] == name
            and record["has_bridge_evidence"]
            for record in records
        )
        for name in ("low", "mid", "high")
    }
    gates = {
        "exact_120_tasks": len(records) == EXPECTED_TASKS,
        "at_least_110_usable_trajectories": (
            usable_count >= MIN_USABLE_TRAJECTORIES
        ),
        "at_least_45_tasks_retrieve_evidence": (
            evidence_task_count >= MIN_TASKS_WITH_EVIDENCE
        ),
        "at_least_20_bridge_evidence_tasks": (
            bridge_task_count >= MIN_BRIDGE_EVIDENCE_TASKS
        ),
        "at_least_8_bridge_gold_tasks": (
            bridge_gold_task_count >= MIN_BRIDGE_GOLD_TASKS
        ),
        "at_least_20_delayed_evidence_tasks": (
            delayed_task_count >= MIN_DELAYED_EVIDENCE_TASKS
        ),
        "at_least_10_bridge_after_miss_tasks": (
            bridge_after_miss_count >= MIN_BRIDGE_AFTER_MISS_TASKS
        ),
        "bridge_tasks_span_all_evidence_bins": all(
            bridge_by_bin[name] >= minimum
            for name, minimum in MIN_BRIDGE_TASKS_BY_BIN.items()
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "interface_version": "browsecomp-plus-path-opportunity-audit-1",
            "manifest_sha256": MANIFEST_SHA256,
            "trajectory_sha256": TRAJECTORY_SHA256,
            "split": "opportunity",
            "task_count": EXPECTED_TASKS,
            "bridge_minimum_enabled_tokens": 2,
            "delayed_first_evidence_minimum_search_index": 3,
            "openrouter_calls": 0,
            "oatml_used": False,
        },
        "summary": {
            "gates": gates,
            "usable_trajectory_count": usable_count,
            "evidence_task_count": evidence_task_count,
            "bridge_evidence_task_count": bridge_task_count,
            "bridge_gold_task_count": bridge_gold_task_count,
            "delayed_evidence_task_count": delayed_task_count,
            "bridge_after_miss_task_count": bridge_after_miss_count,
            "bridge_evidence_task_count_by_bin": bridge_by_bin,
        },
        "records": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--parquet-dir", type=Path, required=True)
    parser.add_argument("--qrel-dir", type=Path, required=True)
    parser.add_argument("--trajectory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run_audit(
        manifest_path=args.manifest,
        parquet_dir=args.parquet_dir,
        qrel_dir=args.qrel_dir,
        trajectory_path=args.trajectory,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
