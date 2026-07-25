#!/usr/bin/env python3
"""Audit exact first-action planning gaps in frozen HoVer opportunity rows."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re
import sqlite3
import unicodedata
from typing import Any, Mapping, Sequence


SOURCE_SHA256 = (
    "67c14858f2d7fcdb96b6fe3d538ffcd6f76e3ba594aa2c0cd4359f601101e89d"
)
RETRIEVAL_SHA256 = (
    "b50a961f63a95ff184986af766b15ff6b1d6c98f7e86b39355b64b1a85fb3745"
)
DATABASE_SHA256 = (
    "c37ee397916ec0bffacfe8902db454a5cda88a7a188409217b2e15231fe5ee2f"
)
MANIFEST_SHA256 = (
    "d64db625ca142b97df990b1831b62c1d4438b1f52555c09388e2f1762f733b13"
)
OPPORTUNITY_HASH = (
    "230ba25090172484250812ae0c24b089b2f3c888c7406702a47d2cd18a570fd0"
)
SOURCE_ROWS = 4_000
OPPORTUNITY_ROWS = 400
CANDIDATE_COUNT = 100
ROOT_COUNT = 10
MAX_PATH_DOCUMENTS = 3


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_hash(values: Sequence[str]) -> str:
    payload = json.dumps(
        list(values),
        ensure_ascii=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.replace("_", " "))
    ascii_text = decomposed.encode("ascii", errors="ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))


def title_aliases(title: str) -> tuple[str, ...]:
    raw_aliases = [title]
    prefix = re.sub(r"\s*\([^()]*\)\s*$", "", title).strip()
    if prefix and prefix != title:
        raw_aliases.append(prefix)
    aliases: list[str] = []
    for raw_alias in raw_aliases:
        normalized = normalize_text(raw_alias)
        if len(normalized) < 5:
            continue
        if " " not in normalized and len(normalized) < 5:
            continue
        aliases.append(normalized)
    return tuple(dict.fromkeys(aliases))


def contains_title(normalized_text: str, title: str) -> bool:
    padded = f" {normalized_text} "
    return any(
        f" {alias} " in padded
        for alias in title_aliases(title)
    )


def build_adjacency(
    titles: Sequence[str],
    text_by_title: Mapping[str, str],
) -> dict[int, tuple[int, ...]]:
    normalized_documents = {
        title: normalize_text(text_by_title[title])
        for title in titles
    }
    return {
        source_index: tuple(
            target_index
            for target_index, target_title in enumerate(titles)
            if target_index != source_index
            and contains_title(
                normalized_documents[source_title],
                target_title,
            )
        )
        for source_index, source_title in enumerate(titles)
    }


def best_path(
    *,
    root_index: int,
    adjacency: Mapping[int, Sequence[int]],
    support_indexes: set[int],
    max_documents: int,
) -> tuple[int, tuple[int, ...]]:
    best_coverage = int(root_index in support_indexes)
    best_indexes = (root_index,)

    def visit(path: tuple[int, ...]) -> None:
        nonlocal best_coverage, best_indexes
        coverage = len(set(path) & support_indexes)
        if coverage > best_coverage or (
            coverage == best_coverage and path < best_indexes
        ):
            best_coverage = coverage
            best_indexes = path
        if len(path) >= max_documents:
            return
        for next_index in adjacency[path[-1]]:
            if next_index not in path:
                visit((*path, next_index))

    visit((root_index,))
    return best_coverage, best_indexes


def _select_root(
    root_rows: Sequence[dict[str, Any]],
    *,
    key,
) -> dict[str, Any]:
    return max(root_rows, key=key)


def analyze_task(
    *,
    task_id: str,
    num_hops: int,
    supporting_titles: Sequence[str],
    candidate_titles: Sequence[str],
    text_by_title: Mapping[str, str],
    root_count: int = ROOT_COUNT,
) -> dict[str, Any]:
    title_to_index = {
        title: index for index, title in enumerate(candidate_titles)
    }
    support_indexes = {
        title_to_index[title]
        for title in supporting_titles
        if title in title_to_index
    }
    adjacency = build_adjacency(candidate_titles, text_by_title)
    roots = []
    for root_index in range(root_count):
        immediate = int(root_index in support_indexes)
        value2, path2 = best_path(
            root_index=root_index,
            adjacency=adjacency,
            support_indexes=support_indexes,
            max_documents=2,
        )
        value3, path3 = best_path(
            root_index=root_index,
            adjacency=adjacency,
            support_indexes=support_indexes,
            max_documents=MAX_PATH_DOCUMENTS,
        )
        roots.append(
            {
                "root_index": root_index,
                "immediate": immediate,
                "value2": value2,
                "value3": value3,
                "path2": path2,
                "path3": path3,
            }
        )

    max_immediate = max(row["immediate"] for row in roots)
    immediate_optimal = [
        row for row in roots if row["immediate"] == max_immediate
    ]
    rank_myopic = min(
        immediate_optimal,
        key=lambda row: row["root_index"],
    )
    robust_myopic = _select_root(
        immediate_optimal,
        key=lambda row: (row["value3"], -row["root_index"]),
    )
    oracle_d2 = _select_root(
        roots,
        key=lambda row: (
            row["value2"],
            row["immediate"],
            -row["root_index"],
        ),
    )
    oracle_d3 = _select_root(
        roots,
        key=lambda row: (
            row["value3"],
            row["value2"],
            row["immediate"],
            -row["root_index"],
        ),
    )
    rank_root_gain = oracle_d3["value3"] - rank_myopic["value3"]
    robust_root_gain = oracle_d3["value3"] - robust_myopic["value3"]
    rank_sensitive = (
        rank_root_gain > 0
        and oracle_d3["root_index"] != rank_myopic["root_index"]
    )
    robust_sacrifice = (
        robust_root_gain > 0
        and oracle_d3["immediate"] < robust_myopic["immediate"]
    )
    graph_edge_count = sum(len(edges) for edges in adjacency.values())
    root_values = [row["value3"] for row in roots]
    return {
        "task_id": task_id,
        "num_hops": num_hops,
        "distinct_support_titles": len(set(supporting_titles)),
        "support_titles_in_candidates": len(support_indexes),
        "graph_edge_count": graph_edge_count,
        "nonempty_graph": graph_edge_count > 0,
        "depth3_support_reachability": oracle_d3["value3"],
        "depth3_root_value_range": max(root_values) - min(root_values),
        "rank_myopic_root_index": rank_myopic["root_index"],
        "rank_myopic_immediate": rank_myopic["immediate"],
        "rank_myopic_value3": rank_myopic["value3"],
        "robust_myopic_root_index": robust_myopic["root_index"],
        "robust_myopic_immediate": robust_myopic["immediate"],
        "robust_myopic_value3": robust_myopic["value3"],
        "oracle_d2_root_index": oracle_d2["root_index"],
        "oracle_d2_value": oracle_d2["value2"],
        "oracle_d3_root_index": oracle_d3["root_index"],
        "oracle_d3_immediate": oracle_d3["immediate"],
        "oracle_d3_value": oracle_d3["value3"],
        "oracle_d3_path_indexes": list(oracle_d3["path3"]),
        "rank_root_gain": rank_root_gain,
        "robust_root_gain": robust_root_gain,
        "rank_sensitive_opportunity": rank_sensitive,
        "robust_sacrifice_opportunity": robust_sacrifice,
        "depth3_exceeds_depth2": (
            oracle_d3["value3"] > oracle_d2["value2"]
        ),
    }


def summarize(
    diagnostics: Sequence[dict[str, Any]],
    *,
    resolved_candidate_rows: int,
) -> dict[str, Any]:
    count = len(diagnostics)
    rank_sensitive = [
        row for row in diagnostics
        if row["rank_sensitive_opportunity"]
    ]
    robust = [
        row for row in diagnostics
        if row["robust_sacrifice_opportunity"]
    ]
    nonempty_graphs = sum(row["nonempty_graph"] for row in diagnostics)
    reachable_dynamic = sum(
        row["depth3_support_reachability"] > 0
        and row["depth3_root_value_range"] >= 1
        for row in diagnostics
    )
    depth3_exceeds = sum(
        row["depth3_exceeds_depth2"] for row in diagnostics
    )
    mean_rank_gain = (
        sum(row["rank_root_gain"] for row in diagnostics) / count
        if count else 0.0
    )
    mean_robust_gain = (
        sum(row["robust_root_gain"] for row in diagnostics) / count
        if count else 0.0
    )
    rank_by_hop = {
        str(hop): sum(
            row["num_hops"] == hop
            and row["rank_sensitive_opportunity"]
            for row in diagnostics
        )
        for hop in (3, 4)
    }
    gates = {
        "exact_400_opportunity_rows": count == OPPORTUNITY_ROWS,
        "resolved_candidate_rows_at_least_390": (
            resolved_candidate_rows >= 390
        ),
        "nonempty_graph_rows_at_least_300": nonempty_graphs >= 300,
        "reachable_dynamic_rows_at_least_250": reachable_dynamic >= 250,
        "rank_sensitive_opportunities_at_least_40": (
            len(rank_sensitive) >= 40
        ),
        "rank_sensitive_three_hop_at_least_15": (
            rank_by_hop["3"] >= 15
        ),
        "rank_sensitive_four_hop_at_least_15": (
            rank_by_hop["4"] >= 15
        ),
        "robust_sacrifice_opportunities_at_least_8": len(robust) >= 8,
        "depth3_exceeds_depth2_rows_at_least_20": depth3_exceeds >= 20,
        "mean_rank_root_gain_at_least_0_10": mean_rank_gain >= 0.10,
        "mean_robust_root_gain_at_least_0_02": mean_robust_gain >= 0.02,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "opportunity_rows": count,
        "resolved_candidate_rows": resolved_candidate_rows,
        "nonempty_graph_rows": nonempty_graphs,
        "reachable_dynamic_rows": reachable_dynamic,
        "rank_sensitive_opportunity_count": len(rank_sensitive),
        "rank_sensitive_opportunity_by_hop": rank_by_hop,
        "robust_sacrifice_opportunity_count": len(robust),
        "depth3_exceeds_depth2_count": depth3_exceeds,
        "mean_rank_root_gain": mean_rank_gain,
        "mean_robust_root_gain": mean_robust_gain,
        "gates": gates,
    }


def load_and_audit(
    *,
    source_path: Path,
    retrieval_path: Path,
    database_path: Path,
    manifest_path: Path,
) -> dict[str, Any]:
    hashes = {
        "source": sha256_file(source_path),
        "retrieval": sha256_file(retrieval_path),
        "database": sha256_file(database_path),
        "manifest": sha256_file(manifest_path),
    }
    expected_hashes = {
        "source": SOURCE_SHA256,
        "retrieval": RETRIEVAL_SHA256,
        "database": DATABASE_SHA256,
        "manifest": MANIFEST_SHA256,
    }
    if hashes != expected_hashes:
        raise ValueError(
            f"HoVer opportunity input hashes changed: {hashes}"
        )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("content_emitted") is not False:
        raise ValueError("HoVer manifest emitted content")
    opportunity_ids = manifest["splits"]["opportunity"]["task_ids"]
    if (
        len(opportunity_ids) != OPPORTUNITY_ROWS
        or ordered_hash(opportunity_ids) != OPPORTUNITY_HASH
    ):
        raise ValueError("HoVer opportunity split changed")

    source_rows = json.loads(source_path.read_text(encoding="utf-8"))
    retrieval_rows = json.loads(
        retrieval_path.read_text(encoding="utf-8")
    )
    if len(source_rows) != SOURCE_ROWS or len(retrieval_rows) != SOURCE_ROWS:
        raise ValueError("HoVer source or retrieval row count changed")
    source_by_id = {str(row["uid"]): row for row in source_rows}
    retrieval_by_id = {str(row["id"]): row for row in retrieval_rows}

    connection = sqlite3.connect(
        f"file:{database_path.resolve()}?mode=ro",
        uri=True,
    )
    text_cache: dict[str, str | None] = {}

    def get_text(title: str) -> str | None:
        if title not in text_cache:
            hit = connection.execute(
                "SELECT text FROM documents WHERE id = ?",
                (title,),
            ).fetchone()
            text_cache[title] = None if hit is None else str(hit[0])
        return text_cache[title]

    diagnostics = []
    resolved_candidate_rows = 0
    unresolved_rows: list[dict[str, Any]] = []
    try:
        for task_id in opportunity_ids:
            source_row = source_by_id[task_id]
            retrieval_row = retrieval_by_id[task_id]
            retrieval_result = retrieval_row["doc_retrieval_results"]
            candidate_titles = [
                str(title) for title in retrieval_result[0][0]
            ]
            if (
                len(candidate_titles) != CANDIDATE_COUNT
                or len(set(candidate_titles)) != CANDIDATE_COUNT
            ):
                unresolved_rows.append(
                    {
                        "task_id": task_id,
                        "reason": "candidate_shape",
                    }
                )
                continue
            text_by_title = {
                title: get_text(title) for title in candidate_titles
            }
            missing = [
                title for title, text in text_by_title.items()
                if text is None
            ]
            if missing:
                unresolved_rows.append(
                    {
                        "task_id": task_id,
                        "reason": "database_title_missing",
                        "missing_count": len(missing),
                    }
                )
                continue
            resolved_candidate_rows += 1
            supporting_titles = list(
                dict.fromkeys(
                    str(fact[0])
                    for fact in source_row["supporting_facts"]
                )
            )
            if len(supporting_titles) != int(source_row["num_hops"]):
                raise ValueError(
                    f"support-title count changed for {task_id}"
                )
            diagnostics.append(
                analyze_task(
                    task_id=task_id,
                    num_hops=int(source_row["num_hops"]),
                    supporting_titles=supporting_titles,
                    candidate_titles=candidate_titles,
                    text_by_title={
                        title: str(text)
                        for title, text in text_by_title.items()
                    },
                )
            )
    finally:
        connection.close()

    summary = summarize(
        diagnostics,
        resolved_candidate_rows=resolved_candidate_rows,
    )
    integrity_gates = {
        "source_sha256_matches": True,
        "retrieval_sha256_matches": True,
        "database_sha256_matches": True,
        "manifest_sha256_matches": True,
        "opportunity_hash_matches": True,
    }
    summary["gates"] = {**integrity_gates, **summary["gates"]}
    summary["gates"]["all_pass"] = all(summary["gates"].values())
    return {
        "interface_version": "hover-path-opportunity-audit-1",
        "input_sha256": hashes,
        "summary": summary,
        "unresolved_rows": unresolved_rows,
        "diagnostics": diagnostics,
        "model_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit exact HoVer path-dependent retrieval opportunity."
    )
    parser.add_argument("--source-json", type=Path, required=True)
    parser.add_argument("--retrieval-json", type=Path, required=True)
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = load_and_audit(
        source_path=args.source_json,
        retrieval_path=args.retrieval_json,
        database_path=args.database,
        manifest_path=args.manifest,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
