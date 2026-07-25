#!/usr/bin/env python3
"""Audit path-dependent clarification opportunities in Interactive SWE-bench."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence


INTERFACE_VERSION = "interactive-swe-path-opportunity-1"
DATASET_REVISION = "a56830ede5eee0925cb7d735d422a207927925bd"
SOURCE_SHA256 = (
    "bdc6663d3f931d9ea04ea2edaec5eca6e406c2f122d7f1956ef0ec859691b5cd"
)
MANIFEST_SHA256 = (
    "7ac50a60bc59046b4406ef65b9ca148554b5b7c596194bd8d2ad66f33d811221"
)
SOURCE_ROWS = 500
SELECTION_SEED = 24_402
OPPORTUNITY_SIZE = 120
DEVELOPMENT_SIZE = 40
HOLDOUT_SIZE = 330
EXCLUDED_IDS = (
    "astropy__astropy-12907",
    "astropy__astropy-13033",
    "astropy__astropy-13236",
    "astropy__astropy-13398",
    "astropy__astropy-13453",
    "astropy__astropy-13579",
    "astropy__astropy-13977",
    "astropy__astropy-14096",
    "astropy__astropy-14182",
    "astropy__astropy-14309",
)
SPLIT_HASHES = {
    "eligible": "3bbd7753014cd1a36cf651c74d26dd5fdc9b0c7d3c0f57cefd64f50dd98e6954",
    "opportunity": "8f361dd221076a7a8bdb14a55978a469a3890f687289d44460322ff7c5fade0f",
    "development": "6f6bfdd0b0715b3cb7375ab7f3a2325f0d1a8e3111139c8fc464965542f842db",
    "holdout": "e945d95d8765d0c6fedb2c3d585ca5b00bc4b9f80f9f5166fa7df0dead238b0c",
}

ROOT_PROBES = (
    "expected desired behavior output result",
    "actual failure error exception traceback symptom",
    "reproduction example input output steps",
    "affected function class method api module component",
    "scope edge cases compatibility constraints",
    "version environment regression platform dependency",
    "implementation file module location change",
    "tests acceptance criteria expected cases",
)
MAX_NOVEL_TERMS = 10
MIN_TARGETS = 5
MIN_ROOT_ANSWERS = 3
MIN_USABLE = 60
MIN_STRICT = 18
MIN_STRICT_REPOS = 6
MIN_MEAN_NORMALIZED_GAIN = 0.025

STOPWORDS = frozenset(
    """
    a about above after again against all am an and any are as at be because
    been before being below between both but by can could did do does doing
    down during each few for from further had has have having he her here
    hers herself him himself his how i if in into is it its itself just me
    more most my myself no nor not now of off on once only or other our ours
    ourselves out over own same she should so some such than that the their
    theirs them themselves then there these they this those through to too
    under until up very was we were what when where which while who whom why
    will with would you your yours yourself yourselves
    issue problem fix change changes changed use uses using used support
    expected actual result error test tests code file files function class
    method module please help need needs make ensure
    """.split()
)


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


def text_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tokens(value: str) -> list[str]:
    return [
        token
        for token in re.findall(
            r"[a-z_][a-z0-9_]{2,}",
            value.casefold(),
        )
        if token not in STOPWORDS
    ]


def split_ids(instance_ids: Iterable[str]) -> dict[str, list[str]]:
    values = list(instance_ids)
    if len(values) != SOURCE_ROWS or len(set(values)) != SOURCE_ROWS:
        raise ValueError("unexpected Interactive SWE-bench ID universe")
    if tuple(values[: len(EXCLUDED_IDS)]) != EXCLUDED_IDS:
        raise ValueError("disclosed preview rows changed")
    eligible = sorted(set(values) - set(EXCLUDED_IDS))
    random.Random(SELECTION_SEED).shuffle(eligible)
    splits = {
        "eligible": eligible,
        "opportunity": eligible[:OPPORTUNITY_SIZE],
        "development": eligible[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": eligible[OPPORTUNITY_SIZE + DEVELOPMENT_SIZE :],
    }
    expected_sizes = {
        "eligible": SOURCE_ROWS - len(EXCLUDED_IDS),
        "opportunity": OPPORTUNITY_SIZE,
        "development": DEVELOPMENT_SIZE,
        "holdout": HOLDOUT_SIZE,
    }
    if {key: len(value) for key, value in splits.items()} != expected_sizes:
        raise ValueError("Interactive SWE-bench split sizes changed")
    if {
        key: ordered_hash(value) for key, value in splits.items()
    } != SPLIT_HASHES:
        raise ValueError("Interactive SWE-bench split hashes changed")
    return splits


def hidden_chunks(value: str) -> list[str]:
    pieces = re.split(
        r"\n\s*\n|(?<=[.!?])\s+(?=[A-Z`#])",
        value,
    )
    chunks = []
    for piece in pieces:
        normalized = " ".join(piece.split())
        if 20 <= len(normalized) <= 1_200 and tokens(normalized):
            chunks.append(normalized)
    return list(dict.fromkeys(chunks))


class BM25Index:
    def __init__(self, documents: Sequence[str]) -> None:
        self.documents = list(documents)
        self.document_tokens = [tokens(value) for value in documents]
        self.lengths = [len(value) for value in self.document_tokens]
        self.average_length = (
            sum(self.lengths) / len(self.lengths) if self.lengths else 0.0
        )
        self.term_frequencies = [Counter(value) for value in self.document_tokens]
        document_frequency: Counter[str] = Counter()
        for value in self.document_tokens:
            document_frequency.update(set(value))
        count = len(self.documents)
        self.idf = {
            token: math.log(1.0 + (count - frequency + 0.5) / (frequency + 0.5))
            for token, frequency in document_frequency.items()
        }

    def scores(self, query: str) -> list[float]:
        query_tokens = tokens(query)
        values = []
        for frequencies, length in zip(self.term_frequencies, self.lengths):
            score = 0.0
            for token in query_tokens:
                frequency = frequencies.get(token, 0)
                if not frequency:
                    continue
                denominator = frequency + 1.5 * (
                    1.0
                    - 0.75
                    + 0.75 * length / max(self.average_length, 1.0)
                )
                score += self.idf.get(token, 0.0) * (
                    frequency * 2.5 / denominator
                )
            values.append(score)
        return values

    def top(self, query: str, *, exclude: set[int] | None = None) -> int:
        excluded = exclude or set()
        candidates = [
            index for index in range(len(self.documents)) if index not in excluded
        ]
        if not candidates:
            raise ValueError("no eligible hidden issue chunk")
        scores = self.scores(query)
        return max(candidates, key=lambda index: (scores[index], -index))


@dataclass(frozen=True)
class RootBranch:
    root_index: int
    probe: str
    answer_index: int
    answer_hash: str
    continuation_indices: tuple[int, ...]
    continuation_query_hashes: tuple[str, ...]


def build_target_blind_tree(
    problem_statement: str,
    original_issue: str,
) -> tuple[list[str], list[RootBranch]]:
    chunks = hidden_chunks(original_issue)
    if not chunks:
        return [], []
    index = BM25Index(chunks)
    selected_answers: set[int] = set()
    roots = []
    visible = set(tokens(problem_statement))
    for probe in ROOT_PROBES:
        answer_index = index.top(probe)
        if answer_index in selected_answers:
            continue
        selected_answers.add(answer_index)
        answer = chunks[answer_index]
        novel = [
            token
            for token, _count in Counter(tokens(answer)).most_common()
            if token not in visible
        ][:MAX_NOVEL_TERMS]
        continuation_queries = list(novel)
        continuation_queries.extend(
            f"{continuation_probe} {' '.join(novel[:2])}".strip()
            for continuation_probe in ROOT_PROBES
        )
        continuation_indices = []
        continuation_hashes = []
        for query in continuation_queries:
            if len(chunks) < 2:
                break
            next_index = index.top(query, exclude={answer_index})
            if next_index in continuation_indices:
                continue
            continuation_indices.append(next_index)
            continuation_hashes.append(text_hash(query))
        roots.append(
            RootBranch(
                root_index=len(roots),
                probe=probe,
                answer_index=answer_index,
                answer_hash=text_hash(answer),
                continuation_indices=tuple(continuation_indices),
                continuation_query_hashes=tuple(continuation_hashes),
            )
        )
    return chunks, roots


def parse_files(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item) for item in value]
    try:
        parsed = json.loads(str(value))
    except json.JSONDecodeError:
        return [item.strip() for item in str(value).split(",") if item.strip()]
    if not isinstance(parsed, list):
        raise ValueError("Interactive SWE-bench files field is not a list")
    return [str(item) for item in parsed]


def hidden_gold_tokens(
    problem_statement: str,
    original_issue: str,
    patch: str,
    test_patch: str,
    files: Any,
) -> set[str]:
    changed_lines = []
    for value in (patch or "", test_patch or ""):
        changed_lines.extend(
            line[1:]
            for line in value.splitlines()
            if line.startswith(("+", "-"))
            and not line.startswith(("+++", "---"))
        )
    gold_text = " ".join(parse_files(files) + changed_lines)
    visible = set(tokens(problem_statement))
    available = set(tokens(original_issue))
    return (set(tokens(gold_text)) & available) - visible


def coverage(chunks: Sequence[str], indices: Iterable[int], target: set[str]) -> int:
    text = " ".join(chunks[index] for index in set(indices))
    return len(set(tokens(text)) & target)


def score_tree(
    chunks: Sequence[str],
    roots: Sequence[RootBranch],
    target: set[str],
) -> dict[str, Any]:
    rows = []
    for root in roots:
        immediate = coverage(chunks, [root.answer_index], target)
        continuation_values = [
            (
                coverage(
                    chunks,
                    [root.answer_index, continuation_index],
                    target,
                ),
                continuation_index,
            )
            for continuation_index in root.continuation_indices
        ]
        best_final, best_continuation = (
            max(continuation_values, key=lambda item: (item[0], -item[1]))
            if continuation_values
            else (immediate, root.answer_index)
        )
        rows.append(
            {
                "root_index": root.root_index,
                "answer_index": root.answer_index,
                "immediate_coverage": immediate,
                "best_final_coverage": best_final,
                "best_continuation_index": best_continuation,
            }
        )
    if not rows:
        return {"roots": [], "greedy": None, "depth_two": None}
    greedy = max(
        rows,
        key=lambda row: (
            row["immediate_coverage"],
            row["best_final_coverage"],
            -row["root_index"],
        ),
    )
    depth_two = max(
        rows,
        key=lambda row: (
            row["best_final_coverage"],
            row["immediate_coverage"],
            -row["root_index"],
        ),
    )
    return {"roots": rows, "greedy": greedy, "depth_two": depth_two}


def build_manifest(source: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    if sha256_file(source) != SOURCE_SHA256:
        raise ValueError("Interactive SWE-bench source hash changed")
    metadata = pq.read_table(source, columns=["instance_id", "repo"]).to_pylist()
    splits = split_ids(row["instance_id"] for row in metadata)
    repos = {row["instance_id"]: row["repo"] for row in metadata}
    return {
        "schema_version": 1,
        "status": "frozen",
        "interface_version": INTERFACE_VERSION,
        "dataset_revision": DATASET_REVISION,
        "source_sha256": SOURCE_SHA256,
        "selection_seed": SELECTION_SEED,
        "excluded_preview_ids": list(EXCLUDED_IDS),
        "split_hashes": SPLIT_HASHES,
        "splits": {
            key: [
                {"instance_id": instance_id, "repo": repos[instance_id]}
                for instance_id in values
            ]
            for key, values in splits.items()
            if key != "eligible"
        },
        "endpoint_columns_read": False,
        "api_calls": 0,
    }


def load_manifest(path: Path) -> dict[str, Any]:
    if sha256_file(path) != MANIFEST_SHA256:
        raise ValueError("Interactive SWE-bench manifest hash changed")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["interface_version"] != INTERFACE_VERSION:
        raise ValueError("Interactive SWE-bench manifest interface changed")
    if payload["source_sha256"] != SOURCE_SHA256:
        raise ValueError("Interactive SWE-bench manifest source changed")
    for split, expected_hash in SPLIT_HASHES.items():
        if split == "eligible":
            continue
        ids = [row["instance_id"] for row in payload["splits"][split]]
        if ordered_hash(ids) != expected_hash:
            raise ValueError(f"Interactive SWE-bench {split} manifest changed")
    return payload


def run_audit(source: Path, manifest_path: Path) -> dict[str, Any]:
    import pyarrow.parquet as pq

    manifest = load_manifest(manifest_path)
    opportunity_ids = [
        row["instance_id"] for row in manifest["splits"]["opportunity"]
    ]
    opportunity_set = set(opportunity_ids)

    # Freeze every action, observation, and continuation before endpoint columns.
    visible_rows = pq.read_table(
        source,
        columns=[
            "instance_id",
            "repo",
            "problem_statement",
            "original_issue",
        ],
    ).to_pylist()
    visible_by_id = {
        row["instance_id"]: row
        for row in visible_rows
        if row["instance_id"] in opportunity_set
    }
    trees = {}
    for instance_id in opportunity_ids:
        row = visible_by_id[instance_id]
        chunks, roots = build_target_blind_tree(
            row["problem_statement"],
            row["original_issue"],
        )
        trees[instance_id] = {"chunks": chunks, "roots": roots}

    endpoint_rows = pq.read_table(
        source,
        columns=["instance_id", "patch", "test_patch", "files"],
    ).to_pylist()
    endpoints = {
        row["instance_id"]: row
        for row in endpoint_rows
        if row["instance_id"] in opportunity_set
    }

    records = []
    for instance_id in opportunity_ids:
        visible = visible_by_id[instance_id]
        endpoint = endpoints[instance_id]
        tree = trees[instance_id]
        target = hidden_gold_tokens(
            visible["problem_statement"],
            visible["original_issue"],
            endpoint["patch"],
            endpoint["test_patch"],
            endpoint["files"],
        )
        scored = score_tree(tree["chunks"], tree["roots"], target)
        unique_answers = len(
            {root.answer_index for root in tree["roots"]}
        )
        usable = (
            len(target) >= MIN_TARGETS
            and unique_answers >= MIN_ROOT_ANSWERS
            and scored["greedy"] is not None
        )
        strict = False
        gain = 0
        normalized_gain = 0.0
        if usable:
            greedy = scored["greedy"]
            depth_two = scored["depth_two"]
            gain = (
                depth_two["best_final_coverage"]
                - greedy["best_final_coverage"]
            )
            normalized_gain = gain / len(target)
            strict = (
                depth_two["root_index"] != greedy["root_index"]
                and depth_two["immediate_coverage"]
                < greedy["immediate_coverage"]
                and gain > 0
            )
        records.append(
            {
                "instance_id": instance_id,
                "repo": visible["repo"],
                "chunk_count": len(tree["chunks"]),
                "target_count": len(target),
                "root_answer_count": unique_answers,
                "usable": usable,
                "strict_opportunity": strict,
                "depth_two_gain": gain,
                "normalized_depth_two_gain": normalized_gain,
                "greedy_root_index": (
                    scored["greedy"]["root_index"]
                    if scored["greedy"] is not None
                    else None
                ),
                "depth_two_root_index": (
                    scored["depth_two"]["root_index"]
                    if scored["depth_two"] is not None
                    else None
                ),
                "greedy_immediate_coverage": (
                    scored["greedy"]["immediate_coverage"]
                    if scored["greedy"] is not None
                    else None
                ),
                "depth_two_immediate_coverage": (
                    scored["depth_two"]["immediate_coverage"]
                    if scored["depth_two"] is not None
                    else None
                ),
                "greedy_final_coverage": (
                    scored["greedy"]["best_final_coverage"]
                    if scored["greedy"] is not None
                    else None
                ),
                "depth_two_final_coverage": (
                    scored["depth_two"]["best_final_coverage"]
                    if scored["depth_two"] is not None
                    else None
                ),
                "tree_digest": text_hash(
                    json.dumps(
                        [
                            {
                                "root_index": root.root_index,
                                "probe": root.probe,
                                "answer_index": root.answer_index,
                                "answer_hash": root.answer_hash,
                                "continuation_indices": root.continuation_indices,
                                "continuation_query_hashes": (
                                    root.continuation_query_hashes
                                ),
                            }
                            for root in tree["roots"]
                        ],
                        sort_keys=True,
                    )
                ),
            }
        )

    usable_rows = [row for row in records if row["usable"]]
    strict_rows = [row for row in records if row["strict_opportunity"]]
    metrics = {
        "opportunity_count": len(records),
        "usable_count": len(usable_rows),
        "strict_opportunity_count": len(strict_rows),
        "strict_repo_count": len({row["repo"] for row in strict_rows}),
        "mean_normalized_depth_two_gain": (
            sum(row["normalized_depth_two_gain"] for row in usable_rows)
            / len(usable_rows)
            if usable_rows
            else 0.0
        ),
        "mean_root_answer_count": (
            sum(row["root_answer_count"] for row in records) / len(records)
            if records
            else 0.0
        ),
    }
    gates = {
        "source_hash_reproduces": sha256_file(source) == SOURCE_SHA256,
        "opportunity_split_reproduces": (
            ordered_hash(opportunity_ids) == SPLIT_HASHES["opportunity"]
        ),
        "exact_opportunity_count": len(records) == OPPORTUNITY_SIZE,
        "usable_count_at_least_60": metrics["usable_count"] >= MIN_USABLE,
        "strict_opportunities_at_least_18": (
            metrics["strict_opportunity_count"] >= MIN_STRICT
        ),
        "strict_repositories_at_least_6": (
            metrics["strict_repo_count"] >= MIN_STRICT_REPOS
        ),
        "mean_normalized_gain_at_least_0_025": (
            metrics["mean_normalized_depth_two_gain"]
            >= MIN_MEAN_NORMALIZED_GAIN
        ),
        "all_strict_rows_sacrifice_immediate_and_gain_final": all(
            row["depth_two_immediate_coverage"]
            < row["greedy_immediate_coverage"]
            and row["depth_two_final_coverage"] > row["greedy_final_coverage"]
            for row in strict_rows
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "interface_version": INTERFACE_VERSION,
        "protocol": {
            "dataset_revision": DATASET_REVISION,
            "source_sha256": SOURCE_SHA256,
            "selection_seed": SELECTION_SEED,
            "split_hashes": SPLIT_HASHES,
            "root_probes": ROOT_PROBES,
            "max_novel_terms": MAX_NOVEL_TERMS,
            "min_targets": MIN_TARGETS,
            "min_root_answers": MIN_ROOT_ANSWERS,
            "endpoint_columns_loaded_after_trees_froze": True,
            "api_calls": 0,
            "oatml_used": False,
        },
        "metrics": metrics,
        "gates": gates,
        "records": records,
    }


def checkpoint(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--manifest-output", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audit-output", type=Path)
    args = parser.parse_args()
    if args.manifest_output is not None:
        manifest = build_manifest(args.source)
        checkpoint(args.manifest_output, manifest)
        print(
            json.dumps(
                {
                    "status": manifest["status"],
                    "split_sizes": {
                        key: len(value)
                        for key, value in manifest["splits"].items()
                    },
                    "split_hashes": manifest["split_hashes"],
                },
                indent=2,
                sort_keys=True,
            )
        )
        return
    if args.manifest is None or args.audit_output is None:
        parser.error("--manifest and --audit-output are required for an audit")
    result = run_audit(args.source, args.manifest)
    checkpoint(args.audit_output, result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": result["metrics"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
