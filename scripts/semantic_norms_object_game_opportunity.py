#!/usr/bin/env python3
"""Zero-call opportunity gate for a human-norm semantic object game."""

from __future__ import annotations

import argparse
from collections import Counter
import csv
from functools import lru_cache
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SOURCE_COMMIT = "d2b940472bb9e88983a5c087594022939cac8895"
SOURCE_FILES = {
    "data/aaltoprod/correspondence.csv": (
        "e922d57ff87fc6c8a4f2276a589c1645386c0e8f296eca6722998dae35763821"
    ),
    "data/aaltoprod/vocab.csv": (
        "a9b4ef36a118e1a23371597e673f20cbb058a00472e8c4cbf5703e3b2154156d"
    ),
    "data/aaltoprod/vectors.csv": (
        "41b620c391b9147a12d63dc9ed1af619e636bd6db4532e56c71a1b144bc54f99"
    ),
    "data/aaltoprod/features.csv": (
        "1d838554bed2dce20adac939d543d87c81947288801367ed568cfd7fc61adf88"
    ),
}
SELECTION_SEED = 40400
NUM_CATEGORIES = 8
OBJECTS_PER_CATEGORY = 4
NUM_OBJECTS = NUM_CATEGORIES * OBJECTS_PER_CATEGORY
MIN_TARGET_MEMBERS = 3
MAX_TARGET_MEMBERS = NUM_OBJECTS - 3

GATES = {
    "eligible_objects_min": 96,
    "unique_target_extensions_min": 64,
    "mid_support_extensions_min": 16,
    "depth3_entropy_gain_nats_min": 0.005,
    "depth3_brier_gain_min": 0.001,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source(source_root: Path) -> dict[str, str]:
    actual: dict[str, str] = {}
    for relative, expected in SOURCE_FILES.items():
        path = source_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing pinned source file: {path}")
        actual[relative] = _sha256(path)
        if actual[relative] != expected:
            raise ValueError(
                f"source hash mismatch for {relative}: "
                f"{actual[relative]} != {expected}"
            )
    git_head = source_root / ".git" / "HEAD"
    if git_head.is_file():
        head = git_head.read_text(encoding="utf-8").strip()
        if len(head) == 40 and head != SOURCE_COMMIT:
            raise ValueError(f"source commit mismatch: {head} != {SOURCE_COMMIT}")
    return actual


def read_correspondence(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_vocab(path: Path) -> list[str]:
    with path.open(encoding="utf-8-sig") as handle:
        return [line.strip() for line in handle if line.strip()]


def eligible_objects(
    correspondence: Sequence[Mapping[str, str]],
    vocab: Sequence[str],
) -> list[dict[str, Any]]:
    by_finnish: dict[str, list[Mapping[str, str]]] = {}
    for row in correspondence:
        key = row.get("aaltoprod", "").strip().casefold()
        if key:
            by_finnish.setdefault(key, []).append(row)

    english_counts = Counter(
        row.get("eng_name", "").strip().casefold()
        for row in correspondence
        if row.get("eng_name", "").strip()
    )
    eligible: list[dict[str, Any]] = []
    for vector_row, finnish_name in enumerate(vocab):
        rows = by_finnish.get(finnish_name.casefold(), [])
        if len(rows) != 1:
            continue
        row = rows[0]
        english_name = row.get("eng_name", "").strip()
        category = row.get("category", "").strip().casefold()
        if (
            not english_name
            or english_counts[english_name.casefold()] != 1
            or category in {"", "abstract"}
            or row.get("word_class", "").strip().casefold() != "noun"
            or row.get("homonym_fin", "").strip()
            or row.get("homonym_eng", "").strip()
        ):
            continue
        eligible.append(
            {
                "source_id": row.get("id", "").strip(),
                "finnish_name": finnish_name,
                "english_name": english_name,
                "category": category,
                "vector_row": vector_row,
            }
        )
    return eligible


def select_object_universe(
    eligible: Sequence[Mapping[str, Any]],
    *,
    seed: int = SELECTION_SEED,
    num_categories: int = NUM_CATEGORIES,
    per_category: int = OBJECTS_PER_CATEGORY,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for item in eligible:
        grouped.setdefault(str(item["category"]), []).append(item)
    category_order = sorted(
        (
            (category, items)
            for category, items in grouped.items()
            if len(items) >= per_category
        ),
        key=lambda pair: (-len(pair[1]), pair[0]),
    )
    if len(category_order) < num_categories:
        raise ValueError(
            f"only {len(category_order)} categories have {per_category}+ objects"
        )

    selected: list[dict[str, Any]] = []
    for category, items in category_order[:num_categories]:
        ordered = sorted(
            items,
            key=lambda item: hashlib.sha256(
                (
                    f"{seed}|{category}|{item['english_name']}|"
                    f"{item['finnish_name']}"
                ).encode("utf-8")
            ).hexdigest(),
        )
        selected.extend(dict(item) for item in ordered[:per_category])
    return selected


def read_selected_vector_rows(
    path: Path,
    selected_rows: Iterable[int],
) -> dict[int, tuple[float, ...]]:
    wanted = set(selected_rows)
    found: dict[int, tuple[float, ...]] = {}
    with path.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for row_index, row in enumerate(reader):
            if row_index in wanted:
                found[row_index] = tuple(float(value) for value in row)
    missing = wanted - set(found)
    if missing:
        raise ValueError(f"missing vector rows: {sorted(missing)}")
    return found


def feature_extensions(
    selected: Sequence[Mapping[str, Any]],
    vector_rows: Mapping[int, Sequence[float]],
    *,
    min_members: int = MIN_TARGET_MEMBERS,
    max_members: int = MAX_TARGET_MEMBERS,
) -> tuple[tuple[int, ...], dict[str, int]]:
    rows = [vector_rows[int(item["vector_row"])] for item in selected]
    widths = {len(row) for row in rows}
    if len(widths) != 1:
        raise ValueError(f"inconsistent vector widths: {sorted(widths)}")

    extension_counts: Counter[int] = Counter()
    for column in range(next(iter(widths))):
        mask = 0
        for object_index, row in enumerate(rows):
            if row[column] > 0.0:
                mask |= 1 << object_index
        members = mask.bit_count()
        if min_members <= members <= max_members:
            extension_counts[mask] += 1
    unique = tuple(sorted(extension_counts))
    return unique, {
        "eligible_feature_columns": sum(extension_counts.values()),
        "duplicate_feature_columns": sum(extension_counts.values()) - len(unique),
    }


def split_state(
    state: tuple[int, ...],
    extensions: Sequence[int],
    query: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    no = tuple(index for index in state if not (extensions[index] >> query) & 1)
    yes = tuple(index for index in state if (extensions[index] >> query) & 1)
    return no, yes


def _terminal_entropy(state: Sequence[int]) -> float:
    return math.log(len(state)) if state else 0.0


def _terminal_brier(state: Sequence[int]) -> float:
    return 1.0 - (1.0 / len(state)) if state else 0.0


def build_policy_evaluator(
    extensions: Sequence[int],
    *,
    exact: bool,
):
    extensions = tuple(extensions)

    @lru_cache(maxsize=None)
    def decision(
        state: tuple[int, ...],
        available: tuple[int, ...],
        depth: int,
    ) -> tuple[float, int | None]:
        if depth <= 0 or len(state) <= 1 or not available:
            return _terminal_entropy(state), None
        total = len(state)
        best_value = math.inf
        best_query: int | None = None
        for query in available:
            no, yes = split_state(state, extensions, query)
            if not no or not yes:
                continue
            next_available = tuple(item for item in available if item != query)
            if exact:
                no_value = decision(no, next_available, depth - 1)[0]
                yes_value = decision(yes, next_available, depth - 1)[0]
            else:
                no_value = _terminal_entropy(no)
                yes_value = _terminal_entropy(yes)
            value = (len(no) * no_value + len(yes) * yes_value) / total
            if value < best_value - 1e-12:
                best_value = value
                best_query = query
        if best_query is None:
            return _terminal_entropy(state), None
        if exact:
            return best_value, best_query
        no, yes = split_state(state, extensions, best_query)
        next_available = tuple(item for item in available if item != best_query)
        recursive = (
            len(no) * decision(no, next_available, depth - 1)[0]
            + len(yes) * decision(yes, next_available, depth - 1)[0]
        ) / total
        return recursive, best_query

    @lru_cache(maxsize=None)
    def brier(
        state: tuple[int, ...],
        available: tuple[int, ...],
        depth: int,
    ) -> float:
        if depth <= 0 or len(state) <= 1 or not available:
            return _terminal_brier(state)
        _, query = decision(state, available, depth)
        if query is None:
            return _terminal_brier(state)
        no, yes = split_state(state, extensions, query)
        next_available = tuple(item for item in available if item != query)
        return (
            len(no) * brier(no, next_available, depth - 1)
            + len(yes) * brier(yes, next_available, depth - 1)
        ) / len(state)

    return decision, brier


def evaluate_planning(
    extensions: Sequence[int],
    *,
    num_queries: int,
    depths: Sequence[int] = (1, 2, 3),
) -> dict[str, Any]:
    state = tuple(range(len(extensions)))
    available = tuple(range(num_queries))
    exact_decision, exact_brier = build_policy_evaluator(extensions, exact=True)
    greedy_decision, greedy_brier = build_policy_evaluator(extensions, exact=False)
    result: dict[str, Any] = {}
    for depth in depths:
        exact_entropy, exact_root = exact_decision(state, available, depth)
        greedy_entropy, greedy_root = greedy_decision(state, available, depth)
        result[f"depth{depth}"] = {
            "exact_root_index": exact_root,
            "greedy_root_index": greedy_root,
            "root_differs": exact_root != greedy_root,
            "exact_terminal_entropy_nats": exact_entropy,
            "greedy_terminal_entropy_nats": greedy_entropy,
            "entropy_gain_nats": greedy_entropy - exact_entropy,
            "exact_terminal_brier": exact_brier(state, available, depth),
            "greedy_terminal_brier": greedy_brier(state, available, depth),
            "brier_gain": (
                greedy_brier(state, available, depth)
                - exact_brier(state, available, depth)
            ),
        }
    return result


def summarize(
    *,
    eligible: Sequence[Mapping[str, Any]],
    selected: Sequence[Mapping[str, Any]],
    extensions: Sequence[int],
    extension_stats: Mapping[str, int],
    planning: Mapping[str, Any],
    source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    support_sizes = [mask.bit_count() for mask in extensions]
    mid_support_count = sum(8 <= count <= 24 for count in support_sizes)
    depth3 = planning["depth3"]
    checks = {
        "eligible_objects": len(eligible) >= GATES["eligible_objects_min"],
        "selected_object_count": len(selected) == NUM_OBJECTS,
        "selected_category_count": (
            len({str(item["category"]) for item in selected}) == NUM_CATEGORIES
        ),
        "unique_target_extensions": (
            len(extensions) >= GATES["unique_target_extensions_min"]
        ),
        "mid_support_extensions": (
            mid_support_count >= GATES["mid_support_extensions_min"]
        ),
        "depth3_root_differs": bool(depth3["root_differs"]),
        "depth3_entropy_gain": (
            depth3["entropy_gain_nats"]
            >= GATES["depth3_entropy_gain_nats_min"]
        ),
        "depth3_brier_gain": (
            depth3["brier_gain"] >= GATES["depth3_brier_gain_min"]
        ),
    }
    public_objects = [
        {
            "source_id": item["source_id"],
            "english_name": item["english_name"],
            "category": item["category"],
            "vector_row": item["vector_row"],
        }
        for item in selected
    ]
    return {
        "schema_version": 1,
        "status": "passed" if all(checks.values()) else "opportunity_failed",
        "source": {
            "repository": "https://github.com/AaltoImagingLanguage/Norms",
            "commit": SOURCE_COMMIT,
            "file_sha256": dict(source_hashes),
            "publication_doi": "10.3758/s13428-023-02311-1",
            "raw_data_redistributed": False,
        },
        "selection": {
            "seed": SELECTION_SEED,
            "num_categories": NUM_CATEGORIES,
            "objects_per_category": OBJECTS_PER_CATEGORY,
            "eligible_object_count": len(eligible),
            "objects": public_objects,
            "object_order_sha256": hashlib.sha256(
                "\n".join(
                    f"{item['source_id']}|{item['english_name']}|{item['category']}"
                    for item in selected
                ).encode("utf-8")
            ).hexdigest(),
        },
        "target_bank": {
            "unique_extension_count": len(extensions),
            "mid_support_extension_count": mid_support_count,
            "support_size_min": min(support_sizes) if support_sizes else None,
            "support_size_median": (
                sorted(support_sizes)[len(support_sizes) // 2]
                if support_sizes
                else None
            ),
            "support_size_max": max(support_sizes) if support_sizes else None,
            "extension_bank_sha256": hashlib.sha256(
                "\n".join(f"{mask:0{NUM_OBJECTS}b}" for mask in extensions).encode(
                    "ascii"
                )
            ).hexdigest(),
            **dict(extension_stats),
        },
        "planning": dict(planning),
        "gates": dict(GATES),
        "checks": checks,
    }


def run(source_root: Path) -> dict[str, Any]:
    source_hashes = validate_source(source_root)
    data_root = source_root / "data" / "aaltoprod"
    correspondence = read_correspondence(data_root / "correspondence.csv")
    vocab = read_vocab(data_root / "vocab.csv")
    eligible = eligible_objects(correspondence, vocab)
    selected = select_object_universe(eligible)
    vectors = read_selected_vector_rows(
        data_root / "vectors.csv",
        (int(item["vector_row"]) for item in selected),
    )
    extensions, extension_stats = feature_extensions(selected, vectors)
    planning = evaluate_planning(extensions, num_queries=len(selected))
    for depth_result in planning.values():
        for prefix in ("exact", "greedy"):
            index = depth_result[f"{prefix}_root_index"]
            depth_result[f"{prefix}_root_object"] = (
                selected[index]["english_name"] if index is not None else None
            )
    return summarize(
        eligible=eligible,
        selected=selected,
        extensions=extensions,
        extension_stats=extension_stats,
        planning=planning,
        source_hashes=source_hashes,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Hash-pinned checkout of AaltoImagingLanguage/Norms.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result = run(args.source_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
