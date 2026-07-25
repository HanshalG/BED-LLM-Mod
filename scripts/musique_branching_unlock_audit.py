#!/usr/bin/env python3
"""Audit two-retrieval unlocks in MuSiQue 4hop3 branching questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
from typing import Any, Iterable, Sequence

from rank_bm25 import BM25Okapi


SOURCE_SHA256 = "83a75b1e11e4e9bb8f8308e72ac40ca617ae4431b3a0d955b61cab259248490a"
SOURCE_ROWS = 19_938
ELIGIBLE_ROWS = 400
SELECTION_SEED = 24_353
OPPORTUNITY_SIZE = 120
DEVELOPMENT_SIZE = 40
SPLIT_HASHES = {
    "opportunity": "d11fbc929ebe04ca01e6fc88b7fe417e970a4ca5912b746a782ae51a3ed16423",
    "development": "619615104c1357f08b51b6363c225ded64313035899626d7d98c1a39117ff8ee",
    "holdout": "175a7dc642b88e843cbcfab8ae3a0ec44e036e93dcc40f9fcc18a3c56030d7b6",
}
EXPECTED_DEPENDENCIES = ((), (1,), (), (2, 3))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def ordered_list_hash(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def tokens(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", value.casefold())


def contains_token_phrase(text: str, phrase: str) -> bool:
    text_tokens = tokens(text)
    phrase_tokens = tokens(phrase)
    if not phrase_tokens:
        return False
    width = len(phrase_tokens)
    return any(
        text_tokens[index : index + width] == phrase_tokens
        for index in range(len(text_tokens) - width + 1)
    )


def dependency_sets(row: dict[str, Any]) -> tuple[tuple[int, ...], ...]:
    decomposition = row.get("question_decomposition")
    if not isinstance(decomposition, list):
        return ()
    return tuple(
        tuple(
            sorted(
                {
                    int(value)
                    for value in re.findall(
                        r"#(\d+)", str(step.get("question", ""))
                    )
                }
            )
        )
        for step in decomposition
        if isinstance(step, dict)
    )


def split_ids(metadata_ids: Iterable[str]) -> dict[str, list[str]]:
    eligible = sorted(
        task_id for task_id in metadata_ids if task_id.startswith("4hop3__")
    )
    random.Random(SELECTION_SEED).shuffle(eligible)
    return {
        "opportunity": eligible[:OPPORTUNITY_SIZE],
        "development": eligible[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": eligible[OPPORTUNITY_SIZE + DEVELOPMENT_SIZE :],
    }


def _bm25_diagnostics(
    *,
    question: str,
    paragraphs: Sequence[dict[str, Any]],
    support_indices: Sequence[int],
) -> dict[str, Any]:
    corpus = [
        tokens(
            f"{paragraph.get('title', '')} "
            f"{paragraph.get('paragraph_text', '')}"
        )
        for paragraph in paragraphs
    ]
    bm25 = BM25Okapi(corpus)
    scores = [float(value) for value in bm25.get_scores(tokens(question))]
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    roles_by_index = {
        support_indices[0]: "deep_root",
        support_indices[1]: "deep_child",
        support_indices[2]: "shallow_root",
        support_indices[3]: "final",
    }
    top_role = roles_by_index.get(order[0], "distractor")
    support_ranks = {
        role: order.index(index) + 1 for index, role in roles_by_index.items()
    }
    return {
        "bm25_top_role": top_role,
        "bm25_support_ranks": support_ranks,
        "shallow_scores_above_deep": (
            scores[support_indices[2]] > scores[support_indices[0]]
        ),
        "deep_root_not_rank_one": support_ranks["deep_root"] != 1,
        "top_is_shallow_or_distractor": top_role
        in {"shallow_root", "distractor"},
    }


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row.get("id", ""))
    base = {"task_id": task_id, "valid_branching_row": False}
    if not task_id.startswith("4hop3__"):
        return {**base, "exclusion": "wrong_id_family"}
    dependencies = dependency_sets(row)
    if dependencies != EXPECTED_DEPENDENCIES:
        return {**base, "exclusion": "wrong_dependency_graph"}
    decomposition = row["question_decomposition"]
    paragraphs = row.get("paragraphs")
    if not isinstance(paragraphs, list) or len(paragraphs) != 20:
        return {**base, "exclusion": "context_not_twenty_paragraphs"}
    support_indices = [
        step.get("paragraph_support_idx") for step in decomposition
    ]
    if (
        any(
            isinstance(index, bool)
            or not isinstance(index, int)
            or index < 0
            or index >= len(paragraphs)
            for index in support_indices
        )
        or len(set(support_indices)) != 4
    ):
        return {**base, "exclusion": "invalid_or_duplicate_support_indices"}

    question = str(row.get("question", ""))
    root_answers_hidden = all(
        not contains_token_phrase(question, str(decomposition[index]["answer"]))
        for index in (0, 2)
    )
    root_titles_hidden = all(
        not contains_token_phrase(
            question, str(paragraphs[support_indices[index]]["title"])
        )
        for index in (0, 2)
    )
    bm25 = _bm25_diagnostics(
        question=question,
        paragraphs=paragraphs,
        support_indices=support_indices,
    )
    deep_first_prefix = 2
    shallow_first_prefix = 1
    return {
        **base,
        "valid_branching_row": True,
        "exclusion": None,
        "dependencies": [list(values) for values in dependencies],
        "support_indices": support_indices,
        "root_answers_hidden": root_answers_hidden,
        "root_titles_hidden": root_titles_hidden,
        "deep_first_connected_prefix": deep_first_prefix,
        "shallow_first_connected_prefix": shallow_first_prefix,
        "connected_prefix_gap": deep_first_prefix - shallow_first_prefix,
        **bm25,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    valid = [row for row in rows if row["valid_branching_row"]]
    root_answers_hidden_count = sum(
        row["root_answers_hidden"] for row in valid
    )
    root_titles_hidden_count = sum(row["root_titles_hidden"] for row in valid)
    shallow_above_count = sum(
        row["shallow_scores_above_deep"] for row in valid
    )
    deep_not_top_count = sum(row["deep_root_not_rank_one"] for row in valid)
    shallow_or_distractor_top_count = sum(
        row["top_is_shallow_or_distractor"] for row in valid
    )
    all_gaps_one = (
        len(valid) == OPPORTUNITY_SIZE
        and all(row["connected_prefix_gap"] == 1 for row in valid)
    )
    gates = {
        "exact_120_opportunity_rows": len(rows) == OPPORTUNITY_SIZE,
        "all_rows_have_exact_branching_structure": len(valid)
        == OPPORTUNITY_SIZE,
        "all_connected_prefix_gaps_equal_one": all_gaps_one,
        "root_answers_hidden_count_at_least_110": (
            root_answers_hidden_count >= 110
        ),
        "shallow_scores_above_deep_count_at_least_45": (
            shallow_above_count >= 45
        ),
        "deep_root_not_rank_one_count_at_least_85": (
            deep_not_top_count >= 85
        ),
        "shallow_or_distractor_top_count_at_least_65": (
            shallow_or_distractor_top_count >= 65
        ),
    }
    gates["all_pass"] = all(gates.values())
    exclusions: dict[str, int] = {}
    for row in rows:
        exclusion = row.get("exclusion")
        if exclusion is not None:
            exclusions[exclusion] = exclusions.get(exclusion, 0) + 1
    return {
        "opportunity_rows": len(rows),
        "valid_branching_rows": len(valid),
        "root_answers_hidden_count": root_answers_hidden_count,
        "root_titles_hidden_count": root_titles_hidden_count,
        "shallow_scores_above_deep_count": shallow_above_count,
        "deep_root_not_rank_one_count": deep_not_top_count,
        "shallow_or_distractor_top_count": shallow_or_distractor_top_count,
        "bm25_top_roles": {
            role: sum(row["bm25_top_role"] == role for row in valid)
            for role in (
                "deep_root",
                "deep_child",
                "shallow_root",
                "final",
                "distractor",
            )
        },
        "mean_deep_root_rank": (
            sum(row["bm25_support_ranks"]["deep_root"] for row in valid)
            / len(valid)
            if valid
            else None
        ),
        "mean_shallow_root_rank": (
            sum(row["bm25_support_ranks"]["shallow_root"] for row in valid)
            / len(valid)
            if valid
            else None
        ),
        "exclusion_counts": exclusions,
        "gates": gates,
    }


def load_and_audit(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("MuSiQue train source hash mismatch")

    row_count = 0
    metadata_ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            row_count += 1
            metadata_ids.append(str(row["id"]))
    if row_count != SOURCE_ROWS:
        raise ValueError("MuSiQue train source row count mismatch")
    if sum(task_id.startswith("4hop3__") for task_id in metadata_ids) != ELIGIBLE_ROWS:
        raise ValueError("MuSiQue 4hop3 eligible row count mismatch")

    splits = split_ids(metadata_ids)
    for name, expected_hash in SPLIT_HASHES.items():
        if ordered_list_hash(splits[name]) != expected_hash:
            raise ValueError(f"MuSiQue {name} split hash mismatch")

    opportunity_ids = set(splits["opportunity"])
    by_id: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            task_id = str(row["id"])
            if task_id in opportunity_ids:
                by_id[task_id] = row
    if set(by_id) != opportunity_ids:
        raise ValueError("MuSiQue opportunity rows are missing")
    diagnostics = [
        analyze_row(by_id[task_id]) for task_id in splits["opportunity"]
    ]
    summary = summarize(diagnostics)
    source_gates = {
        "source_sha256_matches": True,
        "source_row_count_matches": True,
        "eligible_row_count_matches": True,
        "opportunity_split_hash_matches": True,
        "development_split_hash_matches": True,
        "holdout_split_hash_matches": True,
        "zero_development_rows_accessed": True,
        "zero_holdout_rows_accessed": True,
        "zero_model_calls": True,
    }
    summary["gates"] = {**source_gates, **summary["gates"]}
    summary["gates"]["all_pass"] = all(summary["gates"].values())
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "dataset": "MuSiQue-Ans v1.0 train",
            "source_sha256": SOURCE_SHA256,
            "source_rows": SOURCE_ROWS,
            "eligible_4hop3_rows": ELIGIBLE_ROWS,
            "selection_seed": SELECTION_SEED,
            "split_sizes": {
                name: len(values) for name, values in splits.items()
            },
            "split_hashes": SPLIT_HASHES,
            "development_rows_accessed": 0,
            "holdout_rows_accessed": 0,
            "model_calls": 0,
        },
        "summary": summary,
        "opportunity_diagnostics": diagnostics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = load_and_audit(args.data)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": payload["status"],
                "output": str(args.output),
                "summary": payload["summary"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
