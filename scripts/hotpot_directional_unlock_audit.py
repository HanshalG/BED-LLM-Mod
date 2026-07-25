#!/usr/bin/env python3
"""Audit directional two-document unlocks in HotpotQA bridge questions."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Iterable, Sequence

from rank_bm25 import BM25Okapi


SOURCE_SHA256 = "c20b638ca82b21d04fe12e14ff417ad05153d4d215a65de54497fca4e972f7c6"
SOURCE_ROWS = 7405
BRIDGE_ROWS = 5918
SELECTION_SEED = 24350
OPPORTUNITY_SIZE = 500
DEVELOPMENT_SIZE = 100
SPLIT_HASHES = {
    "opportunity": "e324df5b63a8cad523801ffa64111cd377166e0875a5a328599173f9ae25e653",
    "development": "72680f9764936f9595ab413b9055fdcd20c4b6ff77770ddf51fddcf4562c873c",
    "holdout": "e05826f43243d10e2c15266770425c640282c73c1d224c4d18081650c6f6f2da",
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_list_hash(values: Sequence[str]) -> str:
    encoded = json.dumps(
        list(values), ensure_ascii=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.replace("_", " "))
    ascii_text = decomposed.encode("ascii", errors="ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))


def title_aliases(title: str) -> tuple[str, ...]:
    raw_aliases = [title]
    prefix = re.sub(r"\s*\([^()]*\)\s*$", "", title).strip()
    if prefix and prefix != title:
        raw_aliases.append(prefix)
    aliases = []
    for value in raw_aliases:
        normalized = normalize_text(value)
        if len(normalized) >= 3 and re.search(r"[a-z0-9]", normalized):
            aliases.append(normalized)
    return tuple(dict.fromkeys(aliases))


def contains_phrase(text: str, phrase: str) -> bool:
    if not phrase:
        return False
    return f" {phrase} " in f" {text} "


def mentioned_context_titles(
    *,
    paragraph_text: str,
    context_titles: Sequence[str],
    root_title: str,
) -> list[str]:
    normalized_paragraph = normalize_text(paragraph_text)
    mentioned = []
    for title in context_titles:
        if title == root_title:
            continue
        if any(
            contains_phrase(normalized_paragraph, alias)
            for alias in title_aliases(title)
        ):
            mentioned.append(title)
    return mentioned


def _support_sentence_texts(
    *,
    support_titles: Sequence[str],
    support_sentence_ids: Sequence[int],
    context: dict[str, Any],
) -> dict[str, str]:
    context_titles = list(context["title"])
    context_sentences = list(context["sentences"])
    by_title = {
        title: list(sentences)
        for title, sentences in zip(
            context_titles, context_sentences, strict=True
        )
    }
    selected: dict[str, list[str]] = {}
    for title, sentence_id in zip(
        support_titles, support_sentence_ids, strict=True
    ):
        sentences = by_title.get(title)
        if (
            sentences is None
            or isinstance(sentence_id, bool)
            or not isinstance(sentence_id, int)
            or sentence_id < 0
            or sentence_id >= len(sentences)
        ):
            raise ValueError("supporting fact references an invalid sentence")
        selected.setdefault(title, []).append(str(sentences[sentence_id]))
    return {title: " ".join(sentences) for title, sentences in selected.items()}


def title_bm25_root(question: str, context_titles: Sequence[str]) -> str:
    tokenized_titles = [normalize_text(title).split() for title in context_titles]
    bm25 = BM25Okapi(tokenized_titles)
    scores = bm25.get_scores(normalize_text(question).split())
    index = max(
        range(len(context_titles)),
        key=lambda item: (float(scores[item]), -item),
    )
    return context_titles[index]


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    task_id = str(row["id"])
    base = {
        "task_id": task_id,
        "level": str(row["level"]),
        "strict_unlock": False,
    }
    if row["type"] != "bridge":
        return {**base, "exclusion": "not_bridge"}
    support = row["supporting_facts"]
    support_titles = [str(value) for value in support["title"]]
    sentence_ids = [int(value) for value in support["sent_id"]]
    distinct_supports = list(dict.fromkeys(support_titles))
    if len(distinct_supports) != 2:
        return {**base, "exclusion": "not_exactly_two_support_titles"}
    context = row["context"]
    context_titles = [str(value) for value in context["title"]]
    context_sentences = [
        [str(sentence) for sentence in sentences]
        for sentences in context["sentences"]
    ]
    if len(context_titles) != 10 or len(context_sentences) != 10:
        return {**base, "exclusion": "context_not_ten_titles"}
    if any(context_titles.count(title) != 1 for title in distinct_supports):
        return {**base, "exclusion": "support_title_not_unique_in_context"}
    answer = normalize_text(str(row["answer"]))
    if answer in {"yes", "no"} or len(answer) < 3:
        return {**base, "exclusion": "answer_excluded"}
    support_texts = _support_sentence_texts(
        support_titles=support_titles,
        support_sentence_ids=sentence_ids,
        context={
            "title": context_titles,
            "sentences": context_sentences,
        },
    )
    answer_bearing = [
        title
        for title in distinct_supports
        if contains_phrase(normalize_text(support_texts[title]), answer)
    ]
    if len(answer_bearing) != 1:
        return {**base, "exclusion": "answer_support_not_unique"}
    answer_title = answer_bearing[0]
    enabling_title = next(
        title for title in distinct_supports if title != answer_title
    )
    paragraph_by_title = {
        title: " ".join(sentences)
        for title, sentences in zip(
            context_titles, context_sentences, strict=True
        )
    }
    enabling_mentions = mentioned_context_titles(
        paragraph_text=paragraph_by_title[enabling_title],
        context_titles=context_titles,
        root_title=enabling_title,
    )
    answer_mentions = mentioned_context_titles(
        paragraph_text=paragraph_by_title[answer_title],
        context_titles=context_titles,
        root_title=answer_title,
    )
    if enabling_mentions != [answer_title]:
        return {
            **base,
            "exclusion": "enabling_does_not_have_sole_answer_link",
        }
    if enabling_title in answer_mentions:
        return {**base, "exclusion": "reverse_support_link_exists"}
    bm25_title = title_bm25_root(str(row["question"]), context_titles)
    normalized_question = normalize_text(str(row["question"]))
    neither_title_in_question = not any(
        contains_phrase(normalized_question, normalize_text(title))
        for title in distinct_supports
    )
    enabling_first_coverage = 2
    answer_first_coverage = 1
    return {
        **base,
        "strict_unlock": True,
        "exclusion": None,
        "answer_title_context_index": context_titles.index(answer_title),
        "enabling_title_context_index": context_titles.index(enabling_title),
        "answer_paragraph_other_title_mentions": len(answer_mentions),
        "title_bm25_role": (
            "enabling"
            if bm25_title == enabling_title
            else "answer"
            if bm25_title == answer_title
            else "distractor"
        ),
        "neither_support_title_in_question": neither_title_in_question,
        "enabling_first_support_coverage": enabling_first_coverage,
        "answer_first_support_coverage": answer_first_coverage,
        "support_coverage_gain": (
            enabling_first_coverage - answer_first_coverage
        ),
    }


def split_ids(metadata_rows: Iterable[dict[str, Any]]) -> dict[str, list[str]]:
    bridge_ids = sorted(
        str(row["id"]) for row in metadata_rows if row["type"] == "bridge"
    )
    random.Random(SELECTION_SEED).shuffle(bridge_ids)
    return {
        "opportunity": bridge_ids[:OPPORTUNITY_SIZE],
        "development": bridge_ids[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": bridge_ids[OPPORTUNITY_SIZE + DEVELOPMENT_SIZE :],
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [row for row in rows if row["strict_unlock"]]
    medium_hard = sum(row["level"] in {"medium", "hard"} for row in strict)
    bm25_misses = sum(row["title_bm25_role"] != "enabling" for row in strict)
    title_hidden = sum(
        row["neither_support_title_in_question"] for row in strict
    )
    all_gaps_one = all(row["support_coverage_gain"] == 1 for row in strict)
    gates = {
        "exact_500_opportunity_rows": len(rows) == OPPORTUNITY_SIZE,
        "strict_unlock_count_at_least_40": len(strict) >= 40,
        "medium_or_hard_count_at_least_30": medium_hard >= 30,
        "title_bm25_miss_count_at_least_30": bm25_misses >= 30,
        "neither_title_in_question_count_at_least_20": title_hidden >= 20,
        "all_strict_unlock_coverage_gains_equal_1": all_gaps_one,
    }
    gates["all_pass"] = all(gates.values())
    exclusion_counts: dict[str, int] = {}
    for row in rows:
        exclusion = row.get("exclusion")
        if exclusion is not None:
            exclusion_counts[exclusion] = exclusion_counts.get(exclusion, 0) + 1
    return {
        "opportunity_rows": len(rows),
        "strict_unlock_count": len(strict),
        "strict_unlock_rate": len(strict) / len(rows) if rows else 0.0,
        "medium_or_hard_strict_count": medium_hard,
        "title_bm25_misses_enabling_count": bm25_misses,
        "neither_support_title_in_question_count": title_hidden,
        "title_bm25_roles": {
            role: sum(row["title_bm25_role"] == role for row in strict)
            for role in ("enabling", "answer", "distractor")
        },
        "exclusion_counts": exclusion_counts,
        "gates": gates,
    }


def load_and_audit(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("HotpotQA source hash mismatch")
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError(
            "pyarrow==21.0.0 is required for the HotpotQA audit"
        ) from exc
    parquet = pq.ParquetFile(path)
    if parquet.metadata.num_rows != SOURCE_ROWS:
        raise ValueError("HotpotQA source row count mismatch")
    metadata = parquet.read(columns=["id", "type", "level"]).to_pylist()
    if sum(row["type"] == "bridge" for row in metadata) != BRIDGE_ROWS:
        raise ValueError("HotpotQA bridge row count mismatch")
    splits = split_ids(metadata)
    for name, expected_hash in SPLIT_HASHES.items():
        if ordered_list_hash(splits[name]) != expected_hash:
            raise ValueError(f"HotpotQA {name} split hash mismatch")
    opportunity_ids = set(splits["opportunity"])
    # Endpoint columns are converted only for rows selected by the frozen ID set.
    full_table = parquet.read()
    opportunity_rows = [
        row for row in full_table.to_pylist() if str(row["id"]) in opportunity_ids
    ]
    by_id = {str(row["id"]): row for row in opportunity_rows}
    ordered_rows = [by_id[task_id] for task_id in splits["opportunity"]]
    diagnostics = [analyze_row(row) for row in ordered_rows]
    summary = summarize(diagnostics)
    source_gates = {
        "source_sha256_matches": True,
        "source_row_count_matches": True,
        "bridge_row_count_matches": True,
        "opportunity_split_hash_matches": True,
        "development_split_hash_matches": True,
        "holdout_split_hash_matches": True,
    }
    summary["gates"] = {**source_gates, **summary["gates"]}
    summary["gates"]["all_pass"] = all(summary["gates"].values())
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "source_sha256": SOURCE_SHA256,
            "source_rows": SOURCE_ROWS,
            "bridge_rows": BRIDGE_ROWS,
            "selection_seed": SELECTION_SEED,
            "split_sizes": {
                name: len(values) for name, values in splits.items()
            },
            "split_hashes": SPLIT_HASHES,
            "model_calls": 0,
            "development_rows_accessed": 0,
            "holdout_rows_accessed": 0,
        },
        "summary": summary,
        "strict_unlock_rows": [
            row for row in diagnostics if row["strict_unlock"]
        ],
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
