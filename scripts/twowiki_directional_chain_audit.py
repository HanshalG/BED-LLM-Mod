#!/usr/bin/env python3
"""Audit directional two-document chains in 2WikiMultiHopQA."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata
from typing import Any, Iterable, Iterator, Sequence

from rank_bm25 import BM25Okapi


SOURCE_SHA256 = (
    "b318dbafbfed51a8029718fa59be8b616600cbff675a3b587694b28c5eedfc13"
)
SOURCE_ROWS = 167_454
TYPE_COUNTS = {
    "bridge_comparison": 34_631,
    "comparison": 51_963,
    "compositional": 76_481,
    "inference": 4_379,
}
ELIGIBLE_TYPES = frozenset({"compositional", "inference"})
ELIGIBLE_ROWS = 80_860
SELECTION_SEED = 24_360
OPPORTUNITY_SIZE = 500
DEVELOPMENT_SIZE = 100
HOLDOUT_SIZE = 1_000
SPLIT_HASHES = {
    "opportunity": (
        "a549809258236c603e1abd11f52409fd0e0a75ff5077bd798ca61d9ad4466fb2"
    ),
    "development": (
        "4b8ad7e698f8b1201cfebb0f7c663cd0d55c84fd368f4dff9a32f63cb04667a1"
    ),
    "holdout": (
        "dedde590777821a56805d83d486e10975f1a733c8ee8aadeb56fb192bae11d58"
    ),
}
SPLIT_TYPE_COUNTS = {
    "opportunity": {"compositional": 471, "inference": 29},
    "development": {"compositional": 94, "inference": 6},
    "holdout": {"compositional": 964, "inference": 36},
}


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


def iter_json_array(
    path: Path, *, chunk_size: int = 1024 * 1024
) -> Iterator[dict[str, Any]]:
    """Decode a top-level JSON array without retaining the complete dataset."""
    decoder = json.JSONDecoder()
    buffer = ""
    started = False
    eof = False
    with path.open(encoding="utf-8") as handle:
        while True:
            buffer = buffer.lstrip()
            if not started:
                if not buffer and not eof:
                    block = handle.read(chunk_size)
                    eof = block == ""
                    buffer += block
                    continue
                if not buffer or buffer[0] != "[":
                    raise ValueError("expected a top-level JSON array")
                buffer = buffer[1:]
                started = True
                continue

            buffer = buffer.lstrip()
            if buffer.startswith(","):
                buffer = buffer[1:]
                continue
            if buffer.startswith("]"):
                if buffer[1:].strip():
                    raise ValueError("unexpected data after JSON array")
                return
            if not buffer:
                if eof:
                    raise ValueError("unterminated JSON array")
                block = handle.read(chunk_size)
                eof = block == ""
                buffer += block
                continue
            try:
                value, end = decoder.raw_decode(buffer)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError("invalid or truncated JSON array") from None
                block = handle.read(chunk_size)
                eof = block == ""
                buffer += block
                continue
            if not isinstance(value, dict):
                raise ValueError("dataset array entries must be objects")
            yield value
            buffer = buffer[end:]


def normalize_text(value: str) -> str:
    decomposed = unicodedata.normalize("NFKD", value.replace("_", " "))
    ascii_text = decomposed.encode("ascii", errors="ignore").decode("ascii")
    return " ".join(re.findall(r"[a-z0-9]+", ascii_text.casefold()))


def tokens(value: str) -> list[str]:
    return normalize_text(value).split()


def contains_phrase(text: str, phrase: str) -> bool:
    normalized_text = normalize_text(text)
    normalized_phrase = normalize_text(phrase)
    if not normalized_phrase:
        return False
    return f" {normalized_phrase} " in f" {normalized_text} "


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


def split_ids(
    metadata_rows: Iterable[tuple[str, str]],
) -> dict[str, list[str]]:
    eligible = sorted(
        (task_id, task_type)
        for task_id, task_type in metadata_rows
        if task_type in ELIGIBLE_TYPES
    )
    random.Random(SELECTION_SEED).shuffle(eligible)
    ids = [task_id for task_id, _task_type in eligible]
    return {
        "opportunity": ids[:OPPORTUNITY_SIZE],
        "development": ids[
            OPPORTUNITY_SIZE : OPPORTUNITY_SIZE + DEVELOPMENT_SIZE
        ],
        "holdout": ids[
            OPPORTUNITY_SIZE
            + DEVELOPMENT_SIZE : OPPORTUNITY_SIZE
            + DEVELOPMENT_SIZE
            + HOLDOUT_SIZE
        ],
    }


def _unique_support_title(
    entity: str, support_titles: Sequence[str]
) -> str | None:
    normalized = normalize_text(entity)
    matches = [
        title for title in support_titles if normalized in title_aliases(title)
    ]
    return matches[0] if len(matches) == 1 else None


def _support_sentence_texts(
    *,
    supporting_facts: Sequence[Sequence[Any]],
    sentences_by_title: dict[str, list[str]],
) -> dict[str, str]:
    selected: dict[str, list[str]] = {}
    for fact in supporting_facts:
        if not isinstance(fact, list) or len(fact) != 2:
            raise ValueError("invalid supporting-fact row")
        title = str(fact[0])
        sentence_id = fact[1]
        sentences = sentences_by_title.get(title)
        if (
            sentences is None
            or isinstance(sentence_id, bool)
            or not isinstance(sentence_id, int)
            or sentence_id < 0
            or sentence_id >= len(sentences)
        ):
            raise ValueError("supporting fact references an invalid sentence")
        selected.setdefault(title, []).append(sentences[sentence_id])
    return {title: " ".join(values) for title, values in selected.items()}


def _bm25_order(
    *, question: str, corpus: Sequence[str]
) -> tuple[list[int], list[float]]:
    model = BM25Okapi([tokens(value) for value in corpus])
    scores = [
        float(value) for value in model.get_scores(tokens(question))
    ]
    order = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    return order, scores


def _base_result(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "task_id": str(row.get("_id", "")),
        "task_type": str(row.get("type", "")),
        "exact_two_triple_chain": False,
        "strict_directional_chain": False,
    }


def analyze_row(row: dict[str, Any]) -> dict[str, Any]:
    base = _base_result(row)
    if row.get("type") not in ELIGIBLE_TYPES:
        return {**base, "exclusion": "ineligible_type"}

    evidences = row.get("evidences")
    if (
        not isinstance(evidences, list)
        or len(evidences) != 2
        or any(not isinstance(item, list) or len(item) != 3 for item in evidences)
    ):
        return {**base, "exclusion": "not_two_evidence_triples"}
    first = [str(value) for value in evidences[0]]
    second = [str(value) for value in evidences[1]]
    if normalize_text(first[2]) != normalize_text(second[0]):
        return {**base, "exclusion": "evidence_triples_do_not_chain"}

    context = row.get("context")
    if not isinstance(context, list) or len(context) != 10:
        return {**base, "exclusion": "context_not_ten_documents"}
    context_titles: list[str] = []
    sentences_by_title: dict[str, list[str]] = {}
    for item in context:
        if not isinstance(item, list) or len(item) != 2:
            return {**base, "exclusion": "invalid_context_document"}
        title = str(item[0])
        sentences = item[1]
        if not isinstance(sentences, list):
            return {**base, "exclusion": "invalid_context_sentences"}
        context_titles.append(title)
        sentences_by_title[title] = [str(value) for value in sentences]

    supporting_facts = row.get("supporting_facts")
    if not isinstance(supporting_facts, list):
        return {**base, "exclusion": "invalid_supporting_facts"}
    support_titles = list(dict.fromkeys(str(item[0]) for item in supporting_facts))
    if (
        len(support_titles) != 2
        or any(context_titles.count(title) != 1 for title in support_titles)
    ):
        return {**base, "exclusion": "support_titles_not_two_unique_documents"}

    root_title = _unique_support_title(first[0], support_titles)
    child_title = _unique_support_title(first[2], support_titles)
    if (
        root_title is None
        or child_title is None
        or root_title == child_title
    ):
        return {**base, "exclusion": "evidence_entities_do_not_map_to_supports"}

    exact = {
        **base,
        "exact_two_triple_chain": True,
        "root_context_index": context_titles.index(root_title),
        "child_context_index": context_titles.index(child_title),
    }
    answer = normalize_text(str(row.get("answer", "")))
    if answer in {"yes", "no"} or len(answer) < 3:
        return {**exact, "exclusion": "answer_excluded"}

    try:
        support_texts = _support_sentence_texts(
            supporting_facts=supporting_facts,
            sentences_by_title=sentences_by_title,
        )
    except ValueError:
        return {**exact, "exclusion": "invalid_support_sentence_reference"}
    answer_bearing = [
        title
        for title in support_titles
        if contains_phrase(support_texts.get(title, ""), answer)
    ]
    if answer_bearing != [child_title]:
        return {**exact, "exclusion": "answer_not_unique_to_child_support"}

    root_text = " ".join(sentences_by_title[root_title])
    child_text = " ".join(sentences_by_title[child_title])
    root_mentions_child = contains_phrase(root_text, first[2]) or any(
        contains_phrase(root_text, alias) for alias in title_aliases(child_title)
    )
    child_mentions_root = contains_phrase(child_text, first[0]) or any(
        contains_phrase(child_text, alias) for alias in title_aliases(root_title)
    )
    if not root_mentions_child:
        return {**exact, "exclusion": "root_does_not_expose_child"}
    if child_mentions_root:
        return {**exact, "exclusion": "reverse_child_to_root_link_exists"}

    question = str(row.get("question", ""))
    title_order, title_scores = _bm25_order(
        question=question, corpus=context_titles
    )
    paragraph_corpus = [
        f"{title} {' '.join(sentences_by_title[title])}"
        for title in context_titles
    ]
    paragraph_order, paragraph_scores = _bm25_order(
        question=question, corpus=paragraph_corpus
    )
    root_index = context_titles.index(root_title)
    child_index = context_titles.index(child_title)
    question_names_root = contains_phrase(question, first[0]) or any(
        contains_phrase(question, alias) for alias in title_aliases(root_title)
    )
    return {
        **exact,
        "strict_directional_chain": True,
        "exclusion": None,
        "question_names_setup_root": question_names_root,
        "title_bm25_root_rank": title_order.index(root_index) + 1,
        "title_bm25_child_rank": title_order.index(child_index) + 1,
        "title_bm25_top_role": (
            "setup_root"
            if title_order[0] == root_index
            else "answer_child"
            if title_order[0] == child_index
            else "distractor"
        ),
        "title_bm25_child_minus_root_score": (
            title_scores[child_index] - title_scores[root_index]
        ),
        "paragraph_bm25_root_rank": paragraph_order.index(root_index) + 1,
        "paragraph_bm25_child_rank": paragraph_order.index(child_index) + 1,
        "paragraph_bm25_top_role": (
            "setup_root"
            if paragraph_order[0] == root_index
            else "answer_child"
            if paragraph_order[0] == child_index
            else "distractor"
        ),
        "paragraph_bm25_child_minus_root_score": (
            paragraph_scores[child_index] - paragraph_scores[root_index]
        ),
        "setup_first_support_coverage": 2,
        "child_first_support_coverage": 1,
        "support_coverage_gain": 1,
    }


def summarize(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    exact = [row for row in rows if row["exact_two_triple_chain"]]
    strict = [row for row in rows if row["strict_directional_chain"]]
    unnamed = sum(not row["question_names_setup_root"] for row in strict)
    title_misses = sum(row["title_bm25_root_rank"] != 1 for row in strict)
    paragraph_misses = sum(
        row["paragraph_bm25_root_rank"] != 1 for row in strict
    )
    all_gaps_one = bool(strict) and all(
        row["support_coverage_gain"] == 1 for row in strict
    )
    gates = {
        "exact_500_opportunity_rows": len(rows) == OPPORTUNITY_SIZE,
        "exact_two_triple_chain_count_at_least_450": len(exact) >= 450,
        "strict_directional_chain_count_at_least_300": len(strict) >= 300,
        "all_strict_chain_coverage_gains_equal_one": all_gaps_one,
        "unnamed_setup_root_count_at_least_50": unnamed >= 50,
        "title_bm25_setup_root_miss_count_at_least_75": title_misses >= 75,
        "paragraph_bm25_setup_root_miss_count_at_least_75": (
            paragraph_misses >= 75
        ),
    }
    gates["all_pass"] = all(gates.values())
    exclusion_counts: dict[str, int] = {}
    for row in rows:
        exclusion = row.get("exclusion")
        if exclusion is not None:
            exclusion_counts[exclusion] = exclusion_counts.get(exclusion, 0) + 1
    return {
        "opportunity_rows": len(rows),
        "exact_two_triple_chain_count": len(exact),
        "strict_directional_chain_count": len(strict),
        "strict_directional_chain_rate": len(strict) / len(rows) if rows else 0.0,
        "unnamed_setup_root_count": unnamed,
        "title_bm25_setup_root_miss_count": title_misses,
        "paragraph_bm25_setup_root_miss_count": paragraph_misses,
        "title_bm25_top_roles": {
            role: sum(row["title_bm25_top_role"] == role for row in strict)
            for role in ("setup_root", "answer_child", "distractor")
        },
        "paragraph_bm25_top_roles": {
            role: sum(row["paragraph_bm25_top_role"] == role for row in strict)
            for role in ("setup_root", "answer_child", "distractor")
        },
        "exclusion_counts": exclusion_counts,
        "gates": gates,
    }


def load_and_audit(path: Path) -> dict[str, Any]:
    if sha256_file(path) != SOURCE_SHA256:
        raise ValueError("2Wiki train source hash mismatch")

    metadata: list[tuple[str, str]] = []
    type_counts: dict[str, int] = {}
    source_rows = 0
    for row in iter_json_array(path):
        source_rows += 1
        task_type = str(row.get("type", ""))
        type_counts[task_type] = type_counts.get(task_type, 0) + 1
        metadata.append((str(row.get("_id", "")), task_type))
    if source_rows != SOURCE_ROWS:
        raise ValueError("2Wiki train row count mismatch")
    if type_counts != TYPE_COUNTS:
        raise ValueError("2Wiki train type counts mismatch")
    if sum(task_type in ELIGIBLE_TYPES for _id, task_type in metadata) != ELIGIBLE_ROWS:
        raise ValueError("2Wiki eligible row count mismatch")

    splits = split_ids(metadata)
    metadata_by_id = dict(metadata)
    for name, expected_hash in SPLIT_HASHES.items():
        if ordered_list_hash(splits[name]) != expected_hash:
            raise ValueError(f"2Wiki {name} split hash mismatch")
        observed_counts = {
            task_type: sum(
                metadata_by_id[task_id] == task_type for task_id in splits[name]
            )
            for task_type in sorted(ELIGIBLE_TYPES)
        }
        if observed_counts != SPLIT_TYPE_COUNTS[name]:
            raise ValueError(f"2Wiki {name} split type counts mismatch")

    opportunity_ids = set(splits["opportunity"])
    selected: dict[str, dict[str, Any]] = {}
    for row in iter_json_array(path):
        task_id = str(row.get("_id", ""))
        if task_id in opportunity_ids:
            selected[task_id] = row
    if set(selected) != opportunity_ids:
        raise ValueError("2Wiki opportunity rows are incomplete")
    diagnostics = [
        analyze_row(selected[task_id]) for task_id in splits["opportunity"]
    ]
    summary = summarize(diagnostics)
    source_gates = {
        "source_sha256_matches": True,
        "source_row_count_matches": True,
        "source_type_counts_match": True,
        "eligible_row_count_matches": True,
        "opportunity_split_hash_matches": True,
        "development_split_hash_matches": True,
        "holdout_split_hash_matches": True,
        "split_type_counts_match": True,
        "development_endpoint_rows_accessed_equals_zero": True,
        "holdout_endpoint_rows_accessed_equals_zero": True,
    }
    summary["gates"] = {**source_gates, **summary["gates"]}
    summary["gates"]["all_pass"] = all(summary["gates"].values())
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "source_sha256": SOURCE_SHA256,
            "source_rows": SOURCE_ROWS,
            "type_counts": TYPE_COUNTS,
            "eligible_types": sorted(ELIGIBLE_TYPES),
            "eligible_rows": ELIGIBLE_ROWS,
            "selection_seed": SELECTION_SEED,
            "split_sizes": {
                name: len(values) for name, values in splits.items()
            },
            "split_hashes": SPLIT_HASHES,
            "split_type_counts": SPLIT_TYPE_COUNTS,
            "model_calls": 0,
            "development_endpoint_rows_accessed": 0,
            "holdout_endpoint_rows_accessed": 0,
        },
        "summary": summary,
        "strict_directional_chain_rows": [
            row for row in diagnostics if row["strict_directional_chain"]
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
