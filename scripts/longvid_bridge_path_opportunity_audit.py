#!/usr/bin/env python3
"""Audit bridge-first two-search opportunity on frozen LongVidSearch tasks."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import re
import sys
from typing import Any, Iterator, Sequence

import pyarrow.parquet as pq

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bright_biology_unlock_audit import BM25Corpus, STOPWORDS, tokenize


SCHEMA_VERSION = 1
SOURCE_REPO_COMMIT = "4aa5620e06ca1bc5cc7cfce2b3e3ed0a5ae82d4e"
QA_SHA256 = "370711f1299202cd40d559440859476bf3a2d2ca74540dad5226786558ee123b"
CAPTION_SHA256 = (
    "0f2ce94265b7050eaee5de239a760c1a0f0762c1754cbb3bd7b45d00774b6c68"
)
EXPECTED_QA_ROWS = 3_000
EXPECTED_CAPTION_ROWS = 40_804

CAUSAL_SEED = 270731
STATE_SEED = 270732
QUARANTINE_ROWS = 100
OPPORTUNITY_SIZE = 40
DEVELOPMENT_SIZE = 20
RESERVE_SIZE = 40
CAUSAL_VIDEO_POOL_SIZE = 270
STATE_AVAILABLE_POOL_SIZE = 142
CAUSAL_POOL_ORDER_HASH = (
    "7be8ff02fe6b8750d712a566073005118c94beef189066dcd0f3224d15eb5980"
)
STATE_POOL_ORDER_HASH = (
    "632b141fd2b36b401e54b0babfd71134aceffc7d58f451a3271e2e66bfab5eea"
)
OPPORTUNITY_ID_HASH = (
    "092b0337cad5d0fce436e1378640bd1e8580a2fe29953a25e189593ba17bbbb6"
)
DEVELOPMENT_ID_HASH = (
    "1c886e1b58c19e6774b7d8874af3feca6db587325f3f9884f46b00fe8fd70dca"
)
RESERVE_ID_HASH = (
    "535ff26012218733c88fadbd0677cf9ab42051a5ef8e9c857b9e4ad670a09a4d"
)
OPPORTUNITY_VIDEO_HASH = (
    "57c0c3a44eb17ac32e3d4fe8c4512881875196d6628a672cd202c111370314e1"
)
FRESH_VIDEO_HASH = (
    "eb964ad3fa8c2f34eba823467d535490ae62fe01a91f2a8c23a1011915c45855"
)
FRESH_VIDEO_IDS = frozenset(
    {
        "-mt8aNy1M00",
        "3-6FG5eaiiU",
        "5dJUUQufzw4",
        "9BR7FaeS01A",
        "9l3fHjMHuhE",
        "AL6U74YgA0Y",
        "Dz0DfKnG5UE",
        "ENuzZNYbJgU",
        "H4jA_SN4hgg",
        "K4s_tnsfhH4",
        "Q7cwC00Gs-Q",
        "QQ2RaGIYqqE",
        "Rslv43vHNcA",
        "VFXJnbnN5ro",
        "Vp8yU0vKti0",
        "Y1YCvEip_ko",
        "YO3iBRvqzac",
        "Z-LhBjlpwr0",
        "cuHgxPe3J7I",
        "d7IqrLV6Tlg",
        "lKxEXO0Ffio",
        "rba81Y5l7rU",
    }
)

ROOT_TOP_K = 1
FOLLOWUP_TOP_K = 1
MAX_ROOTS = 20
MAX_FOLLOWUPS = 24
OBSERVATION_CHAR_CAP = 2_000
PER_CLIP_TERMS = 10

MIN_ROOTS = 5
MIN_ANSWER_TERMS = 2
MIN_DIVERSE_TASKS = 30
MIN_PAIR_GAIN_TASKS = 15
MIN_ORDERED_CHAIN_TASKS = 10
MIN_MEAN_ORACLE_PAIR_COVERAGE = 0.40
MIN_MEAN_PAIR_COVERAGE_GAIN = 0.15
MIN_STRICT_OPPORTUNITIES = 5
MIN_STRICT_TOTAL_GAP = 5
MIN_MEAN_STRICT_ANSWER_SACRIFICE = 0.15

SENTENCE_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CLAUSE_PATTERN = re.compile(r"[,;:]|\s+[—-]\s+")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _id_hash(values: Sequence[Any]) -> str:
    return hashlib.sha256(
        "\n".join(str(value) for value in values).encode("utf-8")
    ).hexdigest()


def normalized(text: str) -> str:
    return " ".join(tokenize(text))


def stream_json_array(
    path: Path,
    *,
    chunk_size: int = 1024 * 1024,
) -> Iterator[tuple[int, Any]]:
    """Yield indexed values from a top-level JSON array."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as handle:
        buffer = ""
        position = 0
        finished = False

        def refill() -> bool:
            nonlocal buffer, position, finished
            if position:
                buffer = buffer[position:]
                position = 0
            chunk = handle.read(chunk_size)
            if not chunk:
                finished = True
                return False
            buffer += chunk
            return True

        refill()
        while True:
            while position >= len(buffer):
                if not refill():
                    raise ValueError("empty JSON array")
            if buffer[position].isspace():
                position += 1
                continue
            if buffer[position] != "[":
                raise ValueError("QA JSON must be a top-level array")
            position += 1
            break

        index = 0
        expect_value = True
        while True:
            while True:
                while position < len(buffer) and (
                    buffer[position].isspace()
                    or (not expect_value and buffer[position] == ",")
                ):
                    if buffer[position] == ",":
                        expect_value = True
                    position += 1
                if position < len(buffer):
                    break
                if not refill():
                    raise ValueError("unterminated JSON array")
            if buffer[position] == "]":
                position += 1
                break
            if not expect_value:
                raise ValueError("expected comma between array values")
            while True:
                try:
                    value, end = decoder.raw_decode(buffer, position)
                    break
                except json.JSONDecodeError:
                    if not refill():
                        raise ValueError(f"invalid JSON value at index {index}")
            position = end
            expect_value = False
            yield index, value
            index += 1

        while True:
            while position < len(buffer) and buffer[position].isspace():
                position += 1
            if position < len(buffer):
                raise ValueError("trailing content after QA array")
            if finished or not refill():
                break


def frozen_split_ids(qa_path: Path) -> dict[str, list[int]]:
    by_category: dict[str, dict[str, list[int]]] = {
        "Causal_Inference": {},
        "State_Mutation": {},
    }
    row_count = 0
    for index, raw in stream_json_array(qa_path):
        row_count += 1
        if not isinstance(raw, dict) or index < QUARANTINE_ROWS:
            continue
        category = str(raw.get("category", ""))
        if (
            str(raw.get("hop_level", "")) != "2-Hop"
            or category not in by_category
        ):
            continue
        video_id = str(raw.get("vid", ""))
        by_category[category].setdefault(video_id, []).append(index)
    if row_count != EXPECTED_QA_ROWS:
        raise ValueError(f"QA file has {row_count} rows, expected {EXPECTED_QA_ROWS}")

    used: set[str] = set()
    split = {"opportunity": [], "development": [], "reserve": []}
    specs = (
        (
            "Causal_Inference",
            CAUSAL_SEED,
            CAUSAL_VIDEO_POOL_SIZE,
            CAUSAL_POOL_ORDER_HASH,
        ),
        (
            "State_Mutation",
            STATE_SEED,
            STATE_AVAILABLE_POOL_SIZE,
            STATE_POOL_ORDER_HASH,
        ),
    )
    for category, seed, expected_pool_size, expected_order_hash in specs:
        pool = sorted(
            video_id
            for video_id in by_category[category]
            if video_id not in used
        )
        if len(pool) != expected_pool_size:
            raise AssertionError(
                f"{category} video pool has {len(pool)}, expected {expected_pool_size}"
            )
        random.Random(seed).shuffle(pool)
        if _id_hash(pool) != expected_order_hash:
            raise AssertionError(f"{category} shuffled pool hash does not reproduce")
        chosen = pool[:50]
        used.update(chosen)
        row_ids = [
            min(by_category[category][video_id]) for video_id in chosen
        ]
        split["opportunity"].extend(row_ids[:20])
        split["development"].extend(row_ids[20:30])
        split["reserve"].extend(row_ids[30:50])

    expected = {
        "opportunity": OPPORTUNITY_ID_HASH,
        "development": DEVELOPMENT_ID_HASH,
        "reserve": RESERVE_ID_HASH,
    }
    for name, row_ids in split.items():
        if _id_hash(row_ids) != expected[name]:
            raise AssertionError(f"{name} row hash does not reproduce")
    return split


def load_selected_qa(
    qa_path: Path,
    selected_ids: set[int],
) -> dict[int, dict[str, Any]]:
    selected: dict[int, dict[str, Any]] = {}
    for index, raw in stream_json_array(qa_path):
        if index not in selected_ids:
            continue
        if not isinstance(raw, dict):
            raise ValueError(f"QA row {index} is not an object")
        required = (
            "question",
            "answer",
            "category",
            "hop_level",
            "evidence_slices",
            "reasoning_chain",
            "vid",
        )
        if any(not raw.get(field) for field in required):
            raise ValueError(f"QA row {index} has an empty required field")
        evidence = [int(value) for value in raw["evidence_slices"]]
        if raw["hop_level"] != "2-Hop" or len(evidence) != 2:
            raise ValueError(f"QA row {index} is not exactly two-hop")
        if len(set(evidence)) != 2:
            raise ValueError(f"QA row {index} repeats an evidence slice")
        selected[index] = raw
    if set(selected) != selected_ids:
        raise ValueError("QA file does not contain every opportunity row")
    return selected


def load_selected_captions(
    caption_path: Path,
    allowed_video_ids: set[str],
) -> dict[str, list[dict[str, str]]]:
    if allowed_video_ids & FRESH_VIDEO_IDS:
        raise AssertionError("fresh caption-only video entered development audit")
    table = pq.read_table(
        caption_path,
        columns=["vid", "slice_num", "cap"],
        filters=[("vid", "in", sorted(allowed_video_ids))],
    )
    returned_ids = set(map(str, table["vid"].to_pylist()))
    if returned_ids != allowed_video_ids:
        missing = sorted(allowed_video_ids - returned_ids)
        extra = sorted(returned_ids - allowed_video_ids)
        raise ValueError(f"caption filter mismatch: missing={missing}, extra={extra}")
    captions: dict[str, list[dict[str, str]]] = {
        video_id: [] for video_id in allowed_video_ids
    }
    for video_id, slice_num, caption in zip(
        table["vid"].to_pylist(),
        table["slice_num"].to_pylist(),
        table["cap"].to_pylist(),
    ):
        text = str(caption).strip()
        if not text:
            raise ValueError(f"{video_id} slice {slice_num} has an empty caption")
        captions[str(video_id)].append(
            {
                "id": str(int(slice_num)),
                "content": text,
                "raw_source": text,
                "title": f"Clip {int(slice_num)}",
            }
        )
    for video_id, rows in captions.items():
        rows.sort(key=lambda row: int(row["id"]))
        slice_ids = [int(row["id"]) for row in rows]
        if slice_ids != list(range(1, len(rows) + 1)):
            raise ValueError(f"{video_id} caption slices are not contiguous")
    return captions


def _query_idf(corpus: BM25Corpus, token: str) -> float:
    return float(corpus.index.idf.get(token, -100.0))


def root_queries(question: str, corpus: BM25Corpus) -> list[str]:
    sentences = [
        " ".join(sentence.split())
        for sentence in SENTENCE_PATTERN.split(question)
        if sentence.strip()
    ]
    clauses = [
        " ".join(clause.split())
        for sentence in sentences
        for clause in CLAUSE_PATTERN.split(sentence)
        if len(tokenize(clause)) >= 4
    ]
    question_tokens = tokenize(question)
    ranked_terms = sorted(
        {
            token
            for token in question_tokens
            if len(token) >= 3
            and token not in STOPWORDS
            and token in corpus.index.idf
        },
        key=lambda token: (-_query_idf(corpus, token), token),
    )
    rare = set(ranked_terms[:10])
    candidates = [
        question,
        *sentences,
        *clauses,
        " ".join(ranked_terms[:12]),
        *ranked_terms[:8],
    ]
    for width in (2, 3, 4):
        for start in range(max(0, len(question_tokens) - width + 1)):
            window = question_tokens[start : start + width]
            if rare.intersection(window):
                candidates.append(" ".join(window))

    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = normalized(candidate)
        if key and key not in seen:
            roots.append(" ".join(candidate.split()))
            seen.add(key)
        if len(roots) >= MAX_ROOTS:
            break
    return roots


def followup_queries(
    question: str,
    root_query: str,
    root_document: dict[str, str],
    corpus: BM25Corpus,
) -> list[tuple[str, tuple[str, ...]]]:
    observation = root_document["raw_source"][:OBSERVATION_CHAR_CAP]
    excluded = set(tokenize(question)).union(tokenize(root_query))
    terms = corpus.highest_idf_terms(
        [observation],
        excluded_terms=excluded,
        count=PER_CLIP_TERMS,
    )
    candidates: list[tuple[str, tuple[str, ...]]] = []
    for term in terms:
        candidates.append((f"{question} {term}", (term,)))
        candidates.append((f"{root_query} {term}", (term,)))
    for start in range(max(0, len(terms) - 1)):
        pair = (terms[start], terms[start + 1])
        candidates.append((f"{question} {' '.join(pair)}", pair))
    if terms:
        candidates.append((f"{root_query} {' '.join(terms)}", tuple(terms)))

    observation_terms = set(tokenize(observation))
    followups: list[tuple[str, tuple[str, ...]]] = []
    seen: set[str] = set()
    for candidate, visible_terms in candidates:
        valid_terms = tuple(
            term
            for term in visible_terms
            if term in observation_terms and term not in excluded
        )
        key = normalized(candidate)
        if key and valid_terms and key not in seen:
            followups.append((" ".join(candidate.split()), valid_terms))
            seen.add(key)
        if len(followups) >= MAX_FOLLOWUPS:
            break
    return followups


def _answer_terms(answer: str) -> set[str]:
    return {
        token
        for token in tokenize(answer)
        if len(token) >= 2 and token not in STOPWORDS
    }


def analyze_task(
    row_index: int,
    row: dict[str, Any],
    documents: Sequence[dict[str, str]],
) -> dict[str, Any]:
    corpus = BM25Corpus(documents)
    question = str(row["question"])
    answer_terms = _answer_terms(str(row["answer"]))
    evidence = tuple(str(int(value)) for value in row["evidence_slices"])
    evidence_set = set(evidence)
    roots = root_queries(question, corpus)
    root_records: list[dict[str, Any]] = []
    for root_index, root_query in enumerate(roots):
        first = corpus.search(root_query, top_k=ROOT_TOP_K)
        first_id = first[0]["id"] if first else None
        first_gold = int(first_id in evidence_set)
        caption_terms = set(tokenize(first[0]["raw_source"])) if first else set()
        direct_answer = (
            len(answer_terms & caption_terms) / len(answer_terms)
            if first_gold and answer_terms
            else 0.0
        )
        continuations = []
        if first:
            for followup_index, (followup, visible_terms) in enumerate(
                followup_queries(question, root_query, first[0], corpus)
            ):
                second = corpus.search(
                    followup,
                    top_k=FOLLOWUP_TOP_K,
                    excluded_ids={first_id},
                )
                second_id = second[0]["id"] if second else None
                pair_count = len(evidence_set & {first_id, second_id})
                continuations.append(
                    {
                        "followup_index": followup_index,
                        "second_id": second_id,
                        "pair_count": pair_count,
                        "observation_term_count": len(visible_terms),
                    }
                )
        best = max(
            continuations,
            key=lambda item: (item["pair_count"], -item["followup_index"]),
            default={
                "followup_index": -1,
                "second_id": None,
                "pair_count": first_gold,
                "observation_term_count": 0,
            },
        )
        root_records.append(
            {
                "root_index": root_index,
                "first_id": first_id,
                "first_gold": first_gold,
                "direct_answer": direct_answer,
                "best_pair_count": int(best["pair_count"]),
                "best_second_id": best["second_id"],
                "best_followup_index": int(best["followup_index"]),
                "best_observation_term_count": int(
                    best["observation_term_count"]
                ),
                "num_followups": len(continuations),
            }
        )

    greedy = max(
        root_records,
        key=lambda item: (
            item["direct_answer"],
            item["first_gold"],
            item["best_pair_count"],
            -item["root_index"],
        ),
    )
    oracle = max(
        root_records,
        key=lambda item: (
            item["best_pair_count"],
            item["direct_answer"],
            item["first_gold"],
            -item["root_index"],
        ),
    )
    ordered_chain = any(
        record["first_id"] == evidence[0]
        and record["best_second_id"] == evidence[1]
        and record["best_observation_term_count"] >= 1
        for record in root_records
    )
    strict = (
        greedy["root_index"] != oracle["root_index"]
        and oracle["direct_answer"] < greedy["direct_answer"]
        and greedy["first_id"] == evidence[1]
        and oracle["first_id"] == evidence[0]
        and oracle["best_second_id"] == evidence[1]
        and oracle["best_pair_count"] == 2
        and greedy["best_pair_count"] < 2
        and oracle["best_observation_term_count"] >= 1
    )
    best_immediate_gold = max(record["first_gold"] for record in root_records)
    gap_count = (
        oracle["best_pair_count"] - greedy["best_pair_count"] if strict else 0
    )
    answer_sacrifice = (
        greedy["direct_answer"] - oracle["direct_answer"] if strict else 0.0
    )
    return {
        "row_index": row_index,
        "video_id": str(row["vid"]),
        "category": str(row["category"]),
        "num_captions": len(documents),
        "num_answer_terms": len(answer_terms),
        "num_roots": len(roots),
        "distinct_root_top1": len(
            {record["first_id"] for record in root_records if record["first_id"]}
        ),
        "best_immediate_gold_count": best_immediate_gold,
        "oracle_pair_count": int(oracle["best_pair_count"]),
        "oracle_pair_coverage": oracle["best_pair_count"] / 2.0,
        "pair_gain_count": int(
            oracle["best_pair_count"] - best_immediate_gold
        ),
        "pair_coverage_gain": (
            oracle["best_pair_count"] - best_immediate_gold
        )
        / 2.0,
        "greedy_root_index": int(greedy["root_index"]),
        "greedy_first_evidence_position": (
            evidence.index(greedy["first_id"])
            if greedy["first_id"] in evidence_set
            else -1
        ),
        "greedy_direct_answer": float(greedy["direct_answer"]),
        "greedy_pair_count": int(greedy["best_pair_count"]),
        "oracle_root_index": int(oracle["root_index"]),
        "oracle_first_evidence_position": (
            evidence.index(oracle["first_id"])
            if oracle["first_id"] in evidence_set
            else -1
        ),
        "oracle_direct_answer": float(oracle["direct_answer"]),
        "ordered_chain_recovered": ordered_chain,
        "nonmyopic_gap_count": int(gap_count),
        "answer_sacrifice": float(answer_sacrifice),
        "strict_opportunity": strict,
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [record for record in records if record["strict_opportunity"]]
    summary = {
        "num_records": len(records),
        "complete_task_count": sum(
            int(
                record["num_captions"] >= 60
                and record["num_answer_terms"] >= MIN_ANSWER_TERMS
                and record["num_roots"] >= MIN_ROOTS
            )
            for record in records
        ),
        "diverse_root_task_count": sum(
            int(record["distinct_root_top1"] >= 3) for record in records
        ),
        "pair_gain_task_count": sum(
            int(record["pair_gain_count"] >= 1) for record in records
        ),
        "ordered_chain_task_count": sum(
            int(record["ordered_chain_recovered"]) for record in records
        ),
        "mean_oracle_pair_coverage": _mean(
            [float(record["oracle_pair_coverage"]) for record in records]
        ),
        "mean_pair_coverage_gain": _mean(
            [float(record["pair_coverage_gain"]) for record in records]
        ),
        "strict_opportunity_count": len(strict),
        "strict_total_gap": sum(
            int(record["nonmyopic_gap_count"]) for record in strict
        ),
        "mean_strict_answer_sacrifice": _mean(
            [float(record["answer_sacrifice"]) for record in strict]
        ),
    }
    gates = {
        "all_tasks_complete": summary["complete_task_count"] == OPPORTUNITY_SIZE,
        "diverse_root_tasks_at_least_30": (
            summary["diverse_root_task_count"] >= MIN_DIVERSE_TASKS
        ),
        "pair_gain_tasks_at_least_15": (
            summary["pair_gain_task_count"] >= MIN_PAIR_GAIN_TASKS
        ),
        "ordered_chain_tasks_at_least_10": (
            summary["ordered_chain_task_count"] >= MIN_ORDERED_CHAIN_TASKS
        ),
        "mean_oracle_pair_coverage_at_least_0_40": (
            summary["mean_oracle_pair_coverage"]
            >= MIN_MEAN_ORACLE_PAIR_COVERAGE
        ),
        "mean_pair_coverage_gain_at_least_0_15": (
            summary["mean_pair_coverage_gain"]
            >= MIN_MEAN_PAIR_COVERAGE_GAIN
        ),
        "strict_opportunities_at_least_5": (
            summary["strict_opportunity_count"] >= MIN_STRICT_OPPORTUNITIES
        ),
        "strict_total_gap_at_least_5": (
            summary["strict_total_gap"] >= MIN_STRICT_TOTAL_GAP
        ),
        "mean_strict_answer_sacrifice_at_least_0_15": (
            summary["mean_strict_answer_sacrifice"]
            >= MIN_MEAN_STRICT_ANSWER_SACRIFICE
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_audit(
    qa_path: Path,
    caption_path: Path,
) -> dict[str, Any]:
    if sha256_file(qa_path) != QA_SHA256:
        raise ValueError("QA SHA-256 does not match frozen source")
    if sha256_file(caption_path) != CAPTION_SHA256:
        raise ValueError("caption SHA-256 does not match frozen source")
    if _id_hash(sorted(FRESH_VIDEO_IDS)) != FRESH_VIDEO_HASH:
        raise AssertionError("fresh video-ID hash does not reproduce")
    split = frozen_split_ids(qa_path)
    selected_ids = set(split["opportunity"])
    rows = load_selected_qa(qa_path, selected_ids)
    ordered_rows = [rows[index] for index in split["opportunity"]]
    video_ids = [str(row["vid"]) for row in ordered_rows]
    if len(video_ids) != len(set(video_ids)):
        raise AssertionError("opportunity tasks are not video-disjoint")
    if _id_hash(video_ids) != OPPORTUNITY_VIDEO_HASH:
        raise AssertionError("opportunity video hash does not reproduce")
    captions = load_selected_captions(caption_path, set(video_ids))
    records = []
    for offset, row_index in enumerate(split["opportunity"], start=1):
        row = rows[row_index]
        documents = captions[str(row["vid"])]
        evidence = {str(int(value)) for value in row["evidence_slices"]}
        if not evidence.issubset({document["id"] for document in documents}):
            raise ValueError(f"row {row_index} evidence is absent from captions")
        records.append(analyze_task(row_index, row, documents))
        print(f"completed={offset}/{len(selected_ids)}", file=sys.stderr)
    summary = summarize(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "gate_passed" if summary["gates"]["all_pass"] else "gate_failed",
        "source": {
            "repository_commit": SOURCE_REPO_COMMIT,
            "qa_sha256": QA_SHA256,
            "caption_sha256": CAPTION_SHA256,
        },
        "split_hashes": {
            "opportunity": OPPORTUNITY_ID_HASH,
            "development": DEVELOPMENT_ID_HASH,
            "reserve": RESERVE_ID_HASH,
            "fresh_video_ids": FRESH_VIDEO_HASH,
        },
        "parameters": {
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "max_roots": MAX_ROOTS,
            "max_followups": MAX_FOLLOWUPS,
            "observation_char_cap": OBSERVATION_CHAR_CAP,
        },
        "summary": summary,
        "records": records,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qa-path", type=Path, required=True)
    parser.add_argument("--caption-path", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    payload = run_audit(args.qa_path, args.caption_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"status={payload['status']}")
    print(f"output={args.output_path}")
    return 0 if payload["status"] == "gate_passed" else 1


if __name__ == "__main__":
    sys.exit(main())
