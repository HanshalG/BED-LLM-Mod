#!/usr/bin/env python3
"""Audit bridge-first three-search opportunity on frozen LongVidSearch tasks."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.bright_biology_unlock_audit import BM25Corpus, tokenize
from scripts.longvid_bridge_path_opportunity_audit import (
    CAPTION_SHA256,
    EXPECTED_QA_ROWS,
    FRESH_VIDEO_HASH,
    FRESH_VIDEO_IDS,
    OBSERVATION_CHAR_CAP,
    PER_CLIP_TERMS,
    QA_SHA256,
    QUARANTINE_ROWS,
    ROOT_TOP_K,
    SOURCE_REPO_COMMIT,
    _answer_terms,
    _id_hash,
    followup_queries,
    frozen_split_ids as frozen_two_hop_split_ids,
    load_selected_captions,
    root_queries,
    sha256_file,
    stream_json_array,
)


SCHEMA_VERSION = 1
CAUSAL_SEED = 270733
STATE_SEED = 270734
OPPORTUNITY_SIZE = 40
DEVELOPMENT_SIZE = 20
RESERVE_SIZE = 40
CAUSAL_VIDEO_POOL_SIZE = 167
STATE_AVAILABLE_POOL_SIZE = 70
CAUSAL_POOL_ORDER_HASH = (
    "dbd097b24ab80762da62683457c493b17e095fc40515a653ccd75b501483ab4e"
)
STATE_POOL_ORDER_HASH = (
    "d8d80766a3225fcb3c0889eea36465603b973c62a24c98dc131d5efe359829cf"
)
OPPORTUNITY_ID_HASH = (
    "1e7ab608a86bf6281a88779c633fc5cdcfc208065fe263ef78970bc3704a142c"
)
DEVELOPMENT_ID_HASH = (
    "e3606098b77a05faabb13d703204eb5de8c0d05352d948708655f5db51cea1a3"
)
RESERVE_ID_HASH = (
    "0b95e40d40d55c78c5e7b7ef7c8a540a536fc83e50b9d71ff23e18e554ce8937"
)
OPPORTUNITY_VIDEO_HASH = (
    "55d038286dad0eafefd6b11fd66a41ca549617639bf420ac8b2a375e61218f30"
)

FOLLOWUP_TOP_K = 1
MAX_ROOTS = 20
MAX_STAGE_FOLLOWUPS = 12

MIN_ROOTS = 5
MIN_ANSWER_TERMS = 2
MIN_CAPTIONS = 60
MIN_DIVERSE_TASKS = 30
MIN_DEPTH_THREE_GAIN_TASKS = 15
MIN_ORDERED_CHAIN_TASKS = 10
MIN_MEAN_ORACLE_COVERAGE = 0.40
MIN_MEAN_COVERAGE_GAIN = 0.20
MIN_STRICT_OPPORTUNITIES = 5
MIN_STRICT_TOTAL_GAP = 5
MIN_MEAN_STRICT_ANSWER_SACRIFICE = 0.15


def frozen_split_ids(qa_path: Path) -> dict[str, list[int]]:
    """Reproduce the caption-unopened, video-disjoint three-hop split."""
    two_hop_opportunity = set(
        frozen_two_hop_split_ids(qa_path)["opportunity"]
    )
    two_hop_videos: set[str] = set()
    by_category: dict[str, dict[str, int]] = {
        "Causal_Inference": {},
        "State_Mutation": {},
    }
    row_count = 0
    for index, raw in stream_json_array(qa_path):
        row_count += 1
        if not isinstance(raw, dict):
            continue
        if index in two_hop_opportunity:
            two_hop_videos.add(str(raw.get("vid", "")))
        if index < QUARANTINE_ROWS:
            continue
        category = str(raw.get("category", ""))
        if (
            str(raw.get("hop_level", "")) != "3-Hop"
            or category not in by_category
        ):
            continue
        video_id = str(raw.get("vid", ""))
        by_category[category].setdefault(video_id, index)
    if row_count != EXPECTED_QA_ROWS:
        raise ValueError(f"QA file has {row_count} rows, expected {EXPECTED_QA_ROWS}")

    excluded = set(two_hop_videos).union(FRESH_VIDEO_IDS)
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
            if video_id not in excluded and video_id not in used
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
        row_ids = [by_category[category][video_id] for video_id in chosen]
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
        if raw["hop_level"] != "3-Hop" or len(evidence) != 3:
            raise ValueError(f"QA row {index} is not exactly three-hop")
        if len(set(evidence)) != 3:
            raise ValueError(f"QA row {index} repeats an evidence slice")
        selected[index] = raw
    if set(selected) != selected_ids:
        raise ValueError("QA file does not contain every opportunity row")
    return selected


def _best_trajectory(
    trajectories: Sequence[dict[str, Any]],
    *,
    first_gold: int,
) -> dict[str, Any]:
    return max(
        trajectories,
        key=lambda item: (
            item["triple_count"],
            -item["first_followup_index"],
            -item["second_followup_index"],
        ),
        default={
            "first_followup_index": -1,
            "second_followup_index": -1,
            "second_id": None,
            "third_id": None,
            "triple_count": first_gold,
            "first_observation_term_count": 0,
            "second_observation_term_count": 0,
        },
    )


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
    roots = root_queries(question, corpus)[:MAX_ROOTS]
    root_records: list[dict[str, Any]] = []
    ordered_chain_recovered = False

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
        trajectories: list[dict[str, Any]] = []
        first_followups = (
            followup_queries(question, root_query, first[0], corpus)[
                :MAX_STAGE_FOLLOWUPS
            ]
            if first
            else []
        )
        for first_followup_index, (
            first_followup,
            first_visible_terms,
        ) in enumerate(first_followups):
            second = corpus.search(
                first_followup,
                top_k=FOLLOWUP_TOP_K,
                excluded_ids={first_id},
            )
            second_id = second[0]["id"] if second else None
            second_followups = (
                followup_queries(
                    question,
                    first_followup,
                    second[0],
                    corpus,
                )[:MAX_STAGE_FOLLOWUPS]
                if second
                else []
            )
            for second_followup_index, (
                second_followup,
                second_visible_terms,
            ) in enumerate(second_followups):
                third = corpus.search(
                    second_followup,
                    top_k=FOLLOWUP_TOP_K,
                    excluded_ids={first_id, second_id},
                )
                third_id = third[0]["id"] if third else None
                triple_count = len(
                    evidence_set & {first_id, second_id, third_id}
                )
                trajectory = {
                    "first_followup_index": first_followup_index,
                    "second_followup_index": second_followup_index,
                    "second_id": second_id,
                    "third_id": third_id,
                    "triple_count": triple_count,
                    "first_observation_term_count": len(first_visible_terms),
                    "second_observation_term_count": len(second_visible_terms),
                }
                trajectories.append(trajectory)
                ordered_chain_recovered = ordered_chain_recovered or (
                    first_id == evidence[0]
                    and second_id == evidence[1]
                    and third_id == evidence[2]
                    and len(first_visible_terms) >= 1
                    and len(second_visible_terms) >= 1
                )

        best = _best_trajectory(trajectories, first_gold=first_gold)
        root_records.append(
            {
                "root_index": root_index,
                "first_id": first_id,
                "first_gold": first_gold,
                "direct_answer": direct_answer,
                "best_triple_count": int(best["triple_count"]),
                "best_second_id": best["second_id"],
                "best_third_id": best["third_id"],
                "best_first_observation_term_count": int(
                    best["first_observation_term_count"]
                ),
                "best_second_observation_term_count": int(
                    best["second_observation_term_count"]
                ),
                "num_first_followups": len(first_followups),
                "num_trajectories": len(trajectories),
            }
        )

    greedy = max(
        root_records,
        key=lambda item: (
            item["direct_answer"],
            item["first_gold"],
            item["best_triple_count"],
            -item["root_index"],
        ),
    )
    oracle = max(
        root_records,
        key=lambda item: (
            item["best_triple_count"],
            item["direct_answer"],
            item["first_gold"],
            -item["root_index"],
        ),
    )
    strict = (
        greedy["root_index"] != oracle["root_index"]
        and oracle["direct_answer"] < greedy["direct_answer"]
        and greedy["first_id"] == evidence[2]
        and oracle["first_id"] == evidence[0]
        and oracle["best_second_id"] == evidence[1]
        and oracle["best_third_id"] == evidence[2]
        and oracle["best_triple_count"] == 3
        and greedy["best_triple_count"] < 3
        and oracle["best_first_observation_term_count"] >= 1
        and oracle["best_second_observation_term_count"] >= 1
    )
    best_immediate_gold = max(record["first_gold"] for record in root_records)
    gap_count = (
        oracle["best_triple_count"] - greedy["best_triple_count"]
        if strict
        else 0
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
        "oracle_triple_count": int(oracle["best_triple_count"]),
        "oracle_triple_coverage": oracle["best_triple_count"] / 3.0,
        "depth_three_gain_count": int(
            oracle["best_triple_count"] - best_immediate_gold
        ),
        "coverage_gain": (
            oracle["best_triple_count"] - best_immediate_gold
        )
        / 3.0,
        "greedy_root_index": int(greedy["root_index"]),
        "greedy_first_evidence_position": (
            evidence.index(greedy["first_id"])
            if greedy["first_id"] in evidence_set
            else -1
        ),
        "greedy_direct_answer": float(greedy["direct_answer"]),
        "greedy_triple_count": int(greedy["best_triple_count"]),
        "oracle_root_index": int(oracle["root_index"]),
        "oracle_first_evidence_position": (
            evidence.index(oracle["first_id"])
            if oracle["first_id"] in evidence_set
            else -1
        ),
        "oracle_direct_answer": float(oracle["direct_answer"]),
        "ordered_chain_recovered": ordered_chain_recovered,
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
                record["num_captions"] >= MIN_CAPTIONS
                and record["num_answer_terms"] >= MIN_ANSWER_TERMS
                and record["num_roots"] >= MIN_ROOTS
            )
            for record in records
        ),
        "diverse_root_task_count": sum(
            int(record["distinct_root_top1"] >= 3) for record in records
        ),
        "depth_three_gain_task_count": sum(
            int(record["depth_three_gain_count"] >= 1) for record in records
        ),
        "ordered_chain_task_count": sum(
            int(record["ordered_chain_recovered"]) for record in records
        ),
        "mean_oracle_triple_coverage": _mean(
            [float(record["oracle_triple_coverage"]) for record in records]
        ),
        "mean_coverage_gain": _mean(
            [float(record["coverage_gain"]) for record in records]
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
        "depth_three_gain_tasks_at_least_15": (
            summary["depth_three_gain_task_count"]
            >= MIN_DEPTH_THREE_GAIN_TASKS
        ),
        "ordered_chain_tasks_at_least_10": (
            summary["ordered_chain_task_count"] >= MIN_ORDERED_CHAIN_TASKS
        ),
        "mean_oracle_triple_coverage_at_least_0_40": (
            summary["mean_oracle_triple_coverage"]
            >= MIN_MEAN_ORACLE_COVERAGE
        ),
        "mean_coverage_gain_at_least_0_20": (
            summary["mean_coverage_gain"] >= MIN_MEAN_COVERAGE_GAIN
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
            "opportunity_video_ids": OPPORTUNITY_VIDEO_HASH,
            "fresh_video_ids": FRESH_VIDEO_HASH,
        },
        "parameters": {
            "root_top_k": ROOT_TOP_K,
            "followup_top_k": FOLLOWUP_TOP_K,
            "max_roots": MAX_ROOTS,
            "max_first_followups": MAX_STAGE_FOLLOWUPS,
            "max_second_followups": MAX_STAGE_FOLLOWUPS,
            "observation_char_cap": OBSERVATION_CHAR_CAP,
            "per_clip_terms": PER_CLIP_TERMS,
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
