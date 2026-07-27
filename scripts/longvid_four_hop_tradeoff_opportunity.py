#!/usr/bin/env python3
"""Audit four-hop semantic root tradeoffs on frozen LongVidSearch tasks."""

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
    QA_SHA256,
    QUARANTINE_ROWS,
    SOURCE_REPO_COMMIT,
    _answer_terms,
    _id_hash,
    followup_queries,
    frozen_split_ids as frozen_two_hop_split_ids,
    load_selected_captions,
    load_selected_qa as load_two_hop_qa,
    root_queries,
    sha256_file,
    stream_json_array,
)
from scripts.longvid_three_hop_opportunity_audit import (
    frozen_split_ids as frozen_three_hop_split_ids,
    load_selected_qa as load_three_hop_qa,
)


SCHEMA_VERSION = 1
STATE_SEED = 270735
CAUSAL_SEED = 270736
GLOBAL_SEED = 270737
OPPORTUNITY_SIZE = 40
STATE_POOL_SIZE = 29
CAUSAL_POOL_SIZE = 43
GLOBAL_POOL_SIZE = 30
STATE_POOL_HASH = (
    "c09872a82f08169463343713b9db4816bd6d085d8f35423230c9bb7fe7a484c8"
)
CAUSAL_POOL_HASH = (
    "f433e0b65f75fe64af406d7228046f87146ae79b4a0233110638fd3f52abe08e"
)
GLOBAL_POOL_HASH = (
    "4a6861a74efc524b4fe3646a67c03b1e8e096b8ee1e118dcb6ada213adcc911a"
)
OPPORTUNITY_ID_HASH = (
    "6c27da94e920e4c7c98df22e549612bef9f6f31e2879f4a30f53e9b7d5bf60b2"
)
CONFIRMATION_ID_HASH = (
    "1cecd8a4318a0c283f44d18feff1551babba6eaff2218f508fc1141141944453"
)
RESERVE_ID_HASH = (
    "fb7ea1295874f273bc8d8ce163630462bd98533b557a30cb946629e546ca9c0c"
)
OPPORTUNITY_VIDEO_HASH = (
    "0ac0cd4bba6f626b12446d69d82e40d34d7651a86caf2eedbfca28be5bdef59a"
)
CONFIRMATION_VIDEO_HASH = (
    "e0959f11ec5a9c5198b9fa468a3b9681e148623ac0cbab7018206f1bc5300d6e"
)
RESERVE_VIDEO_HASH = (
    "35c1a2747b5e4376c3e2c3dc067a2eec08dd29b4bc9b8421cdebc24a2d17ca7f"
)

MAX_ROOTS = 20
MAX_STAGE_FOLLOWUPS = 8
MIN_CAPTIONS = 60
MIN_ROOTS = 5
MIN_ANSWER_TERMS = 2
MIN_DIVERSE_TASKS = 30
MIN_DEPTH_FOUR_GAIN_TASKS = 20
MIN_MEAN_ORACLE_COVERAGE = 0.45
MIN_MEAN_COVERAGE_GAIN = 0.25
MIN_STRICT_TRADEOFFS = 6
MIN_STRICT_TOTAL_GAP = 6
MIN_MEAN_STRICT_ANSWER_SACRIFICE = 0.15


def _prior_split_videos(qa_path: Path) -> set[str]:
    two_hop = frozen_two_hop_split_ids(qa_path)
    two_ids = set(sum(two_hop.values(), []))
    two_rows = load_two_hop_qa(qa_path, two_ids)
    three_hop = frozen_three_hop_split_ids(qa_path)
    three_ids = set(sum(three_hop.values(), []))
    three_rows = load_three_hop_qa(qa_path, three_ids)
    return {
        str(row["vid"]) for row in [*two_rows.values(), *three_rows.values()]
    }


def frozen_split_ids(qa_path: Path) -> dict[str, list[int]]:
    excluded = _prior_split_videos(qa_path).union(FRESH_VIDEO_IDS)
    categories = ("State_Mutation", "Causal_Inference", "Global_Summary")
    first_rows: dict[str, dict[str, int]] = {
        category: {} for category in categories
    }
    row_count = 0
    for index, raw in stream_json_array(qa_path):
        row_count += 1
        if (
            not isinstance(raw, dict)
            or index < QUARANTINE_ROWS
            or str(raw.get("hop_level", "")) != "4-Hop"
        ):
            continue
        category = str(raw.get("category", ""))
        video_id = str(raw.get("vid", ""))
        if category in first_rows and video_id not in excluded:
            first_rows[category].setdefault(video_id, index)
    if row_count != EXPECTED_QA_ROWS:
        raise ValueError(f"QA file has {row_count} rows, expected {EXPECTED_QA_ROWS}")

    specs = (
        ("State_Mutation", STATE_SEED, STATE_POOL_SIZE, STATE_POOL_HASH, 14, 14),
        (
            "Causal_Inference",
            CAUSAL_SEED,
            CAUSAL_POOL_SIZE,
            CAUSAL_POOL_HASH,
            16,
            16,
        ),
        ("Global_Summary", GLOBAL_SEED, GLOBAL_POOL_SIZE, GLOBAL_POOL_HASH, 10, 10),
    )
    used: set[str] = set()
    split = {"opportunity": [], "confirmation": [], "reserve": []}
    split_videos = {"opportunity": [], "confirmation": [], "reserve": []}
    for category, seed, expected_size, pool_hash, n_opp, n_confirm in specs:
        pool = sorted(
            video_id
            for video_id in first_rows[category]
            if video_id not in used
        )
        if len(pool) != expected_size:
            raise AssertionError(
                f"{category} video pool has {len(pool)}, expected {expected_size}"
            )
        random.Random(seed).shuffle(pool)
        if _id_hash(pool) != pool_hash:
            raise AssertionError(f"{category} shuffled pool hash does not reproduce")
        used.update(pool)
        partitions = {
            "opportunity": pool[:n_opp],
            "confirmation": pool[n_opp : n_opp + n_confirm],
            "reserve": pool[n_opp + n_confirm :],
        }
        for name, videos in partitions.items():
            split_videos[name].extend(videos)
            split[name].extend(first_rows[category][video] for video in videos)

    expected_rows = {
        "opportunity": OPPORTUNITY_ID_HASH,
        "confirmation": CONFIRMATION_ID_HASH,
        "reserve": RESERVE_ID_HASH,
    }
    expected_videos = {
        "opportunity": OPPORTUNITY_VIDEO_HASH,
        "confirmation": CONFIRMATION_VIDEO_HASH,
        "reserve": RESERVE_VIDEO_HASH,
    }
    for name in split:
        if _id_hash(split[name]) != expected_rows[name]:
            raise AssertionError(f"{name} row hash does not reproduce")
        if _id_hash(split_videos[name]) != expected_videos[name]:
            raise AssertionError(f"{name} video hash does not reproduce")
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
        if raw["hop_level"] != "4-Hop" or len(evidence) != 4:
            raise ValueError(f"QA row {index} is not exactly four-hop")
        if len(set(evidence)) != 4:
            raise ValueError(f"QA row {index} repeats an evidence slice")
        selected[index] = raw
    if set(selected) != selected_ids:
        raise ValueError("QA file does not contain every selected row")
    return selected


def _trajectory_key(item: dict[str, Any]) -> tuple[Any, ...]:
    return (item["final_count"], *(-value for value in item["candidate_order"]))


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
        first = corpus.search(root_query, top_k=1)
        first_id = first[0]["id"] if first else None
        first_gold = int(first_id in evidence_set)
        caption_terms = set(tokenize(first[0]["raw_source"])) if first else set()
        direct_answer = (
            len(answer_terms & caption_terms) / len(answer_terms)
            if first_gold and answer_terms
            else 0.0
        )
        best: dict[str, Any] | None = None
        trajectory_count = 0

        def walk(
            previous_query: str,
            previous_document: dict[str, str],
            retrieved_ids: tuple[str, ...],
            candidate_order: tuple[int, ...],
            observation_term_counts: tuple[int, ...],
        ) -> None:
            nonlocal best, ordered_chain_recovered, trajectory_count
            if len(retrieved_ids) == 4:
                trajectory_count += 1
                record = {
                    "final_count": len(evidence_set & set(retrieved_ids)),
                    "retrieved_ids": retrieved_ids,
                    "candidate_order": candidate_order,
                    "observation_term_counts": observation_term_counts,
                }
                if best is None or _trajectory_key(record) > _trajectory_key(best):
                    best = record
                ordered_chain_recovered = ordered_chain_recovered or (
                    retrieved_ids == evidence
                    and all(count >= 1 for count in observation_term_counts)
                )
                return
            followups = followup_queries(
                question,
                previous_query,
                previous_document,
                corpus,
            )[:MAX_STAGE_FOLLOWUPS]
            for followup_index, (followup, visible_terms) in enumerate(followups):
                result = corpus.search(
                    followup,
                    top_k=1,
                    excluded_ids=set(retrieved_ids),
                )
                if not result:
                    continue
                walk(
                    followup,
                    result[0],
                    (*retrieved_ids, result[0]["id"]),
                    (*candidate_order, followup_index),
                    (*observation_term_counts, len(visible_terms)),
                )

        if first:
            walk(root_query, first[0], (first_id,), (), ())
        if best is None:
            best = {
                "final_count": first_gold,
                "retrieved_ids": (first_id,),
                "candidate_order": (),
                "observation_term_counts": (),
            }
        best_ids = tuple(best["retrieved_ids"])
        root_records.append(
            {
                "root_index": root_index,
                "first_id": first_id,
                "first_gold": first_gold,
                "direct_answer": direct_answer,
                "best_final_count": int(best["final_count"]),
                "best_path_positions": tuple(
                    evidence.index(slice_id) if slice_id in evidence_set else -1
                    for slice_id in best_ids
                ),
                "best_observation_term_counts": tuple(
                    int(value) for value in best["observation_term_counts"]
                ),
                "num_trajectories": trajectory_count,
            }
        )

    greedy = max(
        root_records,
        key=lambda item: (
            item["direct_answer"],
            item["first_gold"],
            item["best_final_count"],
            -item["root_index"],
        ),
    )
    oracle = max(
        root_records,
        key=lambda item: (
            item["best_final_count"],
            item["direct_answer"],
            item["first_gold"],
            -item["root_index"],
        ),
    )
    strict = (
        greedy["root_index"] != oracle["root_index"]
        and oracle["direct_answer"] < greedy["direct_answer"]
        and oracle["best_final_count"] > greedy["best_final_count"]
        and len(oracle["best_observation_term_counts"]) == 3
        and all(
            count >= 1 for count in oracle["best_observation_term_counts"]
        )
    )
    best_immediate_gold = max(record["first_gold"] for record in root_records)
    gap = (
        oracle["best_final_count"] - greedy["best_final_count"] if strict else 0
    )
    sacrifice = (
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
        "oracle_final_count": int(oracle["best_final_count"]),
        "oracle_final_coverage": oracle["best_final_count"] / 4.0,
        "depth_four_gain_count": int(
            oracle["best_final_count"] - best_immediate_gold
        ),
        "coverage_gain": (
            oracle["best_final_count"] - best_immediate_gold
        )
        / 4.0,
        "greedy_root_index": int(greedy["root_index"]),
        "greedy_first_evidence_position": (
            evidence.index(greedy["first_id"])
            if greedy["first_id"] in evidence_set
            else -1
        ),
        "greedy_direct_answer": float(greedy["direct_answer"]),
        "greedy_final_count": int(greedy["best_final_count"]),
        "oracle_root_index": int(oracle["root_index"]),
        "oracle_first_evidence_position": (
            evidence.index(oracle["first_id"])
            if oracle["first_id"] in evidence_set
            else -1
        ),
        "oracle_direct_answer": float(oracle["direct_answer"]),
        "oracle_path_positions": list(oracle["best_path_positions"]),
        "ordered_chain_recovered": ordered_chain_recovered,
        "strict_tradeoff": strict,
        "strict_gap_count": int(gap),
        "strict_answer_sacrifice": float(sacrifice),
    }


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [record for record in records if record["strict_tradeoff"]]
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
        "depth_four_gain_task_count": sum(
            int(record["depth_four_gain_count"] >= 1) for record in records
        ),
        "ordered_chain_task_count": sum(
            int(record["ordered_chain_recovered"]) for record in records
        ),
        "mean_oracle_final_coverage": _mean(
            [float(record["oracle_final_coverage"]) for record in records]
        ),
        "mean_coverage_gain": _mean(
            [float(record["coverage_gain"]) for record in records]
        ),
        "strict_tradeoff_count": len(strict),
        "strict_total_gap": sum(
            int(record["strict_gap_count"]) for record in strict
        ),
        "mean_strict_answer_sacrifice": _mean(
            [float(record["strict_answer_sacrifice"]) for record in strict]
        ),
    }
    gates = {
        "all_tasks_complete": summary["complete_task_count"] == OPPORTUNITY_SIZE,
        "diverse_root_tasks_at_least_30": (
            summary["diverse_root_task_count"] >= MIN_DIVERSE_TASKS
        ),
        "depth_four_gain_tasks_at_least_20": (
            summary["depth_four_gain_task_count"]
            >= MIN_DEPTH_FOUR_GAIN_TASKS
        ),
        "mean_oracle_final_coverage_at_least_0_45": (
            summary["mean_oracle_final_coverage"]
            >= MIN_MEAN_ORACLE_COVERAGE
        ),
        "mean_coverage_gain_at_least_0_25": (
            summary["mean_coverage_gain"] >= MIN_MEAN_COVERAGE_GAIN
        ),
        "strict_tradeoffs_at_least_6": (
            summary["strict_tradeoff_count"] >= MIN_STRICT_TRADEOFFS
        ),
        "strict_total_gap_at_least_6": (
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


def run_audit(qa_path: Path, caption_path: Path) -> dict[str, Any]:
    if sha256_file(qa_path) != QA_SHA256:
        raise ValueError("QA SHA-256 does not match frozen source")
    if sha256_file(caption_path) != CAPTION_SHA256:
        raise ValueError("caption SHA-256 does not match frozen source")
    if _id_hash(sorted(FRESH_VIDEO_IDS)) != FRESH_VIDEO_HASH:
        raise AssertionError("fresh video-ID hash does not reproduce")
    split = frozen_split_ids(qa_path)
    selected_order = split["opportunity"]
    selected_ids = set(selected_order)
    rows = load_selected_qa(qa_path, selected_ids)
    ordered_rows = [rows[index] for index in selected_order]
    video_ids = [str(row["vid"]) for row in ordered_rows]
    if len(video_ids) != len(set(video_ids)):
        raise AssertionError("opportunity tasks are not video-disjoint")
    if _id_hash(video_ids) != OPPORTUNITY_VIDEO_HASH:
        raise AssertionError("opportunity video hash does not reproduce")
    captions = load_selected_captions(caption_path, set(video_ids))

    records = []
    for offset, row_index in enumerate(selected_order, start=1):
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
            "opportunity_rows": OPPORTUNITY_ID_HASH,
            "confirmation_rows": CONFIRMATION_ID_HASH,
            "reserve_rows": RESERVE_ID_HASH,
            "opportunity_video_ids": OPPORTUNITY_VIDEO_HASH,
            "confirmation_video_ids": CONFIRMATION_VIDEO_HASH,
            "reserve_video_ids": RESERVE_VIDEO_HASH,
        },
        "parameters": {
            "max_roots": MAX_ROOTS,
            "max_stage_followups": MAX_STAGE_FOLLOWUPS,
            "search_depth": 4,
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

