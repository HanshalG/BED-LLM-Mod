#!/usr/bin/env python3
"""Confirm four-hop tradeoffs with frozen answer-scorable eligibility."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.longvid_bridge_path_opportunity_audit import (
    CAPTION_SHA256,
    FRESH_VIDEO_HASH,
    FRESH_VIDEO_IDS,
    QA_SHA256,
    SOURCE_REPO_COMMIT,
    _id_hash,
    load_selected_captions,
    sha256_file,
)
from scripts.longvid_four_hop_tradeoff_opportunity import (
    CONFIRMATION_ID_HASH,
    CONFIRMATION_VIDEO_HASH,
    MAX_ROOTS,
    MAX_STAGE_FOLLOWUPS,
    MIN_ANSWER_TERMS,
    MIN_CAPTIONS,
    MIN_DEPTH_FOUR_GAIN_TASKS,
    MIN_DIVERSE_TASKS,
    MIN_MEAN_COVERAGE_GAIN,
    MIN_MEAN_ORACLE_COVERAGE,
    MIN_MEAN_STRICT_ANSWER_SACRIFICE,
    MIN_ROOTS,
    MIN_STRICT_TOTAL_GAP,
    MIN_STRICT_TRADEOFFS,
    OPPORTUNITY_SIZE,
    RESERVE_ID_HASH,
    RESERVE_VIDEO_HASH,
    analyze_task,
    frozen_split_ids,
    load_selected_qa,
)


SCHEMA_VERSION = 2
MIN_ANSWER_SCORABLE_TASKS = 38


def apply_scorable_eligibility(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    scorable = int(record["num_answer_terms"]) >= MIN_ANSWER_TERMS
    eligible_strict = scorable and bool(record["strict_tradeoff"])
    enriched["answer_scorable"] = scorable
    enriched["eligible_strict_tradeoff"] = eligible_strict
    enriched["eligible_strict_gap_count"] = (
        int(record["strict_gap_count"]) if eligible_strict else 0
    )
    enriched["eligible_strict_answer_sacrifice"] = (
        float(record["strict_answer_sacrifice"]) if eligible_strict else 0.0
    )
    return enriched


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [
        record for record in records if record["eligible_strict_tradeoff"]
    ]
    summary = {
        "num_records": len(records),
        "retrieval_complete_task_count": sum(
            int(
                record["num_captions"] >= MIN_CAPTIONS
                and record["num_roots"] >= MIN_ROOTS
            )
            for record in records
        ),
        "answer_scorable_task_count": sum(
            int(record["answer_scorable"]) for record in records
        ),
        "diverse_root_task_count": sum(
            int(record["distinct_root_top1"] >= 3) for record in records
        ),
        "depth_four_gain_task_count": sum(
            int(record["depth_four_gain_count"] >= 1) for record in records
        ),
        "mean_oracle_final_coverage": _mean(
            [float(record["oracle_final_coverage"]) for record in records]
        ),
        "mean_coverage_gain": _mean(
            [float(record["coverage_gain"]) for record in records]
        ),
        "eligible_strict_tradeoff_count": len(strict),
        "eligible_strict_total_gap": sum(
            int(record["eligible_strict_gap_count"]) for record in strict
        ),
        "mean_eligible_strict_answer_sacrifice": _mean(
            [
                float(record["eligible_strict_answer_sacrifice"])
                for record in strict
            ]
        ),
    }
    gates = {
        "all_tasks_retrieval_complete": (
            summary["retrieval_complete_task_count"] == OPPORTUNITY_SIZE
        ),
        "answer_scorable_tasks_at_least_38": (
            summary["answer_scorable_task_count"] >= MIN_ANSWER_SCORABLE_TASKS
        ),
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
        "eligible_strict_tradeoffs_at_least_6": (
            summary["eligible_strict_tradeoff_count"]
            >= MIN_STRICT_TRADEOFFS
        ),
        "eligible_strict_total_gap_at_least_6": (
            summary["eligible_strict_total_gap"] >= MIN_STRICT_TOTAL_GAP
        ),
        "mean_eligible_strict_answer_sacrifice_at_least_0_15": (
            summary["mean_eligible_strict_answer_sacrifice"]
            >= MIN_MEAN_STRICT_ANSWER_SACRIFICE
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_confirmation(qa_path: Path, caption_path: Path) -> dict[str, Any]:
    if sha256_file(qa_path) != QA_SHA256:
        raise ValueError("QA SHA-256 does not match frozen source")
    if sha256_file(caption_path) != CAPTION_SHA256:
        raise ValueError("caption SHA-256 does not match frozen source")
    if _id_hash(sorted(FRESH_VIDEO_IDS)) != FRESH_VIDEO_HASH:
        raise AssertionError("fresh video-ID hash does not reproduce")
    split = frozen_split_ids(qa_path)
    selected_order = split["confirmation"]
    if _id_hash(selected_order) != CONFIRMATION_ID_HASH:
        raise AssertionError("confirmation row hash does not reproduce")
    selected_ids = set(selected_order)
    rows = load_selected_qa(qa_path, selected_ids)
    ordered_rows = [rows[index] for index in selected_order]
    video_ids = [str(row["vid"]) for row in ordered_rows]
    if len(video_ids) != len(set(video_ids)):
        raise AssertionError("confirmation tasks are not video-disjoint")
    if _id_hash(video_ids) != CONFIRMATION_VIDEO_HASH:
        raise AssertionError("confirmation video hash does not reproduce")
    captions = load_selected_captions(caption_path, set(video_ids))

    records = []
    for offset, row_index in enumerate(selected_order, start=1):
        row = rows[row_index]
        documents = captions[str(row["vid"])]
        evidence = {str(int(value)) for value in row["evidence_slices"]}
        if not evidence.issubset({document["id"] for document in documents}):
            raise ValueError(f"row {row_index} evidence is absent from captions")
        record = analyze_task(row_index, row, documents)
        records.append(apply_scorable_eligibility(record))
        print(f"completed={offset}/{len(selected_ids)}", file=sys.stderr)

    summary = summarize(records)
    return {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "confirmation_passed"
            if summary["gates"]["all_pass"]
            else "confirmation_failed"
        ),
        "source": {
            "repository_commit": SOURCE_REPO_COMMIT,
            "qa_sha256": QA_SHA256,
            "caption_sha256": CAPTION_SHA256,
        },
        "split_hashes": {
            "confirmation_rows": CONFIRMATION_ID_HASH,
            "confirmation_video_ids": CONFIRMATION_VIDEO_HASH,
            "reserve_rows": RESERVE_ID_HASH,
            "reserve_video_ids": RESERVE_VIDEO_HASH,
            "fresh_video_ids": FRESH_VIDEO_HASH,
        },
        "parameters": {
            "max_roots": MAX_ROOTS,
            "max_stage_followups": MAX_STAGE_FOLLOWUPS,
            "search_depth": 4,
            "min_answer_scorable_tasks": MIN_ANSWER_SCORABLE_TASKS,
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
    payload = run_confirmation(args.qa_path, args.caption_path)
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    args.output_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["summary"], indent=2, sort_keys=True))
    print(f"status={payload['status']}")
    print(f"output={args.output_path}")
    return 0 if payload["status"] == "confirmation_passed" else 1


if __name__ == "__main__":
    sys.exit(main())

