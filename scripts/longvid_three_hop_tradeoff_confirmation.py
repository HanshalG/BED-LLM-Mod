#!/usr/bin/env python3
"""Confirm a semantic three-hop root tradeoff on frozen LongVid reserve tasks."""

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
from scripts.longvid_three_hop_opportunity_audit import (
    MAX_ROOTS,
    MAX_STAGE_FOLLOWUPS,
    MIN_ANSWER_TERMS,
    MIN_CAPTIONS,
    MIN_DIVERSE_TASKS,
    MIN_MEAN_COVERAGE_GAIN,
    MIN_MEAN_ORACLE_COVERAGE,
    MIN_ROOTS,
    OPPORTUNITY_SIZE,
    RESERVE_ID_HASH,
    analyze_task,
    frozen_split_ids,
    load_selected_qa,
)


SCHEMA_VERSION = 1
RESERVE_VIDEO_HASH = (
    "49973b0c7c1e1d480e2035229f6ff326aae88f5d3eabba7dbbf70e35bde0388a"
)
MIN_DEPTH_THREE_GAIN_TASKS = 15
MIN_STRICT_TRADEOFFS = 5
MIN_STRICT_TOTAL_GAP = 5
MIN_MEAN_STRICT_ANSWER_SACRIFICE = 0.15


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def add_semantic_tradeoff(record: dict[str, Any]) -> dict[str, Any]:
    enriched = dict(record)
    strict = (
        record["greedy_root_index"] != record["oracle_root_index"]
        and record["oracle_direct_answer"] < record["greedy_direct_answer"]
        and record["oracle_triple_count"] > record["greedy_triple_count"]
    )
    enriched["semantic_tradeoff"] = strict
    enriched["semantic_tradeoff_gap_count"] = (
        int(record["oracle_triple_count"] - record["greedy_triple_count"])
        if strict
        else 0
    )
    enriched["semantic_tradeoff_answer_sacrifice"] = (
        float(record["greedy_direct_answer"] - record["oracle_direct_answer"])
        if strict
        else 0.0
    )
    return enriched


def summarize(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    strict = [record for record in records if record["semantic_tradeoff"]]
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
        "mean_oracle_triple_coverage": _mean(
            [float(record["oracle_triple_coverage"]) for record in records]
        ),
        "mean_coverage_gain": _mean(
            [float(record["coverage_gain"]) for record in records]
        ),
        "semantic_tradeoff_count": len(strict),
        "semantic_tradeoff_total_gap": sum(
            int(record["semantic_tradeoff_gap_count"]) for record in strict
        ),
        "mean_semantic_tradeoff_answer_sacrifice": _mean(
            [
                float(record["semantic_tradeoff_answer_sacrifice"])
                for record in strict
            ]
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
        "mean_oracle_triple_coverage_at_least_0_40": (
            summary["mean_oracle_triple_coverage"]
            >= MIN_MEAN_ORACLE_COVERAGE
        ),
        "mean_coverage_gain_at_least_0_20": (
            summary["mean_coverage_gain"] >= MIN_MEAN_COVERAGE_GAIN
        ),
        "semantic_tradeoffs_at_least_5": (
            summary["semantic_tradeoff_count"] >= MIN_STRICT_TRADEOFFS
        ),
        "semantic_tradeoff_total_gap_at_least_5": (
            summary["semantic_tradeoff_total_gap"]
            >= MIN_STRICT_TOTAL_GAP
        ),
        "mean_semantic_tradeoff_answer_sacrifice_at_least_0_15": (
            summary["mean_semantic_tradeoff_answer_sacrifice"]
            >= MIN_MEAN_STRICT_ANSWER_SACRIFICE
        ),
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return summary


def run_confirmation(
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
    selected_order = split["reserve"]
    if _id_hash(selected_order) != RESERVE_ID_HASH:
        raise AssertionError("reserve row hash does not reproduce")
    selected_ids = set(selected_order)
    rows = load_selected_qa(qa_path, selected_ids)
    ordered_rows = [rows[index] for index in selected_order]
    video_ids = [str(row["vid"]) for row in ordered_rows]
    if len(video_ids) != len(set(video_ids)):
        raise AssertionError("reserve tasks are not video-disjoint")
    if _id_hash(video_ids) != RESERVE_VIDEO_HASH:
        raise AssertionError("reserve video hash does not reproduce")
    captions = load_selected_captions(caption_path, set(video_ids))

    records = []
    for offset, row_index in enumerate(selected_order, start=1):
        row = rows[row_index]
        documents = captions[str(row["vid"])]
        evidence = {str(int(value)) for value in row["evidence_slices"]}
        if not evidence.issubset({document["id"] for document in documents}):
            raise ValueError(f"row {row_index} evidence is absent from captions")
        record = analyze_task(row_index, row, documents)
        records.append(add_semantic_tradeoff(record))
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
            "reserve_rows": RESERVE_ID_HASH,
            "reserve_video_ids": RESERVE_VIDEO_HASH,
            "fresh_video_ids": FRESH_VIDEO_HASH,
        },
        "parameters": {
            "max_roots": MAX_ROOTS,
            "max_first_followups": MAX_STAGE_FOLLOWUPS,
            "max_second_followups": MAX_STAGE_FOLLOWUPS,
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

