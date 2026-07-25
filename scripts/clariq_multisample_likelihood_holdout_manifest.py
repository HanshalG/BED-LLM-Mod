#!/usr/bin/env python3
"""Freeze structurally evaluable ClariQ holdout topics without utilities."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clariq_multisample_likelihood_v2_manifest import (
    MIN_VALID_ROOTS,
    SAMPLES_PER_QUESTION,
    _structural_roots,
)
from scripts.clariq_topic_level_train_opportunity import (
    _load_split_archive,
    _load_tar_pickle,
    build_split,
    verify_source,
)


MIN_SELECTED_TOPICS = 8
MAX_SELECTED_TOPICS = 12


def build_holdout_manifest(source_root: Path) -> dict:
    paths = verify_source(source_root)
    with paths["train"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    split = build_split(row["topic_id"] for row in rows)
    holdout = split["holdout"]
    by_topic = defaultdict(list)
    for row in rows:
        if row["topic_id"] in holdout:
            by_topic[row["topic_id"]].append(row)

    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation_payload = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )
    evaluation = evaluation_payload["NDCG20"]
    selected = []
    candidates = []
    for topic_id in holdout:
        topic_rows = by_topic[topic_id]
        facets = {
            row["facet_id"]: row["facet_desc"] for row in topic_rows
        }
        questions = {
            row["question_id"]: row["question"] for row in topic_rows
        }
        if not 3 <= len(facets) <= 6 or len(questions) < MIN_VALID_ROOTS:
            candidates.append(
                {
                    "topic_id": topic_id,
                    "status": "metadata_ineligible",
                    "facet_count": len(facets),
                    "question_count": len(questions),
                }
            )
            continue
        question_ids = {
            (int(topic_id), row["question"]): row["question_id"]
            for row in topic_rows
        }
        valid_root_ids = _structural_roots(
            synthetic,
            evaluation,
            question_ids,
            int(topic_id),
            sorted(facets),
        )
        status = (
            "eligible"
            if len(valid_root_ids) >= MIN_VALID_ROOTS
            else "structurally_ineligible"
        )
        candidates.append(
            {
                "topic_id": topic_id,
                "status": status,
                "facet_count": len(facets),
                "question_count": len(questions),
                "valid_root_count": len(valid_root_ids),
            }
        )
        if status == "eligible" and len(selected) < MAX_SELECTED_TOPICS:
            requests = {row["initial_request"] for row in topic_rows}
            if len(requests) != 1:
                raise ValueError("ClariQ topic has multiple initial requests")
            selected.append(
                {
                    "topic_id": topic_id,
                    "initial_request": requests.pop(),
                    "facets": [
                        {
                            "facet_id": facet_id,
                            "description": facets[facet_id],
                        }
                        for facet_id in sorted(facets)
                    ],
                    "questions": [
                        {
                            "question_id": question_id,
                            "question": questions[question_id],
                        }
                        for question_id in valid_root_ids
                    ],
                }
            )
    status = (
        "passed"
        if len(selected) >= MIN_SELECTED_TOPICS
        else "gate_failed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "protocol": {
            "split": "holdout",
            "min_selected_topics": MIN_SELECTED_TOPICS,
            "max_selected_topics": MAX_SELECTED_TOPICS,
            "min_valid_roots": MIN_VALID_ROOTS,
            "samples_per_question": SAMPLES_PER_QUESTION,
            "evaluation_keys_checked": True,
            "evaluation_values_read": False,
            "api_calls": 0,
        },
        "selected_topics": selected,
        "selected_topic_ids": [task["topic_id"] for task in selected],
        "expected_likelihood_requests": (
            sum(len(task["questions"]) for task in selected)
            * SAMPLES_PER_QUESTION
        ),
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_holdout_manifest(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_topic_ids": result["selected_topic_ids"],
                "valid_root_counts": [
                    len(task["questions"])
                    for task in result["selected_topics"]
                ],
                "expected_likelihood_requests": result[
                    "expected_likelihood_requests"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
