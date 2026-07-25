#!/usr/bin/env python3
"""Freeze structurally evaluable ClariQ development topics without utilities."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clariq_topic_level_train_opportunity import (
    _history_key,
    _load_split_archive,
    _load_tar_pickle,
    build_split,
    verify_source,
)


EXCLUDED_V1_TOPIC_IDS = {"46", "177", "117"}
SELECTED_TOPIC_COUNT = 3
MIN_VALID_ROOTS = 6
SAMPLES_PER_QUESTION = 5


def _structural_roots(
    synthetic: dict[Any, dict[str, Any]],
    evaluation: dict[Any, dict[str, Any]],
    question_ids: dict[tuple[int, str], str],
    topic_id: int,
    facets: Sequence[str],
) -> list[str]:
    states: dict[
        tuple[int, str, tuple[tuple[str, str], ...]], Any
    ] = {}
    questions: dict[
        tuple[int, str, tuple[tuple[str, str], ...]], set[str]
    ] = defaultdict(set)
    answers = {}
    for row in synthetic.values():
        if int(row["topic_id"]) != topic_id:
            continue
        history = _history_key(row["conversation_context"])
        state = (topic_id, str(row["facet_id"]), history)
        states[state] = row["context_id"]
        questions[state].add(str(row["question"]))
        answers[(state, str(row["question"]))] = str(row["answer"])

    if any((topic_id, facet, ()) not in states for facet in facets):
        return []
    common_roots = set.intersection(
        *(questions[(topic_id, facet, ())] for facet in facets)
    )
    valid_root_ids = []
    for root_question in common_roots:
        root_id = question_ids.get((topic_id, root_question))
        if root_id is None:
            continue
        groups: dict[str, list[tuple[str, Any]]] = defaultdict(list)
        valid = True
        for facet in facets:
            initial = (topic_id, facet, ())
            initial_context = states[initial]
            if root_id not in evaluation.get(initial_context, {}):
                valid = False
                break
            answer = answers[(initial, root_question)]
            successor = (
                topic_id,
                facet,
                ((root_question, answer),),
            )
            if successor not in states:
                valid = False
                break
            groups[answer].append((facet, states[successor]))
        if not valid:
            continue
        for answer, members in groups.items():
            followups = set.intersection(
                *(
                    questions[
                        (
                            topic_id,
                            facet,
                            ((root_question, answer),),
                        )
                    ]
                    for facet, _context_id in members
                )
            )
            followups.discard(root_question)
            has_valid_followup = False
            for followup in followups:
                followup_id = question_ids.get((topic_id, followup))
                if followup_id is not None and all(
                    followup_id in evaluation.get(context_id, {})
                    for _facet, context_id in members
                ):
                    has_valid_followup = True
                    break
            if not has_valid_followup:
                valid = False
                break
        if valid:
            valid_root_ids.append(root_id)
    return sorted(set(valid_root_ids))


def build_manifest(source_root: Path) -> dict[str, Any]:
    paths = verify_source(source_root)
    with paths["train"].open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    split = build_split(row["topic_id"] for row in rows)
    development = split["development"]
    by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["topic_id"] in development:
            by_topic[row["topic_id"]].append(row)

    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation_payload = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )
    evaluation = evaluation_payload["NDCG20"]
    selected = []
    candidates = []
    for topic_id in development:
        if topic_id in EXCLUDED_V1_TOPIC_IDS:
            candidates.append(
                {"topic_id": topic_id, "status": "excluded_v1"}
            )
            continue
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
        if status == "eligible" and len(selected) < SELECTED_TOPIC_COUNT:
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
    expected_requests = (
        sum(len(task["questions"]) for task in selected)
        * SAMPLES_PER_QUESTION
    )
    status = (
        "passed"
        if len(selected) == SELECTED_TOPIC_COUNT
        else "gate_failed"
    )
    return {
        "schema_version": 1,
        "status": status,
        "protocol": {
            "development_split_order_used": True,
            "excluded_v1_topic_ids": sorted(EXCLUDED_V1_TOPIC_IDS),
            "min_valid_roots": MIN_VALID_ROOTS,
            "selected_topic_count": SELECTED_TOPIC_COUNT,
            "samples_per_question": SAMPLES_PER_QUESTION,
            "evaluation_keys_checked": True,
            "evaluation_values_read": False,
            "api_calls": 0,
        },
        "selected_topics": selected,
        "selected_topic_ids": [task["topic_id"] for task in selected],
        "expected_likelihood_requests": expected_requests,
        "candidates": candidates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_manifest(args.source_root)
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
