#!/usr/bin/env python3
"""Freeze ClariQ tasks for path-dependent open-support planning."""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import hashlib
import json
from pathlib import Path
import string
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.clariq_topic_level_train_opportunity import (
    _history_key,
    _load_split_archive,
    _load_tar_pickle,
    build_split,
    verify_source,
)


SCHEMA_VERSION = 1
SPLIT_SEED = 24_399
MECHANICS_TOPIC_IDS = ("38",)
DEVELOPMENT_TOPIC_IDS = ("148",)
SEALED_HOLDOUT_TOPIC_IDS = ("102", "11", "103", "141")
PRIOR_FIXED_SUPPORT_TOPIC_IDS = (
    "46",
    "177",
    "117",
    "136",
    "125",
    "149",
    "115",
    "10",
    "104",
    "14",
    "135",
    "122",
    "105",
    "131",
    "113",
    "137",
    "119",
    "116",
)
EXPECTED_ROOT_COUNTS = {"38": 13, "148": 13}
EXPECTED_BRANCH_COUNTS = {"38": 39, "148": 52}
EXPECTED_MODEL_REQUESTS = {"38": 40, "148": 53}


def _ordered_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(
        json.dumps(list(values), separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _response_options(answers: Sequence[str]) -> list[dict[str, str]]:
    ordered = sorted(set(answers))
    if not ordered or len(ordered) > len(string.ascii_uppercase):
        raise ValueError("ClariQ response alphabet is invalid")
    return [
        {"code": string.ascii_uppercase[index], "answer": answer}
        for index, answer in enumerate(ordered)
    ]


def build_topic_structure(
    topic_id: str,
    metadata_rows: Sequence[Mapping[str, str]],
    synthetic: Mapping[Any, Mapping[str, Any]],
    evaluation: Mapping[Any, Mapping[str, Any]],
) -> dict[str, Any]:
    """Build a key-only task graph without emitting latent facet profiles."""

    numeric_topic_id = int(topic_id)
    requests = {str(row["initial_request"]) for row in metadata_rows}
    if len(requests) != 1:
        raise ValueError("ClariQ topic has multiple initial requests")
    question_text_by_id: dict[str, str] = {}
    question_id_by_text: dict[str, str] = {}
    for row in metadata_rows:
        question_id = str(row["question_id"])
        question = str(row["question"])
        old_text = question_text_by_id.setdefault(question_id, question)
        old_id = question_id_by_text.setdefault(question, question_id)
        if old_text != question or old_id != question_id:
            raise ValueError("ClariQ question metadata is inconsistent")

    states: dict[
        tuple[int, str, tuple[tuple[str, str], ...]], Any
    ] = {}
    available_questions: dict[
        tuple[int, str, tuple[tuple[str, str], ...]], set[str]
    ] = defaultdict(set)
    answers: dict[
        tuple[
            tuple[int, str, tuple[tuple[str, str], ...]],
            str,
        ],
        str,
    ] = {}
    global_answers: dict[str, set[str]] = defaultdict(set)
    facets: set[str] = set()
    for row in synthetic.values():
        if int(row["topic_id"]) != numeric_topic_id:
            continue
        history = _history_key(row["conversation_context"])
        facet_id = str(row["facet_id"])
        state = (numeric_topic_id, facet_id, history)
        question = str(row["question"])
        answer = str(row["answer"])
        states[state] = row["context_id"]
        available_questions[state].add(question)
        answers[(state, question)] = answer
        global_answers[question].add(answer)
        if not history:
            facets.add(facet_id)
    if len(facets) < 2:
        raise ValueError("ClariQ topic has fewer than two latent facets")

    common_roots = set.intersection(
        *(
            available_questions[(numeric_topic_id, facet_id, ())]
            for facet_id in sorted(facets)
        )
    )
    roots: list[dict[str, Any]] = []
    valid_question_ids: set[str] = set()
    for root_question in sorted(common_roots):
        root_id = question_id_by_text.get(root_question)
        if root_id is None:
            continue
        response_groups: dict[
            str, list[tuple[str, Any]]
        ] = defaultdict(list)
        valid = True
        for facet_id in sorted(facets):
            initial_state = (numeric_topic_id, facet_id, ())
            context_id = states[initial_state]
            if root_id not in evaluation.get(context_id, {}):
                valid = False
                break
            answer = answers[(initial_state, root_question)]
            successor_state = (
                numeric_topic_id,
                facet_id,
                ((root_question, answer),),
            )
            if successor_state not in states:
                valid = False
                break
            response_groups[answer].append(
                (facet_id, states[successor_state])
            )
        if not valid:
            continue

        root_options = _response_options(global_answers[root_question])
        root_code_by_answer = {
            option["answer"]: option["code"] for option in root_options
        }
        branches: list[dict[str, Any]] = []
        for answer, members in sorted(response_groups.items()):
            followup_questions = set.intersection(
                *(
                    available_questions[
                        (
                            numeric_topic_id,
                            facet_id,
                            ((root_question, answer),),
                        )
                    ]
                    for facet_id, _context_id in members
                )
            )
            followup_questions.discard(root_question)
            legal_followup_ids = []
            for followup_question in sorted(followup_questions):
                followup_id = question_id_by_text.get(followup_question)
                if followup_id is None:
                    continue
                if all(
                    followup_id in evaluation.get(context_id, {})
                    for _facet_id, context_id in members
                ):
                    legal_followup_ids.append(followup_id)
            if not legal_followup_ids:
                valid = False
                break
            branches.append(
                {
                    "response_code": root_code_by_answer[answer],
                    "answer": answer,
                    "legal_followup_question_ids": sorted(
                        legal_followup_ids
                    ),
                }
            )
            valid_question_ids.update(legal_followup_ids)
        if valid:
            valid_question_ids.add(root_id)
            roots.append(
                {
                    "question_id": root_id,
                    "branches": sorted(
                        branches,
                        key=lambda branch: branch["response_code"],
                    ),
                }
            )

    roots.sort(key=lambda root: root["question_id"])
    valid_root_ids = {root["question_id"] for root in roots}
    if valid_question_ids != valid_root_ids:
        raise ValueError("ClariQ followup bank differs from valid roots")
    question_bank = []
    for question_id in sorted(valid_root_ids):
        question = question_text_by_id[question_id]
        question_bank.append(
            {
                "question_id": question_id,
                "question": question,
                "response_options": _response_options(
                    global_answers[question]
                ),
            }
        )
    return {
        "topic_id": topic_id,
        "initial_request": requests.pop(),
        "question_bank": question_bank,
        "roots": roots,
        "root_count": len(roots),
        "branch_count": sum(len(root["branches"]) for root in roots),
        "expected_model_requests": 1
        + sum(len(root["branches"]) for root in roots),
        "latent_facet_descriptions_emitted": False,
        "latent_cross_question_profiles_emitted": False,
        "evaluation_values_read": False,
    }


def build_manifest(source_root: Path) -> dict[str, Any]:
    paths = verify_source(source_root)
    with paths["train"].open(encoding="utf-8", newline="") as handle:
        metadata = list(csv.DictReader(handle, delimiter="\t"))
    split = build_split(row["topic_id"] for row in metadata)
    if any(
        topic_id not in split["opportunity"]
        for topic_id in MECHANICS_TOPIC_IDS
    ):
        raise ValueError("ClariQ mechanics selection changed")
    if any(
        topic_id not in split["development"]
        for topic_id in DEVELOPMENT_TOPIC_IDS
    ):
        raise ValueError("ClariQ development selection changed")
    if any(
        topic_id not in split["holdout"]
        for topic_id in SEALED_HOLDOUT_TOPIC_IDS
    ):
        raise ValueError("ClariQ sealed holdout selection changed")
    if set(DEVELOPMENT_TOPIC_IDS) & set(PRIOR_FIXED_SUPPORT_TOPIC_IDS):
        raise ValueError("ClariQ development topic was previously used")
    if set(SEALED_HOLDOUT_TOPIC_IDS) & set(PRIOR_FIXED_SUPPORT_TOPIC_IDS):
        raise ValueError("ClariQ holdout topic was previously used")

    by_topic: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in metadata:
        if row["topic_id"] in (
            set(MECHANICS_TOPIC_IDS) | set(DEVELOPMENT_TOPIC_IDS)
        ):
            by_topic[row["topic_id"]].append(row)
    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )["NDCG20"]
    stages = {
        "mechanics": [
            build_topic_structure(
                topic_id,
                by_topic[topic_id],
                synthetic,
                evaluation,
            )
            for topic_id in MECHANICS_TOPIC_IDS
        ],
        "development": [
            build_topic_structure(
                topic_id,
                by_topic[topic_id],
                synthetic,
                evaluation,
            )
            for topic_id in DEVELOPMENT_TOPIC_IDS
        ],
    }
    for tasks in stages.values():
        for task in tasks:
            topic_id = task["topic_id"]
            if task["root_count"] != EXPECTED_ROOT_COUNTS[topic_id]:
                raise ValueError("ClariQ valid root count changed")
            if task["branch_count"] != EXPECTED_BRANCH_COUNTS[topic_id]:
                raise ValueError("ClariQ valid branch count changed")
            if (
                task["expected_model_requests"]
                != EXPECTED_MODEL_REQUESTS[topic_id]
            ):
                raise ValueError("ClariQ model request count changed")

    selected_ids = (
        list(MECHANICS_TOPIC_IDS)
        + list(DEVELOPMENT_TOPIC_IDS)
        + list(SEALED_HOLDOUT_TOPIC_IDS)
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "passed",
        "protocol": {
            "split_seed": SPLIT_SEED,
            "mechanics_selection_is_endpoint_disclosed": True,
            "development_utility_values_read": False,
            "holdout_content_or_utility_values_read": False,
            "latent_facet_descriptions_emitted": False,
            "latent_cross_question_profiles_emitted": False,
            "api_calls": 0,
        },
        "stages": stages,
        "sealed_holdout": {
            "topic_ids": list(SEALED_HOLDOUT_TOPIC_IDS),
            "content_emitted": False,
            "utility_values_read": False,
        },
        "prior_fixed_support_topic_ids": list(
            PRIOR_FIXED_SUPPORT_TOPIC_IDS
        ),
        "selected_topic_order_sha256": _ordered_hash(selected_ids),
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
                "mechanics": [
                    {
                        "topic_id": task["topic_id"],
                        "roots": task["root_count"],
                        "branches": task["branch_count"],
                        "requests": task["expected_model_requests"],
                    }
                    for task in result["stages"]["mechanics"]
                ],
                "development": [
                    {
                        "topic_id": task["topic_id"],
                        "roots": task["root_count"],
                        "branches": task["branch_count"],
                        "requests": task["expected_model_requests"],
                    }
                    for task in result["stages"]["development"]
                ],
                "sealed_holdout_topic_ids": result["sealed_holdout"][
                    "topic_ids"
                ],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
