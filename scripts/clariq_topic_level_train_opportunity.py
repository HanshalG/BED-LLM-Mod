#!/usr/bin/env python3
"""Audit topic-level two-turn opportunity on fresh ClariQ train topics."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import pickle
from pathlib import Path
import random
import subprocess
import tarfile
import tempfile
from typing import Any, Iterable, Sequence


SOURCE_COMMIT = "46885a544581a0af8aff0681d29e4971807e2912"
TRAIN_SHA256 = (
    "65d3da13b2d6ea77e7eaa45290894ffc162a5bd000e7640decd1b0a272a6e9d1"
)
SYNTHETIC_SHA256 = (
    "04c7312fddb79494696b6d3b965c1892d9ff6767456f7200c48b018ab785e90c"
)
EVAL_PART_SHA256 = (
    "fb9562b7ac1810b3ee74a04eea2936242929169a451ca7591e5c67af5d5982e2",
    "903ad459772c716734c2f1fccd2610148d1d7414c7d5dcd9360a4bafe8e6a306",
)
EVAL_COMBINED_SHA256 = (
    "4611b0d2f551da4cfddf13fbaddac1d0cac5b388f1d9c806d9528836bd4cc509"
)
SPLIT_SEED = 24_399
SPLIT_SIZES = {
    "opportunity": 93,
    "development": 31,
    "holdout": 63,
}
SPLIT_SHA256 = {
    "opportunity": (
        "52e8cfe007e7204800fde86eed429715c25f02274c87d63f1a23db91adeca52b"
    ),
    "development": (
        "9bc6d027d463ad5dc357be0bb299c22d25bb29376d20f67bff97875fab48612e"
    ),
    "holdout": (
        "b4e1809fcaa35708a46472da0793d12c0ea20a58a9c7abe63b4e1d7619f2e711"
    ),
}
MEANINGFUL_GAIN = 0.005


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _id_hash(values: Sequence[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def build_split(topic_ids: Iterable[str]) -> dict[str, list[str]]:
    values = sorted(set(topic_ids), key=int)
    random.Random(SPLIT_SEED).shuffle(values)
    first = SPLIT_SIZES["opportunity"]
    second = first + SPLIT_SIZES["development"]
    split = {
        "opportunity": values[:first],
        "development": values[first:second],
        "holdout": values[second:],
    }
    if {name: len(ids) for name, ids in split.items()} != SPLIT_SIZES:
        raise ValueError("ClariQ split sizes do not reproduce")
    if any(_id_hash(split[name]) != SPLIT_SHA256[name] for name in split):
        raise ValueError("ClariQ split hashes do not reproduce")
    return split


def _load_tar_pickle(path: Path) -> Any:
    with tarfile.open(path, "r:gz") as archive:
        members = [
            member
            for member in archive.getmembers()
            if not Path(member.name).name.startswith("._")
        ]
        if len(members) != 1:
            raise ValueError(f"{path.name} has unexpected archive members")
        stream = archive.extractfile(members[0])
        if stream is None:
            raise ValueError(f"{path.name} pickle member cannot be read")
        return pickle.load(stream)


def _load_split_archive(parts: Sequence[Path]) -> Any:
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".tar.gz") as combined:
        for part in parts:
            with part.open("rb") as source:
                while chunk := source.read(1024 * 1024):
                    digest.update(chunk)
                    combined.write(chunk)
        combined.flush()
        if digest.hexdigest() != EVAL_COMBINED_SHA256:
            raise ValueError("combined ClariQ evaluation hash does not match")
        return _load_tar_pickle(Path(combined.name))


def verify_source(source_root: Path) -> dict[str, Path]:
    commit = subprocess.check_output(
        ["git", "-C", str(source_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()
    if commit != SOURCE_COMMIT:
        raise ValueError("ClariQ source commit does not match")
    data = source_root / "data"
    paths = {
        "train": data / "train.tsv",
        "synthetic": data / "train_synthetic.pkl.tar.gz",
        "eval_a": data / "multi_turn_train_eval.pkl.tar.gz.aa",
        "eval_b": data / "multi_turn_train_eval.pkl.tar.gz.ab",
    }
    expected = {
        "train": TRAIN_SHA256,
        "synthetic": SYNTHETIC_SHA256,
        "eval_a": EVAL_PART_SHA256[0],
        "eval_b": EVAL_PART_SHA256[1],
    }
    for name, path in paths.items():
        if _sha256(path) != expected[name]:
            raise ValueError(f"ClariQ {name} hash does not match")
    return paths


def _history_key(history: Sequence[dict[str, str]]) -> tuple[tuple[str, str], ...]:
    return tuple((item["question"], item["answer"]) for item in history)


def analyze_topics(
    synthetic: dict[Any, dict[str, Any]],
    evaluation: dict[Any, dict[str, Any]],
    question_ids: dict[tuple[int, str], str],
    opportunity_ids: Sequence[str],
) -> dict[str, Any]:
    opportunity = {int(value) for value in opportunity_ids}
    states: dict[tuple[int, str, tuple[tuple[str, str], ...]], Any] = {}
    questions: dict[
        tuple[int, str, tuple[tuple[str, str], ...]], set[str]
    ] = defaultdict(set)
    answers: dict[
        tuple[
            tuple[int, str, tuple[tuple[str, str], ...]],
            str,
        ],
        str,
    ] = {}
    topic_facets: dict[int, set[str]] = defaultdict(set)

    for row in synthetic.values():
        topic_id = int(row["topic_id"])
        if topic_id not in opportunity:
            continue
        history = _history_key(row["conversation_context"])
        state = (topic_id, str(row["facet_id"]), history)
        states[state] = row["context_id"]
        questions[state].add(str(row["question"]))
        answers[(state, str(row["question"]))] = str(row["answer"])
        if not history:
            topic_facets[topic_id].add(str(row["facet_id"]))

    records = []
    exclusions: Counter[str] = Counter()
    accessed_context_ids: set[Any] = set()
    for topic_id in map(int, opportunity_ids):
        facets = sorted(topic_facets.get(topic_id, set()))
        if len(facets) < 2:
            exclusions["fewer_than_two_facets"] += 1
            continue
        root_questions = set.intersection(
            *(questions[(topic_id, facet, ())] for facet in facets)
        )
        if len(root_questions) < 2:
            exclusions["fewer_than_two_common_roots"] += 1
            continue

        root_rows = []
        for root_question in sorted(root_questions):
            question_id = question_ids.get((topic_id, root_question))
            if question_id is None:
                continue
            immediate_values = []
            observation_groups: dict[str, list[tuple[str, Any]]] = defaultdict(
                list
            )
            valid = True
            for facet in facets:
                initial_state = (topic_id, facet, ())
                context_id = states[initial_state]
                if question_id not in evaluation.get(context_id, {}):
                    valid = False
                    break
                answer = answers[(initial_state, root_question)]
                successor_state = (
                    topic_id,
                    facet,
                    ((root_question, answer),),
                )
                if successor_state not in states:
                    valid = False
                    break
                immediate_values.append(
                    evaluation[context_id][question_id]["with_answer"]
                )
                accessed_context_ids.add(context_id)
                observation_groups[answer].append(
                    (facet, states[successor_state])
                )
            if not valid:
                continue

            terminal_sum = 0.0
            branch_rows = []
            for answer, members in sorted(observation_groups.items()):
                followup_questions = set.intersection(
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
                followup_questions.discard(root_question)
                options = []
                for followup_question in followup_questions:
                    followup_id = question_ids.get(
                        (topic_id, followup_question)
                    )
                    if followup_id is None:
                        continue
                    values = []
                    for _facet, context_id in members:
                        if followup_id not in evaluation.get(context_id, {}):
                            break
                        values.append(
                            evaluation[context_id][followup_id]["with_answer"]
                        )
                    if len(values) == len(members):
                        options.append(
                            (
                                sum(values) / len(values),
                                followup_id,
                            )
                        )
                if not options:
                    valid = False
                    break
                best_value, best_question_id = min(
                    options,
                    key=lambda item: (-item[0], item[1]),
                )
                terminal_sum += len(members) * best_value
                accessed_context_ids.update(
                    context_id for _facet, context_id in members
                )
                branch_rows.append(
                    {
                        "answer": answer,
                        "facet_count": len(members),
                        "best_followup_question_id": best_question_id,
                        "mean_terminal_utility": best_value,
                    }
                )
            if valid:
                root_rows.append(
                    {
                        "question_id": question_id,
                        "immediate_utility": (
                            sum(immediate_values) / len(facets)
                        ),
                        "terminal_utility": terminal_sum / len(facets),
                        "observation_branch_count": len(observation_groups),
                        "branches": branch_rows,
                    }
                )

        if len(root_rows) < 2:
            exclusions["fewer_than_two_valid_roots"] += 1
            continue
        greedy = min(
            root_rows,
            key=lambda row: (-row["immediate_utility"], row["question_id"]),
        )
        depth_two = min(
            root_rows,
            key=lambda row: (-row["terminal_utility"], row["question_id"]),
        )
        gap = depth_two["terminal_utility"] - greedy["terminal_utility"]
        records.append(
            {
                "topic_id": str(topic_id),
                "facet_count": len(facets),
                "valid_root_count": len(root_rows),
                "greedy_question_id": greedy["question_id"],
                "depth_two_question_id": depth_two["question_id"],
                "greedy_immediate_utility": greedy["immediate_utility"],
                "depth_two_immediate_utility": depth_two[
                    "immediate_utility"
                ],
                "greedy_terminal_utility": greedy["terminal_utility"],
                "depth_two_terminal_utility": depth_two["terminal_utility"],
                "terminal_gain": gap,
                "roots": root_rows,
            }
        )

    gains = [record["terminal_gain"] for record in records]
    summary = {
        "opportunity_topic_count": len(opportunity_ids),
        "usable_topic_count": len(records),
        "excluded_topic_count": len(opportunity_ids) - len(records),
        "exclusion_reasons": dict(exclusions),
        "roots_differ_count": sum(
            row["greedy_question_id"] != row["depth_two_question_id"]
            for row in records
        ),
        "positive_gain_count": sum(value > 1e-12 for value in gains),
        "meaningful_gain_count": sum(
            value >= MEANINGFUL_GAIN - 1e-12 for value in gains
        ),
        "mean_terminal_gain": (
            sum(gains) / len(gains) if gains else 0.0
        ),
        "max_terminal_gain": max(gains, default=0.0),
        "accessed_evaluation_context_count": len(accessed_context_ids),
    }
    gates = {
        "usable_topics_at_least_50": summary["usable_topic_count"] >= 50,
        "roots_differ_at_least_30": summary["roots_differ_count"] >= 30,
        "positive_gain_at_least_25": summary["positive_gain_count"] >= 25,
        "meaningful_gain_at_least_15": (
            summary["meaningful_gain_count"] >= 15
        ),
        "mean_gain_at_least_0_005": (
            summary["mean_terminal_gain"] >= MEANINGFUL_GAIN
        ),
        "max_gain_at_least_0_02": summary["max_terminal_gain"] >= 0.02,
    }
    gates["all_pass"] = all(gates.values())
    summary["gates"] = gates
    return {
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "summary": summary,
        "records": records,
    }


def run(source_root: Path) -> dict[str, Any]:
    paths = verify_source(source_root)
    with paths["train"].open(encoding="utf-8", newline="") as handle:
        metadata_reader = csv.DictReader(handle, delimiter="\t")
        topic_ids = [row["topic_id"] for row in metadata_reader]
    split = build_split(topic_ids)
    opportunity = set(split["opportunity"])
    question_ids: dict[tuple[int, str], str] = {}
    with paths["train"].open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if row["topic_id"] in opportunity:
                key = (int(row["topic_id"]), row["question"])
                previous = question_ids.setdefault(key, row["question_id"])
                if previous != row["question_id"]:
                    raise ValueError("topic question text maps to multiple IDs")

    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation_payload = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )
    evaluation = evaluation_payload["NDCG20"]
    analysis = analyze_topics(
        synthetic,
        evaluation,
        question_ids,
        split["opportunity"],
    )
    return {
        "schema_version": 1,
        "status": analysis["status"],
        "protocol": {
            "source_commit": SOURCE_COMMIT,
            "split_seed": SPLIT_SEED,
            "split_sizes": {name: len(ids) for name, ids in split.items()},
            "split_sha256": {
                name: _id_hash(ids) for name, ids in split.items()
            },
            "opportunity_ids": split["opportunity"],
            "utility": "NDCG20.with_answer",
            "facet_prior": "uniform within topic",
            "identical_answers_share_one_observation_branch": True,
            "root_repeat_disallowed": True,
            "question_id_tie_break": "lexicographically smallest",
            "api_calls": 0,
            "development_and_holdout_entries_indexed": False,
        },
        "summary": analysis["summary"],
        "records": analysis["records"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.source_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
