#!/usr/bin/env python3
"""Replay failed ClariQ V2 as joint intent-response particles."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts import browsecomp_plus_semantic_mechanics as flat
from scripts.clariq_dynamic_support_smoke import (
    Support,
    _load_split_archive,
    _load_tar_pickle,
    _question_payload,
    policy_scores,
    spearman,
)
from scripts.clariq_dynamic_support_v2_serving import load_manifest
from scripts.clariq_topic_level_train_opportunity import (
    analyze_topics,
    verify_source,
)


RAW_SHA256 = (
    "752718c675f3e2c9d7f89dcb94b2a26286e563c9e68a1ea0c8237d8ff78bab9e"
)
EXPECTED_REQUESTS = 91
EXPECTED_BRANCHES = 90
HYPOTHESIS_COUNT = 8


def parse_particle_support(
    text: str,
    *,
    questions: Sequence[Mapping[str, Any]],
) -> Support:
    lines = flat._response_lines(text)
    if len(lines) != HYPOTHESIS_COUNT:
        raise ValueError("wrong particle-support line count")
    question_ids = tuple(str(row["question_id"]) for row in questions)
    allowed_codes = [
        {str(option["code"]) for option in row["response_options"]}
        for row in questions
    ]
    masses: list[int] = []
    hypotheses: list[str] = []
    predictions: list[str] = []
    for index, line in enumerate(lines, start=1):
        parts = line.split("|")
        if len(parts) != 4 or parts[0] != f"H{index:02d}":
            raise ValueError("invalid particle-support line")
        masses.append(
            flat._canonical_integer(
                parts[1],
                minimum=0,
                maximum=100,
            )
        )
        tokens = parts[2].split(" ")
        if (
            parts[2] != " ".join(tokens)
            or len(tokens) != len(questions)
            or any(len(token) != 1 for token in tokens)
        ):
            raise ValueError("particle response-code shape changed")
        if any(
            code not in allowed
            for code, allowed in zip(tokens, allowed_codes, strict=True)
        ):
            raise ValueError("particle response code is invalid")
        prediction = "".join(tokens)
        hypothesis = " ".join(parts[3].split())
        if not hypothesis:
            raise ValueError("particle intent is empty")
        hypotheses.append(hypothesis)
        predictions.append(prediction)
    identities = {
        (flat.normalize_answer(hypothesis), prediction)
        for hypothesis, prediction in zip(
            hypotheses,
            predictions,
            strict=True,
        )
    }
    if len(identities) != HYPOTHESIS_COUNT:
        raise ValueError("joint intent-response particles must be distinct")
    total = sum(masses)
    if total <= 0:
        raise ValueError("particle support must have positive total mass")
    return Support(
        question_ids=question_ids,
        hypotheses=tuple(hypotheses),
        probabilities=tuple(mass / total for mass in masses),
        predictions=tuple(predictions),
    )


def run_replay(
    *,
    source_root: Path,
    manifest_path: Path,
    raw_path: Path,
) -> dict[str, Any]:
    if hashlib.sha256(raw_path.read_bytes()).hexdigest() != RAW_SHA256:
        raise ValueError("ClariQ failed-tree raw hash changed")
    task = load_manifest(manifest_path)
    raw = json.loads(raw_path.read_text(encoding="utf-8"))
    if len(raw["initial_responses"]) != 1:
        raise ValueError("ClariQ replay initial response count changed")
    if (
        len(raw["branch_keys"]) != EXPECTED_BRANCHES
        or len(raw["branch_responses"]) != EXPECTED_BRANCHES
    ):
        raise ValueError("ClariQ replay branch response count changed")
    question_ids = tuple(
        str(root["question_id"]) for root in task["roots"]
    )
    initial = parse_particle_support(
        raw["initial_responses"][0],
        questions=_question_payload(task, question_ids),
    )
    branch_lookup = {
        (str(root["question_id"]), str(branch["response_code"])): branch
        for root in task["roots"]
        for branch in root["branches"]
    }
    branches = {
        tuple(key): parse_particle_support(
            response,
            questions=_question_payload(
                task,
                branch_lookup[tuple(key)][
                    "legal_followup_question_ids"
                ],
            ),
        )
        for key, response in zip(
            raw["branch_keys"],
            raw["branch_responses"],
            strict=True,
        )
    }
    if len(branches) + 1 != EXPECTED_REQUESTS:
        raise ValueError("ClariQ replay support count changed")
    scores = policy_scores(task, initial, branches)

    description_collision_count = 0
    joint_collision_count = 0
    support_change_count = 0
    profile_diversity_count = 0
    positive_continuation_count = 0
    initial_descriptions = {
        flat.normalize_answer(hypothesis)
        for hypothesis, probability in zip(
            initial.hypotheses,
            initial.probabilities,
            strict=True,
        )
        if probability > 0
    }
    for key, support in branches.items():
        normalized = [
            flat.normalize_answer(hypothesis)
            for hypothesis in support.hypotheses
        ]
        description_collision_count += len(normalized) - len(set(normalized))
        joint = list(zip(normalized, support.predictions, strict=True))
        joint_collision_count += len(joint) - len(set(joint))
        descriptions = {
            description
            for description, probability in zip(
                normalized,
                support.probabilities,
                strict=True,
            )
            if probability > 0
        }
        if descriptions != initial_descriptions:
            support_change_count += 1
        profiles = {
            prediction
            for probability, prediction in zip(
                support.probabilities,
                support.predictions,
                strict=True,
            )
            if probability > 0
        }
        if len(profiles) >= 2:
            profile_diversity_count += 1
        root_id, code = key
        if scores["branch_dynamic_gains"][root_id][code] > 1e-12:
            positive_continuation_count += 1

    # All particle supports and policy scores are fixed before this point.
    paths = verify_source(source_root)
    synthetic = _load_tar_pickle(paths["synthetic"])
    evaluation = _load_split_archive(
        [paths["eval_a"], paths["eval_b"]]
    )["NDCG20"]
    question_ids_by_text = {
        (int(task["topic_id"]), str(question["question"])): str(
            question["question_id"]
        )
        for question in task["question_bank"]
    }
    external = analyze_topics(
        synthetic,
        evaluation,
        question_ids_by_text,
        [str(task["topic_id"])],
    )
    if len(external["records"]) != 1:
        raise ValueError("ClariQ particle replay endpoint changed")
    endpoints = {
        str(root["question_id"]): float(root["terminal_utility"])
        for root in external["records"][0]["roots"]
    }
    if set(endpoints) != set(question_ids):
        raise ValueError("ClariQ particle replay roots changed")
    score_keys = {
        "myopic": "myopic_scores",
        "fixed_depth_two": "fixed_depth_two_scores",
        "dynamic_depth_two": "dynamic_depth_two_scores",
        "shuffled_dynamic": "shuffled_dynamic_scores",
    }
    correlations = {
        name: spearman(
            [scores[key][question_id] for question_id in question_ids],
            [endpoints[question_id] for question_id in question_ids],
        )
        for name, key in score_keys.items()
    }
    selected_ids = {
        "myopic": scores["myopic_question_id"],
        "fixed_depth_two": scores["fixed_depth_two_question_id"],
        "dynamic_depth_two": scores["dynamic_depth_two_question_id"],
        "shuffled_dynamic": scores["shuffled_dynamic_question_id"],
    }
    selected_endpoints = {
        name: endpoints[question_id]
        for name, question_id in selected_ids.items()
    }
    future_values = [
        scores["dynamic_depth_two_scores"][question_id]
        - scores["myopic_scores"][question_id]
        for question_id in question_ids
    ]
    myopic_values = list(scores["myopic_scores"].values())
    metrics = {
        "description_collision_count": description_collision_count,
        "joint_particle_collision_count": joint_collision_count,
        "branch_support_change_count": support_change_count,
        "branch_profile_diversity_count": profile_diversity_count,
        "positive_continuation_count": positive_continuation_count,
        "myopic_score_range": max(myopic_values) - min(myopic_values),
        "dynamic_future_range": max(future_values) - min(future_values),
        "maximum_dynamic_fixed_score_difference": max(
            abs(
                scores["dynamic_depth_two_scores"][question_id]
                - scores["fixed_depth_two_scores"][question_id]
            )
            for question_id in question_ids
        ),
        "selected_question_ids": selected_ids,
        "selected_terminal_ndcg20": selected_endpoints,
        "score_terminal_ndcg20_spearman": correlations,
        "dynamic_gain_over_myopic": (
            selected_endpoints["dynamic_depth_two"]
            - selected_endpoints["myopic"]
        ),
        "dynamic_gain_over_fixed": (
            selected_endpoints["dynamic_depth_two"]
            - selected_endpoints["fixed_depth_two"]
        ),
        "dynamic_gain_over_shuffled": (
            selected_endpoints["dynamic_depth_two"]
            - selected_endpoints["shuffled_dynamic"]
        ),
    }
    gates = {
        "all_91_joint_particle_supports_parse": len(branches) + 1 == 91,
        "no_joint_particle_collisions": joint_collision_count == 0,
        "description_collisions_at_most_5": (
            description_collision_count <= 5
        ),
        "branch_support_changes_at_least_80": support_change_count >= 80,
        "branch_profile_diversity_at_least_75": (
            profile_diversity_count >= 75
        ),
        "positive_continuations_at_least_75": (
            positive_continuation_count >= 75
        ),
        "myopic_score_range_at_least_0_05": (
            metrics["myopic_score_range"] >= 0.05
        ),
        "dynamic_future_range_at_least_0_05": (
            metrics["dynamic_future_range"] >= 0.05
        ),
        "dynamic_fixed_scores_differ": (
            metrics["maximum_dynamic_fixed_score_difference"] >= 0.02
        ),
        "dynamic_changes_root_from_myopic": (
            selected_ids["dynamic_depth_two"] != selected_ids["myopic"]
        ),
        "dynamic_spearman_at_least_0_20": (
            correlations["dynamic_depth_two"] >= 0.20
        ),
        "dynamic_spearman_beats_myopic_and_fixed_by_0_02": (
            correlations["dynamic_depth_two"]
            >= max(
                correlations["myopic"],
                correlations["fixed_depth_two"],
            )
            + 0.02
        ),
        "dynamic_endpoint_strictly_beats_myopic_and_fixed": (
            selected_endpoints["dynamic_depth_two"]
            > max(
                selected_endpoints["myopic"],
                selected_endpoints["fixed_depth_two"],
            )
            + 1e-12
        ),
        "dynamic_endpoint_nonworse_than_shuffled": (
            selected_endpoints["dynamic_depth_two"]
            >= selected_endpoints["shuffled_dynamic"] - 1e-12
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 1,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "raw_sha256": RAW_SHA256,
            "model_calls": 0,
            "posthoc_failed_tree_replay": True,
            "joint_particle_identity": (
                "normalized intent text + predicted response profile"
            ),
            "exact_duplicate_pairs_rejected": True,
            "scores_computed_before_endpoint_load": True,
            "development_or_holdout_loaded": False,
            "claim_authorized": False,
        },
        "policy": scores,
        "endpoint_by_question_id": endpoints,
        "metrics": metrics,
        "gates": gates,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run_replay(
        source_root=args.source_root,
        manifest_path=args.manifest,
        raw_path=args.raw,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "status": result["status"],
                "metrics": result["metrics"],
                "gates": result["gates"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
