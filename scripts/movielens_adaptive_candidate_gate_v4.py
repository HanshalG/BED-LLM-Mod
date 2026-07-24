#!/usr/bin/env python3
"""Test semantic profile dynamics with adaptive MovieLens candidate pools."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import (
    FORMAL_EXPECTED_REQUESTS,
    FORMAL_USER_IDS as V1_FORMAL_USER_IDS,
    HELDOUT_COUNT,
    INITIAL_MOVIE_IDS,
    ITEMS_SHA256,
    PROFILE_COUNT,
    RATINGS_SHA256,
    README_SHA256,
    SMOKE_EXPECTED_REQUESTS,
    _history_payload,
    _movie_payload,
    _usage,
    _write_raw_checkpoint,
    immediate_eig_values,
    load_movielens,
    merge_branch_profiles,
    parse_profiles,
    parse_rating_likelihoods,
    predictive_nll,
    profile_messages,
    summarize,
)
from scripts.movielens_profile_dynamics_gate_v2 import (
    ALL_SELECTED_USER_IDS as V2_SELECTED_USER_IDS,
    _build_models,
    _text_hash,
    parse_refreshed_profiles,
    profile_only_rating_likelihood_messages,
    refreshed_profile_messages,
)
from scripts.movielens_profile_dynamics_gate_v3 import (
    ALL_SELECTED_USER_IDS as V3_SELECTED_USER_IDS,
)


SELECTION_SEED = 24305
SMOKE_USER_IDS = (113, 130)
FORMAL_USER_IDS = (158, 194, 227, 234, 323, 468, 494, 551, 579, 679, 710, 854)
FORMAL_SCREEN_USER_IDS = FORMAL_USER_IDS
PROSPECTIVE_ENROLLMENT_COUNT: int | None = None
SEMANTIC_RANKING_MESSAGES: Any | None = None
SEMANTIC_RANKING_PARSE: Any | None = None
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_USER_IDS
CANDIDATE_POOL_SIZE = 16
SELECTED_CANDIDATE_COUNT = 4


def selected_user_ids(ratings: dict[int, dict[int, int]]) -> tuple[int, ...]:
    excluded = (
        set(V1_FORMAL_USER_IDS)
        | set(V2_SELECTED_USER_IDS)
        | set(V3_SELECTED_USER_IDS)
    )
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if user_id not in excluded
        and all(movie_id in user_ratings for movie_id in INITIAL_MOVIE_IDS)
        and sum(movie_id not in INITIAL_MOVIE_IDS for movie_id in user_ratings)
        >= CANDIDATE_POOL_SIZE + HELDOUT_COUNT
    )
    chosen = tuple(
        sorted(
            int(value)
            for value in np.random.default_rng(SELECTION_SEED).choice(
                eligible,
                size=len(ALL_SELECTED_USER_IDS),
                replace=False,
            )
        )
    )
    if chosen != tuple(sorted(ALL_SELECTED_USER_IDS)):
        raise ValueError("frozen v4 MovieLens user selection does not reproduce")
    return chosen


def candidate_and_heldout_ids(
    user_id: int,
    ratings: dict[int, dict[int, int]],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    popularity: dict[int, int] = {}
    for user_ratings in ratings.values():
        for movie_id in user_ratings:
            popularity[movie_id] = popularity.get(movie_id, 0) + 1
    user_ratings = ratings[user_id]
    available = [
        movie_id
        for movie_id in user_ratings
        if movie_id not in INITIAL_MOVIE_IDS
    ]
    candidate_pool = tuple(
        sorted(available, key=lambda movie_id: (-popularity[movie_id], movie_id))[
            :CANDIDATE_POOL_SIZE
        ]
    )
    remaining = np.asarray(
        sorted(set(available) - set(candidate_pool)),
        dtype=int,
    )
    if len(candidate_pool) != CANDIDATE_POOL_SIZE or len(remaining) < HELDOUT_COUNT:
        raise ValueError("selected user lacks candidate or held-out capacity")
    heldout = np.random.default_rng(SELECTION_SEED * 1000 + user_id).choice(
        remaining,
        size=HELDOUT_COUNT,
        replace=False,
    )
    return candidate_pool, tuple(sorted(int(value) for value in heldout))


def select_candidate_indices(
    candidate_ids: Sequence[int],
    eig_values: Sequence[float],
) -> tuple[int, ...]:
    if len(candidate_ids) != CANDIDATE_POOL_SIZE or len(eig_values) != len(
        candidate_ids
    ):
        raise ValueError("candidate pool and EIG values do not align")
    return tuple(
        sorted(
            range(len(candidate_ids)),
            key=lambda index: (-float(eig_values[index]), candidate_ids[index]),
        )[:SELECTED_CANDIDATE_COUNT]
    )


def run_gate(
    config: Config,
    *,
    data_dir: str | Path,
    likelihood_model: str,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    if stage not in {"serving_smoke", "formal"}:
        raise ValueError("stage must be serving_smoke or formal")
    ratings, items = load_movielens(data_dir)
    selected_user_ids(ratings)
    user_ids = (
        SMOKE_USER_IDS if stage == "serving_smoke" else FORMAL_SCREEN_USER_IDS
    )
    generator, likelihood = _build_models(config, likelihood_model)
    raw: dict[str, Any] = {}

    histories = [
        _history_payload(INITIAL_MOVIE_IDS, ratings[user_id], items)
        for user_id in user_ids
    ]
    initial_raw = generator.chat_complete_messages_batched(
        [profile_messages(history) for history in histories],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial_profiles"] = initial_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, user_ids=user_ids, raw=raw
    )
    initial_profiles = [parse_profiles(response) for response in initial_raw]

    pools_and_heldout = [
        candidate_and_heldout_ids(user_id, ratings) for user_id in user_ids
    ]
    candidate_pools = [value[0] for value in pools_and_heldout]
    heldout_ids_many = [value[1] for value in pools_and_heldout]
    query_ids_many = [
        [*candidate_ids, *heldout_ids]
        for candidate_ids, heldout_ids in zip(
            candidate_pools, heldout_ids_many, strict=True
        )
    ]
    initial_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                profiles,
                [items[movie_id] for movie_id in query_ids],
            )
            for profiles, query_ids in zip(
                initial_profiles, query_ids_many, strict=True
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial_likelihoods"] = initial_likelihood_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, user_ids=user_ids, raw=raw
    )
    initial_likelihoods = [
        parse_rating_likelihoods(
            response,
            profile_count=PROFILE_COUNT,
            movie_count=len(query_ids),
        )
        for response, query_ids in zip(
            initial_likelihood_raw, query_ids_many, strict=True
        )
    ]

    all_pool_eigs = [
        immediate_eig_values(matrix[:, :CANDIDATE_POOL_SIZE, :])
        for matrix in initial_likelihoods
    ]
    selected_indices = [
        select_candidate_indices(candidate_ids, eig_values)
        for candidate_ids, eig_values in zip(
            candidate_pools, all_pool_eigs, strict=True
        )
    ]
    selected_movie_ids = [
        tuple(candidate_pools[user_index][index] for index in indices)
        for user_index, indices in enumerate(selected_indices)
    ]
    if stage == "formal":
        max_eigs = [max(values) for values in all_pool_eigs]
        mean_max_eig = float(np.mean(max_eigs))
        users_above_threshold = sum(value >= 0.02 for value in max_eigs)
        enrollment_indices = [
            index for index, value in enumerate(max_eigs) if value >= 0.02
        ]
        insufficient_enrollment = (
            PROSPECTIVE_ENROLLMENT_COUNT is not None
            and len(enrollment_indices) < PROSPECTIVE_ENROLLMENT_COUNT
        )
        population_gate_failed = (
            PROSPECTIVE_ENROLLMENT_COUNT is None
            and (mean_max_eig < 0.02 or users_above_threshold < 8)
        )
        if insufficient_enrollment or population_gate_failed:
            usage = _usage(generator, likelihood)
            expected_screen_requests = 2 * len(user_ids)
            gates = {
                "all_users_completed": len(user_ids) == len(FORMAL_SCREEN_USER_IDS),
                "exact_sensitivity_screen_request_count": (
                    int(usage["physical_requests"]) == expected_screen_requests
                ),
                "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
                "mean_max_immediate_eig_at_least_0_02": mean_max_eig >= 0.02,
                "at_least_eight_users_have_max_eig_at_least_0_02": (
                    users_above_threshold >= 8
                ),
            }
            if PROSPECTIVE_ENROLLMENT_COUNT is not None:
                gates["prospective_enrollment_complete"] = (
                    len(enrollment_indices) >= PROSPECTIVE_ENROLLMENT_COUNT
                )
            gates["all_pass"] = False
            return {
                "schema_version": 4,
                "status": "gate_failed",
                "protocol": {
                    "stage": stage,
                    "selection_seed": SELECTION_SEED,
                    "ratings_sha256": RATINGS_SHA256,
                    "items_sha256": ITEMS_SHA256,
                    "readme_sha256": README_SHA256,
                    "user_ids": list(user_ids),
                    "initial_movie_ids": list(INITIAL_MOVIE_IDS),
                    "candidate_pool_size": CANDIDATE_POOL_SIZE,
                    "selected_candidate_count": SELECTED_CANDIDATE_COUNT,
                    "candidate_pool_uses_presence_and_popularity_only": True,
                    "heldout_count": HELDOUT_COUNT,
                    "profile_count": PROFILE_COUNT,
                    "likelihood_model": likelihood_model,
                    "likelihood_history_hidden": True,
                    "v1_v2_v3_users_excluded": True,
                    "formal_sensitivity_futility_stop": True,
                    "prospective_enrollment_count": PROSPECTIVE_ENROLLMENT_COUNT,
                    "candidate_outcomes_not_read": True,
                    "heldout_ratings_not_read": True,
                    "raw_responses_private_and_untracked": True,
                    "committed_profile_text_omitted": True,
                },
                "summary": {
                    "num_users": len(user_ids),
                    "num_branches": 0,
                    "mean_max_immediate_eig": mean_max_eig,
                    "users_with_max_immediate_eig_at_least_0_02": (
                        users_above_threshold
                    ),
                    "gates": gates,
                },
                "records": [
                    {
                        "user_id": user_id,
                        "initial_profile_count": len(initial_profiles[user_index]),
                        "initial_profile_hashes": [
                            _text_hash(value)
                            for value in initial_profiles[user_index]
                        ],
                        "candidate_pool_size": CANDIDATE_POOL_SIZE,
                        "candidate_movies": [
                            _movie_payload(items[movie_id])
                            for movie_id in selected_movie_ids[user_index]
                        ],
                        "immediate_eig_values": [
                            all_pool_eigs[user_index][index]
                            for index in selected_indices[user_index]
                        ],
                    }
                    for user_index, user_id in enumerate(user_ids)
                ],
                "usage": usage,
            }
        if PROSPECTIVE_ENROLLMENT_COUNT is not None:
            enrollment_indices = enrollment_indices[:PROSPECTIVE_ENROLLMENT_COUNT]
            screen_user_ids = tuple(user_ids)
            user_ids = tuple(user_ids[index] for index in enrollment_indices)
            histories = [histories[index] for index in enrollment_indices]
            initial_profiles = [
                initial_profiles[index] for index in enrollment_indices
            ]
            heldout_ids_many = [
                heldout_ids_many[index] for index in enrollment_indices
            ]
            candidate_pools = [
                candidate_pools[index] for index in enrollment_indices
            ]
            query_ids_many = [
                query_ids_many[index] for index in enrollment_indices
            ]
            initial_likelihoods = [
                initial_likelihoods[index] for index in enrollment_indices
            ]
            all_pool_eigs = [
                all_pool_eigs[index] for index in enrollment_indices
            ]
            selected_indices = [
                selected_indices[index] for index in enrollment_indices
            ]
            selected_movie_ids = [
                selected_movie_ids[index] for index in enrollment_indices
            ]
        else:
            screen_user_ids = tuple(user_ids)
    else:
        screen_user_ids = tuple(user_ids)
    semantic_scores_many: list[list[float]] | None = None
    if SEMANTIC_RANKING_MESSAGES is not None:
        ranking_prompts = [
            SEMANTIC_RANKING_MESSAGES(
                initial_profiles[user_index],
                [
                    items[selected_movie_ids[user_index][candidate_index]]
                    for candidate_index in range(SELECTED_CANDIDATE_COUNT)
                ],
                [items[movie_id] for movie_id in heldout_ids_many[user_index]],
                initial_likelihoods[user_index][
                    :,
                    list(selected_indices[user_index]),
                    :,
                ],
            )
            for user_index in range(len(user_ids))
        ]
        ranking_raw = generator.chat_complete_messages_batched(
            ranking_prompts,
            temperature=0.0,
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["semantic_ranking"] = ranking_raw
        _write_raw_checkpoint(
            raw_checkpoint_path, stage=stage, user_ids=user_ids, raw=raw
        )
        semantic_scores_many = [
            SEMANTIC_RANKING_PARSE(response) for response in ranking_raw
        ]
    branch_movie_ids_many = [
        movie_ids[:1] if stage == "serving_smoke" else movie_ids
        for movie_ids in selected_movie_ids
    ]
    flat_user_indices = [
        user_index
        for user_index, movie_ids in enumerate(branch_movie_ids_many)
        for _movie_id in movie_ids
    ]
    flat_movie_ids = [
        movie_id for movie_ids in branch_movie_ids_many for movie_id in movie_ids
    ]
    refresh_histories = [
        [
            *histories[user_index],
            {
                **_movie_payload(items[movie_id]),
                "rating": ratings[user_ids[user_index]][movie_id],
            },
        ]
        for user_index, movie_id in zip(
            flat_user_indices, flat_movie_ids, strict=True
        )
    ]
    refresh_prompts = [
        refreshed_profile_messages(
            history,
            initial_profiles[user_index],
        )
        for history, user_index in zip(
            refresh_histories, flat_user_indices, strict=True
        )
    ]
    refresh_raw = generator.chat_complete_messages_batched(
        refresh_prompts,
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["refreshed_profiles"] = refresh_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, user_ids=user_ids, raw=raw
    )
    refreshed_rows = [
        parse_refreshed_profiles(response, initial_profiles[user_index])
        for response, user_index in zip(
            refresh_raw, flat_user_indices, strict=True
        )
    ]
    generated_profiles = [
        [row["description"] for row in rows] for rows in refreshed_rows
    ]

    branch_profiles: list[list[str]] = []
    for generated, user_index, movie_id in zip(
        generated_profiles,
        flat_user_indices,
        flat_movie_ids,
        strict=True,
    ):
        pool_index = candidate_pools[user_index].index(movie_id)
        observed_rating = ratings[user_ids[user_index]][movie_id]
        old_probabilities = initial_likelihoods[user_index][
            :, pool_index, observed_rating - 1
        ]
        branch_profiles.append(
            merge_branch_profiles(
                generated,
                initial_profiles[user_index],
                old_probabilities,
            )
        )

    branch_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                profiles,
                [items[movie_id] for movie_id in heldout_ids_many[user_index]],
            )
            for profiles, user_index in zip(
                branch_profiles,
                flat_user_indices,
                strict=True,
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["branch_likelihoods"] = branch_likelihood_raw
    _write_raw_checkpoint(
        raw_checkpoint_path, stage=stage, user_ids=user_ids, raw=raw
    )
    branch_likelihoods = [
        parse_rating_likelihoods(
            response,
            profile_count=len(profiles),
            movie_count=HELDOUT_COUNT,
        )
        for response, profiles in zip(
            branch_likelihood_raw, branch_profiles, strict=True
        )
    ]

    replay_rows: list[list[dict[str, str]]] = []
    if stage == "serving_smoke":
        replay_raw = generator.chat_complete_messages_batched(
            refresh_prompts,
            temperature=float(config.generation_temperature_diverse),
            block_size=config.batched_block_size,
            max_new_tokens=config.openrouter_max_output_tokens,
        )
        raw["refresh_replays"] = replay_raw
        _write_raw_checkpoint(
            raw_checkpoint_path, stage=stage, user_ids=user_ids, raw=raw
        )
        replay_rows = [
            parse_refreshed_profiles(response, initial_profiles[user_index])
            for response, user_index in zip(
                replay_raw, flat_user_indices, strict=True
            )
        ]

    records: list[dict[str, Any]] = []
    offset = 0
    for user_index, user_id in enumerate(user_ids):
        heldout_ids = heldout_ids_many[user_index]
        heldout_ratings = [ratings[user_id][movie_id] for movie_id in heldout_ids]
        initial_matrix = initial_likelihoods[user_index]
        heldout_matrix = initial_matrix[:, CANDIDATE_POOL_SIZE:, :]
        chosen_indices = selected_indices[user_index]
        chosen_eigs = [all_pool_eigs[user_index][index] for index in chosen_indices]
        movie_ids = branch_movie_ids_many[user_index]
        branches = []
        for local_index, movie_id in enumerate(movie_ids):
            flat_index = offset + local_index
            generated = refreshed_rows[flat_index]
            branches.append(
                {
                    "branch_index": local_index,
                    "movie": _movie_payload(items[movie_id]),
                    "generated_profile_count": len(generated),
                    "generated_profile_hashes": [
                        _text_hash(row["description"]) for row in generated
                    ],
                    "new_evidence_effect_hashes": [
                        _text_hash(row["new_evidence_effect"]) for row in generated
                    ],
                    "retained_support_count": len(branch_profiles[flat_index]),
                    "retained_support_hashes": [
                        _text_hash(value) for value in branch_profiles[flat_index]
                    ],
                    "exact_initial_copy_count": 0,
                    "heldout_nll": predictive_nll(
                        branch_likelihoods[flat_index],
                        heldout_ratings,
                    ),
                }
            )
        record = {
            "user_id": user_id,
            "initial_profile_count": len(initial_profiles[user_index]),
            "initial_profile_hashes": [
                _text_hash(value) for value in initial_profiles[user_index]
            ],
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "candidate_movies": [
                _movie_payload(items[movie_id])
                for movie_id in selected_movie_ids[user_index]
            ],
            "immediate_eig_values": chosen_eigs,
            "immediate_eig_selected_branch": 0,
            "initial_heldout_nll": predictive_nll(
                heldout_matrix,
                heldout_ratings,
            ),
            "branches": branches,
        }
        if semantic_scores_many is not None:
            record["semantic_lookahead_scores"] = semantic_scores_many[user_index]
            record["semantic_lookahead_selected_branch"] = int(
                np.argmax(semantic_scores_many[user_index])
            )
        if stage == "serving_smoke":
            record["replay_profiles"] = [
                _text_hash(row["description"]) for row in replay_rows[user_index]
            ]
        records.append(record)
        offset += len(movie_ids)

    usage = _usage(generator, likelihood)
    summary = summarize(records, usage, stage=stage)
    if stage == "serving_smoke" and SEMANTIC_RANKING_MESSAGES is not None:
        summary["gates"]["exact_physical_request_count"] = (
            int(usage["physical_requests"]) == 12
        )
        summary["gates"]["all_pass"] = all(
            value
            for name, value in summary["gates"].items()
            if name != "all_pass"
        )
    if stage == "formal" and PROSPECTIVE_ENROLLMENT_COUNT is not None:
        expected_requests = (
            2 * len(FORMAL_SCREEN_USER_IDS)
            + 8 * PROSPECTIVE_ENROLLMENT_COUNT
            + (
                PROSPECTIVE_ENROLLMENT_COUNT
                if SEMANTIC_RANKING_MESSAGES is not None
                else 0
            )
        )
        summary["gates"]["exact_physical_request_count"] = (
            int(usage["physical_requests"]) == expected_requests
        )
        summary["gates"]["prospective_enrollment_complete"] = (
            len(records) == PROSPECTIVE_ENROLLMENT_COUNT
        )
        summary["gates"]["all_pass"] = all(
            value
            for name, value in summary["gates"].items()
            if name != "all_pass"
        )
    return {
        "schema_version": 4,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "ratings_sha256": RATINGS_SHA256,
            "items_sha256": ITEMS_SHA256,
            "readme_sha256": README_SHA256,
            "user_ids": list(user_ids),
            "screen_user_ids": list(screen_user_ids),
            "prospective_enrollment_count": PROSPECTIVE_ENROLLMENT_COUNT,
            "semantic_ranking_enabled": SEMANTIC_RANKING_MESSAGES is not None,
            "initial_movie_ids": list(INITIAL_MOVIE_IDS),
            "candidate_pool_size": CANDIDATE_POOL_SIZE,
            "selected_candidate_count": SELECTED_CANDIDATE_COUNT,
            "candidate_pool_uses_presence_and_popularity_only": True,
            "heldout_count": HELDOUT_COUNT,
            "profile_count": PROFILE_COUNT,
            "likelihood_model": likelihood_model,
            "likelihood_history_hidden": True,
            "v1_v2_v3_users_excluded": True,
            "smoke_user_ids_disjoint_from_formal": True,
            "recorded_ratings_are_only_outcomes": True,
            "heldout_ratings_hidden_from_all_model_prompts": True,
            "user_ids_hidden_from_all_model_prompts": True,
            "raw_responses_private_and_untracked": True,
            "committed_profile_text_omitted": True,
            "source_ratings_omitted_from_persisted_artifacts": True,
        },
        "summary": summary,
        "records": records,
        "usage": usage,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--likelihood-model", default="openai/gpt-5.4-mini")
    parser.add_argument(
        "--stage",
        choices=("serving_smoke", "formal"),
        default="formal",
    )
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_run_dir = args.private_raw_dir / args.run_id
    private_run_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_run_dir / "RAW_RESPONSES.json"
    output_name = "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = run_gate(
            config,
            data_dir=args.data_dir,
            likelihood_model=args.likelihood_model,
            stage=args.stage,
            raw_checkpoint_path=raw_path,
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": 4,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
