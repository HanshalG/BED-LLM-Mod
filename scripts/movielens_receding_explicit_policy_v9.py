#!/usr/bin/env python3
"""Evaluate a two-round receding explicit-regeneration MovieLens policy."""

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
    HELDOUT_COUNT,
    PROFILE_COUNT,
    _history_payload,
    _movie_payload,
    _parse_json_object,
    _usage,
    _write_raw_checkpoint,
    immediate_eig_values,
    load_movielens,
    merge_branch_profiles,
    parse_profiles,
    parse_rating_likelihoods,
    predictive_nll,
    profile_messages,
)
from scripts.movielens_profile_dynamics_gate_v2 import (
    parse_refreshed_profiles,
    profile_only_rating_likelihood_messages,
    refreshed_profile_messages,
)
import scripts.movielens_adaptive_candidate_gate_v4 as base
import scripts.movielens_depth2_semantic_policy_v8 as v8
import scripts.movielens_explicit_rollout_ranking_v7 as v7


SELECTION_SEED = 24310
INITIAL_HISTORY_MOVIE_IDS = v7.INITIAL_HISTORY_MOVIE_IDS
SMOKE_USER_IDS = (684, 643)
FORMAL_SCREEN_USER_IDS = (
    880,
    94,
    868,
    745,
    301,
    246,
    59,
    308,
    184,
    216,
    465,
    632,
    497,
    339,
    121,
    715,
    295,
    312,
    457,
    287,
    913,
    664,
    458,
    1,
    23,
    97,
    561,
    493,
    311,
    292,
    806,
    249,
    896,
    77,
    274,
    435,
    344,
    682,
    363,
    593,
    883,
    514,
    222,
    535,
    343,
    805,
    330,
    44,
)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_SCREEN_USER_IDS
ENROLLMENT_COUNT = 8
FORMAL_CANDIDATE_COUNT = 4
SMOKE_CANDIDATE_COUNT = 1
SMOKE_EXPECTED_REQUESTS = 22
LIKELIHOOD_SUM_TOLERANCE = 0.10
POLICIES = ("receding_explicit", "immediate_eig", "seeded_random")


def selected_user_ids(
    ratings: dict[int, dict[int, int]],
) -> tuple[int, ...]:
    excluded = (
        v8.v7._prior_users()
        | set(v8.v7.ALL_SELECTED_USER_IDS)
        | set(v8.ALL_SELECTED_USER_IDS)
    )
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if user_id not in excluded
        and all(movie_id in user_ratings for movie_id in INITIAL_HISTORY_MOVIE_IDS)
        and sum(
            movie_id not in INITIAL_HISTORY_MOVIE_IDS
            for movie_id in user_ratings
        )
        >= base.CANDIDATE_POOL_SIZE + HELDOUT_COUNT
    )
    chosen = tuple(
        int(value)
        for value in np.random.default_rng(SELECTION_SEED).choice(
            eligible,
            size=len(ALL_SELECTED_USER_IDS),
            replace=False,
        )
    )
    if chosen != ALL_SELECTED_USER_IDS:
        raise ValueError("frozen v9 MovieLens user selection does not reproduce")
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
        if movie_id not in INITIAL_HISTORY_MOVIE_IDS
    ]
    candidate_pool = tuple(
        sorted(
            available,
            key=lambda movie_id: (-popularity[movie_id], movie_id),
        )[: base.CANDIDATE_POOL_SIZE]
    )
    remaining = np.asarray(
        sorted(set(available) - set(candidate_pool)),
        dtype=int,
    )
    if (
        len(candidate_pool) != base.CANDIDATE_POOL_SIZE
        or len(remaining) < HELDOUT_COUNT
    ):
        raise ValueError("selected user lacks candidate or held-out capacity")
    heldout = np.random.default_rng(SELECTION_SEED * 1000 + user_id).choice(
        remaining,
        size=HELDOUT_COUNT,
        replace=False,
    )
    return candidate_pool, tuple(sorted(int(value) for value in heldout))


def top_candidate_indices(
    candidate_ids: Sequence[int],
    likelihoods: np.ndarray,
    *,
    count: int,
) -> tuple[int, ...]:
    eig = immediate_eig_values(likelihoods[:, : len(candidate_ids), :])
    return tuple(
        sorted(
            range(len(candidate_ids)),
            key=lambda index: (-eig[index], candidate_ids[index]),
        )[:count]
    )


def parse_likelihood(
    text: str,
    *,
    profile_count: int,
    movie_count: int,
) -> tuple[np.ndarray, int]:
    matrix = parse_rating_likelihoods(
        text,
        profile_count=profile_count,
        movie_count=movie_count,
        sum_tolerance=LIKELIHOOD_SUM_TOLERANCE,
    )
    normalized_rows = 0
    for profile in _parse_json_object(text)["profiles"]:
        for probabilities in profile["ratings"]:
            if not np.isclose(
                float(np.sum(np.asarray(probabilities, dtype=float))),
                1.0,
                atol=1e-12,
                rtol=0.0,
            ):
                normalized_rows += 1
    return matrix, normalized_rows


def expected_downstream_entropies(
    *,
    current_likelihoods: np.ndarray,
    candidate_indices: Sequence[int],
    branch_likelihoods: Sequence[np.ndarray],
    branches_per_candidate: int = 5,
) -> list[float]:
    if len(branch_likelihoods) != len(candidate_indices) * branches_per_candidate:
        raise ValueError("candidate branch likelihoods do not align")
    values: list[float] = []
    for action_index, candidate_index in enumerate(candidate_indices):
        probabilities = np.mean(
            current_likelihoods[:, candidate_index, :],
            axis=0,
        )
        probabilities = probabilities / float(np.sum(probabilities))
        branch_entropies = [
            v7.mean_predictive_entropy(
                branch_likelihoods[
                    action_index * branches_per_candidate + rating_index
                ]
            )
            for rating_index in range(branches_per_candidate)
        ]
        values.append(float(np.dot(probabilities, branch_entropies)))
    return values


def _checkpoint(
    path: Path | None,
    *,
    stage: str,
    user_ids: Sequence[int],
    raw: dict[str, Any],
) -> None:
    _write_raw_checkpoint(
        path,
        stage=stage,
        user_ids=user_ids,
        raw=raw,
    )


_build_models = v7._build_models


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
    screen_user_ids = (
        (SMOKE_USER_IDS[0],)
        if stage == "serving_smoke"
        else FORMAL_SCREEN_USER_IDS
    )
    candidate_count = (
        SMOKE_CANDIDATE_COUNT
        if stage == "serving_smoke"
        else FORMAL_CANDIDATE_COUNT
    )
    generator, likelihood = _build_models(config, likelihood_model)
    raw: dict[str, Any] = {}
    normalized_probability_rows = 0

    histories = [
        _history_payload(
            INITIAL_HISTORY_MOVIE_IDS,
            ratings[user_id],
            items,
        )
        for user_id in screen_user_ids
    ]
    initial_profile_raw = generator.chat_complete_messages_batched(
        [profile_messages(history) for history in histories],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial_profiles"] = initial_profile_raw
    _checkpoint(
        raw_checkpoint_path,
        stage=stage,
        user_ids=screen_user_ids,
        raw=raw,
    )
    initial_profiles = [parse_profiles(response) for response in initial_profile_raw]
    pools_and_heldout = [
        candidate_and_heldout_ids(user_id, ratings)
        for user_id in screen_user_ids
    ]
    candidate_pools = [value[0] for value in pools_and_heldout]
    heldout_ids_many = [value[1] for value in pools_and_heldout]
    initial_query_ids = [
        [*pool, *heldout]
        for pool, heldout in zip(
            candidate_pools,
            heldout_ids_many,
            strict=True,
        )
    ]
    initial_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                profiles,
                [items[movie_id] for movie_id in query_ids],
            )
            for profiles, query_ids in zip(
                initial_profiles,
                initial_query_ids,
                strict=True,
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["initial_likelihoods"] = initial_likelihood_raw
    _checkpoint(
        raw_checkpoint_path,
        stage=stage,
        user_ids=screen_user_ids,
        raw=raw,
    )
    initial_likelihoods = []
    for response in initial_likelihood_raw:
        matrix, count = parse_likelihood(
            response,
            profile_count=PROFILE_COUNT,
            movie_count=base.CANDIDATE_POOL_SIZE + HELDOUT_COUNT,
        )
        initial_likelihoods.append(matrix)
        normalized_probability_rows += count

    max_initial_eigs = [
        max(immediate_eig_values(matrix[:, : base.CANDIDATE_POOL_SIZE, :]))
        for matrix in initial_likelihoods
    ]
    if stage == "formal":
        enrollment_indices = [
            index
            for index, value in enumerate(max_initial_eigs)
            if value >= 0.02
        ][:ENROLLMENT_COUNT]
        if len(enrollment_indices) < ENROLLMENT_COUNT:
            usage = _usage(generator, likelihood)
            gates = {
                "exact_screen_request_count": (
                    usage["physical_requests"] == 2 * len(screen_user_ids)
                ),
                "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
                "prospective_enrollment_complete": False,
                "all_pass": False,
            }
            return {
                "schema_version": 9,
                "status": "gate_failed",
                "protocol": {
                    "stage": stage,
                    "selection_seed": SELECTION_SEED,
                    "screen_user_ids": list(screen_user_ids),
                    "candidate_outcomes_not_read": True,
                    "heldout_ratings_not_read": True,
                    "normalized_probability_rows": normalized_probability_rows,
                    "raw_responses_private_and_untracked": True,
                },
                "summary": {
                    "num_screen_users": len(screen_user_ids),
                    "num_enrolled_users": len(enrollment_indices),
                    "gates": gates,
                },
                "records": [],
                "usage": usage,
            }
    else:
        enrollment_indices = [0]

    user_ids = tuple(screen_user_ids[index] for index in enrollment_indices)
    histories = [histories[index] for index in enrollment_indices]
    initial_profiles = [initial_profiles[index] for index in enrollment_indices]
    candidate_pools = [candidate_pools[index] for index in enrollment_indices]
    heldout_ids_many = [heldout_ids_many[index] for index in enrollment_indices]
    initial_likelihoods = [
        initial_likelihoods[index] for index in enrollment_indices
    ]
    max_initial_eigs = [
        max_initial_eigs[index] for index in enrollment_indices
    ]
    q1_indices_many = [
        top_candidate_indices(
            candidate_ids,
            matrix,
            count=candidate_count,
        )
        for candidate_ids, matrix in zip(
            candidate_pools,
            initial_likelihoods,
            strict=True,
        )
    ]

    q1_keys: list[tuple[int, int, int]] = []
    q1_histories: list[list[dict[str, Any]]] = []
    for user_index, q1_indices in enumerate(q1_indices_many):
        for action_index, pool_index in enumerate(q1_indices):
            movie_id = candidate_pools[user_index][pool_index]
            for rating in range(1, 6):
                q1_keys.append((user_index, action_index, rating))
                q1_histories.append(
                    [
                        *histories[user_index],
                        {
                            **_movie_payload(items[movie_id]),
                            "rating": rating,
                        },
                    ]
                )
    q1_profile_raw = generator.chat_complete_messages_batched(
        [
            refreshed_profile_messages(
                history,
                initial_profiles[user_index],
            )
            for history, (user_index, _action_index, _rating) in zip(
                q1_histories,
                q1_keys,
                strict=True,
            )
        ],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["q1_profiles"] = q1_profile_raw
    _checkpoint(
        raw_checkpoint_path,
        stage=stage,
        user_ids=user_ids,
        raw=raw,
    )
    q1_generated = [
        parse_refreshed_profiles(response, initial_profiles[user_index])
        for response, (user_index, _action_index, _rating) in zip(
            q1_profile_raw,
            q1_keys,
            strict=True,
        )
    ]
    q1_supports: list[list[str]] = []
    q1_remaining_ids: list[tuple[int, ...]] = []
    for rows, (user_index, action_index, rating) in zip(
        q1_generated,
        q1_keys,
        strict=True,
    ):
        pool_index = q1_indices_many[user_index][action_index]
        q1_supports.append(
            merge_branch_profiles(
                [row["description"] for row in rows],
                initial_profiles[user_index],
                initial_likelihoods[user_index][:, pool_index, rating - 1],
            )
        )
        q1_remaining_ids.append(
            tuple(
                movie_id
                for index, movie_id in enumerate(candidate_pools[user_index])
                if index != pool_index
            )
        )
    q1_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                support,
                [
                    items[movie_id]
                    for movie_id in [
                        *remaining,
                        *heldout_ids_many[user_index],
                    ]
                ],
            )
            for support, remaining, (
                user_index,
                _action_index,
                _rating,
            ) in zip(
                q1_supports,
                q1_remaining_ids,
                q1_keys,
                strict=True,
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["q1_likelihoods"] = q1_likelihood_raw
    _checkpoint(
        raw_checkpoint_path,
        stage=stage,
        user_ids=user_ids,
        raw=raw,
    )
    q1_likelihoods = []
    for response, support, remaining in zip(
        q1_likelihood_raw,
        q1_supports,
        q1_remaining_ids,
        strict=True,
    ):
        matrix, count = parse_likelihood(
            response,
            profile_count=len(support),
            movie_count=len(remaining) + HELDOUT_COUNT,
        )
        q1_likelihoods.append(matrix)
        normalized_probability_rows += count

    q1_branch_count_per_user = candidate_count * 5
    q1_explicit_values: list[list[float]] = []
    q1_policy_branches: list[dict[str, int]] = []
    for user_index, user_id in enumerate(user_ids):
        branch_start = user_index * q1_branch_count_per_user
        branch_matrices = q1_likelihoods[
            branch_start : branch_start + q1_branch_count_per_user
        ]
        heldout_matrices = [
            matrix[:, len(q1_remaining_ids[branch_start + index]) :, :]
            for index, matrix in enumerate(branch_matrices)
        ]
        explicit_values = expected_downstream_entropies(
            current_likelihoods=initial_likelihoods[user_index],
            candidate_indices=q1_indices_many[user_index],
            branch_likelihoods=heldout_matrices,
        )
        q1_explicit_values.append(explicit_values)
        selected_actions = {
            "receding_explicit": int(np.argmin(explicit_values)),
            "immediate_eig": 0,
            "seeded_random": int(
                np.random.default_rng(
                    SELECTION_SEED * 1000 + user_id + 1
                ).integers(candidate_count)
            ),
        }
        policy_branches: dict[str, int] = {}
        for policy, action_index in selected_actions.items():
            pool_index = q1_indices_many[user_index][action_index]
            movie_id = candidate_pools[user_index][pool_index]
            rating = ratings[user_id][movie_id]
            policy_branches[policy] = (
                branch_start + action_index * 5 + rating - 1
            )
        q1_policy_branches.append(policy_branches)

    unique_q2_branches: list[int] = []
    q2_state_index: dict[int, int] = {}
    for policy_branches in q1_policy_branches:
        for policy in POLICIES:
            branch_index = policy_branches[policy]
            if branch_index not in q2_state_index:
                q2_state_index[branch_index] = len(unique_q2_branches)
                unique_q2_branches.append(branch_index)
    q2_indices_many = [
        top_candidate_indices(
            q1_remaining_ids[branch_index],
            q1_likelihoods[branch_index],
            count=candidate_count,
        )
        for branch_index in unique_q2_branches
    ]
    q2_keys: list[tuple[int, int, int]] = []
    q2_histories: list[list[dict[str, Any]]] = []
    for state_index, branch_index in enumerate(unique_q2_branches):
        for action_index, remaining_index in enumerate(
            q2_indices_many[state_index]
        ):
            movie_id = q1_remaining_ids[branch_index][remaining_index]
            for rating in range(1, 6):
                q2_keys.append((state_index, action_index, rating))
                q2_histories.append(
                    [
                        *q1_histories[branch_index],
                        {
                            **_movie_payload(items[movie_id]),
                            "rating": rating,
                        },
                    ]
                )
    q2_profile_raw = generator.chat_complete_messages_batched(
        [
            refreshed_profile_messages(
                history,
                q1_supports[unique_q2_branches[state_index]],
            )
            for history, (state_index, _action_index, _rating) in zip(
                q2_histories,
                q2_keys,
                strict=True,
            )
        ],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["q2_profiles"] = q2_profile_raw
    _checkpoint(
        raw_checkpoint_path,
        stage=stage,
        user_ids=user_ids,
        raw=raw,
    )
    q2_generated = [
        parse_refreshed_profiles(
            response,
            q1_supports[unique_q2_branches[state_index]],
        )
        for response, (state_index, _action_index, _rating) in zip(
            q2_profile_raw,
            q2_keys,
            strict=True,
        )
    ]
    q2_supports: list[list[str]] = []
    for rows, (state_index, action_index, rating) in zip(
        q2_generated,
        q2_keys,
        strict=True,
    ):
        branch_index = unique_q2_branches[state_index]
        remaining_index = q2_indices_many[state_index][action_index]
        q2_supports.append(
            merge_branch_profiles(
                [row["description"] for row in rows],
                q1_supports[branch_index],
                q1_likelihoods[branch_index][
                    :,
                    remaining_index,
                    rating - 1,
                ],
            )
        )
    q2_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                support,
                [
                    items[movie_id]
                    for movie_id in heldout_ids_many[
                        q1_keys[unique_q2_branches[state_index]][0]
                    ]
                ],
            )
            for support, (state_index, _action_index, _rating) in zip(
                q2_supports,
                q2_keys,
                strict=True,
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    raw["q2_likelihoods"] = q2_likelihood_raw
    _checkpoint(
        raw_checkpoint_path,
        stage=stage,
        user_ids=user_ids,
        raw=raw,
    )
    q2_likelihoods = []
    for response, support in zip(
        q2_likelihood_raw,
        q2_supports,
        strict=True,
    ):
        matrix, count = parse_likelihood(
            response,
            profile_count=len(support),
            movie_count=HELDOUT_COUNT,
        )
        q2_likelihoods.append(matrix)
        normalized_probability_rows += count

    q2_branch_count_per_state = candidate_count * 5
    q2_explicit_values: list[list[float]] = []
    for state_index, branch_index in enumerate(unique_q2_branches):
        start = state_index * q2_branch_count_per_state
        q2_explicit_values.append(
            expected_downstream_entropies(
                current_likelihoods=q1_likelihoods[branch_index],
                candidate_indices=q2_indices_many[state_index],
                branch_likelihoods=q2_likelihoods[
                    start : start + q2_branch_count_per_state
                ],
            )
        )

    pending_paths: list[dict[str, Any]] = []
    records: list[dict[str, Any]] = []
    for user_index, user_id in enumerate(user_ids):
        paths: dict[str, dict[str, Any]] = {}
        for policy in POLICIES:
            q1_branch_index = q1_policy_branches[user_index][policy]
            state_index = q2_state_index[q1_branch_index]
            if policy == "receding_explicit":
                action_index = int(np.argmin(q2_explicit_values[state_index]))
            elif policy == "immediate_eig":
                action_index = 0
            else:
                action_index = int(
                    np.random.default_rng(
                        SELECTION_SEED * 1000 + user_id + 2
                    ).integers(candidate_count)
                )
            remaining_index = q2_indices_many[state_index][action_index]
            q2_movie_id = q1_remaining_ids[q1_branch_index][remaining_index]
            q2_rating = ratings[user_id][q2_movie_id]
            q2_branch_index = (
                state_index * q2_branch_count_per_state
                + action_index * 5
                + q2_rating
                - 1
            )
            q1_user_index, q1_action_index, _q1_rating = q1_keys[
                q1_branch_index
            ]
            q1_pool_index = q1_indices_many[q1_user_index][q1_action_index]
            q1_movie_id = candidate_pools[user_index][q1_pool_index]
            paths[policy] = {
                "query_movie_ids": [q1_movie_id, q2_movie_id],
                "query_movies": [
                    _movie_payload(items[q1_movie_id]),
                    _movie_payload(items[q2_movie_id]),
                ],
                "round1_matrix": q1_likelihoods[q1_branch_index][
                    :,
                    len(q1_remaining_ids[q1_branch_index]) :,
                    :,
                ],
                "final_matrix": q2_likelihoods[q2_branch_index],
            }
        pending_paths.append(
            {
                "user_id": user_id,
                "heldout_ids": heldout_ids_many[user_index],
                "paths": paths,
            }
        )
        q1_eig = immediate_eig_values(
            initial_likelihoods[user_index][
                :,
                : base.CANDIDATE_POOL_SIZE,
                :,
            ]
        )
        records.append(
            {
                "user_id": user_id,
                "max_initial_immediate_eig": max_initial_eigs[user_index],
                "q1_candidate_movies": [
                    _movie_payload(
                        items[candidate_pools[user_index][pool_index]]
                    )
                    for pool_index in q1_indices_many[user_index]
                ],
                "q1_immediate_eig_values": [
                    q1_eig[pool_index]
                    for pool_index in q1_indices_many[user_index]
                ],
                "q1_explicit_expected_entropies": q1_explicit_values[user_index],
                "policy_paths": {},
            }
        )

    for record, pending in zip(records, pending_paths, strict=True):
        heldout_ratings = [
            ratings[pending["user_id"]][movie_id]
            for movie_id in pending["heldout_ids"]
        ]
        for policy, path in pending["paths"].items():
            if len(set(path["query_movie_ids"])) != 2:
                raise ValueError("policy repeated a query")
            record["policy_paths"][policy] = {
                "query_movie_ids": path["query_movie_ids"],
                "query_movies": path["query_movies"],
                "round1_heldout_nll": predictive_nll(
                    path["round1_matrix"],
                    heldout_ratings,
                ),
                "final_heldout_nll": predictive_nll(
                    path["final_matrix"],
                    heldout_ratings,
                ),
            }

    usage = _usage(generator, likelihood)
    unique_state_count = len(unique_q2_branches)
    expected_requests = (
        2 * len(screen_user_ids)
        + len(user_ids) * candidate_count * 10
        + unique_state_count * candidate_count * 10
    )
    final_nlls = {
        policy: [
            record["policy_paths"][policy]["final_heldout_nll"]
            for record in records
        ]
        for policy in POLICIES
    }
    mean_nlls = {
        policy: float(np.mean(values))
        for policy, values in final_nlls.items()
    }
    wins_vs_immediate = sum(
        explicit < immediate
        for explicit, immediate in zip(
            final_nlls["receding_explicit"],
            final_nlls["immediate_eig"],
            strict=True,
        )
    )
    if stage == "serving_smoke":
        gates = {
            "exact_physical_request_count": (
                usage["physical_requests"] == SMOKE_EXPECTED_REQUESTS
            ),
            "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
            "one_unique_round2_state": unique_state_count == 1,
            "all_policy_paths_have_two_distinct_queries": all(
                len(set(path["query_movie_ids"])) == 2
                for path in records[0]["policy_paths"].values()
            ),
        }
    else:
        gates = {
            "exact_dynamic_request_count": (
                usage["physical_requests"] == expected_requests
            ),
            "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
            "prospective_enrollment_complete": len(records) == ENROLLMENT_COUNT,
            "unique_round2_state_count_in_range": (
                ENROLLMENT_COUNT
                <= unique_state_count
                <= len(POLICIES) * ENROLLMENT_COUNT
            ),
            "all_policy_paths_have_two_distinct_queries": all(
                len(set(path["query_movie_ids"])) == 2
                for record in records
                for path in record["policy_paths"].values()
            ),
            "explicit_improves_mean_nll_by_0_03_vs_immediate": (
                mean_nlls["immediate_eig"]
                - mean_nlls["receding_explicit"]
                >= 0.03
            ),
            "explicit_wins_on_five_vs_immediate": wins_vs_immediate >= 5,
            "explicit_no_worse_than_random": (
                mean_nlls["receding_explicit"]
                <= mean_nlls["seeded_random"]
            ),
        }
    gates["all_pass"] = all(gates.values())
    summary = {
        "num_screen_users": len(screen_user_ids),
        "num_enrolled_users": len(records),
        "num_unique_round2_policy_states": unique_state_count,
        "expected_physical_requests": expected_requests,
        "mean_final_heldout_nll": mean_nlls,
        "explicit_mean_nll_improvement_vs_immediate": (
            mean_nlls["immediate_eig"] - mean_nlls["receding_explicit"]
        ),
        "explicit_wins_vs_immediate": wins_vs_immediate,
        "gates": gates,
    }
    return {
        "schema_version": 9,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "screen_user_ids": list(screen_user_ids),
            "user_ids": list(user_ids),
            "initial_movie_ids": list(INITIAL_HISTORY_MOVIE_IDS),
            "candidate_pool_size": base.CANDIDATE_POOL_SIZE,
            "candidate_count_per_state": candidate_count,
            "outcomes_per_candidate": 5,
            "heldout_count": HELDOUT_COUNT,
            "policies": list(POLICIES),
            "common_tree_for_identical_policy_states": True,
            "candidate_outcomes_read_only_after_each_round_tree_frozen": True,
            "heldout_ratings_read_only_after_policy_paths_fixed": True,
            "likelihood_history_hidden": True,
            "likelihood_sum_tolerance": LIKELIHOOD_SUM_TOLERANCE,
            "normalized_probability_rows": normalized_probability_rows,
            "source_ratings_omitted_from_persisted_artifacts": True,
            "raw_responses_private_and_untracked": True,
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
    parser.add_argument("--stage", choices=("serving_smoke", "formal"), required=True)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
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
            "schema_version": 9,
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
