#!/usr/bin/env python3
"""Compare depth-2, depth-1, and immediate-EIG MovieLens policies."""

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
import scripts.movielens_explicit_rollout_ranking_v7 as v7


SELECTION_SEED = 24309
INITIAL_HISTORY_MOVIE_IDS = v7.INITIAL_HISTORY_MOVIE_IDS
SMOKE_USER_IDS = (250, 870)
FORMAL_SCREEN_USER_IDS = (
    455,
    933,
    49,
    395,
    606,
    620,
    18,
    887,
    503,
    916,
    109,
    630,
    262,
    804,
    72,
    276,
    99,
    230,
    660,
    889,
    658,
    622,
    291,
    786,
    198,
    478,
    773,
    380,
    618,
    738,
    790,
    374,
    751,
    830,
    41,
    92,
    727,
    886,
    177,
    65,
    892,
    151,
    256,
    903,
    174,
    757,
    763,
    58,
)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_SCREEN_USER_IDS
ENROLLMENT_COUNT = 4
FORMAL_Q1_CANDIDATE_COUNT = 4
SMOKE_Q1_CANDIDATE_COUNT = 1
FORMAL_EXPECTED_REQUESTS = 1056
SMOKE_EXPECTED_REQUESTS = 62
RECOVERY_RAW_SHA256 = (
    "11f1311024b559008db543d037b518f5e89a35b2d5b800c1b8ce7494ffc87cfb"
)
DEFAULT_PROBABILITY_SUM_TOLERANCE = 0.02
RECOVERY_TERMINAL_SUM_TOLERANCE = 0.10
RECOVERY_RELAXED_TERMINAL_ROWS = 2
RECOVERY_COST_USD = 5.00894919


def selected_user_ids(
    ratings: dict[int, dict[int, int]],
) -> tuple[int, ...]:
    excluded = v7._prior_users() | set(v7.ALL_SELECTED_USER_IDS)
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
        raise ValueError("frozen v8 MovieLens user selection does not reproduce")
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


def top_immediate_eig_indices(
    candidate_ids: Sequence[int],
    likelihoods: np.ndarray,
    *,
    count: int,
) -> tuple[int, ...]:
    if likelihoods.ndim != 3 or likelihoods.shape[1] < len(candidate_ids):
        raise ValueError("candidate likelihoods do not align")
    if count <= 0 or count > len(candidate_ids):
        raise ValueError("candidate count is invalid")
    eig = immediate_eig_values(likelihoods[:, : len(candidate_ids), :])
    return tuple(
        sorted(
            range(len(candidate_ids)),
            key=lambda index: (-eig[index], candidate_ids[index]),
        )[:count]
    )


def aggregate_depth_scores(
    q1_probabilities: np.ndarray,
    q1_heldout_entropies: np.ndarray,
    q2_probabilities: np.ndarray,
    terminal_entropies: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if q1_probabilities.shape != q1_heldout_entropies.shape:
        raise ValueError("first-level probabilities and entropies do not align")
    if q2_probabilities.shape != terminal_entropies.shape:
        raise ValueError("second-level probabilities and entropies do not align")
    if q2_probabilities.shape[:2] != q1_probabilities.shape:
        raise ValueError("first- and second-level branches do not align")
    depth1 = np.sum(q1_probabilities * q1_heldout_entropies, axis=1)
    conditional_terminal = np.sum(
        q2_probabilities * terminal_entropies,
        axis=2,
    )
    depth2 = np.sum(q1_probabilities * conditional_terminal, axis=1)
    return depth1, depth2


def _normalized_mixture(
    likelihoods: np.ndarray,
    movie_index: int,
) -> np.ndarray:
    values = np.mean(likelihoods[:, movie_index, :], axis=0)
    total = float(np.sum(values))
    if total <= 0.0:
        raise ValueError("predictive outcome distribution has zero mass")
    return values / total


def count_rows_outside_sum_tolerance(
    responses: Sequence[str],
    *,
    profile_count: int,
    movie_count: int,
    tolerance: float,
) -> int:
    count = 0
    for response in responses:
        parse_rating_likelihoods(
            response,
            profile_count=profile_count,
            movie_count=movie_count,
            sum_tolerance=RECOVERY_TERMINAL_SUM_TOLERANCE,
        )
        rows = _parse_json_object(response)["profiles"]
        for profile in rows:
            for probabilities in profile["ratings"]:
                total = float(np.sum(np.asarray(probabilities, dtype=float)))
                if (
                    total < 1.0 - tolerance - 1e-12
                    or total > 1.0 + tolerance + 1e-12
                ):
                    count += 1
    return count


class _ReplayBatches:
    def __init__(
        self,
        batches: Sequence[Sequence[str]],
        usage: dict[str, Any],
    ) -> None:
        self._batches = [list(batch) for batch in batches]
        self._usage = dict(usage)
        self._index = 0

    def chat_complete_messages_batched(
        self,
        batch_messages: Sequence[Sequence[dict[str, str]]],
        **_kwargs: Any,
    ) -> list[str]:
        if self._index >= len(self._batches):
            raise ValueError("replay requested an unexpected response batch")
        responses = self._batches[self._index]
        self._index += 1
        if len(batch_messages) != len(responses):
            raise ValueError("replay batch length does not match frozen responses")
        return responses

    def usage_snapshot(self) -> dict[str, Any]:
        return dict(self._usage)

    def assert_exhausted(self) -> None:
        if self._index != len(self._batches):
            raise ValueError("replay did not consume every frozen response batch")


def _usage_by_model_from_log(path: Path) -> dict[str, dict[str, Any]]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    events = [row for row in rows if row.get("event") == "llm_token_usage"]
    if len(events) != FORMAL_EXPECTED_REQUESTS:
        raise ValueError("recovery log does not contain exactly 1,056 requests")
    if sum(int(row["reasoning_tokens"]) for row in events) != 0:
        raise ValueError("recovery log contains reasoning tokens")
    if not np.isclose(
        sum(float(row["cost_usd"]) for row in events),
        RECOVERY_COST_USD,
        atol=1e-9,
        rtol=0.0,
    ):
        raise ValueError("recovery log cost does not match the frozen run")
    result: dict[str, dict[str, Any]] = {}
    for row in events:
        model = str(row["model"])
        usage = result.setdefault(
            model,
            {
                "adapter_requests": 0,
                "adapter_prompt_tokens": 0,
                "adapter_completion_tokens": 0,
                "adapter_reasoning_tokens": 0,
                "adapter_cost_usd": 0.0,
            },
        )
        usage["adapter_requests"] += 1
        usage["adapter_prompt_tokens"] += int(row["prompt_tokens"])
        usage["adapter_completion_tokens"] += int(row["completion_tokens"])
        usage["adapter_reasoning_tokens"] += int(row["reasoning_tokens"])
        usage["adapter_cost_usd"] += float(row["cost_usd"])
    return result


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
    terminal_sum_tolerance: float = DEFAULT_PROBABILITY_SUM_TOLERANCE,
    expected_relaxed_terminal_rows: int = 0,
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
    q1_candidate_count = (
        SMOKE_Q1_CANDIDATE_COUNT
        if stage == "serving_smoke"
        else FORMAL_Q1_CANDIDATE_COUNT
    )
    generator, likelihood = _build_models(config, likelihood_model)
    raw: dict[str, Any] = {}

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
        [*candidate_ids, *heldout_ids]
        for candidate_ids, heldout_ids in zip(
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
    initial_likelihoods = [
        parse_rating_likelihoods(
            response,
            profile_count=PROFILE_COUNT,
            movie_count=len(query_ids),
        )
        for response, query_ids in zip(
            initial_likelihood_raw,
            initial_query_ids,
            strict=True,
        )
    ]

    max_initial_eigs = [
        max(
            immediate_eig_values(
                matrix[:, : base.CANDIDATE_POOL_SIZE, :]
            )
        )
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
                "schema_version": 8,
                "status": "gate_failed",
                "protocol": {
                    "stage": stage,
                    "selection_seed": SELECTION_SEED,
                    "screen_user_ids": list(screen_user_ids),
                    "initial_movie_ids": list(INITIAL_HISTORY_MOVIE_IDS),
                    "candidate_outcomes_not_read": True,
                    "heldout_ratings_not_read": True,
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
        top_immediate_eig_indices(
            candidate_ids,
            matrix,
            count=q1_candidate_count,
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
                        *remaining_ids,
                        *heldout_ids_many[user_index],
                    ]
                ],
            )
            for support, remaining_ids, (
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
    q1_likelihoods = [
        parse_rating_likelihoods(
            response,
            profile_count=len(support),
            movie_count=len(remaining_ids) + HELDOUT_COUNT,
        )
        for response, support, remaining_ids in zip(
            q1_likelihood_raw,
            q1_supports,
            q1_remaining_ids,
            strict=True,
        )
    ]
    q2_indices = [
        top_immediate_eig_indices(
            remaining_ids,
            matrix,
            count=1,
        )[0]
        for remaining_ids, matrix in zip(
            q1_remaining_ids,
            q1_likelihoods,
            strict=True,
        )
    ]

    q2_keys: list[tuple[int, int]] = []
    q2_histories: list[list[dict[str, Any]]] = []
    for q1_branch_index, (
        q1_history,
        remaining_ids,
        q2_index,
    ) in enumerate(
        zip(
            q1_histories,
            q1_remaining_ids,
            q2_indices,
            strict=True,
        )
    ):
        q2_movie_id = remaining_ids[q2_index]
        for rating in range(1, 6):
            q2_keys.append((q1_branch_index, rating))
            q2_histories.append(
                [
                    *q1_history,
                    {
                        **_movie_payload(items[q2_movie_id]),
                        "rating": rating,
                    },
                ]
            )
    q2_profile_raw = generator.chat_complete_messages_batched(
        [
            refreshed_profile_messages(
                history,
                q1_supports[q1_branch_index],
            )
            for history, (q1_branch_index, _rating) in zip(
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
            q1_supports[q1_branch_index],
        )
        for response, (q1_branch_index, _rating) in zip(
            q2_profile_raw,
            q2_keys,
            strict=True,
        )
    ]
    q2_supports: list[list[str]] = []
    for rows, (q1_branch_index, rating) in zip(
        q2_generated,
        q2_keys,
        strict=True,
    ):
        q2_supports.append(
            merge_branch_profiles(
                [row["description"] for row in rows],
                q1_supports[q1_branch_index],
                q1_likelihoods[q1_branch_index][
                    :,
                    q2_indices[q1_branch_index],
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
                        q1_keys[q1_branch_index][0]
                    ]
                ],
            )
            for support, (q1_branch_index, _rating) in zip(
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
    relaxed_terminal_rows = count_rows_outside_sum_tolerance(
        q2_likelihood_raw,
        profile_count=8,
        movie_count=HELDOUT_COUNT,
        tolerance=DEFAULT_PROBABILITY_SUM_TOLERANCE,
    )
    if relaxed_terminal_rows != expected_relaxed_terminal_rows:
        raise ValueError(
            "unexpected count of terminal rows outside the default tolerance"
        )
    q2_likelihoods = [
        parse_rating_likelihoods(
            response,
            profile_count=len(support),
            movie_count=HELDOUT_COUNT,
            sum_tolerance=terminal_sum_tolerance,
        )
        for response, support in zip(
            q2_likelihood_raw,
            q2_supports,
            strict=True,
        )
    ]

    records: list[dict[str, Any]] = []
    pending_paths: list[dict[str, Any]] = []
    q1_branches_per_user = q1_candidate_count * 5
    for user_index, user_id in enumerate(user_ids):
        q1_probabilities = np.empty((q1_candidate_count, 5), dtype=float)
        q1_entropies = np.empty((q1_candidate_count, 5), dtype=float)
        q2_probabilities = np.empty((q1_candidate_count, 5, 5), dtype=float)
        terminal_entropies = np.empty((q1_candidate_count, 5, 5), dtype=float)
        for action_index in range(q1_candidate_count):
            pool_index = q1_indices_many[user_index][action_index]
            q1_probabilities[action_index] = _normalized_mixture(
                initial_likelihoods[user_index],
                pool_index,
            )
            for rating_index in range(5):
                q1_branch_index = (
                    user_index * q1_branches_per_user
                    + action_index * 5
                    + rating_index
                )
                remaining_count = len(q1_remaining_ids[q1_branch_index])
                q1_entropies[action_index, rating_index] = (
                    v7.mean_predictive_entropy(
                        q1_likelihoods[q1_branch_index][
                            :,
                            remaining_count:,
                            :,
                        ]
                    )
                )
                q2_probabilities[action_index, rating_index] = (
                    _normalized_mixture(
                        q1_likelihoods[q1_branch_index],
                        q2_indices[q1_branch_index],
                    )
                )
                q2_offset = q1_branch_index * 5
                terminal_entropies[action_index, rating_index] = [
                    v7.mean_predictive_entropy(q2_likelihoods[q2_offset + index])
                    for index in range(5)
                ]
        depth1_values, depth2_values = aggregate_depth_scores(
            q1_probabilities,
            q1_entropies,
            q2_probabilities,
            terminal_entropies,
        )
        policy_action_indices = {
            "depth2": int(np.argmin(depth2_values)),
            "depth1": int(np.argmin(depth1_values)),
            "immediate_eig": 0,
        }
        paths: dict[str, dict[str, Any]] = {}
        for policy, action_index in policy_action_indices.items():
            q1_pool_index = q1_indices_many[user_index][action_index]
            q1_movie_id = candidate_pools[user_index][q1_pool_index]
            q1_rating = ratings[user_id][q1_movie_id]
            q1_branch_index = (
                user_index * q1_branches_per_user
                + action_index * 5
                + q1_rating
                - 1
            )
            q2_movie_id = q1_remaining_ids[q1_branch_index][
                q2_indices[q1_branch_index]
            ]
            q2_rating = ratings[user_id][q2_movie_id]
            q2_branch_index = q1_branch_index * 5 + q2_rating - 1
            paths[policy] = {
                "query_movie_ids": [q1_movie_id, q2_movie_id],
                "query_movies": [
                    _movie_payload(items[q1_movie_id]),
                    _movie_payload(items[q2_movie_id]),
                ],
                "q1_branch_index": q1_branch_index,
                "q2_branch_index": q2_branch_index,
                "round1_heldout_nll_matrix": q1_likelihoods[q1_branch_index][
                    :,
                    len(q1_remaining_ids[q1_branch_index]) :,
                    :,
                ],
                "final_heldout_nll_matrix": q2_likelihoods[q2_branch_index],
            }
        pending_paths.append(
            {
                "user_id": user_id,
                "heldout_ids": heldout_ids_many[user_index],
                "paths": paths,
            }
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
                    immediate_eig_values(
                        initial_likelihoods[user_index][
                            :,
                            : base.CANDIDATE_POOL_SIZE,
                            :,
                        ]
                    )[pool_index]
                    for pool_index in q1_indices_many[user_index]
                ],
                "q1_depth1_expected_entropies": depth1_values.tolist(),
                "q1_depth2_expected_terminal_entropies": depth2_values.tolist(),
                "policy_paths": {},
            }
        )

    # Policy paths are fixed before held-out ratings are accessed.
    for record, pending in zip(records, pending_paths, strict=True):
        heldout_ratings = [
            ratings[pending["user_id"]][movie_id]
            for movie_id in pending["heldout_ids"]
        ]
        for policy, path in pending["paths"].items():
            query_ids = path["query_movie_ids"]
            if len(query_ids) != 2 or len(set(query_ids)) != 2:
                raise ValueError("policy path did not contain two distinct queries")
            record["policy_paths"][policy] = {
                "query_movie_ids": query_ids,
                "query_movies": path["query_movies"],
                "round1_heldout_nll": predictive_nll(
                    path["round1_heldout_nll_matrix"],
                    heldout_ratings,
                ),
                "final_heldout_nll": predictive_nll(
                    path["final_heldout_nll_matrix"],
                    heldout_ratings,
                ),
            }

    usage = _usage(generator, likelihood)
    expected_requests = (
        SMOKE_EXPECTED_REQUESTS
        if stage == "serving_smoke"
        else FORMAL_EXPECTED_REQUESTS
    )
    if stage == "serving_smoke":
        gates = {
            "exact_physical_request_count": (
                usage["physical_requests"] == expected_requests
            ),
            "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
            "all_five_q1_outcomes_completed": len(q1_keys) == 5,
            "all_twenty_five_q2_outcomes_completed": len(q2_keys) == 25,
            "all_policy_paths_have_two_distinct_queries": all(
                len(
                    {
                        movie_id
                        for movie_id in path["query_movie_ids"]
                    }
                )
                == 2
                for path in records[0]["policy_paths"].values()
            ),
        }
        summary: dict[str, Any] = {
            "num_screen_users": 1,
            "num_enrolled_users": 1,
            "num_q1_branches": len(q1_keys),
            "num_q2_branches": len(q2_keys),
            "gates": gates,
        }
    else:
        final_nlls = {
            policy: [
                record["policy_paths"][policy]["final_heldout_nll"]
                for record in records
            ]
            for policy in ("depth2", "depth1", "immediate_eig")
        }
        mean_nlls = {
            policy: float(np.mean(values))
            for policy, values in final_nlls.items()
        }
        depth2_wins_depth1 = sum(
            depth2 < depth1
            for depth2, depth1 in zip(
                final_nlls["depth2"],
                final_nlls["depth1"],
                strict=True,
            )
        )
        depth2_wins_immediate = sum(
            depth2 < immediate
            for depth2, immediate in zip(
                final_nlls["depth2"],
                final_nlls["immediate_eig"],
                strict=True,
            )
        )
        gates = {
            "exact_physical_request_count": (
                usage["physical_requests"] == expected_requests
            ),
            "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
            "prospective_enrollment_complete": len(records) == ENROLLMENT_COUNT,
            "all_policy_paths_have_two_distinct_queries": all(
                len(
                    {
                        movie_id
                        for movie_id in path["query_movie_ids"]
                    }
                )
                == 2
                for record in records
                for path in record["policy_paths"].values()
            ),
            "depth2_improves_mean_nll_by_0_03_vs_depth1": (
                mean_nlls["depth1"] - mean_nlls["depth2"] >= 0.03
            ),
            "depth2_wins_on_three_vs_depth1": depth2_wins_depth1 >= 3,
            "depth2_improves_mean_nll_by_0_03_vs_immediate": (
                mean_nlls["immediate_eig"] - mean_nlls["depth2"] >= 0.03
            ),
            "depth2_wins_on_three_vs_immediate": depth2_wins_immediate >= 3,
        }
        summary = {
            "num_screen_users": len(screen_user_ids),
            "num_enrolled_users": len(records),
            "mean_final_heldout_nll": mean_nlls,
            "depth2_mean_nll_improvement_vs_depth1": (
                mean_nlls["depth1"] - mean_nlls["depth2"]
            ),
            "depth2_mean_nll_improvement_vs_immediate": (
                mean_nlls["immediate_eig"] - mean_nlls["depth2"]
            ),
            "depth2_wins_vs_depth1": depth2_wins_depth1,
            "depth2_wins_vs_immediate": depth2_wins_immediate,
            "gates": gates,
        }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 8,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "screen_user_ids": list(screen_user_ids),
            "user_ids": list(user_ids),
            "initial_movie_ids": list(INITIAL_HISTORY_MOVIE_IDS),
            "candidate_pool_size": base.CANDIDATE_POOL_SIZE,
            "q1_candidate_count": q1_candidate_count,
            "q1_outcomes_per_candidate": 5,
            "q2_rollout_policy": "immediate_eig",
            "q2_outcomes_per_q1_branch": 5,
            "heldout_count": HELDOUT_COUNT,
            "profile_count_initial": PROFILE_COUNT,
            "profile_count_transition": 8,
            "likelihood_model": likelihood_model,
            "likelihood_history_hidden": True,
            "common_transition_tree_for_all_policies": True,
            "candidate_outcomes_read_only_after_tree_frozen": True,
            "heldout_ratings_read_only_after_policy_paths_fixed": True,
            "source_ratings_omitted_from_persisted_artifacts": True,
            "raw_responses_private_and_untracked": True,
            "terminal_probability_sum_tolerance": terminal_sum_tolerance,
            "relaxed_terminal_probability_rows": relaxed_terminal_rows,
            "parser_recovery_applied": expected_relaxed_terminal_rows > 0,
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
    parser.add_argument("--resume-raw", type=Path)
    parser.add_argument("--resume-log", type=Path)
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    terminal_sum_tolerance = DEFAULT_PROBABILITY_SUM_TOLERANCE
    expected_relaxed_terminal_rows = 0
    replay_adapters: tuple[_ReplayBatches, _ReplayBatches] | None = None
    global _build_models
    if args.resume_raw or args.resume_log:
        if args.stage != "formal":
            raise ValueError("replay recovery is formal-only")
        if not args.resume_raw or not args.resume_log:
            raise ValueError("both replay paths are required")
        if hashlib.sha256(args.resume_raw.read_bytes()).hexdigest() != RECOVERY_RAW_SHA256:
            raise ValueError("recovery raw hash does not match the frozen run")
        frozen = json.loads(args.resume_raw.read_text(encoding="utf-8"))
        responses = frozen["responses"]
        usage_by_model = _usage_by_model_from_log(args.resume_log)
        generator_model = config.model_pairs[0].questioner.model
        replay_adapters = (
            _ReplayBatches(
                [
                    responses["initial_profiles"],
                    responses["q1_profiles"],
                    responses["q2_profiles"],
                ],
                usage_by_model[generator_model],
            ),
            _ReplayBatches(
                [
                    responses["initial_likelihoods"],
                    responses["q1_likelihoods"],
                    responses["q2_likelihoods"],
                ],
                usage_by_model[args.likelihood_model],
            ),
        )

        def replay_builder(config: Config, likelihood_model: str):
            del config, likelihood_model
            assert replay_adapters is not None
            return replay_adapters

        _build_models = replay_builder
        terminal_sum_tolerance = RECOVERY_TERMINAL_SUM_TOLERANCE
        expected_relaxed_terminal_rows = RECOVERY_RELAXED_TERMINAL_ROWS
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
            terminal_sum_tolerance=terminal_sum_tolerance,
            expected_relaxed_terminal_rows=expected_relaxed_terminal_rows,
        )
        if replay_adapters is not None:
            for adapter in replay_adapters:
                adapter.assert_exhausted()
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": 8,
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
