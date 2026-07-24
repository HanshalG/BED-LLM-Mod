#!/usr/bin/env python3
"""Rank MovieLens queries by explicit semantic-belief regeneration rollouts."""

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
    FORMAL_USER_IDS as V1_USERS,
    HELDOUT_COUNT,
    PROFILE_COUNT,
    SMOKE_USER_IDS as V1_SMOKE_USERS,
    _average_ranks,
    _correlation,
    _history_payload,
    _movie_payload,
    _usage,
    _write_raw_checkpoint,
    immediate_eig_values,
    load_movielens,
    merge_branch_profiles,
    parse_profiles,
    parse_rating_likelihoods,
    profile_messages,
)
from scripts.movielens_profile_dynamics_gate_v2 import (
    ALL_SELECTED_USER_IDS as V2_USERS,
    parse_refreshed_profiles,
    profile_only_rating_likelihood_messages,
    refreshed_profile_messages,
)
from scripts.movielens_profile_dynamics_gate_v3 import (
    ALL_SELECTED_USER_IDS as V3_USERS,
)
import scripts.movielens_adaptive_candidate_gate_v4 as base
from scripts.movielens_adaptive_candidate_gate_v4 import (
    ALL_SELECTED_USER_IDS as V4_USERS,
)
from scripts.movielens_uncertainty_enriched_gate_v5 import (
    ALL_SELECTED_USER_IDS as V5_USERS,
)
from scripts.movielens_semantic_ranking_gate_v6 import (
    FORMAL_SCREEN_USER_IDS as V6_USERS,
)


SELECTION_SEED = 24308
INITIAL_HISTORY_MOVIE_IDS = (50, 100, 1, 98)
SMOKE_USER_IDS = (253, 654)
FORMAL_SCREEN_USER_IDS = (
    201,
    749,
    454,
    910,
    407,
    325,
    248,
    929,
    747,
    305,
    313,
    553,
    429,
    96,
    399,
    577,
    64,
    263,
    5,
    389,
)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_SCREEN_USER_IDS
ENROLLMENT_COUNT = 4
EXPLICIT_ROLLOUT_REQUESTS_PER_USER = 40
FORMAL_EXPECTED_REQUESTS = 232
SMOKE_EXPECTED_REQUESTS = 12


def _prior_users() -> set[int]:
    return (
        set(V1_USERS)
        | set(V1_SMOKE_USERS)
        | set(V2_USERS)
        | set(V3_USERS)
        | set(V4_USERS)
        | set(V5_USERS)
        | set(V6_USERS)
    )


def selected_user_ids(
    ratings: dict[int, dict[int, int]],
) -> tuple[int, ...]:
    excluded = _prior_users()
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
        raise ValueError("frozen v7 MovieLens user selection does not reproduce")
    return chosen


def mean_predictive_entropy(likelihoods: np.ndarray) -> float:
    if likelihoods.ndim != 3 or likelihoods.shape[2] != 5:
        raise ValueError("likelihoods must have shape profiles x movies x 5")
    mixture = np.mean(likelihoods, axis=0)
    positive = np.where(mixture > 0.0, mixture, 1.0)
    entropies = -np.sum(
        np.where(mixture > 0.0, mixture * np.log(positive), 0.0),
        axis=1,
    )
    return float(np.mean(entropies))


def explicit_rollout_scorer(
    *,
    generator: Any,
    likelihood: Any,
    config: Config,
    histories: Sequence[Sequence[dict[str, Any]]],
    initial_profiles: Sequence[Sequence[str]],
    candidate_pools: Sequence[Sequence[int]],
    selected_indices: Sequence[Sequence[int]],
    selected_movie_ids: Sequence[Sequence[int]],
    heldout_ids_many: Sequence[Sequence[int]],
    initial_likelihoods: Sequence[np.ndarray],
    user_ids: Sequence[int],
    items: dict[int, dict[str, Any]],
    stage: str = "formal",
    raw: dict[str, Any] | None = None,
    raw_checkpoint_path: Path | None = None,
) -> tuple[list[list[float]], dict[str, Any]]:
    num_users = len(user_ids)
    aligned = (
        histories,
        initial_profiles,
        candidate_pools,
        selected_indices,
        selected_movie_ids,
        heldout_ids_many,
        initial_likelihoods,
    )
    if any(len(values) != num_users for values in aligned):
        raise ValueError("explicit rollout inputs do not align")

    flat_keys: list[tuple[int, int, int]] = []
    hypothetical_histories: list[list[dict[str, Any]]] = []
    for user_index, movie_ids in enumerate(selected_movie_ids):
        if len(movie_ids) != len(selected_indices[user_index]):
            raise ValueError("selected candidates and pool indices do not align")
        for candidate_index, movie_id in enumerate(movie_ids):
            for rating in range(1, 6):
                flat_keys.append((user_index, candidate_index, rating))
                hypothetical_histories.append(
                    [
                        *histories[user_index],
                        {
                            **_movie_payload(items[movie_id]),
                            "rating": rating,
                        },
                    ]
                )

    generation_raw = generator.chat_complete_messages_batched(
        [
            refreshed_profile_messages(
                history,
                initial_profiles[user_index],
            )
            for history, (user_index, _candidate_index, _rating) in zip(
                hypothetical_histories,
                flat_keys,
                strict=True,
            )
        ],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    explicit_raw: dict[str, Any] = {
        "hypothetical_refreshed_profiles": generation_raw,
    }
    if raw is not None:
        raw["explicit_rollouts"] = explicit_raw
        _write_raw_checkpoint(
            raw_checkpoint_path,
            stage=stage,
            user_ids=user_ids,
            raw=raw,
        )
    generated_rows = [
        parse_refreshed_profiles(response, initial_profiles[user_index])
        for response, (user_index, _candidate_index, _rating) in zip(
            generation_raw,
            flat_keys,
            strict=True,
        )
    ]

    branch_profiles: list[list[str]] = []
    for rows, (user_index, candidate_index, rating) in zip(
        generated_rows,
        flat_keys,
        strict=True,
    ):
        pool_index = selected_indices[user_index][candidate_index]
        old_probabilities = initial_likelihoods[user_index][
            :, pool_index, rating - 1
        ]
        branch_profiles.append(
            merge_branch_profiles(
                [row["description"] for row in rows],
                initial_profiles[user_index],
                old_probabilities,
            )
        )

    likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                profiles,
                [items[movie_id] for movie_id in heldout_ids_many[user_index]],
            )
            for profiles, (user_index, _candidate_index, _rating) in zip(
                branch_profiles,
                flat_keys,
                strict=True,
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    explicit_raw["hypothetical_downstream_likelihoods"] = likelihood_raw
    if raw is not None:
        _write_raw_checkpoint(
            raw_checkpoint_path,
            stage=stage,
            user_ids=user_ids,
            raw=raw,
        )
    branch_likelihoods = [
        parse_rating_likelihoods(
            response,
            profile_count=len(profiles),
            movie_count=len(heldout_ids_many[user_index]),
        )
        for response, profiles, (user_index, _candidate_index, _rating) in zip(
            likelihood_raw,
            branch_profiles,
            flat_keys,
            strict=True,
        )
    ]
    branch_entropies = [
        mean_predictive_entropy(matrix) for matrix in branch_likelihoods
    ]

    scores: list[list[float]] = []
    outcome_probabilities: list[list[list[float]]] = []
    entropy_rows: list[list[list[float]]] = []
    offset = 0
    for user_index, movie_ids in enumerate(selected_movie_ids):
        user_scores: list[float] = []
        user_probabilities: list[list[float]] = []
        user_entropies: list[list[float]] = []
        for candidate_index, _movie_id in enumerate(movie_ids):
            pool_index = selected_indices[user_index][candidate_index]
            probabilities = np.mean(
                initial_likelihoods[user_index][:, pool_index, :],
                axis=0,
            )
            probabilities = probabilities / float(np.sum(probabilities))
            entropies = np.asarray(branch_entropies[offset : offset + 5])
            if len(entropies) != 5:
                raise ValueError("explicit rollout lost a rating branch")
            user_scores.append(-float(np.dot(probabilities, entropies)))
            user_probabilities.append(probabilities.tolist())
            user_entropies.append(entropies.tolist())
            offset += 5
        scores.append(user_scores)
        outcome_probabilities.append(user_probabilities)
        entropy_rows.append(user_entropies)
    if offset != len(flat_keys):
        raise ValueError("explicit rollout branch accounting does not close")

    explicit_raw["outcome_probabilities"] = outcome_probabilities
    explicit_raw["branch_mean_predictive_entropies"] = entropy_rows
    return scores, explicit_raw


_build_models = base._build_models


def _base_overrides() -> dict[str, Any]:
    return {
        "SELECTION_SEED": SELECTION_SEED,
        "SMOKE_USER_IDS": SMOKE_USER_IDS,
        "FORMAL_SCREEN_USER_IDS": FORMAL_SCREEN_USER_IDS,
        "PROSPECTIVE_ENROLLMENT_COUNT": ENROLLMENT_COUNT,
        "ALL_SELECTED_USER_IDS": ALL_SELECTED_USER_IDS,
        "SEMANTIC_RANKING_MESSAGES": None,
        "SEMANTIC_RANKING_PARSE": None,
        "EXPLICIT_ROLLOUT_SCORER": explicit_rollout_scorer,
        "EXPLICIT_ROLLOUT_REQUESTS_PER_USER": (
            EXPLICIT_ROLLOUT_REQUESTS_PER_USER
        ),
        "INITIAL_HISTORY_MOVIE_IDS": INITIAL_HISTORY_MOVIE_IDS,
        "selected_user_ids": selected_user_ids,
        "_build_models": _build_models,
    }


def _ranking_summary(result: dict[str, Any]) -> dict[str, Any]:
    scores: list[float] = []
    negative_nlls: list[float] = []
    explicit_regrets: list[float] = []
    immediate_regrets: list[float] = []
    random_regrets: list[float] = []
    explicit_beats = 0
    for record in result["records"]:
        nlls = [float(row["heldout_nll"]) for row in record["branches"]]
        oracle = min(nlls)
        explicit = int(record["semantic_lookahead_selected_branch"])
        immediate = int(record["immediate_eig_selected_branch"])
        random_index = int(
            np.random.default_rng(
                SELECTION_SEED * 1000 + record["user_id"]
            ).integers(len(nlls))
        )
        explicit_regrets.append(nlls[explicit] - oracle)
        immediate_regrets.append(nlls[immediate] - oracle)
        random_regrets.append(nlls[random_index] - oracle)
        explicit_beats += nlls[explicit] < nlls[immediate]
        scores.extend(float(value) for value in record["semantic_lookahead_scores"])
        negative_nlls.extend(-value for value in nlls)
    correlation = _correlation(
        _average_ranks(scores),
        _average_ranks(negative_nlls),
    )
    mean_explicit = float(np.mean(explicit_regrets))
    mean_immediate = float(np.mean(immediate_regrets))
    mean_random = float(np.mean(random_regrets))
    gates = {
        "exact_physical_request_count": (
            result["usage"]["physical_requests"] == FORMAL_EXPECTED_REQUESTS
        ),
        "zero_reasoning_tokens": result["usage"]["reasoning_tokens"] == 0,
        "prospective_enrollment_complete": len(result["records"]) == ENROLLMENT_COUNT,
        "explicit_rollout_spearman_at_least_0_25": (
            correlation is not None and correlation >= 0.25
        ),
        "explicit_rollout_regret_improves_by_0_02": (
            mean_immediate - mean_explicit >= 0.02
        ),
        "explicit_rollout_beats_immediate_on_two": explicit_beats >= 2,
        "explicit_rollout_no_worse_than_random": mean_explicit <= mean_random,
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_users": ENROLLMENT_COUNT,
        "num_branches": ENROLLMENT_COUNT * base.SELECTED_CANDIDATE_COUNT,
        "explicit_rollout_spearman_vs_negative_branch_nll": correlation,
        "mean_explicit_rollout_top1_regret": mean_explicit,
        "mean_immediate_eig_top1_regret": mean_immediate,
        "mean_seeded_random_top1_regret": mean_random,
        "explicit_rollout_beats_immediate_count": explicit_beats,
        "gates": gates,
    }


def run_formal(
    config: Config,
    *,
    data_dir: str | Path,
    likelihood_model: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    overrides = _base_overrides()
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        result = base.run_gate(
            config,
            data_dir=data_dir,
            likelihood_model=likelihood_model,
            stage="formal",
            raw_checkpoint_path=raw_checkpoint_path,
        )
    finally:
        for name, value in previous.items():
            setattr(base, name, value)
    result["schema_version"] = 7
    result["protocol"].update(
        {
            "selection_seed": SELECTION_SEED,
            "explicit_regeneration_rollout_ranking": True,
            "initial_movie_ids": list(INITIAL_HISTORY_MOVIE_IDS),
            "v1_through_v6_users_excluded": True,
            "hypothetical_outcomes_per_candidate": 5,
            "source_outcomes_hidden_from_rollout_scorer": True,
        }
    )
    if result["records"] and "branches" in result["records"][0]:
        result["summary"] = _ranking_summary(result)
        result["status"] = (
            "passed" if result["summary"]["gates"]["all_pass"] else "gate_failed"
        )
    return result


def run_smoke(
    config: Config,
    *,
    data_dir: str | Path,
    likelihood_model: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    ratings, items = load_movielens(data_dir)
    selected_user_ids(ratings)
    user_id = SMOKE_USER_IDS[0]
    generator, likelihood = _build_models(config, likelihood_model)
    history = _history_payload(
        INITIAL_HISTORY_MOVIE_IDS,
        ratings[user_id],
        items,
    )
    initial_raw = generator.chat_complete_messages_batched(
        [profile_messages(history)],
        temperature=float(config.generation_temperature_diverse),
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    initial_profiles = [parse_profiles(initial_raw[0])]

    previous_history_ids = base.INITIAL_HISTORY_MOVIE_IDS
    previous_seed = base.SELECTION_SEED
    try:
        base.INITIAL_HISTORY_MOVIE_IDS = INITIAL_HISTORY_MOVIE_IDS
        base.SELECTION_SEED = SELECTION_SEED
        candidate_pool, heldout_ids = base.candidate_and_heldout_ids(
            user_id,
            ratings,
        )
    finally:
        base.INITIAL_HISTORY_MOVIE_IDS = previous_history_ids
        base.SELECTION_SEED = previous_seed
    query_ids = [*candidate_pool, *heldout_ids]
    initial_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                initial_profiles[0],
                [items[movie_id] for movie_id in query_ids],
            )
        ],
        temperature=0.0,
        block_size=config.batched_block_size,
        max_new_tokens=config.openrouter_max_output_tokens,
    )
    initial_likelihoods = [
        parse_rating_likelihoods(
            initial_likelihood_raw[0],
            profile_count=PROFILE_COUNT,
            movie_count=len(query_ids),
        )
    ]
    eig_values = immediate_eig_values(
        initial_likelihoods[0][:, : base.CANDIDATE_POOL_SIZE, :]
    )
    selected_index = base.select_candidate_indices(
        candidate_pool,
        eig_values,
    )[:1]
    selected_movie_ids = [(candidate_pool[selected_index[0]],)]
    scores, explicit_raw = explicit_rollout_scorer(
        generator=generator,
        likelihood=likelihood,
        config=config,
        histories=[history],
        initial_profiles=initial_profiles,
        candidate_pools=[candidate_pool],
        selected_indices=[selected_index],
        selected_movie_ids=selected_movie_ids,
        heldout_ids_many=[heldout_ids],
        initial_likelihoods=initial_likelihoods,
        user_ids=[user_id],
        items=items,
    )
    raw = {
        "initial_profiles": initial_raw,
        "initial_likelihoods": initial_likelihood_raw,
        "explicit_rollouts": explicit_raw,
    }
    _write_raw_checkpoint(
        raw_checkpoint_path,
        stage="serving_smoke",
        user_ids=[user_id],
        raw=raw,
    )
    usage = _usage(generator, likelihood)
    entropies = explicit_raw["branch_mean_predictive_entropies"][0][0]
    gates = {
        "exact_physical_request_count": (
            usage["physical_requests"] == SMOKE_EXPECTED_REQUESTS
        ),
        "zero_reasoning_tokens": usage["reasoning_tokens"] == 0,
        "initial_profile_count_is_six": len(initial_profiles[0]) == PROFILE_COUNT,
        "all_five_hypothetical_paths_completed": len(entropies) == 5,
        "explicit_score_is_finite": bool(np.isfinite(scores[0][0])),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "schema_version": 7,
        "status": "passed" if gates["all_pass"] else "gate_failed",
        "protocol": {
            "stage": "serving_smoke",
            "selection_seed": SELECTION_SEED,
            "user_ids": [user_id],
            "reserve_smoke_user_ids": [SMOKE_USER_IDS[1]],
            "initial_movie_ids": list(INITIAL_HISTORY_MOVIE_IDS),
            "candidate_pool_size": base.CANDIDATE_POOL_SIZE,
            "selected_candidate_count": 1,
            "heldout_count": HELDOUT_COUNT,
            "likelihood_model": likelihood_model,
            "likelihood_history_hidden": True,
            "candidate_and_heldout_outcomes_not_read": True,
            "raw_responses_private_and_untracked": True,
        },
        "summary": {
            "num_users": 1,
            "num_candidates": 1,
            "num_hypothetical_branches": 5,
            "explicit_rollout_score": scores[0][0],
            "branch_mean_predictive_entropies": entropies,
            "gates": gates,
        },
        "records": [
            {
                "user_id": user_id,
                "initial_profile_hashes": [
                    hashlib.sha256(value.encode("utf-8")).hexdigest()
                    for value in initial_profiles[0]
                ],
                "candidate_movie": _movie_payload(items[selected_movie_ids[0][0]]),
                "immediate_eig": eig_values[selected_index[0]],
                "explicit_rollout_score": scores[0][0],
            }
        ],
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
        if args.stage == "serving_smoke":
            payload = run_smoke(
                config,
                data_dir=args.data_dir,
                likelihood_model=args.likelihood_model,
                raw_checkpoint_path=raw_path,
            )
        else:
            payload = run_formal(
                config,
                data_dir=args.data_dir,
                likelihood_model=args.likelihood_model,
                raw_checkpoint_path=raw_path,
            )
    except Exception as exc:
        failure = {
            "schema_version": 7,
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
    payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
