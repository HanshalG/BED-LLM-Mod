#!/usr/bin/env python3
"""Test load-bearing semantic profile updates on recorded MovieLens ratings."""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter
from scripts.movielens_profile_dynamics_gate import (
    CANDIDATE_MOVIE_IDS,
    FORMAL_EXPECTED_REQUESTS,
    FORMAL_USER_IDS as V1_FORMAL_USER_IDS,
    HELDOUT_COUNT,
    INITIAL_MOVIE_IDS,
    ITEMS_SHA256,
    PROFILE_COUNT,
    RATINGS_SHA256,
    README_SHA256,
    RETAINED_PROFILE_COUNT,
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


SELECTION_SEED = 24303
SMOKE_USER_IDS = (294, 327)
FORMAL_USER_IDS = (378, 387, 416, 450, 470, 488, 533, 537, 580, 650, 676, 699)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_USER_IDS


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def selected_user_ids(ratings: dict[int, dict[int, int]]) -> tuple[int, ...]:
    fixed = INITIAL_MOVIE_IDS + CANDIDATE_MOVIE_IDS
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if user_id not in V1_FORMAL_USER_IDS
        and all(movie_id in user_ratings for movie_id in fixed)
        and sum(movie_id not in fixed for movie_id in user_ratings) >= HELDOUT_COUNT
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
        raise ValueError("frozen v2 MovieLens user selection does not reproduce")
    return chosen


def heldout_movie_ids(
    user_id: int,
    user_ratings: dict[int, int],
) -> tuple[int, ...]:
    fixed = set(INITIAL_MOVIE_IDS + CANDIDATE_MOVIE_IDS)
    available = np.asarray(
        sorted(movie_id for movie_id in user_ratings if movie_id not in fixed),
        dtype=int,
    )
    if len(available) < HELDOUT_COUNT:
        raise ValueError("selected user lacks enough held-out ratings")
    chosen = np.random.default_rng(SELECTION_SEED * 1000 + user_id).choice(
        available,
        size=HELDOUT_COUNT,
        replace=False,
    )
    return tuple(sorted(int(value) for value in chosen))


def refreshed_profile_messages(
    history: Sequence[dict[str, Any]],
    previous_profiles: Sequence[str],
) -> list[dict[str, str]]:
    schema = {
        "profiles": [
            {
                "id": f"p{index + 1}",
                "description": "...",
                "new_evidence_effect": "...",
            }
            for index in range(PROFILE_COUNT)
        ]
    }
    payload = {
        "observed_movie_ratings": list(history),
        "previous_candidate_profiles_for_context_only": list(previous_profiles),
    }
    return [
        {
            "role": "system",
            "content": (
                "Rebuild diverse latent movie-taste hypotheses after new recorded "
                "evidence. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {PROFILE_COUNT} replacement preference profiles "
                "from all observed ratings. Previous profiles are context only: do not "
                "copy any previous description verbatim. Every new description must "
                "materially revise, qualify, split, or replace an earlier generalization "
                "in light of the newest rating. For each profile, new_evidence_effect "
                "must state specifically how the newest rating changed that hypothesis. "
                "Profiles must remain mutually distinct and state favored and disliked "
                "genres, tones, narrative styles, or eras. Do not predict unobserved "
                "ratings, make demographic guesses, or mention a user ID. Preserve "
                "profile IDs and return "
                + json.dumps(schema, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_refreshed_profiles(
    text: str,
    previous_profiles: Sequence[str],
) -> list[dict[str, str]]:
    from scripts.movielens_profile_dynamics_gate import _parse_json_object

    rows = _parse_json_object(text).get("profiles")
    if not isinstance(rows, list) or len(rows) != PROFILE_COUNT:
        raise ValueError(f"profiles must contain exactly {PROFILE_COUNT} rows")
    previous_keys = {_clean_text(value).casefold() for value in previous_profiles}
    descriptions: set[str] = set()
    parsed: list[dict[str, str]] = []
    for index, row in enumerate(rows):
        expected_id = f"p{index + 1}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("profile IDs or order changed")
        description = row.get("description")
        effect = row.get("new_evidence_effect")
        if not isinstance(description, str) or not _clean_text(description):
            raise ValueError("profile description must be nonempty")
        if not isinstance(effect, str) or not _clean_text(effect):
            raise ValueError("new_evidence_effect must be nonempty")
        clean_description = _clean_text(description)
        key = clean_description.casefold()
        if key in previous_keys:
            raise ValueError("refreshed profile copied a previous description")
        if key in descriptions:
            raise ValueError("refreshed profile descriptions must be unique")
        descriptions.add(key)
        parsed.append(
            {
                "description": clean_description,
                "new_evidence_effect": _clean_text(effect),
            }
        )
    return parsed


def profile_only_rating_likelihood_messages(
    profiles: Sequence[str],
    movies: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    profile_rows = [
        {"id": f"p{index + 1}", "description": description}
        for index, description in enumerate(profiles)
    ]
    schema = {
        "profiles": [
            {
                "id": f"p{index + 1}",
                "ratings": [[0.2, 0.2, 0.2, 0.2, 0.2] for _movie in movies],
            }
            for index in range(len(profiles))
        ]
    }
    payload = {
        "candidate_profiles": profile_rows,
        "movies_in_fixed_order": [_movie_payload(movie) for movie in movies],
    }
    return [
        {
            "role": "system",
            "content": (
                "Estimate movie-rating likelihoods from semantic preference profiles. "
                "Return calibrated probabilities and strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For every profile and movie, return probabilities for ratings "
                "[1,2,3,4,5] in that order. Each five-number row must sum to 1. The "
                "profile is the complete conditioning information: no raw rating "
                "history is available. Preserve profile and movie order. Return "
                + json.dumps(schema, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def _build_models(config: Config, likelihood_model: str) -> tuple[Any, Any]:
    if len(config.model_pairs) != 1:
        raise ValueError("profile gate requires exactly one model pair")
    generator_spec = config.model_pairs[0].questioner
    generator = build_model_adapter(generator_spec, config)
    likelihood_spec = replace(
        generator_spec,
        model=likelihood_model,
        thinking=None,
        reasoning_effort="none",
        reasoning_max_tokens=None,
        thinking_max_new_tokens=None,
        thinking_final_max_new_tokens=None,
    )
    likelihood = build_model_adapter(likelihood_spec, config)
    return generator, likelihood


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
    user_ids = SMOKE_USER_IDS if stage == "serving_smoke" else FORMAL_USER_IDS
    branch_movie_ids = (
        CANDIDATE_MOVIE_IDS[:1]
        if stage == "serving_smoke"
        else CANDIDATE_MOVIE_IDS
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

    heldout_ids_many = [
        heldout_movie_ids(user_id, ratings[user_id]) for user_id in user_ids
    ]
    initial_query_ids = [
        [*CANDIDATE_MOVIE_IDS, *heldout_ids]
        for heldout_ids in heldout_ids_many
    ]
    initial_likelihood_raw = likelihood.chat_complete_messages_batched(
        [
            profile_only_rating_likelihood_messages(
                profiles,
                [items[movie_id] for movie_id in query_ids],
            )
            for profiles, query_ids in zip(
                initial_profiles, initial_query_ids, strict=True
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
            initial_likelihood_raw, initial_query_ids, strict=True
        )
    ]

    flat_user_indices = [
        user_index
        for user_index, _user_id in enumerate(user_ids)
        for _movie_id in branch_movie_ids
    ]
    flat_movie_ids = [
        movie_id for _user_id in user_ids for movie_id in branch_movie_ids
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
        candidate_index = CANDIDATE_MOVIE_IDS.index(movie_id)
        observed_rating = ratings[user_ids[user_index]][movie_id]
        old_probabilities = initial_likelihoods[user_index][
            :, candidate_index, observed_rating - 1
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
        candidate_matrix = initial_matrix[:, : len(CANDIDATE_MOVIE_IDS), :]
        heldout_matrix = initial_matrix[:, len(CANDIDATE_MOVIE_IDS) :, :]
        eig_values = immediate_eig_values(candidate_matrix)
        branches = []
        for local_index, movie_id in enumerate(branch_movie_ids):
            flat_index = offset + local_index
            branches.append(
                {
                    "branch_index": local_index,
                    "movie": _movie_payload(items[movie_id]),
                    "generated_profiles": refreshed_rows[flat_index],
                    "retained_support": branch_profiles[flat_index],
                    "heldout_nll": predictive_nll(
                        branch_likelihoods[flat_index],
                        heldout_ratings,
                    ),
                }
            )
        record = {
            "user_id": user_id,
            "initial_profiles": initial_profiles[user_index],
            "candidate_movies": [
                _movie_payload(items[movie_id]) for movie_id in CANDIDATE_MOVIE_IDS
            ],
            "immediate_eig_values": eig_values,
            "immediate_eig_selected_branch": int(np.argmax(eig_values)),
            "initial_heldout_nll": predictive_nll(
                heldout_matrix,
                heldout_ratings,
            ),
            "branches": branches,
        }
        if stage == "serving_smoke":
            record["replay_profiles"] = replay_rows[user_index]
        records.append(record)
        offset += len(branch_movie_ids)

    usage = _usage(generator, likelihood)
    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": 2,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "ratings_sha256": RATINGS_SHA256,
            "items_sha256": ITEMS_SHA256,
            "readme_sha256": README_SHA256,
            "user_ids": list(user_ids),
            "smoke_user_ids_disjoint_from_formal": True,
            "v1_users_excluded": True,
            "initial_movie_ids": list(INITIAL_MOVIE_IDS),
            "candidate_movie_ids": list(CANDIDATE_MOVIE_IDS),
            "heldout_count": HELDOUT_COUNT,
            "profile_count": PROFILE_COUNT,
            "retained_profile_count": RETAINED_PROFILE_COUNT,
            "likelihood_model": likelihood_model,
            "likelihood_history_hidden": True,
            "recorded_ratings_are_only_outcomes": True,
            "heldout_ratings_hidden_from_all_model_prompts": True,
            "user_ids_hidden_from_all_model_prompts": True,
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
    config.log_path = args.output_dir / "run.log"
    raw_path = args.output_dir / "RAW_RESPONSES.json"
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
    except Exception as exc:
        failure = {
            "schema_version": 2,
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
