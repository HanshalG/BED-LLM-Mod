#!/usr/bin/env python3
"""Test path-dependent semantic preference profiles on recorded MovieLens ratings."""

from __future__ import annotations

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from model_factory import build_model_adapter


SELECTION_SEED = 24302
RATINGS_SHA256 = "06416e597f82b7342361e41163890c81036900f418ad91315590814211dca490"
ITEMS_SHA256 = "553841ebc7de3a0fd0d6b62a204ea30c1e651aacfb2814c7a6584ac52f2c5701"
README_SHA256 = "4883b8cf340ed0059971b33f56ad3adc88a40792108ebca3d24aafc4c82e8b52"
FORMAL_USER_IDS = (13, 62, 141, 213, 422, 447, 479, 552, 588, 592, 655, 919)
SMOKE_USER_IDS = FORMAL_USER_IDS[:2]
INITIAL_MOVIE_IDS = (50, 100, 294, 286)
CANDIDATE_MOVIE_IDS = (258, 181, 288, 1)
HELDOUT_COUNT = 8
PROFILE_COUNT = 6
RETAINED_PROFILE_COUNT = 2
RATING_IDS = (1, 2, 3, 4, 5)
SMOKE_EXPECTED_REQUESTS = 10
FORMAL_EXPECTED_REQUESTS = 120
GENRES = (
    "unknown",
    "Action",
    "Adventure",
    "Animation",
    "Children's",
    "Comedy",
    "Crime",
    "Documentary",
    "Drama",
    "Fantasy",
    "Film-Noir",
    "Horror",
    "Musical",
    "Mystery",
    "Romance",
    "Sci-Fi",
    "Thriller",
    "War",
    "Western",
)


def _remove_json_trailing_commas(text: str) -> str:
    repaired: list[str] = []
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            repaired.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            repaired.append(char)
            continue
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                continue
        repaired.append(char)
    return "".join(repaired)


def _loads_json_with_trailing_comma_repair(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        repaired = _remove_json_trailing_commas(text)
        if repaired == text:
            raise
        return json.loads(repaired)


def _parse_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    fenced = re.fullmatch(
        r"```(?:json)?\s*(\{.*\})\s*```",
        stripped,
        flags=re.DOTALL | re.IGNORECASE,
    )
    if fenced:
        stripped = fenced.group(1)
    try:
        payload = _loads_json_with_trailing_comma_repair(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("response does not contain a JSON object")
        payload = _loads_json_with_trailing_comma_repair(
            stripped[start : end + 1]
        )
    if not isinstance(payload, dict):
        raise ValueError("response JSON must be an object")
    return payload


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def parse_profiles(text: str, count: int = PROFILE_COUNT) -> list[str]:
    rows = _parse_json_object(text).get("profiles")
    if not isinstance(rows, list) or len(rows) != count:
        raise ValueError(f"profiles must contain exactly {count} rows")
    profiles: list[str] = []
    seen: set[str] = set()
    for index, row in enumerate(rows):
        expected_id = f"p{index + 1}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("profile IDs or order changed")
        description = row.get("description")
        if not isinstance(description, str) or not _clean_text(description):
            raise ValueError("profile description must be nonempty")
        description = _clean_text(description)
        key = description.casefold()
        if key in seen:
            raise ValueError("profile descriptions must be unique")
        seen.add(key)
        profiles.append(description)
    return profiles


def parse_rating_likelihoods(
    text: str,
    *,
    profile_count: int,
    movie_count: int,
    sum_tolerance: float = 0.02,
) -> np.ndarray:
    if not 0.0 <= sum_tolerance < 1.0:
        raise ValueError("probability sum tolerance must be in [0,1)")
    rows = _parse_json_object(text).get("profiles")
    if not isinstance(rows, list) or len(rows) != profile_count:
        raise ValueError("likelihood response has the wrong profile count")
    matrix = np.empty((profile_count, movie_count, len(RATING_IDS)), dtype=float)
    for profile_index, row in enumerate(rows):
        expected_id = f"p{profile_index + 1}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("likelihood profile IDs or order changed")
        ratings = row.get("ratings")
        if not isinstance(ratings, list) or len(ratings) != movie_count:
            raise ValueError("likelihood response has the wrong movie count")
        for movie_index, probabilities in enumerate(ratings):
            if not isinstance(probabilities, list) or len(probabilities) != 5:
                raise ValueError("each rating row must contain five probabilities")
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float))
                for value in probabilities
            ):
                raise ValueError("rating probabilities must be numeric")
            values = np.asarray(probabilities, dtype=float)
            if not np.all(np.isfinite(values)) or np.any(values < 0.0):
                raise ValueError("rating probabilities must be finite and nonnegative")
            total = float(values.sum())
            if (
                total < 1.0 - sum_tolerance - 1e-12
                or total > 1.0 + sum_tolerance + 1e-12
            ):
                raise ValueError("rating probabilities must sum to one")
            matrix[profile_index, movie_index] = values / total
    return matrix


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_movielens(
    data_dir: str | Path,
) -> tuple[dict[int, dict[int, int]], dict[int, dict[str, Any]]]:
    root = Path(data_dir)
    ratings_path = root / "u.data"
    items_path = root / "u.item"
    readme_path = root / "README"
    expected = (
        (ratings_path, RATINGS_SHA256),
        (items_path, ITEMS_SHA256),
        (readme_path, README_SHA256),
    )
    for path, digest in expected:
        observed = _sha256(path)
        if observed != digest:
            raise ValueError(
                f"MovieLens file hash mismatch for {path.name}: "
                f"expected {digest}, got {observed}"
            )

    ratings: dict[int, dict[int, int]] = {}
    row_count = 0
    for line in ratings_path.read_text(encoding="ascii").splitlines():
        user_id, movie_id, rating, _timestamp = map(int, line.split())
        if rating not in RATING_IDS:
            raise ValueError("MovieLens rating is outside 1..5")
        ratings.setdefault(user_id, {})[movie_id] = rating
        row_count += 1
    if row_count != 100_000 or len(ratings) != 943:
        raise ValueError("MovieLens ratings file has unexpected dimensions")

    items: dict[int, dict[str, Any]] = {}
    for line in items_path.read_text(encoding="latin-1").splitlines():
        fields = line.split("|")
        if len(fields) != 24:
            raise ValueError("MovieLens item row has unexpected width")
        movie_id = int(fields[0])
        active_genres = [
            genre for genre, flag in zip(GENRES, fields[5:], strict=True) if flag == "1"
        ]
        items[movie_id] = {
            "movie_id": movie_id,
            "title": fields[1],
            "genres": active_genres,
        }
    if len(items) != 1682:
        raise ValueError("MovieLens item file has unexpected dimensions")

    fixed = INITIAL_MOVIE_IDS + CANDIDATE_MOVIE_IDS
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if all(movie_id in user_ratings for movie_id in fixed)
        and sum(movie_id not in fixed for movie_id in user_ratings) >= HELDOUT_COUNT
    )
    selected = tuple(
        sorted(
            int(value)
            for value in np.random.default_rng(SELECTION_SEED).choice(
                eligible, size=len(FORMAL_USER_IDS), replace=False
            )
        )
    )
    if selected != FORMAL_USER_IDS:
        raise ValueError("frozen MovieLens user selection does not reproduce")
    return ratings, items


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


def _movie_payload(movie: dict[str, Any]) -> dict[str, Any]:
    return {"title": movie["title"], "genres": movie["genres"]}


def _history_payload(
    movie_ids: Sequence[int],
    user_ratings: dict[int, int],
    items: dict[int, dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {
            **_movie_payload(items[movie_id]),
            "rating": user_ratings[movie_id],
        }
        for movie_id in movie_ids
    ]


def profile_messages(
    history: Sequence[dict[str, Any]],
    *,
    prior_profiles: Sequence[str] = (),
) -> list[dict[str, str]]:
    schema = {
        "profiles": [
            {"id": f"p{index + 1}", "description": "..."}
            for index in range(PROFILE_COUNT)
        ]
    }
    payload: dict[str, Any] = {"observed_movie_ratings": list(history)}
    if prior_profiles:
        payload["previous_candidate_profiles"] = list(prior_profiles)
    return [
        {
            "role": "system",
            "content": (
                "Infer diverse latent movie-taste hypotheses from recorded ratings. "
                "Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {PROFILE_COUNT} distinct, complete preference "
                "profiles that could explain all observed ratings. Each profile must "
                "state favored and disliked genres, tones, narrative styles, or eras "
                "and explain conditional tastes. Profiles should represent genuinely "
                "different plausible generalizations, not demographic guesses or "
                "paraphrases. If previous profiles are supplied, revise the space in "
                "light of all ratings and introduce alternatives suggested by the newest "
                "rating. Do not predict unobserved ratings or mention a user ID. Preserve "
                "profile IDs and return "
                + json.dumps(schema, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def rating_likelihood_messages(
    history: Sequence[dict[str, Any]],
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
        "observed_movie_ratings": list(history),
        "candidate_profiles": profile_rows,
        "movies_in_fixed_order": [_movie_payload(movie) for movie in movies],
    }
    return [
        {
            "role": "system",
            "content": (
                "Estimate movie-rating likelihoods under semantic preference "
                "hypotheses. Return calibrated probabilities and strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For every profile and movie, return probabilities for ratings "
                "[1,2,3,4,5] in that order. Each five-number row must sum to 1. Use "
                "the observed history and the stated profile, without inventing hidden "
                "ratings. Preserve profile and movie order. Return "
                + json.dumps(schema, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def entropy(probabilities: np.ndarray) -> float:
    values = np.asarray(probabilities, dtype=float)
    positive = values[values > 0.0]
    return float(-np.sum(positive * np.log(positive)))


def immediate_eig_values(candidate_likelihoods: np.ndarray) -> list[float]:
    if candidate_likelihoods.ndim != 3 or candidate_likelihoods.shape[2] != 5:
        raise ValueError("candidate likelihoods must have shape profiles x movies x 5")
    values = []
    for movie_index in range(candidate_likelihoods.shape[1]):
        conditional = candidate_likelihoods[:, movie_index, :]
        marginal = conditional.mean(axis=0)
        values.append(
            entropy(marginal)
            - float(np.mean([entropy(row) for row in conditional]))
        )
    return values


def predictive_nll(
    likelihoods: np.ndarray,
    observed_ratings: Sequence[int],
) -> float:
    if likelihoods.ndim != 3 or likelihoods.shape[1] != len(observed_ratings):
        raise ValueError("likelihood matrix and ratings do not align")
    mixture = likelihoods.mean(axis=0)
    terms = [
        -math.log(max(float(mixture[index, rating - 1]), 1e-12))
        for index, rating in enumerate(observed_ratings)
    ]
    return float(np.mean(terms))


def merge_branch_profiles(
    generated: Sequence[str],
    previous: Sequence[str],
    previous_rating_probabilities: Sequence[float],
) -> list[str]:
    if len(previous) != len(previous_rating_probabilities):
        raise ValueError("previous profiles and probabilities do not align")
    retained_indices = sorted(
        range(len(previous)),
        key=lambda index: float(previous_rating_probabilities[index]),
        reverse=True,
    )[:RETAINED_PROFILE_COUNT]
    merged: list[str] = []
    seen: set[str] = set()
    for description in [
        *generated,
        *(previous[index] for index in retained_indices),
    ]:
        clean = _clean_text(description)
        key = clean.casefold()
        if clean and key not in seen:
            merged.append(clean)
            seen.add(key)
    if len(merged) < PROFILE_COUNT:
        raise ValueError("branch support lost generated profile capacity")
    return merged


def _average_ranks(values: Sequence[float]) -> list[float]:
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        rank = (start + 1 + end) / 2.0
        for index in ordered[start:end]:
            ranks[index] = rank
        start = end
    return ranks


def _correlation(left: Sequence[float], right: Sequence[float]) -> float | None:
    if len(left) != len(right) or len(left) < 2:
        return None
    x = np.asarray(left, dtype=float)
    y = np.asarray(right, dtype=float)
    x -= x.mean()
    y -= y.mean()
    denominator = float(np.linalg.norm(x) * np.linalg.norm(y))
    if denominator == 0.0:
        return None
    return float(np.dot(x, y) / denominator)


def summarize(
    records: Sequence[dict[str, Any]],
    usage: dict[str, Any],
    *,
    stage: str,
) -> dict[str, Any]:
    expected_users = 2 if stage == "serving_smoke" else 12
    expected_branches = 2 if stage == "serving_smoke" else 48
    expected_requests = (
        SMOKE_EXPECTED_REQUESTS
        if stage == "serving_smoke"
        else FORMAL_EXPECTED_REQUESTS
    )
    branch_count = sum(len(record["branches"]) for record in records)
    base_gates = {
        "all_users_completed": len(records) == expected_users,
        "all_branches_completed": branch_count == expected_branches,
        "exact_physical_request_count": (
            int(usage["physical_requests"]) == expected_requests
        ),
        "zero_reasoning_tokens": int(usage["reasoning_tokens"]) == 0,
    }
    if stage == "serving_smoke":
        gates = {**base_gates, "all_replays_completed": all(
            len(record.get("replay_profiles", [])) == PROFILE_COUNT
            for record in records
        )}
        gates["all_pass"] = all(gates.values())
        return {
            "num_users": len(records),
            "num_branches": branch_count,
            "gates": gates,
        }

    oracle_improvements: list[float] = []
    spreads: list[float] = []
    eig_regrets: list[float] = []
    eig_values: list[float] = []
    negative_nlls: list[float] = []
    max_eigs: list[float] = []
    for record in records:
        branch_nlls = [float(branch["heldout_nll"]) for branch in record["branches"]]
        initial_nll = float(record["initial_heldout_nll"])
        oracle_nll = min(branch_nlls)
        selected_index = int(record["immediate_eig_selected_branch"])
        record_eigs = [
            float(value) for value in record["immediate_eig_values"]
        ]
        oracle_improvements.append(initial_nll - oracle_nll)
        spreads.append(max(branch_nlls) - min(branch_nlls))
        eig_regrets.append(branch_nlls[selected_index] - oracle_nll)
        eig_values.extend(record_eigs)
        max_eigs.append(max(record_eigs))
        negative_nlls.extend(-value for value in branch_nlls)

    summary = {
        "num_users": len(records),
        "num_branches": branch_count,
        "mean_oracle_heldout_nll_improvement": float(np.mean(oracle_improvements)),
        "users_with_oracle_improvement_at_least_0_05": sum(
            value >= 0.05 for value in oracle_improvements
        ),
        "mean_within_user_branch_nll_spread": float(np.mean(spreads)),
        "users_with_branch_nll_spread_at_least_0_10": sum(
            value >= 0.10 for value in spreads
        ),
        "mean_immediate_eig_heldout_nll_regret": float(np.mean(eig_regrets)),
        "users_with_immediate_eig_regret_at_least_0_05": sum(
            value >= 0.05 for value in eig_regrets
        ),
        "mean_max_immediate_eig": float(np.mean(max_eigs)),
        "users_with_max_immediate_eig_at_least_0_02": sum(
            value >= 0.02 for value in max_eigs
        ),
        "spearman_immediate_eig_vs_negative_branch_nll": _correlation(
            _average_ranks(eig_values),
            _average_ranks(negative_nlls),
        ),
    }
    gates = {
        **base_gates,
        "mean_oracle_improvement_at_least_0_05": (
            summary["mean_oracle_heldout_nll_improvement"] >= 0.05
        ),
        "at_least_six_users_improve_by_0_05": (
            summary["users_with_oracle_improvement_at_least_0_05"] >= 6
        ),
        "at_least_six_users_have_branch_spread": (
            summary["users_with_branch_nll_spread_at_least_0_10"] >= 6
        ),
        "mean_immediate_eig_regret_at_least_0_03": (
            summary["mean_immediate_eig_heldout_nll_regret"] >= 0.03
        ),
        "at_least_four_users_have_eig_regret": (
            summary["users_with_immediate_eig_regret_at_least_0_05"] >= 4
        ),
        "semantic_profiles_affect_rating_likelihoods": (
            summary["mean_max_immediate_eig"] >= 0.02
            and summary["users_with_max_immediate_eig_at_least_0_02"] >= 8
        ),
    }
    gates["all_pass"] = all(gates.values())
    return {**summary, "gates": gates}


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


def _usage(generator: Any, likelihood: Any) -> dict[str, Any]:
    snapshots = {
        "generator": generator.usage_snapshot(),
        "likelihood": likelihood.usage_snapshot(),
    }
    return {
        "physical_requests": sum(
            int(snapshot["adapter_requests"]) for snapshot in snapshots.values()
        ),
        "reasoning_tokens": sum(
            int(snapshot["adapter_reasoning_tokens"])
            for snapshot in snapshots.values()
        ),
        "adapter_cost_usd": sum(
            float(snapshot["adapter_cost_usd"]) for snapshot in snapshots.values()
        ),
        "by_role": snapshots,
    }


def _write_raw_checkpoint(
    path: Path | None,
    *,
    stage: str,
    user_ids: Sequence[int],
    raw: dict[str, Any],
) -> None:
    if path is None:
        return
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "protocol_stage": stage,
                "user_ids": list(user_ids),
                "responses": raw,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
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
            rating_likelihood_messages(
                history,
                profiles,
                [items[movie_id] for movie_id in query_ids],
            )
            for history, profiles, query_ids in zip(
                histories, initial_profiles, initial_query_ids, strict=True
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
        profile_messages(
            history,
            prior_profiles=initial_profiles[user_index],
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
    generated_profiles = [parse_profiles(response) for response in refresh_raw]

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
            rating_likelihood_messages(
                history,
                profiles,
                [items[movie_id] for movie_id in heldout_ids_many[user_index]],
            )
            for history, profiles, user_index in zip(
                refresh_histories,
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

    replay_profiles: list[list[str]] = []
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
        replay_profiles = [parse_profiles(response) for response in replay_raw]

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
                    "generated_profiles": generated_profiles[flat_index],
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
            record["replay_profiles"] = replay_profiles[user_index]
        records.append(record)
        offset += len(branch_movie_ids)

    usage = _usage(generator, likelihood)
    summary = summarize(records, usage, stage=stage)
    return {
        "schema_version": 1,
        "status": "passed" if summary["gates"]["all_pass"] else "gate_failed",
        "protocol": {
            "stage": stage,
            "selection_seed": SELECTION_SEED,
            "ratings_sha256": RATINGS_SHA256,
            "items_sha256": ITEMS_SHA256,
            "readme_sha256": README_SHA256,
            "user_ids": list(user_ids),
            "initial_movie_ids": list(INITIAL_MOVIE_IDS),
            "candidate_movie_ids": list(CANDIDATE_MOVIE_IDS),
            "heldout_count": HELDOUT_COUNT,
            "profile_count": PROFILE_COUNT,
            "retained_profile_count": RETAINED_PROFILE_COUNT,
            "likelihood_model": likelihood_model,
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
            "schema_version": 1,
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
