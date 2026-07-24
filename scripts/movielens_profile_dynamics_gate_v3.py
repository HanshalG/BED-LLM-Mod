#!/usr/bin/env python3
"""Test candidate-contrastive semantic profile updates on MovieLens ratings."""

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
    CANDIDATE_MOVIE_IDS,
    FORMAL_USER_IDS as V1_FORMAL_USER_IDS,
    HELDOUT_COUNT,
    INITIAL_MOVIE_IDS,
    PROFILE_COUNT,
    _parse_json_object,
)
import scripts.movielens_profile_dynamics_gate_v2 as base
from scripts.movielens_profile_dynamics_gate_v2 import (
    ALL_SELECTED_USER_IDS as V2_SELECTED_USER_IDS,
)


SELECTION_SEED = 24304
SMOKE_USER_IDS = (26, 63)
FORMAL_USER_IDS = (144, 178, 268, 293, 303, 345, 417, 425, 486, 487, 624, 663)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_USER_IDS
REACTIONS = ("appeal", "avoid", "uncertain")
DESIGN_MOVIES = (
    {
        "id": "q1",
        "title": "Contact (1997)",
        "genres": ["Drama", "Sci-Fi"],
    },
    {
        "id": "q2",
        "title": "Return of the Jedi (1983)",
        "genres": ["Action", "Adventure", "Romance", "Sci-Fi", "War"],
    },
    {
        "id": "q3",
        "title": "Scream (1996)",
        "genres": ["Horror", "Thriller"],
    },
    {
        "id": "q4",
        "title": "Toy Story (1995)",
        "genres": ["Animation", "Children's", "Comedy"],
    },
)


def _clean_text(value: str) -> str:
    return " ".join(value.strip().split())


def selected_user_ids(ratings: dict[int, dict[int, int]]) -> tuple[int, ...]:
    fixed = INITIAL_MOVIE_IDS + CANDIDATE_MOVIE_IDS
    excluded = set(V1_FORMAL_USER_IDS) | set(V2_SELECTED_USER_IDS)
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if user_id not in excluded
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
        raise ValueError("frozen v3 MovieLens user selection does not reproduce")
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


def _profile_schema(*, refreshed: bool) -> dict[str, Any]:
    rows = []
    for index in range(PROFILE_COUNT):
        row: dict[str, Any] = {
            "id": f"p{index + 1}",
            "description": "...",
            "candidate_contrasts": [
                {
                    "movie_id": movie["id"],
                    "reaction": "appeal|avoid|uncertain",
                    "rationale": "...",
                }
                for movie in DESIGN_MOVIES
            ],
        }
        if refreshed:
            row["new_evidence_effect"] = "..."
        rows.append(row)
    return {"profiles": rows}


def profile_messages(
    history: Sequence[dict[str, Any]],
) -> list[dict[str, str]]:
    payload = {
        "observed_movie_ratings": list(history),
        "fixed_design_movies_without_ratings": list(DESIGN_MOVIES),
    }
    return [
        {
            "role": "system",
            "content": (
                "Infer contrastive latent movie-taste hypotheses for experimental "
                "design. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {PROFILE_COUNT} complete and mutually distinct "
                "preference hypotheses consistent with the observed ratings. Broad "
                "paraphrases are not useful. Use the four unrated design movies to "
                "expose unresolved taste distinctions: every profile must give one "
                "qualitative reaction (appeal, avoid, or uncertain) and a rationale "
                "for every movie. Across the six profiles, every design movie must "
                "receive at least two distinct reaction labels. Within each profile, "
                "use at least two distinct labels. Do not invent or assign numeric "
                "ratings, mention a user ID, or assume any design-movie outcome. "
                "Preserve all IDs and order. Return "
                + json.dumps(_profile_schema(refreshed=False), separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def refreshed_profile_messages(
    history: Sequence[dict[str, Any]],
    previous_profiles: Sequence[str],
) -> list[dict[str, str]]:
    payload = {
        "observed_movie_ratings": list(history),
        "previous_candidate_profiles_for_context_only": list(previous_profiles),
        "fixed_design_movies": list(DESIGN_MOVIES),
    }
    return [
        {
            "role": "system",
            "content": (
                "Rebuild contrastive latent movie-taste hypotheses after new recorded "
                "evidence. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Generate exactly {PROFILE_COUNT} replacement preference hypotheses "
                "from all observed ratings. Previous profiles are context only: do not "
                "copy a previous description or candidate-contrast set verbatim. Every "
                "profile must explain how the newest rating revises it, then give one "
                "qualitative reaction (appeal, avoid, or uncertain) and rationale for "
                "every fixed design movie. Across the six profiles, every movie must "
                "receive at least two distinct reaction labels; within each profile, "
                "use at least two labels. Keep alternatives plausible under all "
                "evidence. Do not invent numeric ratings for unobserved movies, mention "
                "a user ID, or omit IDs. Preserve all IDs and order. Return "
                + json.dumps(_profile_schema(refreshed=True), separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def _parse_contrastive_rows(
    text: str,
    *,
    refreshed: bool,
) -> list[dict[str, str]]:
    rows = _parse_json_object(text).get("profiles")
    if not isinstance(rows, list) or len(rows) != PROFILE_COUNT:
        raise ValueError(f"profiles must contain exactly {PROFILE_COUNT} rows")
    parsed: list[dict[str, str]] = []
    description_keys: set[str] = set()
    reactions_by_movie = {movie["id"]: set() for movie in DESIGN_MOVIES}
    for index, row in enumerate(rows):
        expected_id = f"p{index + 1}"
        if not isinstance(row, dict) or row.get("id") != expected_id:
            raise ValueError("profile IDs or order changed")
        description = row.get("description")
        if not isinstance(description, str) or not _clean_text(description):
            raise ValueError("profile description must be nonempty")
        description = _clean_text(description)
        contrasts = row.get("candidate_contrasts")
        if not isinstance(contrasts, list) or len(contrasts) != len(DESIGN_MOVIES):
            raise ValueError("candidate_contrasts must cover all design movies")
        reaction_rows = []
        profile_reactions: set[str] = set()
        for movie, contrast in zip(DESIGN_MOVIES, contrasts, strict=True):
            if not isinstance(contrast, dict) or contrast.get("movie_id") != movie["id"]:
                raise ValueError("candidate contrast IDs or order changed")
            reaction = contrast.get("reaction")
            rationale = contrast.get("rationale")
            if reaction not in REACTIONS:
                raise ValueError("candidate reaction is invalid")
            if not isinstance(rationale, str) or not _clean_text(rationale):
                raise ValueError("candidate rationale must be nonempty")
            profile_reactions.add(reaction)
            reactions_by_movie[movie["id"]].add(reaction)
            reaction_rows.append(
                f"{movie['title']}: {reaction} because {_clean_text(rationale)}"
            )
        if len(profile_reactions) < 2:
            raise ValueError("each profile must contrast at least two reactions")
        combined = description + " Candidate implications: " + "; ".join(reaction_rows)
        key = combined.casefold()
        if key in description_keys:
            raise ValueError("profile hypotheses must be unique")
        description_keys.add(key)
        parsed_row = {"description": combined}
        if refreshed:
            effect = row.get("new_evidence_effect")
            if not isinstance(effect, str) or not _clean_text(effect):
                raise ValueError("new_evidence_effect must be nonempty")
            parsed_row["new_evidence_effect"] = _clean_text(effect)
        parsed.append(parsed_row)
    if any(len(labels) < 2 for labels in reactions_by_movie.values()):
        raise ValueError("every design movie must vary across profile reactions")
    return parsed


def parse_profiles(text: str) -> list[str]:
    return [
        row["description"]
        for row in _parse_contrastive_rows(text, refreshed=False)
    ]


def parse_refreshed_profiles(
    text: str,
    previous_profiles: Sequence[str],
) -> list[dict[str, str]]:
    rows = _parse_contrastive_rows(text, refreshed=True)
    previous_keys = {_clean_text(value).casefold() for value in previous_profiles}
    if any(row["description"].casefold() in previous_keys for row in rows):
        raise ValueError("refreshed profile copied a previous hypothesis")
    return rows


_build_models = base._build_models


def run_gate(
    config: Config,
    *,
    data_dir: str | Path,
    likelihood_model: str,
    stage: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    overrides = {
        "SELECTION_SEED": SELECTION_SEED,
        "SMOKE_USER_IDS": SMOKE_USER_IDS,
        "FORMAL_USER_IDS": FORMAL_USER_IDS,
        "ALL_SELECTED_USER_IDS": ALL_SELECTED_USER_IDS,
        "selected_user_ids": selected_user_ids,
        "heldout_movie_ids": heldout_movie_ids,
        "profile_messages": profile_messages,
        "refreshed_profile_messages": refreshed_profile_messages,
        "parse_profiles": parse_profiles,
        "parse_refreshed_profiles": parse_refreshed_profiles,
        "_build_models": _build_models,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        payload = base.run_gate(
            config,
            data_dir=data_dir,
            likelihood_model=likelihood_model,
            stage=stage,
            raw_checkpoint_path=raw_checkpoint_path,
        )
    finally:
        for name, value in previous.items():
            setattr(base, name, value)
    payload["schema_version"] = 3
    payload["protocol"].update(
        {
            "selection_seed": SELECTION_SEED,
            "candidate_contrastive_profiles": True,
            "candidate_reaction_labels": list(REACTIONS),
            "v1_and_v2_users_excluded": True,
        }
    )
    payload["protocol"].pop("v1_users_excluded", None)
    return payload


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
            "schema_version": 3,
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
