#!/usr/bin/env python3
"""Prospectively enroll high-uncertainty MovieLens semantic beliefs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from helpers import Config, load_config
from scripts.movielens_profile_dynamics_gate import INITIAL_MOVIE_IDS
from scripts.movielens_profile_dynamics_gate import FORMAL_USER_IDS as V1_USERS
from scripts.movielens_profile_dynamics_gate_v2 import (
    ALL_SELECTED_USER_IDS as V2_USERS,
)
from scripts.movielens_profile_dynamics_gate_v3 import (
    ALL_SELECTED_USER_IDS as V3_USERS,
)
import scripts.movielens_adaptive_candidate_gate_v4 as base
from scripts.movielens_adaptive_candidate_gate_v4 import (
    ALL_SELECTED_USER_IDS as V4_USERS,
)


SELECTION_SEED = 24306
SMOKE_USER_IDS = (123, 781)
FORMAL_SCREEN_USER_IDS = (
    176, 318, 296, 864, 459, 145, 548, 21, 288, 907, 401, 629,
    236, 825, 764, 680, 6, 43, 653, 665, 758, 104, 406, 232,
    634, 796, 881, 7, 834, 28, 936, 666, 430, 82, 328, 452,
    119, 280, 608, 347, 299, 717, 10, 625, 789, 474, 940, 271,
)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_SCREEN_USER_IDS
ENROLLMENT_COUNT = 12


def selected_user_ids(ratings: dict[int, dict[int, int]]) -> tuple[int, ...]:
    excluded = set(V1_USERS) | set(V2_USERS) | set(V3_USERS) | set(V4_USERS)
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if user_id not in excluded
        and all(movie_id in user_ratings for movie_id in INITIAL_MOVIE_IDS)
        and sum(movie_id not in INITIAL_MOVIE_IDS for movie_id in user_ratings)
        >= base.CANDIDATE_POOL_SIZE + 8
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
        raise ValueError("frozen v5 MovieLens user selection does not reproduce")
    return chosen


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
        "FORMAL_SCREEN_USER_IDS": FORMAL_SCREEN_USER_IDS,
        "PROSPECTIVE_ENROLLMENT_COUNT": ENROLLMENT_COUNT,
        "ALL_SELECTED_USER_IDS": ALL_SELECTED_USER_IDS,
        "selected_user_ids": selected_user_ids,
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
    payload["schema_version"] = 5
    payload["protocol"].update(
        {
            "selection_seed": SELECTION_SEED,
            "prospective_uncertainty_enrichment": True,
            "enrollment_threshold_max_eig": 0.02,
            "prospective_enrollment_count": ENROLLMENT_COUNT,
            "v1_v2_v3_v4_users_excluded": True,
        }
    )
    payload["protocol"].pop("v1_v2_v3_users_excluded", None)
    return payload


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
            "schema_version": 5,
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
