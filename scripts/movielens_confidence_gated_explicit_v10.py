#!/usr/bin/env python3
"""Fresh confirmation of confidence-gated explicit belief rollouts."""

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
import scripts.movielens_adaptive_candidate_gate_v4 as base
import scripts.movielens_depth2_semantic_policy_v8 as v8
import scripts.movielens_explicit_rollout_ranking_v7 as v7
import scripts.movielens_receding_explicit_policy_v9 as v9
from scripts.movielens_profile_dynamics_gate import (
    HELDOUT_COUNT,
    _average_ranks,
    _correlation,
    load_movielens,
)


SCHEMA_VERSION = 10
SELECTION_SEED = 24326
INITIAL_HISTORY_MOVIE_IDS = (258, 294, 286, 288)
SMOKE_USER_IDS = (750, 782)
FORMAL_SCREEN_USER_IDS = (
    200, 755, 66, 587, 100, 743, 787, 102, 193, 451, 646,
    875, 489, 724, 529, 853, 223, 683, 116, 589, 515, 752,
    324, 893, 673, 126, 159, 783, 721, 24, 404, 863, 626,
    733, 197, 181, 544, 464, 668, 460, 802, 510, 827, 392,
)
ALL_SELECTED_USER_IDS = SMOKE_USER_IDS + FORMAL_SCREEN_USER_IDS
ENROLLMENT_COUNT = 16
CONFIDENCE_MARGIN = 0.02
FORMAL_EXPECTED_REQUESTS = 856
SMOKE_EXPECTED_REQUESTS = 12
BOOTSTRAP_DRAWS = 10000
PRIOR_USER_IDS = frozenset(
    v7._prior_users()
    | set(v7.ALL_SELECTED_USER_IDS)
    | set(v8.ALL_SELECTED_USER_IDS)
    | set(v9.ALL_SELECTED_USER_IDS)
)


def _prior_users() -> set[int]:
    return set(PRIOR_USER_IDS)


def selected_user_ids(
    ratings: dict[int, dict[int, int]],
) -> tuple[int, ...]:
    eligible = sorted(
        user_id
        for user_id, user_ratings in ratings.items()
        if user_id not in _prior_users()
        and all(
            movie_id in user_ratings
            for movie_id in INITIAL_HISTORY_MOVIE_IDS
        )
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
        raise ValueError("frozen V10 MovieLens user selection does not reproduce")
    return chosen


def _bootstrap_ci(values: list[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    draws = np.random.default_rng(SELECTION_SEED + 100).choice(
        array,
        size=(BOOTSTRAP_DRAWS, len(array)),
        replace=True,
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.05, 0.95])
    return float(low), float(high)


def confidence_summary(result: dict[str, Any]) -> dict[str, Any]:
    improvements: list[float] = []
    active_improvements: list[float] = []
    gated_regrets: list[float] = []
    immediate_regrets: list[float] = []
    random_regrets: list[float] = []
    scores: list[float] = []
    negative_nlls: list[float] = []
    active = 0
    wins = 0
    losses = 0
    per_user = []
    for record in result["records"]:
        nlls = [float(row["heldout_nll"]) for row in record["branches"]]
        rollout_scores = [
            float(value) for value in record["semantic_lookahead_scores"]
        ]
        explicit_index = int(np.argmax(rollout_scores))
        margin = rollout_scores[explicit_index] - rollout_scores[0]
        use_explicit = (
            explicit_index != 0 and margin >= CONFIDENCE_MARGIN
        )
        selected_index = explicit_index if use_explicit else 0
        improvement = nlls[0] - nlls[selected_index]
        oracle = min(nlls)
        random_index = int(
            np.random.default_rng(
                SELECTION_SEED * 1000 + int(record["user_id"])
            ).integers(len(nlls))
        )
        improvements.append(improvement)
        gated_regrets.append(nlls[selected_index] - oracle)
        immediate_regrets.append(nlls[0] - oracle)
        random_regrets.append(nlls[random_index] - oracle)
        if use_explicit:
            active += 1
            active_improvements.append(improvement)
        wins += improvement > 1.0e-12
        losses += improvement < -1.0e-12
        scores.extend(rollout_scores)
        negative_nlls.extend(-value for value in nlls)
        per_user.append(
            {
                "user_id": record["user_id"],
                "explicit_index": explicit_index,
                "selected_index": selected_index,
                "confidence_margin": margin,
                "active": use_explicit,
                "nll_improvement_vs_immediate": improvement,
            }
        )
    ci = _bootstrap_ci(improvements)
    mean_improvement = float(np.mean(improvements))
    mean_active = (
        float(np.mean(active_improvements))
        if active_improvements
        else 0.0
    )
    correlation = _correlation(
        _average_ranks(scores),
        _average_ranks(negative_nlls),
    )
    gates = {
        "exact_physical_request_count": int(
            result["usage"]["physical_requests"]
        )
        == FORMAL_EXPECTED_REQUESTS,
        "zero_reasoning_tokens": int(result["usage"]["reasoning_tokens"]) == 0,
        "prospective_enrollment_complete": len(result["records"])
        == ENROLLMENT_COUNT,
        "confidence_gate_active_on_at_least_five": active >= 5,
        "mean_nll_improvement_at_least_0_015": mean_improvement >= 0.015,
        "paired_bootstrap_90_lower_positive": ci[0] > 0.0,
        "wins_exceed_losses_by_two": wins >= losses + 2,
        "active_mean_improvement_at_least_0_03": mean_active >= 0.03,
        "gated_regret_no_worse_than_random": float(
            np.mean(gated_regrets)
        )
        <= float(np.mean(random_regrets)),
    }
    gates["all_pass"] = all(gates.values())
    return {
        "num_users": len(result["records"]),
        "num_branches": len(result["records"])
        * base.SELECTED_CANDIDATE_COUNT,
        "confidence_margin_threshold": CONFIDENCE_MARGIN,
        "active_user_count": active,
        "wins_ties_losses_vs_immediate": [
            wins,
            len(improvements) - wins - losses,
            losses,
        ],
        "mean_nll_improvement_vs_immediate": mean_improvement,
        "active_mean_nll_improvement_vs_immediate": mean_active,
        "nll_improvement_bootstrap_90_ci": list(ci),
        "mean_gated_top1_regret": float(np.mean(gated_regrets)),
        "mean_immediate_top1_regret": float(np.mean(immediate_regrets)),
        "mean_seeded_random_top1_regret": float(np.mean(random_regrets)),
        "rollout_score_spearman_vs_negative_branch_nll": correlation,
        "per_user": per_user,
        "gates": gates,
    }


def _base_overrides() -> dict[str, Any]:
    return {
        "SELECTION_SEED": SELECTION_SEED,
        "SMOKE_USER_IDS": SMOKE_USER_IDS,
        "FORMAL_SCREEN_USER_IDS": FORMAL_SCREEN_USER_IDS,
        "PROSPECTIVE_ENROLLMENT_COUNT": ENROLLMENT_COUNT,
        "SENSITIVITY_EIG_THRESHOLD": 0.02,
        "ALL_SELECTED_USER_IDS": ALL_SELECTED_USER_IDS,
        "SEMANTIC_RANKING_MESSAGES": None,
        "SEMANTIC_RANKING_PARSE": None,
        "EXPLICIT_ROLLOUT_SCORER": v7.explicit_rollout_scorer,
        "EXPLICIT_ROLLOUT_REQUESTS_PER_USER": (
            v7.EXPLICIT_ROLLOUT_REQUESTS_PER_USER
        ),
        "INITIAL_HISTORY_MOVIE_IDS": INITIAL_HISTORY_MOVIE_IDS,
        "selected_user_ids": selected_user_ids,
        "_build_models": v7._build_models,
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
    result["schema_version"] = SCHEMA_VERSION
    result["protocol"].update(
        {
            "selection_seed": SELECTION_SEED,
            "confidence_gated_explicit_rollout": True,
            "confidence_margin_threshold": CONFIDENCE_MARGIN,
            "initial_movie_ids": list(INITIAL_HISTORY_MOVIE_IDS),
            "v1_through_v9_users_excluded": True,
            "source_outcomes_hidden_until_tree_frozen": True,
        }
    )
    if result["records"] and "branches" in result["records"][0]:
        result["summary"] = confidence_summary(result)
        result["status"] = (
            "passed"
            if result["summary"]["gates"]["all_pass"]
            else "gate_failed"
        )
    return result


def run_smoke(
    config: Config,
    *,
    data_dir: str | Path,
    likelihood_model: str,
    raw_checkpoint_path: Path | None = None,
) -> dict[str, Any]:
    ratings, _items = load_movielens(data_dir)
    selected_user_ids(ratings)
    overrides = {
        "SELECTION_SEED": SELECTION_SEED,
        "SMOKE_USER_IDS": SMOKE_USER_IDS,
        "FORMAL_SCREEN_USER_IDS": FORMAL_SCREEN_USER_IDS,
        "ALL_SELECTED_USER_IDS": ALL_SELECTED_USER_IDS,
        "INITIAL_HISTORY_MOVIE_IDS": INITIAL_HISTORY_MOVIE_IDS,
        "selected_user_ids": selected_user_ids,
    }
    previous = {name: getattr(v7, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(v7, name, value)
        result = v7.run_smoke(
            config,
            data_dir=data_dir,
            likelihood_model=likelihood_model,
            raw_checkpoint_path=raw_checkpoint_path,
        )
    finally:
        for name, value in previous.items():
            setattr(v7, name, value)
    result["schema_version"] = SCHEMA_VERSION
    result["protocol"].update(
        {
            "selection_seed": SELECTION_SEED,
            "v1_through_v9_users_excluded": True,
            "confidence_gate_not_scored_in_smoke": True,
        }
    )
    if int(result["usage"]["physical_requests"]) != SMOKE_EXPECTED_REQUESTS:
        raise ValueError("V10 smoke request count changed")
    return result


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
        required=True,
    )
    args = parser.parse_args()

    config = load_config(str(args.config))
    config.run_id = args.run_id
    if args.stage == "serving_smoke":
        config.openrouter_projected_cost_usd = 0.20
        config.openrouter_run_budget_usd = 0.75
    else:
        config.openrouter_projected_cost_usd = 3.60
        config.openrouter_run_budget_usd = 7.00
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    output_name = (
        "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    )
    failure_name = (
        "SERVING_SMOKE_FAILURE.json"
        if args.stage == "serving_smoke"
        else "GATE_FAILURE.json"
    )
    try:
        payload = (
            run_smoke(
                config,
                data_dir=args.data_dir,
                likelihood_model=args.likelihood_model,
                raw_checkpoint_path=raw_path,
            )
            if args.stage == "serving_smoke"
            else run_formal(
                config,
                data_dir=args.data_dir,
                likelihood_model=args.likelihood_model,
                raw_checkpoint_path=raw_path,
            )
        )
        payload["protocol"]["private_raw_sha256"] = hashlib.sha256(
            raw_path.read_bytes()
        ).hexdigest()
    except Exception as exc:
        failure = {
            "schema_version": SCHEMA_VERSION,
            "status": "failed_closed",
            "stage": args.stage,
            "error": f"{type(exc).__name__}: {exc}",
            "raw_responses_path": str(raw_path),
        }
        if raw_path.exists():
            failure["private_raw_sha256"] = hashlib.sha256(
                raw_path.read_bytes()
            ).hexdigest()
        (args.output_dir / failure_name).write_text(
            json.dumps(failure, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        raise
    (args.output_dir / output_name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {"status": payload["status"], **payload["summary"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
