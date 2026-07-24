#!/usr/bin/env python3
"""Rank belief-regenerating MovieLens queries with semantic lookahead."""

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

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import INITIAL_MOVIE_IDS
from scripts.movielens_profile_dynamics_gate import (
    _average_ranks,
    _correlation,
    _movie_payload,
    _parse_json_object,
)
import scripts.movielens_adaptive_candidate_gate_v4 as base


SELECTION_SEED = 24307
SMOKE_USER_IDS = (123, 781)
FORMAL_SCREEN_USER_IDS = (
    391, 540, 2, 500, 329, 823, 456, 546, 424, 379, 297, 554, 637, 937, 569,
)
ENROLLMENT_COUNT = 4


def selected_user_ids(ratings):
    for user_id in FORMAL_SCREEN_USER_IDS:
        user_ratings = ratings[user_id]
        if not all(movie_id in user_ratings for movie_id in INITIAL_MOVIE_IDS):
            raise ValueError("frozen v6 user lacks initial history")
        if sum(movie_id not in INITIAL_MOVIE_IDS for movie_id in user_ratings) < 24:
            raise ValueError("frozen v6 user lacks candidate capacity")
    return tuple(sorted(FORMAL_SCREEN_USER_IDS))


def ranking_messages(
    profiles: Sequence[str],
    candidates: Sequence[dict[str, Any]],
    heldout_movies: Sequence[dict[str, Any]],
    likelihoods: np.ndarray,
) -> list[dict[str, str]]:
    schema = {
        "candidates": [
            {"id": f"q{i + 1}", "score": 0.5, "rationale": "..."}
            for i in range(4)
        ]
    }
    payload = {
        "semantic_profiles": list(profiles),
        "candidate_movies": [
            {"id": f"q{i + 1}", **_movie_payload(movie)}
            for i, movie in enumerate(candidates)
        ],
        "rating_probabilities_by_profile_candidate": likelihoods.tolist(),
        "downstream_movie_metadata": [_movie_payload(movie) for movie in heldout_movies],
    }
    return [
        {
            "role": "system",
            "content": (
                "Score experimental queries by expected downstream semantic-belief "
                "revision. Return strict JSON only."
            ),
        },
        {
            "role": "user",
            "content": (
                "For each candidate, score 0 to 1 how useful observing its rating will "
                "be for inducing a regenerated profile support that improves predictions "
                "on the downstream movies. Value revealing distinctions that reorganize "
                "or correct the semantic hypotheses, not merely immediate response "
                "entropy. Do not assume an outcome. Preserve order and return "
                + json.dumps(schema, separators=(",", ":"))
                + ".\n"
                + json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
            ),
        },
    ]


def parse_ranking(text: str) -> list[float]:
    rows = _parse_json_object(text).get("candidates")
    if not isinstance(rows, list) or len(rows) != 4:
        raise ValueError("ranking must contain four candidates")
    scores = []
    for i, row in enumerate(rows):
        if not isinstance(row, dict) or row.get("id") != f"q{i + 1}":
            raise ValueError("ranking IDs or order changed")
        score = row.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)):
            raise ValueError("ranking score must be numeric")
        if not 0.0 <= float(score) <= 1.0:
            raise ValueError("ranking score must be in [0,1]")
        if not isinstance(row.get("rationale"), str) or not row["rationale"].strip():
            raise ValueError("ranking rationale must be nonempty")
        scores.append(float(score))
    return scores


_build_models = base._build_models


class _ReplayFirstBatch:
    def __init__(self, inner, responses, prefix_usage):
        self.inner = inner
        self.responses = list(responses)
        self.prefix_usage = dict(prefix_usage)

    def chat_complete_messages_batched(self, batch_messages, **kwargs):
        if self.responses:
            if len(batch_messages) != len(self.responses):
                raise ValueError("resume batch does not match frozen responses")
            responses, self.responses = self.responses, []
            return responses
        return self.inner.chat_complete_messages_batched(batch_messages, **kwargs)

    def usage_snapshot(self):
        result = self.inner.usage_snapshot()
        for key, value in self.prefix_usage.items():
            result[key] = result.get(key, 0) + value
        return result


def run_gate(config, *, data_dir, likelihood_model, stage, raw_checkpoint_path=None):
    overrides = {
        "SELECTION_SEED": SELECTION_SEED,
        "SMOKE_USER_IDS": SMOKE_USER_IDS,
        "FORMAL_SCREEN_USER_IDS": FORMAL_SCREEN_USER_IDS,
        "PROSPECTIVE_ENROLLMENT_COUNT": ENROLLMENT_COUNT,
        "SEMANTIC_RANKING_MESSAGES": ranking_messages,
        "SEMANTIC_RANKING_PARSE": parse_ranking,
        "selected_user_ids": selected_user_ids,
        "_build_models": _build_models,
    }
    previous = {name: getattr(base, name) for name in overrides}
    try:
        for name, value in overrides.items():
            setattr(base, name, value)
        result = base.run_gate(
            config,
            data_dir=data_dir,
            likelihood_model=likelihood_model,
            stage=stage,
            raw_checkpoint_path=raw_checkpoint_path,
        )
    finally:
        for name, value in previous.items():
            setattr(base, name, value)
    result["schema_version"] = 6
    result["protocol"]["semantic_ranking_gate"] = True
    if stage != "formal" or not result["records"] or "branches" not in result["records"][0]:
        return result

    scores, negative_nlls = [], []
    semantic_regrets, immediate_regrets, random_regrets = [], [], []
    semantic_beats = 0
    for record in result["records"]:
        nlls = [float(row["heldout_nll"]) for row in record["branches"]]
        oracle = min(nlls)
        semantic = int(record["semantic_lookahead_selected_branch"])
        immediate = int(record["immediate_eig_selected_branch"])
        random_index = int(
            np.random.default_rng(SELECTION_SEED * 1000 + record["user_id"]).integers(4)
        )
        semantic_regrets.append(nlls[semantic] - oracle)
        immediate_regrets.append(nlls[immediate] - oracle)
        random_regrets.append(nlls[random_index] - oracle)
        semantic_beats += nlls[semantic] < nlls[immediate]
        scores.extend(record["semantic_lookahead_scores"])
        negative_nlls.extend(-value for value in nlls)
    correlation = _correlation(_average_ranks(scores), _average_ranks(negative_nlls))
    mean_semantic = float(np.mean(semantic_regrets))
    mean_immediate = float(np.mean(immediate_regrets))
    mean_random = float(np.mean(random_regrets))
    gates = {
        "exact_physical_request_count": result["usage"]["physical_requests"] == 66,
        "zero_reasoning_tokens": result["usage"]["reasoning_tokens"] == 0,
        "prospective_enrollment_complete": len(result["records"]) == 4,
        "semantic_spearman_at_least_0_25": correlation is not None and correlation >= 0.25,
        "semantic_regret_improves_by_0_02": mean_immediate - mean_semantic >= 0.02,
        "semantic_beats_immediate_on_two": semantic_beats >= 2,
        "semantic_no_worse_than_random": mean_semantic <= mean_random,
    }
    gates["all_pass"] = all(gates.values())
    result["summary"] = {
        "num_users": 4,
        "num_branches": 16,
        "semantic_spearman_vs_negative_branch_nll": correlation,
        "mean_semantic_top1_regret": mean_semantic,
        "mean_immediate_eig_top1_regret": mean_immediate,
        "mean_seeded_random_top1_regret": mean_random,
        "semantic_beats_immediate_count": semantic_beats,
        "gates": gates,
    }
    result["status"] = "passed" if gates["all_pass"] else "gate_failed"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--private-raw-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--likelihood-model", default="openai/gpt-5.4-mini")
    parser.add_argument("--stage", choices=("serving_smoke", "formal"), required=True)
    parser.add_argument("--resume-initial-raw", type=Path)
    parser.add_argument("--resume-initial-log", type=Path)
    args = parser.parse_args()
    config = load_config(str(args.config))
    config.run_id = args.run_id
    args.output_dir.mkdir(parents=True, exist_ok=True)
    private_dir = args.private_raw_dir / args.run_id
    private_dir.mkdir(parents=True, exist_ok=True)
    config.log_path = args.output_dir / "run.log"
    raw_path = private_dir / "RAW_RESPONSES.json"
    global _build_models
    if args.resume_initial_raw or args.resume_initial_log:
        if not args.resume_initial_raw or not args.resume_initial_log:
            raise ValueError("both resume paths are required")
        frozen = json.loads(args.resume_initial_raw.read_text())["responses"]
        events = [
            json.loads(line) for line in args.resume_initial_log.read_text().splitlines()
        ]
        if len(events) != 30:
            raise ValueError("resume log must contain exactly 30 initial requests")
        original_builder = _build_models

        def resumed_builder(config, likelihood_model):
            generator, likelihood = original_builder(config, likelihood_model)
            def prefix(rows):
                return {
                    "adapter_requests": len(rows),
                    "adapter_prompt_tokens": sum(r["prompt_tokens"] for r in rows),
                    "adapter_completion_tokens": sum(r["completion_tokens"] for r in rows),
                    "adapter_reasoning_tokens": sum(r["reasoning_tokens"] for r in rows),
                    "adapter_cost_usd": sum(r["cost_usd"] for r in rows),
                }
            return (
                _ReplayFirstBatch(
                    generator, frozen["initial_profiles"], prefix(events[:15])
                ),
                _ReplayFirstBatch(
                    likelihood, frozen["initial_likelihoods"], prefix(events[15:])
                ),
            )
        _build_models = resumed_builder
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
    name = "SERVING_SMOKE.json" if args.stage == "serving_smoke" else "GATE.json"
    (args.output_dir / name).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": payload["status"], **payload["summary"]}, indent=2))


if __name__ == "__main__":
    main()
