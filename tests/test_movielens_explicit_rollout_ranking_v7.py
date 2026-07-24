from __future__ import annotations

import json
import math

import numpy as np
import pytest

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import PROFILE_COUNT, load_movielens
import scripts.movielens_explicit_rollout_ranking_v7 as gate


def _profile_response(prefix: str, *, refreshed: bool) -> str:
    rows = []
    for index in range(PROFILE_COUNT):
        row = {
            "id": f"p{index + 1}",
            "description": f"{prefix} profile {index}",
        }
        if refreshed:
            row["new_evidence_effect"] = f"effect {prefix} {index}"
        rows.append(row)
    return json.dumps({"profiles": rows})


def _likelihood_response(
    profile_count: int,
    movie_count: int,
    probabilities: list[float],
) -> str:
    return json.dumps(
        {
            "profiles": [
                {
                    "id": f"p{index + 1}",
                    "ratings": [probabilities] * movie_count,
                }
                for index in range(profile_count)
            ]
        }
    )


class _ScorerGenerator:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(self, batch_messages, **_kwargs):
        responses = [
            _profile_response(f"branch {index}", refreshed=True)
            for index, _messages in enumerate(batch_messages)
        ]
        self.requests += len(responses)
        return responses


class _ScorerLikelihood:
    DISTRIBUTIONS = (
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25, 0.0],
        [0.8, 0.05, 0.05, 0.05, 0.05],
    )

    def __init__(self) -> None:
        self.requests = 0
        self.saw_history = False

    def chat_complete_messages_batched(self, batch_messages, **_kwargs):
        responses = []
        for index, messages in enumerate(batch_messages):
            self.saw_history |= "observed_movie_ratings" in messages[-1]["content"]
            responses.append(
                _likelihood_response(8, 8, self.DISTRIBUTIONS[index])
            )
        self.requests += len(responses)
        return responses


class _SmokeGenerator:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(self, batch_messages, **_kwargs):
        responses = []
        for messages in batch_messages:
            refreshed = "new_evidence_effect" in messages[-1]["content"]
            responses.append(
                _profile_response(
                    f"{'refreshed' if refreshed else 'initial'} {self.requests}",
                    refreshed=refreshed,
                )
            )
            self.requests += 1
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


class _SmokeLikelihood:
    def __init__(self) -> None:
        self.requests = 0
        self.saw_history = False

    def chat_complete_messages_batched(self, batch_messages, **_kwargs):
        responses = []
        for messages in batch_messages:
            content = messages[-1]["content"]
            self.saw_history |= "observed_movie_ratings" in content
            schema_text = content.split("Return ", 1)[1].split(".\n", 1)[0]
            responses.append(json.dumps(json.loads(schema_text)))
            self.requests += 1
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_v7_selection_is_frozen_and_fresh() -> None:
    ratings, _items = load_movielens("external/ml-100k")
    assert gate.selected_user_ids(ratings) == gate.ALL_SELECTED_USER_IDS
    assert not set(gate.ALL_SELECTED_USER_IDS) & gate._prior_users()


def test_mean_predictive_entropy_matches_manual_mixture() -> None:
    likelihoods = np.asarray(
        [
            [[[1.0, 0.0, 0.0, 0.0, 0.0]]],
            [[[0.0, 1.0, 0.0, 0.0, 0.0]]],
        ]
    ).reshape(2, 1, 5)
    assert gate.mean_predictive_entropy(likelihoods) == pytest.approx(math.log(2))


def test_explicit_rollout_weights_all_five_hypothetical_paths() -> None:
    config = load_config(
        "configs/config_movielens_explicit_rollout_ranking_v7_openrouter.yaml"
    )
    generator = _ScorerGenerator()
    likelihood = _ScorerLikelihood()
    probabilities = np.asarray([0.1, 0.2, 0.3, 0.2, 0.2])
    initial_matrix = np.tile(probabilities, (PROFILE_COUNT, 24, 1))
    item_ids = list(range(1, 25))
    items = {
        movie_id: {"movie_id": movie_id, "title": f"movie {movie_id}", "genres": []}
        for movie_id in item_ids
    }
    scores, raw = gate.explicit_rollout_scorer(
        generator=generator,
        likelihood=likelihood,
        config=config,
        histories=[[{"movie_id": 99, "title": "history", "rating": 4}]],
        initial_profiles=[[f"old profile {index}" for index in range(PROFILE_COUNT)]],
        candidate_pools=[item_ids[:16]],
        selected_indices=[(2,)],
        selected_movie_ids=[(item_ids[2],)],
        heldout_ids_many=[item_ids[16:]],
        initial_likelihoods=[initial_matrix],
        user_ids=[123],
        items=items,
    )
    entropies = np.asarray(
        [
            0.0,
            math.log(2),
            math.log(5),
            math.log(4),
            -(0.8 * math.log(0.8) + 4 * 0.05 * math.log(0.05)),
        ]
    )
    assert scores == [[pytest.approx(-float(np.dot(probabilities, entropies)))]]
    assert raw["outcome_probabilities"][0][0] == pytest.approx(
        probabilities.tolist()
    )
    assert generator.requests == 5
    assert likelihood.requests == 5
    assert likelihood.saw_history is False


def test_v7_smoke_routes_exact_twelve_requests(monkeypatch) -> None:
    generator = _SmokeGenerator()
    likelihood = _SmokeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_explicit_rollout_ranking_v7_openrouter.yaml"
    )
    result = gate.run_smoke(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == gate.SMOKE_EXPECTED_REQUESTS
    assert generator.requests == 6
    assert likelihood.requests == 6
    assert likelihood.saw_history is False
    assert result["protocol"]["candidate_and_heldout_outcomes_not_read"] is True
    assert result["summary"]["num_hypothetical_branches"] == 5
