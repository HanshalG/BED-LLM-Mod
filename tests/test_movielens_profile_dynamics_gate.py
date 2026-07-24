from __future__ import annotations

import json

import numpy as np
import pytest

from helpers import load_config
import scripts.movielens_profile_dynamics_gate as gate_module
from scripts.movielens_profile_dynamics_gate import (
    CANDIDATE_MOVIE_IDS,
    FORMAL_EXPECTED_REQUESTS,
    FORMAL_USER_IDS,
    HELDOUT_COUNT,
    INITIAL_MOVIE_IDS,
    PROFILE_COUNT,
    README_SHA256,
    RATINGS_SHA256,
    ITEMS_SHA256,
    SELECTION_SEED,
    SMOKE_EXPECTED_REQUESTS,
    heldout_movie_ids,
    immediate_eig_values,
    load_movielens,
    merge_branch_profiles,
    parse_profiles,
    parse_rating_likelihoods,
    predictive_nll,
    profile_messages,
    rating_likelihood_messages,
    run_gate,
    summarize,
)


def test_frozen_config_and_protocol() -> None:
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(0.75)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.30)
    assert config.openrouter_concurrency == 64
    assert config.openrouter_max_output_tokens == 4096
    assert SELECTION_SEED == 24302
    assert len(FORMAL_USER_IDS) == 12
    assert len(INITIAL_MOVIE_IDS) == 4
    assert len(CANDIDATE_MOVIE_IDS) == 4
    assert FORMAL_EXPECTED_REQUESTS == 120
    assert SMOKE_EXPECTED_REQUESTS == 10
    assert len(RATINGS_SHA256) == len(ITEMS_SHA256) == len(README_SHA256) == 64


def test_pinned_movielens_data_and_selection_load() -> None:
    ratings, items = load_movielens("external/ml-100k")
    assert len(ratings) == 943
    assert len(items) == 1682
    assert items[50]["title"] == "Star Wars (1977)"
    for user_id in FORMAL_USER_IDS:
        assert all(
            movie_id in ratings[user_id]
            for movie_id in INITIAL_MOVIE_IDS + CANDIDATE_MOVIE_IDS
        )
        heldout = heldout_movie_ids(user_id, ratings[user_id])
        assert len(heldout) == HELDOUT_COUNT
        assert not set(heldout) & set(INITIAL_MOVIE_IDS + CANDIDATE_MOVIE_IDS)


def test_profile_and_likelihood_parsers_fail_closed() -> None:
    profile_rows = [
        {"id": f"p{index + 1}", "description": f"Profile {index}"}
        for index in range(PROFILE_COUNT)
    ]
    profiles = parse_profiles(json.dumps({"profiles": profile_rows}))
    assert len(profiles) == PROFILE_COUNT
    with pytest.raises(ValueError, match="unique"):
        duplicate = [dict(row, description="same") for row in profile_rows]
        parse_profiles(json.dumps({"profiles": duplicate}))

    likelihood_rows = [
        {
            "id": f"p{index + 1}",
            "ratings": [[0.1, 0.2, 0.3, 0.2, 0.2] for _ in range(2)],
        }
        for index in range(PROFILE_COUNT)
    ]
    matrix = parse_rating_likelihoods(
        json.dumps({"profiles": likelihood_rows}),
        profile_count=PROFILE_COUNT,
        movie_count=2,
    )
    assert matrix.shape == (PROFILE_COUNT, 2, 5)
    likelihood_rows[0]["ratings"][0] = [0.42, 0.24, 0.16, 0.08, 0.08]
    parse_rating_likelihoods(
        json.dumps({"profiles": likelihood_rows}),
        profile_count=PROFILE_COUNT,
        movie_count=2,
    )
    likelihood_rows[0]["ratings"][0] = [0.1] * 5
    with pytest.raises(ValueError, match="sum to one"):
        parse_rating_likelihoods(
            json.dumps({"profiles": likelihood_rows}),
            profile_count=PROFILE_COUNT,
            movie_count=2,
        )


def test_json_parser_repairs_only_structural_trailing_commas() -> None:
    payload = gate_module._parse_json_object(
        '{"profiles":[{"description":"literal,} text",},],}'
    )
    assert payload == {
        "profiles": [{"description": "literal,} text"}],
    }
    with pytest.raises(json.JSONDecodeError):
        gate_module._parse_json_object('{"profiles":[not-json]}')


def test_prompts_do_not_expose_user_id_or_unobserved_ratings() -> None:
    history = [{"title": "Observed", "genres": ["Drama"], "rating": 4}]
    profiles = [f"Profile {index}" for index in range(PROFILE_COUNT)]
    movies = [{"title": "Hidden", "genres": ["Comedy"]}]
    generation_text = json.dumps(profile_messages(history))
    likelihood_text = json.dumps(
        rating_likelihood_messages(history, profiles, movies)
    )
    assert "user_id" not in generation_text
    assert "user_id" not in likelihood_text
    assert "heldout" not in generation_text
    assert "heldout" not in likelihood_text
    assert '"rating":4' in profile_messages(history)[1]["content"]
    assert "Hidden" in likelihood_text


def test_eig_and_predictive_nll_match_manual_values() -> None:
    likelihoods = np.asarray(
        [
            [[0.8, 0.2, 0.0, 0.0, 0.0]],
            [[0.2, 0.8, 0.0, 0.0, 0.0]],
        ],
        dtype=float,
    )
    eig = immediate_eig_values(likelihoods)[0]
    manual = -sum(value * np.log(value) for value in (0.5, 0.5))
    manual -= -sum(value * np.log(value) for value in (0.8, 0.2))
    assert eig == pytest.approx(manual)
    assert predictive_nll(likelihoods, [1]) == pytest.approx(-np.log(0.5))


def test_branch_merge_keeps_generated_and_best_old_profiles() -> None:
    generated = [f"new {index}" for index in range(PROFILE_COUNT)]
    previous = ["old low", "old high", "old middle"]
    merged = merge_branch_profiles(generated, previous, [0.1, 0.9, 0.5])
    assert merged[:PROFILE_COUNT] == generated
    assert merged[-2:] == ["old high", "old middle"]


def _record(
    *,
    initial_nll: float,
    branch_nlls: list[float],
    selected: int,
) -> dict[str, object]:
    return {
        "initial_heldout_nll": initial_nll,
        "immediate_eig_values": [0.4, 0.3, 0.2, 0.1],
        "immediate_eig_selected_branch": selected,
        "branches": [{"heldout_nll": value} for value in branch_nlls],
    }


def test_formal_summary_applies_frozen_gates() -> None:
    records = [
        _record(
            initial_nll=1.0,
            branch_nlls=[0.95, 0.8, 1.1, 1.0],
            selected=0,
        )
        for _ in range(12)
    ]
    result = summarize(
        records,
        {"physical_requests": FORMAL_EXPECTED_REQUESTS, "reasoning_tokens": 0},
        stage="formal",
    )
    assert result["mean_oracle_heldout_nll_improvement"] == pytest.approx(0.2)
    assert result["mean_immediate_eig_heldout_nll_regret"] == pytest.approx(0.15)
    assert result["mean_max_immediate_eig"] == pytest.approx(0.4)
    assert result["gates"]["all_pass"] is True


def test_smoke_summary_requires_exact_requests_and_replays() -> None:
    records = [
        {"branches": [{}], "replay_profiles": ["x"] * PROFILE_COUNT},
        {"branches": [{}], "replay_profiles": ["x"] * PROFILE_COUNT},
    ]
    result = summarize(
        records,
        {"physical_requests": SMOKE_EXPECTED_REQUESTS, "reasoning_tokens": 0},
        stage="serving_smoke",
    )
    assert result["gates"]["all_pass"] is True
    result = summarize(
        records,
        {"physical_requests": SMOKE_EXPECTED_REQUESTS + 1, "reasoning_tokens": 0},
        stage="serving_smoke",
    )
    assert result["gates"]["all_pass"] is False


class _FakeGenerator:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        responses = []
        for batch_index, _messages in enumerate(batch_messages):
            responses.append(
                json.dumps(
                    {
                        "profiles": [
                            {
                                "id": f"p{index + 1}",
                                "description": (
                                    f"request {self.requests} batch {batch_index} "
                                    f"profile {index}"
                                ),
                            }
                            for index in range(PROFILE_COUNT)
                        ]
                    }
                )
            )
        self.requests += len(batch_messages)
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


class _FakeLikelihood:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        responses = []
        for messages in batch_messages:
            content = messages[-1]["content"]
            schema_text = content.split("Return ", 1)[1].split(".\n", 1)[0]
            schema = json.loads(schema_text)
            responses.append(json.dumps(schema))
        self.requests += len(batch_messages)
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_smoke_run_routes_exactly_ten_calls_without_model_outcome_oracle(
    monkeypatch,
) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate_module,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_openrouter.yaml"
    )
    result = run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="serving_smoke",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == SMOKE_EXPECTED_REQUESTS
    assert generator.requests == 6
    assert likelihood.requests == 4
    serialized = json.dumps(result)
    assert "observed_rating" not in serialized
    assert "heldout_movies" not in serialized
    assert "initial_history" not in serialized


def test_formal_run_routes_all_four_branches_per_user(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate_module,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_openrouter.yaml"
    )
    result = run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["status"] == "gate_failed"
    assert result["usage"]["physical_requests"] == FORMAL_EXPECTED_REQUESTS
    assert generator.requests == 60
    assert likelihood.requests == 60
    assert len(result["records"]) == len(FORMAL_USER_IDS)
    assert all(len(record["branches"]) == 4 for record in result["records"])
