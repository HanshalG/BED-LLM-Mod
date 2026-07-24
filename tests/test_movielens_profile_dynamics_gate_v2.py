from __future__ import annotations

import json

import pytest

from helpers import load_config
import scripts.movielens_profile_dynamics_gate_v2 as gate_module
from scripts.movielens_profile_dynamics_gate import (
    FORMAL_EXPECTED_REQUESTS,
    PROFILE_COUNT,
    SMOKE_EXPECTED_REQUESTS,
    load_movielens,
)
from scripts.movielens_profile_dynamics_gate_v2 import (
    ALL_SELECTED_USER_IDS,
    FORMAL_USER_IDS,
    SELECTION_SEED,
    SMOKE_USER_IDS,
    parse_refreshed_profiles,
    profile_only_rating_likelihood_messages,
    refreshed_profile_messages,
    run_gate,
    selected_user_ids,
)


def test_v2_frozen_config_and_disjoint_selection() -> None:
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_v2_openrouter.yaml"
    )
    assert config.openrouter_budget_usd == pytest.approx(70.38480269545715)
    assert config.openrouter_run_budget_usd == pytest.approx(0.75)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.30)
    assert config.openrouter_concurrency == 64
    assert SELECTION_SEED == 24303
    assert len(SMOKE_USER_IDS) == 2
    assert len(FORMAL_USER_IDS) == 12
    assert not set(SMOKE_USER_IDS) & set(FORMAL_USER_IDS)
    ratings, _items = load_movielens("external/ml-100k")
    assert selected_user_ids(ratings) == tuple(sorted(ALL_SELECTED_USER_IDS))


def test_refresh_parser_rejects_copies_and_requires_evidence_effect() -> None:
    previous = [f"old profile {index}" for index in range(PROFILE_COUNT)]
    rows = [
        {
            "id": f"p{index + 1}",
            "description": f"new profile {index}",
            "new_evidence_effect": f"rating changed dimension {index}",
        }
        for index in range(PROFILE_COUNT)
    ]
    parsed = parse_refreshed_profiles(
        json.dumps({"profiles": rows}),
        previous,
    )
    assert len(parsed) == PROFILE_COUNT

    copied = [dict(row) for row in rows]
    copied[0]["description"] = previous[0]
    with pytest.raises(ValueError, match="copied"):
        parse_refreshed_profiles(json.dumps({"profiles": copied}), previous)

    missing_effect = [dict(row) for row in rows]
    missing_effect[0]["new_evidence_effect"] = ""
    with pytest.raises(ValueError, match="new_evidence_effect"):
        parse_refreshed_profiles(
            json.dumps({"profiles": missing_effect}),
            previous,
        )


def test_likelihood_prompt_has_profiles_but_no_observed_history() -> None:
    profiles = [f"profile {index}" for index in range(PROFILE_COUNT)]
    movies = [{"title": "Movie", "genres": ["Drama"]}]
    prompt = profile_only_rating_likelihood_messages(profiles, movies)
    text = json.dumps(prompt)
    assert "candidate_profiles" in text
    assert "movies_in_fixed_order" in text
    assert "observed_movie_ratings" not in text
    assert '"rating": 4' not in text

    history = [{"title": "Observed", "genres": ["Drama"], "rating": 4}]
    refresh_text = json.dumps(refreshed_profile_messages(history, profiles))
    assert "observed_movie_ratings" in refresh_text
    assert "new_evidence_effect" in refresh_text


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
        for batch_index, messages in enumerate(batch_messages):
            refresh = "new_evidence_effect" in messages[-1]["content"]
            rows = []
            for index in range(PROFILE_COUNT):
                row = {
                    "id": f"p{index + 1}",
                    "description": (
                        f"{'refreshed' if refresh else 'initial'} request "
                        f"{self.requests} batch {batch_index} profile {index}"
                    ),
                }
                if refresh:
                    row["new_evidence_effect"] = f"effect {index}"
                rows.append(row)
            responses.append(json.dumps({"profiles": rows}))
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
        self.saw_observed_history = False

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
            self.saw_observed_history |= "observed_movie_ratings" in content
            schema_text = content.split("Return ", 1)[1].split(".\n", 1)[0]
            responses.append(json.dumps(json.loads(schema_text)))
        self.requests += len(batch_messages)
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_v2_smoke_routes_exact_calls_without_likelihood_history(
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
        "configs/config_movielens_profile_dynamics_gate_v2_openrouter.yaml"
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
    assert likelihood.saw_observed_history is False
    assert result["protocol"]["likelihood_history_hidden"] is True


def test_v2_formal_routes_all_branches_without_likelihood_history(
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
        "configs/config_movielens_profile_dynamics_gate_v2_openrouter.yaml"
    )
    result = run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["usage"]["physical_requests"] == FORMAL_EXPECTED_REQUESTS
    assert generator.requests == 60
    assert likelihood.requests == 60
    assert likelihood.saw_observed_history is False
    assert len(result["records"]) == len(FORMAL_USER_IDS)
    assert all(len(record["branches"]) == 4 for record in result["records"])
