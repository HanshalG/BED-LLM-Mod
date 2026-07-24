from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import (
    FORMAL_EXPECTED_REQUESTS,
    PROFILE_COUNT,
    SMOKE_EXPECTED_REQUESTS,
    load_movielens,
)
import scripts.movielens_profile_dynamics_gate_v3 as gate


def _profile_rows(*, refreshed: bool, collapse_movie: str | None = None):
    rows = []
    reactions = ("appeal", "avoid", "uncertain")
    for profile_index in range(PROFILE_COUNT):
        contrasts = []
        for movie_index, movie in enumerate(gate.DESIGN_MOVIES):
            reaction = reactions[(profile_index + movie_index) % len(reactions)]
            if movie["id"] == collapse_movie:
                reaction = "appeal"
            contrasts.append(
                {
                    "movie_id": movie["id"],
                    "reaction": reaction,
                    "rationale": f"reason {profile_index} {movie_index}",
                }
            )
        row = {
            "id": f"p{profile_index + 1}",
            "description": f"profile {profile_index}",
            "candidate_contrasts": contrasts,
        }
        if refreshed:
            row["new_evidence_effect"] = f"effect {profile_index}"
        rows.append(row)
    return rows


def test_v3_frozen_fresh_selection_and_config() -> None:
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_v3_openrouter.yaml"
    )
    assert config.openrouter_projected_cost_usd == pytest.approx(0.50)
    assert config.openrouter_concurrency == 64
    ratings, _items = load_movielens("external/ml-100k")
    assert gate.selected_user_ids(ratings) == tuple(
        sorted(gate.ALL_SELECTED_USER_IDS)
    )
    assert not set(gate.SMOKE_USER_IDS) & set(gate.FORMAL_USER_IDS)


def test_contrastive_parser_requires_candidate_disagreement() -> None:
    payload = json.dumps(
        {"profiles": _profile_rows(refreshed=False)}
    )
    parsed = gate.parse_profiles(payload)
    assert len(parsed) == PROFILE_COUNT
    assert "Candidate implications:" in parsed[0]
    assert "Contact (1997): appeal" in parsed[0]

    collapsed = json.dumps(
        {"profiles": _profile_rows(refreshed=False, collapse_movie="q1")}
    )
    with pytest.raises(ValueError, match="vary across"):
        gate.parse_profiles(collapsed)


def test_refreshed_parser_requires_effect_and_noncopy() -> None:
    rows = _profile_rows(refreshed=True)
    payload = json.dumps({"profiles": rows})
    parsed = gate.parse_refreshed_profiles(payload, ["unrelated"] * PROFILE_COUNT)
    assert len(parsed) == PROFILE_COUNT
    rows[0]["new_evidence_effect"] = ""
    with pytest.raises(ValueError, match="new_evidence_effect"):
        gate.parse_refreshed_profiles(
            json.dumps({"profiles": rows}),
            ["unrelated"] * PROFILE_COUNT,
        )


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
        for messages in batch_messages:
            refreshed = "new_evidence_effect" in messages[-1]["content"]
            rows = _profile_rows(refreshed=refreshed)
            for index, row in enumerate(rows):
                row["description"] += f" request {self.requests} row {index}"
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
        self.saw_history = False

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
            self.saw_history |= "observed_movie_ratings" in content
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


def test_v3_smoke_routes_exact_calls_with_history_free_likelihood(
    monkeypatch,
) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_v3_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="serving_smoke",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == SMOKE_EXPECTED_REQUESTS
    assert generator.requests == 6
    assert likelihood.requests == 4
    assert likelihood.saw_history is False
    assert result["schema_version"] == 3
    assert result["protocol"]["candidate_contrastive_profiles"] is True


def test_v3_formal_routes_all_branches(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_profile_dynamics_gate_v3_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["usage"]["physical_requests"] == FORMAL_EXPECTED_REQUESTS
    assert generator.requests == 60
    assert likelihood.requests == 60
    assert len(result["records"]) == len(gate.FORMAL_USER_IDS)
    assert all(len(record["branches"]) == 4 for record in result["records"])
