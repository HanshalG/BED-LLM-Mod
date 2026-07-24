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
import scripts.movielens_adaptive_candidate_gate_v4 as gate


def test_v4_selection_and_candidate_split_are_frozen() -> None:
    config = load_config(
        "configs/config_movielens_adaptive_candidate_gate_v4_openrouter.yaml"
    )
    assert config.openrouter_run_budget_usd == pytest.approx(0.90)
    ratings, _items = load_movielens("external/ml-100k")
    assert gate.selected_user_ids(ratings) == tuple(
        sorted(gate.ALL_SELECTED_USER_IDS)
    )
    pool, heldout = gate.candidate_and_heldout_ids(
        gate.SMOKE_USER_IDS[0], ratings
    )
    assert len(pool) == gate.CANDIDATE_POOL_SIZE
    assert len(heldout) == 8
    assert not set(pool) & set(heldout)


def test_candidate_selection_uses_descending_eig_and_movie_tiebreak() -> None:
    movie_ids = list(range(100, 100 + gate.CANDIDATE_POOL_SIZE))
    eig = [0.0] * gate.CANDIDATE_POOL_SIZE
    eig[3] = 0.4
    eig[5] = 0.4
    eig[8] = 0.3
    eig[9] = 0.2
    assert gate.select_candidate_indices(movie_ids, eig) == (3, 5, 8, 9)


class _FakeGenerator:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(
        self, batch_messages, temperature, block_size, max_new_tokens=None
    ):
        del temperature, block_size, max_new_tokens
        responses = []
        for batch_index, messages in enumerate(batch_messages):
            refreshed = "new_evidence_effect" in messages[-1]["content"]
            rows = []
            for index in range(PROFILE_COUNT):
                row = {
                    "id": f"p{index + 1}",
                    "description": (
                        f"{'refresh' if refreshed else 'initial'} "
                        f"{self.requests} {batch_index} {index}"
                    ),
                }
                if refreshed:
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
        self.saw_history = False

    def chat_complete_messages_batched(
        self, batch_messages, temperature, block_size, max_new_tokens=None
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


def test_v4_smoke_routes_exact_calls(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_adaptive_candidate_gate_v4_openrouter.yaml"
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
    assert all(record["candidate_pool_size"] == 16 for record in result["records"])


def test_v4_formal_routes_four_adaptive_branches(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_adaptive_candidate_gate_v4_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["status"] == "gate_failed"
    assert result["usage"]["physical_requests"] == 24
    assert generator.requests == 12
    assert likelihood.requests == 12
    assert result["protocol"]["formal_sensitivity_futility_stop"] is True
    assert result["protocol"]["candidate_outcomes_not_read"] is True
