from __future__ import annotations

import json

import numpy as np
import pytest

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import PROFILE_COUNT, load_movielens
import scripts.movielens_depth2_semantic_policy_v8 as gate


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


class _FakeGenerator:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(self, batch_messages, **_kwargs):
        responses = []
        for messages in batch_messages:
            refreshed = "new_evidence_effect" in messages[-1]["content"]
            responses.append(
                _profile_response(
                    f"{'refresh' if refreshed else 'initial'} {self.requests}",
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


class _FakeLikelihood:
    def __init__(self, *, informative_initial: bool = False) -> None:
        self.requests = 0
        self.saw_history = False
        self.informative_initial = informative_initial

    def chat_complete_messages_batched(self, batch_messages, **_kwargs):
        responses = []
        for messages in batch_messages:
            content = messages[-1]["content"]
            self.saw_history |= "observed_movie_ratings" in content
            schema_text = content.split("Return ", 1)[1].split(".\n", 1)[0]
            payload = json.loads(schema_text)
            rows = payload["profiles"]
            if (
                self.informative_initial
                and len(rows) == PROFILE_COUNT
                and len(rows[0]["ratings"]) == 24
            ):
                for profile_index, row in enumerate(rows):
                    row["ratings"][0] = (
                        [1.0, 0.0, 0.0, 0.0, 0.0]
                        if profile_index < PROFILE_COUNT // 2
                        else [0.0, 0.0, 0.0, 0.0, 1.0]
                    )
            responses.append(json.dumps(payload))
            self.requests += 1
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_v8_selection_is_frozen_and_excludes_v7() -> None:
    ratings, _items = load_movielens("external/ml-100k")
    assert gate.selected_user_ids(ratings) == gate.ALL_SELECTED_USER_IDS
    assert not set(gate.ALL_SELECTED_USER_IDS) & (
        gate.v7._prior_users() | set(gate.v7.ALL_SELECTED_USER_IDS)
    )


def test_depth_score_aggregation_matches_nested_expectation() -> None:
    q1_probabilities = np.asarray([[0.25, 0.75], [0.6, 0.4]])
    q1_entropies = np.asarray([[1.0, 2.0], [3.0, 1.0]])
    q2_probabilities = np.asarray(
        [
            [[0.5, 0.5], [0.2, 0.8]],
            [[0.7, 0.3], [0.4, 0.6]],
        ]
    )
    terminal_entropies = np.asarray(
        [
            [[1.0, 3.0], [2.0, 4.0]],
            [[5.0, 1.0], [2.0, 1.0]],
        ]
    )
    depth1, depth2 = gate.aggregate_depth_scores(
        q1_probabilities,
        q1_entropies,
        q2_probabilities,
        terminal_entropies,
    )
    assert depth1 == pytest.approx([1.75, 2.2])
    assert depth2 == pytest.approx(
        [
            0.25 * (0.5 * 1.0 + 0.5 * 3.0)
            + 0.75 * (0.2 * 2.0 + 0.8 * 4.0),
            0.6 * (0.7 * 5.0 + 0.3 * 1.0)
            + 0.4 * (0.4 * 2.0 + 0.6 * 1.0),
        ]
    )


def test_v8_smoke_routes_exact_common_tree_calls(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_depth2_semantic_policy_v8_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="serving_smoke",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == gate.SMOKE_EXPECTED_REQUESTS
    assert generator.requests == 31
    assert likelihood.requests == 31
    assert likelihood.saw_history is False
    assert result["summary"]["num_q1_branches"] == 5
    assert result["summary"]["num_q2_branches"] == 25
    assert result["protocol"]["common_transition_tree_for_all_policies"] is True
    assert {
        tuple(path["query_movie_ids"])
        for path in result["records"][0]["policy_paths"].values()
    }


def test_v8_formal_routes_exact_full_tree_calls(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood(informative_initial=True)
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_depth2_semantic_policy_v8_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["usage"]["physical_requests"] == gate.FORMAL_EXPECTED_REQUESTS
    assert generator.requests == gate.FORMAL_EXPECTED_REQUESTS // 2
    assert likelihood.requests == gate.FORMAL_EXPECTED_REQUESTS // 2
    assert likelihood.saw_history is False
    assert result["summary"]["num_enrolled_users"] == gate.ENROLLMENT_COUNT
    assert result["summary"]["gates"]["exact_physical_request_count"] is True
    assert len(result["records"]) == gate.ENROLLMENT_COUNT
    assert all(
        len(record["policy_paths"]) == 3 for record in result["records"]
    )
