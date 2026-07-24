from __future__ import annotations

import json
import math

import numpy as np
import pytest

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import PROFILE_COUNT, load_movielens
import scripts.movielens_receding_explicit_policy_v9 as gate


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


def test_v9_selection_is_frozen_and_uses_final_fresh_pool() -> None:
    ratings, _items = load_movielens("external/ml-100k")
    assert gate.selected_user_ids(ratings) == gate.ALL_SELECTED_USER_IDS
    assert len(gate.ALL_SELECTED_USER_IDS) == 50
    assert not set(gate.ALL_SELECTED_USER_IDS) & set(gate.v8.ALL_SELECTED_USER_IDS)


def test_expected_downstream_entropies_match_manual_weighting() -> None:
    current = np.tile(
        np.asarray([0.1, 0.2, 0.3, 0.2, 0.2]),
        (PROFILE_COUNT, 1, 1),
    )
    distributions = (
        [1.0, 0.0, 0.0, 0.0, 0.0],
        [0.5, 0.5, 0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2, 0.2, 0.2],
        [0.25, 0.25, 0.25, 0.25, 0.0],
        [0.8, 0.05, 0.05, 0.05, 0.05],
    )
    branches = [
        np.tile(np.asarray(values), (8, 3, 1))
        for values in distributions
    ]
    entropies = np.asarray(
        [
            0.0,
            math.log(2),
            math.log(5),
            math.log(4),
            -(0.8 * math.log(0.8) + 4 * 0.05 * math.log(0.05)),
        ]
    )
    assert gate.expected_downstream_entropies(
        current_likelihoods=current,
        candidate_indices=(0,),
        branch_likelihoods=branches,
    ) == [pytest.approx(float(current[0, 0] @ entropies))]


def test_v9_probability_parser_normalizes_and_counts_wide_rows() -> None:
    response = json.dumps(
        {
            "profiles": [
                {
                    "id": "p1",
                    "ratings": [[0.24, 0.3, 0.2, 0.1, 0.06]],
                }
            ]
        }
    )
    matrix, count = gate.parse_likelihood(
        response,
        profile_count=1,
        movie_count=1,
    )
    assert matrix[0, 0].sum() == pytest.approx(1.0)
    assert count == 1


def test_v9_smoke_routes_exact_shared_state_calls(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood()
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_receding_explicit_policy_v9_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="serving_smoke",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == gate.SMOKE_EXPECTED_REQUESTS
    assert generator.requests == 11
    assert likelihood.requests == 11
    assert likelihood.saw_history is False
    assert result["summary"]["num_unique_round2_policy_states"] == 1


def test_v9_formal_dynamic_request_accounting(monkeypatch) -> None:
    generator = _FakeGenerator()
    likelihood = _FakeLikelihood(informative_initial=True)
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    config = load_config(
        "configs/config_movielens_receding_explicit_policy_v9_openrouter.yaml"
    )
    result = gate.run_gate(
        config,
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["usage"]["physical_requests"] == result["summary"][
        "expected_physical_requests"
    ]
    assert generator.requests == likelihood.requests
    assert likelihood.saw_history is False
    assert result["summary"]["num_enrolled_users"] == gate.ENROLLMENT_COUNT
    assert result["summary"]["gates"]["exact_dynamic_request_count"] is True
    assert all(
        len(record["policy_paths"]) == len(gate.POLICIES)
        for record in result["records"]
    )
