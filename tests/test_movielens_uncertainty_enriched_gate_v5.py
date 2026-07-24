from __future__ import annotations

import json

from helpers import load_config
from scripts.movielens_profile_dynamics_gate import PROFILE_COUNT, load_movielens
import scripts.movielens_uncertainty_enriched_gate_v5 as gate


def test_v5_frozen_selection() -> None:
    ratings, _items = load_movielens("external/ml-100k")
    assert gate.selected_user_ids(ratings) == gate.ALL_SELECTED_USER_IDS
    assert len(gate.FORMAL_SCREEN_USER_IDS) == 48
    assert len(set(gate.ALL_SELECTED_USER_IDS)) == 50


class _Fake:
    def __init__(self, likelihood: bool = False, discriminative: bool = False) -> None:
        self.requests = 0
        self.likelihood = likelihood
        self.discriminative = discriminative

    def chat_complete_messages_batched(
        self, batch_messages, temperature, block_size, max_new_tokens=None
    ):
        del temperature, block_size, max_new_tokens
        out = []
        for messages in batch_messages:
            content = messages[-1]["content"]
            if self.likelihood:
                schema = json.loads(content.split("Return ", 1)[1].split(".\n", 1)[0])
                if self.discriminative:
                    for profile_index, profile in enumerate(schema["profiles"]):
                        peak = profile_index % 5
                        probabilities = [0.025] * 5
                        probabilities[peak] = 0.9
                        profile["ratings"] = [
                            list(probabilities) for _ in profile["ratings"]
                        ]
                out.append(json.dumps(schema))
            else:
                refreshed = "new_evidence_effect" in content
                rows = []
                for i in range(PROFILE_COUNT):
                    row = {
                        "id": f"p{i + 1}",
                        "description": f"{'r' if refreshed else 'i'} {self.requests} {i}",
                    }
                    if refreshed:
                        row["new_evidence_effect"] = f"effect {i}"
                    rows.append(row)
                out.append(json.dumps({"profiles": rows}))
        self.requests += len(batch_messages)
        return out

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_v5_uniform_screen_fails_before_outcomes(monkeypatch) -> None:
    generator, likelihood = _Fake(), _Fake(likelihood=True)
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    result = gate.run_gate(
        load_config("configs/config_movielens_uncertainty_enriched_gate_v5_openrouter.yaml"),
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["status"] == "gate_failed"
    assert result["usage"]["physical_requests"] == 96
    assert result["protocol"]["candidate_outcomes_not_read"] is True
    assert result["protocol"]["prospective_uncertainty_enrichment"] is True


def test_v5_discriminative_screen_executes_full_enrolled_path(monkeypatch) -> None:
    generator = _Fake()
    likelihood = _Fake(likelihood=True, discriminative=True)
    monkeypatch.setattr(
        gate,
        "_build_models",
        lambda config, likelihood_model: (generator, likelihood),
    )
    result = gate.run_gate(
        load_config("configs/config_movielens_uncertainty_enriched_gate_v5_openrouter.yaml"),
        data_dir="external/ml-100k",
        likelihood_model="fake",
        stage="formal",
    )
    assert result["usage"]["physical_requests"] == 192
    assert generator.requests == likelihood.requests == 96
    assert len(result["records"]) == gate.ENROLLMENT_COUNT
    assert len(result["protocol"]["screen_user_ids"]) == 48
    assert result["summary"]["gates"]["prospective_enrollment_complete"] is True
