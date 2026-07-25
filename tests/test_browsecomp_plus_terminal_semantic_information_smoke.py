from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts import browsecomp_plus_semantic_mechanics as mechanics
from scripts import browsecomp_plus_terminal_semantic_information_smoke as smoke


def _terminal_lines(request: int = 1) -> str:
    masses = (40, 20, 10, 8, 7, 6, 5, 0)
    return "\n".join(
        f"H{index:02d}|{mass}|answer {request} candidate {index}"
        for index, mass in enumerate(masses, start=1)
    )


def test_terminal_mass_parser_normalizes_and_allows_zero():
    belief = smoke.parse_terminal_masses(_terminal_lines())
    assert sum(belief.probabilities) == pytest.approx(1.0)
    assert belief.probabilities[-1] == 0.0
    with pytest.raises(ValueError, match="positive total"):
        smoke.parse_terminal_masses(
            "\n".join(
                f"H{index:02d}|0|answer {index}"
                for index in range(1, 9)
            )
        )


def test_terminal_messages_do_not_emit_endpoint_fields():
    initial = mechanics.Belief(
        hypotheses=tuple(f"candidate {index}" for index in range(8)),
        weights=(40, 20, 10, 8, 7, 6, 5, 4),
    )
    branch = mechanics.Branch(
        root_index=0,
        strategy=mechanics.Strategy("first query", 50, "future"),
        root_documents=[{"docid": "a", "text": "first observation"}],
        root_belief=initial,
        adaptive_query="second query",
        adaptive_documents=[
            {"docid": "b", "text": "second observation"}
        ],
    )
    messages = smoke.terminal_mass_messages(
        {
            "query": "question",
            "answer": "SECRET_ENDPOINT",
            "gold_docs": [{"docid": "secret"}],
        },
        initial_belief=initial,
        branch=branch,
    )
    serialized = json.dumps(messages)
    assert "SECRET_ENDPOINT" not in serialized
    assert "gold_docs" not in serialized


def test_config_is_nonreasoning_fail_closed_and_capped():
    config = load_config(
        "configs/"
        "config_browsecomp_plus_terminal_semantic_information_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_max_retries == 0
    assert config.openrouter_concurrency == 30
    assert config.openrouter_run_budget_usd == pytest.approx(0.75)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.30)
    assert config.openrouter_budget_usd == pytest.approx(105.0)


class FakeAdapter:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(self, messages, **kwargs):
        del kwargs
        responses = []
        for _ in messages:
            self.requests += 1
            responses.append(_terminal_lines(self.requests))
        return responses

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "adapter_cost_usd": 0.01,
        }


def test_model_stage_has_exact_30_call_shape(tmp_path, monkeypatch):
    initial = mechanics.Belief(
        hypotheses=tuple(f"candidate {index}" for index in range(8)),
        weights=(40, 20, 10, 8, 7, 6, 5, 4),
    )
    reconstructed = {
        "tasks": [
            {"query_id": task_id, "query": f"question {task_id}"}
            for task_id in mechanics.TASK_IDS
        ],
        "initials": [(initial, []) for _ in mechanics.TASK_IDS],
        "branches": [],
    }
    for _ in mechanics.TASK_IDS:
        branches = []
        for root_index in range(mechanics.ROOT_COUNT):
            branches.append(
                mechanics.Branch(
                    root_index=root_index,
                    strategy=mechanics.Strategy(
                        f"root {root_index}",
                        50,
                        "future",
                    ),
                    root_documents=[
                        {"docid": "a", "text": "first observation"}
                    ],
                    root_belief=initial,
                    adaptive_query=f"adaptive {root_index}",
                    adaptive_documents=[
                        {"docid": "b", "text": "second observation"}
                    ],
                )
            )
        reconstructed["branches"].append(branches)
    monkeypatch.setattr(
        smoke.posthoc,
        "reconstruct",
        lambda **kwargs: reconstructed,
    )
    adapter = FakeAdapter()
    config = type(
        "Config",
        (),
        {
            "openrouter_concurrency": 30,
            "openrouter_max_output_tokens": 768,
        },
    )()
    _, beliefs, usage = smoke.run_model_stage(
        config,
        source_path=tmp_path / "source.jsonl",
        cached_raw_path=tmp_path / "cached.json",
        raw_path=tmp_path / "raw.json",
        model_adapter=adapter,
    )
    assert usage["physical_requests"] == smoke.EXPECTED_REQUESTS == 30
    assert len(beliefs) == 5
    assert all(len(task_beliefs) == 6 for task_beliefs in beliefs)
