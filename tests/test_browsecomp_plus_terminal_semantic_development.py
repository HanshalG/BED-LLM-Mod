from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts import browsecomp_plus_semantic_mechanics as mechanics
from scripts import browsecomp_plus_terminal_semantic_development as development


def _belief_lines(request: int) -> list[str]:
    weights = (40, 20, 10, 8, 7, 6, 5, 4)
    return [
        f"H{index:02d}|{weight}|answer {request} candidate {index}"
        for index, weight in enumerate(weights, start=1)
    ]


def test_initial_parser_has_six_distinct_unscored_queries():
    text = "\n".join(
        _belief_lines(1)
        + [
            f"Q{index:02d}|root query {index}"
            for index in range(1, 7)
        ]
    )
    belief, queries = development.parse_initial(text)
    assert belief.weights[0] == 40
    assert queries[-1] == "root query 6"
    with pytest.raises(ValueError, match="distinct"):
        development.parse_initial(
            text.replace("root query 6", "root query 1")
        )


def test_initial_messages_do_not_request_value_scores():
    messages = development.initial_messages(
        {"query": "Which answer is supported?"}
    )
    serialized = json.dumps(messages).lower()
    assert "do not score" in serialized
    assert "future score" not in serialized
    assert "direct score" not in serialized


def test_config_is_capped_nonreasoning_and_uses_stage_concurrency():
    config = load_config(
        "configs/"
        "config_browsecomp_plus_terminal_semantic_development_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_max_retries == 0
    assert config.openrouter_concurrency == 60
    assert config.openrouter_run_budget_usd == pytest.approx(1.0)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.78)
    assert config.openrouter_budget_usd == pytest.approx(105.0)


class FakeAdapter:
    def __init__(self) -> None:
        self.requests = 0

    def chat_complete_messages_batched(self, messages, **kwargs):
        del kwargs
        responses = []
        for message_list in messages:
            self.requests += 1
            payload = json.loads(message_list[-1]["content"])
            grammar = payload["exact_output_lines"]
            lines = []
            for specification in grammar:
                identifier = specification.split("|", 1)[0]
                if identifier.startswith("H"):
                    weights = (40, 20, 10, 8, 7, 6, 5, 4)
                    lines.append(
                        f"{identifier}|"
                        f"{weights[int(identifier[1:]) - 1]}|"
                        f"answer {self.requests} {identifier}"
                    )
                elif identifier.startswith("Q"):
                    lines.append(
                        f"{identifier}|root query {self.requests} {identifier}"
                    )
                elif identifier == "A01":
                    lines.append(
                        f"A01|adaptive query {self.requests}"
                    )
                else:
                    raise AssertionError(specification)
            responses.append("\n".join(lines))
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


def test_model_stage_has_exact_130_call_shape(tmp_path):
    tasks = []
    for task_id in development.DEVELOPMENT_IDS:
        documents = [
            {
                "docid": f"{task_id}-{index}",
                "text": f"document {index} root adaptive evidence",
            }
            for index in range(10)
        ]
        tasks.append(
            {
                "query_id": task_id,
                "query": f"Question {task_id}",
                "answer": f"answer {task_id}",
                "gold_docs": documents[:1],
                "evidence_docs": documents[:4],
                "negative_docs": documents[4:],
            }
        )
    config = type(
        "Config",
        (),
        {
            "openrouter_concurrency": 60,
            "openrouter_max_output_tokens": 768,
        },
    )()
    adapter = FakeAdapter()
    reconstructed, terminals, usage = development.run_model_stage(
        config,
        manifest_path=tmp_path / "manifest.json",
        parquet_dir=tmp_path,
        raw_path=tmp_path / "raw.json",
        model_adapter=adapter,
        tasks_override=tasks,
    )
    assert usage["physical_requests"] == development.EXPECTED_REQUESTS
    assert usage["physical_requests"] == 130
    assert len(reconstructed["branches"]) == 10
    assert all(len(branches) == 6 for branches in reconstructed["branches"])
    assert all(len(task_terminals) == 6 for task_terminals in terminals)
