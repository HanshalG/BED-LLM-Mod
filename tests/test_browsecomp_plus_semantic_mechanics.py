from __future__ import annotations

import json

import pytest

from scripts import browsecomp_plus_semantic_mechanics as mechanics


def _belief_lines(prefix: str = "answer") -> list[str]:
    weights = (40, 20, 10, 8, 7, 6, 5, 4)
    return [
        f"H{index:02d}|{weight}|{prefix} {index}"
        for index, weight in enumerate(weights, start=1)
    ]


def test_flat_parsers_are_exact_and_require_distinct_roots():
    text = "\n".join(
        _belief_lines()
        + [
            f"S{index:02d}|{index * 10}|root query {index}|future intent {index}"
            for index in range(1, 7)
        ]
    )
    belief, strategies = mechanics.parse_initial(text)
    assert belief.weights[0] == 40
    assert len(strategies) == 6
    assert strategies[-1].direct_score == 60

    duplicate = text.replace("root query 6", "root query 1")
    with pytest.raises(ValueError, match="distinct"):
        mechanics.parse_initial(duplicate)
    with pytest.raises(ValueError, match="line count"):
        mechanics.parse_initial(text + "\nextra")


def test_refresh_future_and_terminal_parsers():
    belief, query = mechanics.parse_refresh(
        "\n".join(_belief_lines("updated") + ["A01|next query"])
    )
    assert query == "next query"
    assert belief.hypotheses[0] == "updated 1"
    assert mechanics.parse_future_scores(
        "\n".join(f"S{i:02d}|{i}" for i in range(1, 7))
    ) == [1, 2, 3, 4, 5, 6]
    assert mechanics.parse_terminal(
        "\n".join(_belief_lines("terminal"))
    ).hypotheses[-1] == "terminal 8"


def test_task_bm25_prefers_matching_document():
    retriever = mechanics.TaskBM25(
        [
            {"docid": "a", "text": "violet archive mountain"},
            {"docid": "b", "text": "financial exchange market"},
            {"docid": "c", "text": "ordinary unrelated page"},
            {"docid": "d", "text": "another unrelated page"},
        ]
    )
    results = retriever.search("violet archive")
    assert results[0]["docid"] == "a"
    assert len(results) == mechanics.RETRIEVAL_TOP_K


def test_bridge_analysis_rewards_aligned_full_ranking():
    assert mechanics._pairwise_accuracy(
        [1, 3, 2],
        [0, 2, 1],
    ) == (1.0, 3)
    assert mechanics._pairwise_accuracy(
        [1, 1],
        [0, 2],
    ) == (0.5, 1)


def test_config_is_nonreasoning_fail_closed_and_budgeted():
    from helpers import load_config

    config = load_config(
        "configs/config_browsecomp_plus_semantic_mechanics_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_max_retries == 0
    assert config.openrouter_concurrency == 24
    assert config.openrouter_run_budget_usd == pytest.approx(0.90)
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
                    lines.append(
                        f"{identifier}|"
                        f"{(40, 20, 10, 8, 7, 6, 5, 4)[int(identifier[1:]) - 1]}"
                        f"|answer {identifier} request {self.requests}"
                    )
                elif identifier.startswith("A"):
                    lines.append(f"{identifier}|adaptive query {self.requests}")
                elif "direct score" in specification:
                    index = int(identifier[1:])
                    lines.append(
                        f"{identifier}|{index * 10}|root query {index}|"
                        f"future intent {index}"
                    )
                else:
                    lines.append(f"{identifier}|{int(identifier[1:]) * 10}")
            responses.append("\n".join(lines))
        self.requests += 0
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


def test_model_stage_has_exact_55_call_shape(tmp_path, monkeypatch):
    tasks = []
    for task_id in mechanics.TASK_IDS:
        documents = [
            {
                "docid": f"{task_id}-{index}",
                "text": f"document {index} root query adaptive evidence",
            }
            for index in range(8)
        ]
        tasks.append(
            {
                "query_id": task_id,
                "query": f"Question {task_id}",
                "answer": f"answer {task_id}",
                "gold_docs": documents[:1],
                "evidence_docs": documents[:3],
                "negative_docs": documents[3:],
            }
        )
    monkeypatch.setattr(mechanics, "load_tasks", lambda path: tasks)
    config = type(
        "Config",
        (),
        {
            "openrouter_concurrency": 24,
            "openrouter_max_output_tokens": 1024,
        },
    )()
    adapter = FakeAdapter()

    parsed, usage = mechanics.run_model_stage(
        config,
        source_path=tmp_path / "source.jsonl",
        raw_path=tmp_path / "raw.json",
        model_adapter=adapter,
    )

    assert usage["physical_requests"] == mechanics.EXPECTED_REQUESTS == 55
    assert len(parsed["branches"]) == 5
    assert all(len(branches) == 6 for branches in parsed["branches"])
    assert (tmp_path / "raw.json").exists()
