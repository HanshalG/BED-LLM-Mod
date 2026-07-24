import json
from types import SimpleNamespace

import pytest

import scripts.paprika_clothing_support_recall_gate as gate_module
from scripts.paprika_clothing_support_recall_gate import (
    ACTIVITY_INDICES,
    CANDIDATE_COUNT,
    EXPECTED_REQUESTS,
    FORMAL_INDICES,
    SMOKE_INDICES,
    SUPPORT_SAMPLE_SIZE,
    SUPPORT_SAMPLES,
    analyze_record,
    binary_entropy,
    parse_candidates,
    parse_endpoint,
    parse_support,
    parse_support_labels,
    ranker_messages,
    run_gate,
    union_supports,
)


def test_parse_support_and_union_dedupe_semantic_spelling():
    values = [f"item {index}" for index in range(SUPPORT_SAMPLE_SIZE)]
    first = parse_support(json.dumps({"possibilities": values}))
    second = [*values[:-1], "Different item"]
    union = union_supports([first, second])
    assert len(union) == SUPPORT_SAMPLE_SIZE + 1
    assert union[-1] == "Different item"


def test_parse_candidates_requires_three_questions():
    questions = [f"Question {index}?" for index in range(CANDIDATE_COUNT)]
    assert parse_candidates(json.dumps({"questions": questions})) == questions
    with pytest.raises(ValueError):
        parse_candidates(json.dumps({"questions": questions[:-1]}))


def test_boolean_parsers_reject_integer_surrogates():
    labels = [[True, False] for _ in range(CANDIDATE_COUNT)]
    assert parse_support_labels(
        json.dumps({"labels": labels}), support_size=2
    ) == labels
    labels[0][0] = 1
    with pytest.raises(ValueError):
        parse_support_labels(
            json.dumps({"labels": labels}), support_size=2
        )

    endpoint = {
        "target_answers": [True, False, True],
        "current_contains_target": False,
        "branch_contains_target": [
            [True, False],
            [False, True],
            [False, False],
        ],
    }
    assert parse_endpoint(json.dumps(endpoint)) == endpoint


def test_ranker_prompt_is_target_blind():
    rows = [
        {
            "question": f"Question {index}?",
            "p_yes": 0.5,
            "support_if_yes": ["A", "B"],
            "support_if_no": ["C", "D"],
        }
        for index in range(CANDIDATE_COUNT)
    ]
    messages = ranker_messages(
        [("Prefix?", True)],
        ["A", "B"],
        rows,
    )
    visible = "\n".join(message["content"] for message in messages)
    assert "unknown target" in visible.lower()
    assert '"target"' not in visible


def test_analysis_uses_truthful_realized_branch():
    record = {
        "candidates": [
            {
                "immediate_eig": 0.6,
                "expected_support_size": 20.0,
            },
            {
                "immediate_eig": 0.5,
                "expected_support_size": 21.0,
            },
            {
                "immediate_eig": 0.4,
                "expected_support_size": 19.0,
            },
        ],
        "endpoint": {
            "target_answers": [True, False, True],
            "current_contains_target": False,
            "branch_contains_target": [
                [False, True],
                [False, True],
                [True, False],
            ],
        },
        "ranker_scores": [0.1, 0.9, 0.2],
    }
    result = analyze_record(record)
    assert result["realized_candidate_coverages"] == [0.0, 1.0, 1.0]
    assert result["ranker_selected_index"] == 1
    assert result["ranker_minus_immediate_eig"] == 1.0
    assert result["ranker_minus_size"] == 0.0


def test_request_counts_match_pipeline():
    per_ranked_case = (
        1
        + SUPPORT_SAMPLES
        + 1
        + 1
        + CANDIDATE_COUNT * 2 * SUPPORT_SAMPLES
        + 1
        + 1
    )
    per_activity_case = per_ranked_case - 1
    assert EXPECTED_REQUESTS["serving_smoke"] == (
        len(SMOKE_INDICES) * per_ranked_case
    )
    assert EXPECTED_REQUESTS["activity"] == (
        len(ACTIVITY_INDICES) * per_activity_case
    )
    assert EXPECTED_REQUESTS["confirmation"] == (
        len(FORMAL_INDICES) * per_ranked_case
    )
    assert binary_entropy(0.5) == pytest.approx(0.6931471805599453)


class _FakeAdapter:
    def __init__(self, role):
        self.role = role
        self.calls = 0
        self.batches = 0

    def chat_complete_messages_batched(self, messages, **_kwargs):
        self.batches += 1
        self.calls += len(messages)
        if self.role == "generator":
            first = messages[0][-1]["content"]
            if "Propose exactly" in first:
                return [
                    json.dumps(
                        {
                            "questions": [
                                "Is it worn on the head?",
                                "Is it made mainly of leather?",
                                "Is it normally worn indoors?",
                            ]
                        }
                    )
                    for _ in messages
                ]
            if "Return exactly one finite score" in first:
                return [
                    json.dumps({"scores": [0.2, 0.8, 0.4]})
                    for _ in messages
                ]
            return [
                json.dumps(
                    {
                        "possibilities": [
                            f"garment {index}"
                            for index in range(SUPPORT_SAMPLE_SIZE)
                        ]
                    }
                )
                for _ in messages
            ]

        if self.batches == 1:
            width = 1
        elif self.batches == 2:
            width = SUPPORT_SAMPLE_SIZE
        else:
            return [
                json.dumps(
                    {
                        "target_answers": [True, False, True],
                        "current_contains_target": False,
                        "branch_contains_target": [
                            [True, False],
                            [False, True],
                            [False, False],
                        ],
                    }
                )
                for _ in messages
            ]
        return [
            json.dumps(
                {
                    "labels": [
                        [index % 2 == 0 for index in range(width)]
                        for _ in range(CANDIDATE_COUNT)
                    ]
                }
            )
            for _ in messages
        ]

    def usage_snapshot(self):
        return {
            "adapter_requests": self.calls,
            "adapter_reasoning_tokens": 0,
            "adapter_cost_usd": 0.0,
        }


def test_smoke_pipeline_executes_exact_target_blind_topology(monkeypatch):
    generator = _FakeAdapter("generator")
    judge = _FakeAdapter("judge")
    monkeypatch.setattr(
        gate_module,
        "_build_models",
        lambda _config: (generator, judge),
    )
    config = SimpleNamespace(
        batched_block_size=256,
        openrouter_max_output_tokens=4096,
    )
    payload = run_gate(
        config,
        data_path=(
            "external/paprika/llm_exploration/game/game_configs/"
            "twenty_questions.json"
        ),
        stage="serving_smoke",
    )
    assert payload["status"] == "passed"
    assert payload["usage"]["physical_requests"] == EXPECTED_REQUESTS[
        "serving_smoke"
    ]
    assert generator.batches == 4
    assert judge.batches == 3
    assert all(
        '"target"' not in json.dumps(
            ranker_messages(
                [
                    (item["question"], item["answer"] == "Yes")
                    for item in record["history"]
                ],
                record["current_support"],
                record["candidates"],
            )
        )
        for record in payload["records"]
    )
