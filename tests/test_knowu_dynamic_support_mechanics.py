from __future__ import annotations

import json

import pytest

from scripts import knowu_dynamic_support_mechanics as mechanics


def _initial_payload() -> dict[str, str]:
    return {
        **{
            f"h{index}": (
                f"Computer choice {index} uses Shop{index} with OS{index} "
                f"for use case {index} within budget {index}."
            )
            for index in range(1, 5)
        },
        **{f"d{index}": f"dimension {index}" for index in range(1, 5)},
        **{
            f"q{index}": f"What is preference dimension {index}?"
            for index in range(1, 5)
        },
    }


def test_parse_initial_accepts_four_atomic_questions():
    parsed = mechanics.parse_initial(json.dumps(_initial_payload()))

    assert len(parsed.hypotheses) == 4
    assert len(parsed.dimensions) == 4
    assert len(parsed.questions) == 4


def test_parse_initial_handles_distinct_chinese_and_fullwidth_question_marks():
    payload = {
        **{
            f"h{index}": f"偏好状态{index}使用平台{index}满足预算{index}"
            for index in range(1, 5)
        },
        "d1": "预算上限",
        "d2": "操作系统",
        "d3": "购买平台",
        "d4": "主要用途",
        "q1": "你的预算上限是多少？",
        "q2": "你偏好什么操作系统？",
        "q3": "你偏好哪个购买平台？",
        "q4": "这台电脑主要用于什么场景？",
    }

    parsed = mechanics.parse_initial(json.dumps(payload, ensure_ascii=False))

    assert parsed.dimensions == (
        "预算上限",
        "操作系统",
        "购买平台",
        "主要用途",
    )


@pytest.mark.parametrize(
    "question",
    [
        "What computer and shopping platform do you want?",
        "Are you a student?",
        "What exact computer should I buy?",
        "Which brand/OS do you prefer?",
        "Which platform do you prefer",
    ],
)
def test_question_validation_rejects_compound_labels_and_terminal_requests(
    question,
):
    payload = _initial_payload()
    payload["q1"] = question

    with pytest.raises(ValueError):
        mechanics.parse_initial(json.dumps(payload))


def test_judgment_parser_enforces_threshold_consistency():
    payload = {}
    for label in ("initial", "q1", "q2", "q3", "q4"):
        payload[f"{label}_best_index"] = 0
        payload[f"{label}_best_score"] = 50
        payload[f"{label}_reason"] = "Missing a required dimension."
    parsed = mechanics.parse_judgment(json.dumps(payload))
    assert parsed.present == (False,) * 5

    payload["q1_best_index"] = 1
    with pytest.raises(ValueError):
        mechanics.parse_judgment(json.dumps(payload))


def test_refresh_message_contains_only_one_clarification():
    policy = mechanics.parse_initial(json.dumps(_initial_payload()))
    messages = mechanics.refresh_messages(
        "Choose a computer.",
        "buy_computer",
        ("A visible behavior log.",),
        policy,
        policy.questions[0],
        "I prefer a Linux laptop.",
    )
    payload = json.loads(messages[-1]["content"])

    assert set(payload["single_clarification"]) == {"question", "answer"}
    assert "truth_packet" not in payload
    assert "PRIVATE_USER_STATE" not in payload


def test_deterministic_serving_gate_has_exact_ten_requests(tmp_path):
    model = mechanics.DeterministicFixtureModel()
    result = mechanics.run_serving_gate(
        model, raw_path=tmp_path / "raw.json"
    )

    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 10
    assert result["gates"]["all_pass"]


def test_deterministic_mechanics_gate_has_isolated_branches(tmp_path):
    fixtures = [
        mechanics.WorldFixture(
            world_id=f"T{1 + index // 3}W{1 + index % 3}",
            task_id=(
                "BuyComputerPreferenceAskUserTask"
                if index < 3
                else "MattermostLeaveNoticeTask"
            ),
            task_kind="buy_computer" if index < 3 else "leave_notice",
            goal_request="Handle the task according to my preferences.",
            profile_id=("developer", "student", "user")[index % 3],
            retrieved_log_indices=tuple(range(8)),
            visible_logs=(f"Visible behavior {index}.",),
            truth_packet={"preference": f"truth {index}"},
        )
        for index in range(6)
    ]
    model = mechanics.DeterministicFixtureModel()
    result = mechanics.run_mechanics_gate(
        fixtures, model, raw_path=tmp_path / "raw.json"
    )

    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 42
    assert result["summary"]["initial_truth_missing_worlds"] == 6
    assert result["summary"]["worlds_with_truth_entry"] == 6
    assert result["gates"]["each_refresh_contains_one_question_answer_pair"]


def test_mechanics_resume_counts_cached_initial_without_reissue(tmp_path):
    fixtures = [
        mechanics.WorldFixture(
            world_id=f"T{1 + index // 3}W{1 + index % 3}",
            task_id="SyntheticTask",
            task_kind="buy_computer",
            goal_request="Handle the task according to my preferences.",
            profile_id=("developer", "student", "user")[index % 3],
            retrieved_log_indices=tuple(range(8)),
            visible_logs=(f"Visible behavior {index}.",),
            truth_packet={"preference": f"truth {index}"},
        )
        for index in range(6)
    ]
    initial_model = mechanics.DeterministicFixtureModel()
    cached = initial_model.chat_complete_messages_batched(
        [
            mechanics.initial_messages(
                fixture.goal_request,
                fixture.task_kind,
                fixture.visible_logs,
            )
            for fixture in fixtures
        ],
        temperature=0.0,
        block_size=6,
    )
    prior_usage = {
        "physical_requests": 6,
        "http_attempts": 6,
        "retry_count": 0,
        "reasoning_tokens": 0,
        "forced_exits": 0,
        "adapter_cost_usd": 0.04,
    }
    continuation_model = mechanics.DeterministicFixtureModel()

    result = mechanics.run_mechanics_gate(
        fixtures,
        continuation_model,
        raw_path=tmp_path / "raw.json",
        cached_initial_responses=cached,
        prior_usage=prior_usage,
    )

    assert continuation_model.requests == 36
    assert result["usage"]["physical_requests"] == 42
    assert result["protocol"]["cached_initial_responses"]
    assert result["status"] == "passed"
