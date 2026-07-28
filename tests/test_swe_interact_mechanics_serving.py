from __future__ import annotations

from dataclasses import replace
import json

import pytest

from scripts.swe_interact_mechanics_serving import (
    JUDGE_KEYS,
    MechanicsTask,
    build_result,
    branch_messages,
    parse_judgement,
    parse_requirement_set,
    task_mechanics,
)


def _task() -> MechanicsTask:
    return MechanicsTask(
        family="fixture",
        task_id="fixture-task",
        persona="private persona",
        requirements=(
            "PRIVATE_REQ_ALPHA",
            "PRIVATE_REQ_BRAVO",
            "PRIVATE_REQ_CHARLIE",
            "PRIVATE_REQ_DELTA",
        ),
        root_a="ask A",
        root_b="ask B",
        review_a="surface A",
        review_b="surface B",
        root_a_expected=frozenset({"R2"}),
        root_b_expected=frozenset({"R4"}),
        review_a_expected=frozenset({"R2"}),
        review_b_expected=frozenset({"R4"}),
    )


def _labels() -> dict[str, frozenset[str]]:
    return {
        "INITIAL": frozenset({"R1"}),
        "GENERIC": frozenset(),
        "ROOT_A_1": frozenset({"R2"}),
        "ROOT_A_2": frozenset({"R2"}),
        "ROOT_B_1": frozenset({"R4"}),
        "ROOT_B_2": frozenset({"R4"}),
        "REVIEW_A": frozenset({"R2"}),
        "REVIEW_B": frozenset({"R4"}),
    }


def _usage(requests: int, *, reasoning: int = 0) -> dict[str, object]:
    return {
        "adapter_requests": requests,
        "http_attempts": requests,
        "retry_count": 0,
        "forced_exits": 0,
        "adapter_reasoning_tokens": reasoning,
        "adapter_cost_usd": 0.1,
    }


def test_parse_judgement_is_keyed_order_insensitive() -> None:
    text = "\n".join(
        (
            "ROOT_B_2|R4",
            "INITIAL|R1",
            "REVIEW_A|R2",
            "GENERIC|NONE",
            "ROOT_A_2|R2",
            "REVIEW_B|R4",
            "ROOT_B_1|R4",
            "ROOT_A_1|R2",
        )
    )
    assert parse_judgement(text, ("R1", "R2", "R3", "R4")) == _labels()


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ("R2,R2", "repeats"),
        ("R4,R2", "not canonical"),
        ("R5", "unknown"),
        ("R2,", "empty"),
    ],
)
def test_parse_requirement_set_fails_closed(value: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_requirement_set(value, ("R1", "R2", "R3", "R4"))


def test_branch_reviews_include_private_snapshot_surface() -> None:
    task = _task()
    review_a = branch_messages(task, "initial", "REVIEW_A")
    review_b = branch_messages(task, "initial", "REVIEW_B")
    assert review_a[-1]["role"] == "system"
    assert "surface A" in review_a[-1]["content"]
    assert "surface B" in review_b[-1]["content"]
    assert review_a[:-1] == review_b[:-1]


def test_task_mechanics_requires_distinct_aligned_stable_branches() -> None:
    row = task_mechanics(_task(), _labels())
    assert row["root_a_aligned"]
    assert row["root_b_aligned"]
    assert row["root_sets_distinct"]
    assert row["root_a_repeat_exact"]
    assert row["root_b_repeat_exact"]
    assert row["review_a_aligned"]
    assert row["review_b_aligned"]
    assert row["review_sets_distinct"]


def test_build_result_passes_only_complete_frozen_gate() -> None:
    task = _task()
    result = build_result(
        (task, task, task),
        {task.task_id: _labels()},
        _usage(24, reasoning=100),
        _usage(3),
        all_replies_nonempty=True,
        private_raw_sha256="abc",
        elapsed_seconds=1.0,
    )
    assert result["status"] == "passed"
    assert result["summary"]["gates"]["all_pass"]

    bad = dict(_labels())
    bad["GENERIC"] = frozenset({"R2"})
    failed = build_result(
        (task, task, task),
        {task.task_id: bad},
        _usage(24, reasoning=100),
        _usage(3),
        all_replies_nonempty=True,
        private_raw_sha256="abc",
        elapsed_seconds=1.0,
    )
    assert failed["status"] == "gate_failed"
    assert not failed["summary"]["gates"]["generic_zero_on_3_of_3"]


def test_public_result_contains_ids_not_private_requirement_text() -> None:
    task = _task()
    result = build_result(
        (task,),
        {task.task_id: _labels()},
        _usage(24, reasoning=100),
        _usage(3),
        all_replies_nonempty=True,
        private_raw_sha256="abc",
        elapsed_seconds=1.0,
    )
    public = json.dumps(result)
    assert "R2" in public
    assert "PRIVATE_REQ_ALPHA" not in public
    assert "PRIVATE_REQ_BRAVO" not in public
