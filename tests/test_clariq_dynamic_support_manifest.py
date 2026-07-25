from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any

from scripts.clariq_dynamic_support_manifest import (
    build_topic_structure,
)


class _PoisonValue(Mapping[str, Any]):
    def __getitem__(self, key: str) -> Any:
        raise AssertionError(f"evaluation value was read: {key}")

    def __iter__(self) -> Iterator[str]:
        return iter(())

    def __len__(self) -> int:
        return 0


def _metadata() -> list[dict[str, str]]:
    return [
        {
            "topic_id": "7",
            "facet_id": facet_id,
            "facet_desc": f"hidden {facet_id}",
            "initial_request": "ambiguous request",
            "question_id": question_id,
            "question": question,
        }
        for facet_id in ("f1", "f2")
        for question_id, question in (
            ("Q01", "first question"),
            ("Q02", "second question"),
        )
    ]


def _synthetic() -> dict[int, dict[str, Any]]:
    rows: dict[int, dict[str, Any]] = {}
    index = 0
    answer_by_facet = {
        "f1": {"first question": "first-a", "second question": "second-a"},
        "f2": {"first question": "first-b", "second question": "second-b"},
    }
    for facet_id in ("f1", "f2"):
        for question in ("first question", "second question"):
            rows[index] = {
                "topic_id": 7,
                "facet_id": facet_id,
                "context_id": f"empty-{facet_id}",
                "conversation_context": [],
                "question": question,
                "answer": answer_by_facet[facet_id][question],
            }
            index += 1
        for root_question in ("first question", "second question"):
            root_answer = answer_by_facet[facet_id][root_question]
            followup = (
                "second question"
                if root_question == "first question"
                else "first question"
            )
            rows[index] = {
                "topic_id": 7,
                "facet_id": facet_id,
                "context_id": f"{root_question}-{facet_id}",
                "conversation_context": [
                    {"question": root_question, "answer": root_answer}
                ],
                "question": followup,
                "answer": answer_by_facet[facet_id][followup],
            }
            index += 1
    return rows


def _evaluation() -> dict[str, dict[str, _PoisonValue]]:
    return {
        "empty-f1": {"Q01": _PoisonValue(), "Q02": _PoisonValue()},
        "empty-f2": {"Q01": _PoisonValue(), "Q02": _PoisonValue()},
        "first question-f1": {"Q02": _PoisonValue()},
        "first question-f2": {"Q02": _PoisonValue()},
        "second question-f1": {"Q01": _PoisonValue()},
        "second question-f2": {"Q01": _PoisonValue()},
    }


def test_topic_structure_is_utility_blind_and_hides_latent_profiles() -> None:
    task = build_topic_structure(
        "7",
        _metadata(),
        _synthetic(),
        _evaluation(),
    )

    assert task["root_count"] == 2
    assert task["branch_count"] == 4
    assert task["expected_model_requests"] == 5
    assert task["latent_facet_descriptions_emitted"] is False
    assert task["latent_cross_question_profiles_emitted"] is False
    assert "facets" not in task
    serialized = repr(task)
    assert "hidden f1" not in serialized
    assert "hidden f2" not in serialized


def test_topic_structure_uses_global_coded_response_alphabets() -> None:
    task = build_topic_structure(
        "7",
        _metadata(),
        _synthetic(),
        _evaluation(),
    )

    questions = {
        row["question_id"]: row for row in task["question_bank"]
    }
    assert questions["Q01"]["response_options"] == [
        {"code": "A", "answer": "first-a"},
        {"code": "B", "answer": "first-b"},
    ]
    assert questions["Q02"]["response_options"] == [
        {"code": "A", "answer": "second-a"},
        {"code": "B", "answer": "second-b"},
    ]
    first = next(root for root in task["roots"] if root["question_id"] == "Q01")
    assert [branch["response_code"] for branch in first["branches"]] == [
        "A",
        "B",
    ]
    assert all(
        branch["legal_followup_question_ids"] == ["Q02"]
        for branch in first["branches"]
    )
