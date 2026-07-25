from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bamboogle_no_search_memory_screen.py"
)
SPEC = importlib.util.spec_from_file_location(
    "bamboogle_no_search_memory_screen",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _task(task_id: str, answer: str) -> dict:
    return {
        "id": task_id,
        "question": "A hidden mechanics question?",
        "golden_answers": [answer],
    }


def test_parse_and_normalize_answer():
    assert MODULE.parse_answer('{"answer":"The Richmond"}') == "The Richmond"
    assert MODULE.normalize_answer("The Richmond.") == "richmond"
    assert MODULE.answer_matches("Richmond!", ["The Richmond"])


def test_prompt_never_contains_gold_answer():
    messages = MODULE.answer_messages("Where is the college?")
    serialized = str(messages)

    assert "gold" not in serialized.lower()
    assert "richmond" not in serialized.lower()
    assert "Where is the college?" in serialized


def test_saturation_gate_passes_only_for_unsaturated_samples():
    tasks = [
        _task(task_id, f"gold-{index}")
        for index, task_id in enumerate(MODULE.MECHANICS_IDS)
    ]
    answers = [
        [f"wrong-{index}"] * MODULE.SAMPLES_PER_TASK
        for index in range(len(tasks))
    ]
    answers[0] = ["gold-0"] * MODULE.SAMPLES_PER_TASK
    answers[1] = ["gold-1"] * MODULE.SAMPLES_PER_TASK
    answers[2] = ["gold-2", "wrong-2", "wrong-2", "wrong-2", "wrong-2"]

    records, summary = MODULE.summarize_records(tasks, answers)

    assert len(records) == 5
    assert summary["modal_correct_count"] == 2
    assert summary["sample_accuracy"] == 11 / 25
    assert summary["tasks_below_point_eight_gold_support"] == 3
    assert summary["passed"]


def test_saturation_gate_fails_at_four_correct_modes():
    tasks = [
        _task(task_id, f"gold-{index}")
        for index, task_id in enumerate(MODULE.MECHANICS_IDS)
    ]
    answers = [
        [f"gold-{index}"] * MODULE.SAMPLES_PER_TASK
        for index in range(len(tasks))
    ]
    answers[-1] = ["wrong"] * MODULE.SAMPLES_PER_TASK

    _, summary = MODULE.summarize_records(tasks, answers)

    assert summary["modal_correct_count"] == 4
    assert not summary["gates"]["modal_correct_at_most_3_of_5"]
    assert not summary["passed"]
