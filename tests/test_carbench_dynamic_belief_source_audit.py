from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import carbench_dynamic_belief_source_audit as audit


def task(index: int, kind: str = "disambiguation_user") -> dict:
    internal = "color" if kind == "disambiguation_internal" else None
    user = "window" if kind == "disambiguation_user" else None
    return {
        "actions": json.dumps([{"name": "get_preferences", "kwargs": {}}]),
        "calendar_id": f"calendar-{index}",
        "context_init_config": json.dumps({"location": f"location-{index}"}),
        "disambiguation_element_internal": internal,
        "disambiguation_element_note": "one choice is ambiguous",
        "disambiguation_element_user": user,
        "instruction": f"perform task {index}",
        "persona": f"persona {index}",
        "removed_part": None,
        "task_id": f"task-{index}",
        "task_type": kind,
    }


def test_structured_nonempty_parses_json_strings() -> None:
    assert audit.structured_nonempty('{"a": 1}')
    assert audit.structured_nonempty('[{"a": 1}]')
    assert not audit.structured_nonempty("{}")
    assert not audit.structured_nonempty("[]")
    assert not audit.structured_nonempty("  ")


def test_appropriate_ambiguity_is_mutually_exclusive() -> None:
    assert audit.appropriate_ambiguity(task(0, "disambiguation_user"))
    assert audit.appropriate_ambiguity(task(1, "disambiguation_internal"))
    invalid = task(2, "disambiguation_user")
    invalid["disambiguation_element_internal"] = "also set"
    assert not audit.appropriate_ambiguity(invalid)


def test_split_train_is_deterministic_complete_and_disjoint() -> None:
    ids = [f"{index:064x}" for index in range(31)]
    first = audit.split_train(ids)
    second = audit.split_train(list(reversed(ids)))
    assert first == second
    assert {name: len(value) for name, value in first.items()} == {"mechanics": 6, "opportunity": 10, "development": 15}
    flat = [item for value in first.values() for item in value]
    assert len(flat) == len(set(flat)) == 31


def test_split_train_rejects_wrong_population() -> None:
    with pytest.raises(ValueError, match="do not cover"):
        audit.split_train(["a"] * 30)


def test_load_rows_rejects_non_object(tmp_path: Path) -> None:
    path = tmp_path / "tasks.jsonl"
    path.write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="task must be an object"):
        audit.load_rows(path)


def test_public_manifest_shape_contains_no_task_values() -> None:
    manifest = {
        "population_counts": {"train": 31, "test": 25},
        "task_type_counts": {"train": {"disambiguation_user": 19}},
        "privacy": {"source_values_serialized": False, "actions_serialized": False},
    }
    encoded = json.dumps(manifest)
    assert "perform task" not in encoded
    assert "persona 0" not in encoded
