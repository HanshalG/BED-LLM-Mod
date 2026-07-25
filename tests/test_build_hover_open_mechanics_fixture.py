from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "build_hover_open_mechanics_fixture.py"
)
SPEC = importlib.util.spec_from_file_location(
    "build_hover_open_mechanics_fixture",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_frozen_mechanics_tasks_are_two_distinct_open_rows():
    assert len(MODULE.TASK_IDS) == 2
    assert len(set(MODULE.TASK_IDS)) == 2


def test_fixture_builder_source_never_names_endpoint_fields():
    source = SCRIPT_PATH.read_text(encoding="utf-8")
    emitted_block = source.split("tasks.append(", 1)[1].split(
        "return {",
        1,
    )[0]

    assert '"label"' not in emitted_block
    assert '"supporting_facts"' not in emitted_block
    assert '"evidence"' not in emitted_block
