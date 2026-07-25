from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "bamboogle_semantic_bed_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "bamboogle_semantic_bed_manifest",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_split_ids_reproduces_frozen_content_blind_partition():
    splits = MODULE.split_ids([f"test_{index}" for index in range(125)])

    assert {key: len(value) for key, value in splits.items()} == {
        "mechanics": 5,
        "opportunity": 20,
        "development": 20,
        "holdout": 80,
    }
    assert {
        key: MODULE.ordered_hash(value) for key, value in splits.items()
    } == MODULE.SPLIT_HASHES
    assert len({task_id for values in splits.values() for task_id in values}) == 125


def test_split_ids_rejects_changed_source_shape():
    try:
        MODULE.split_ids([f"test_{index}" for index in range(124)])
    except ValueError as exc:
        assert "expected 125" in str(exc)
    else:
        raise AssertionError("changed Bamboogle source shape should fail")


def test_load_rows_rejects_unexpected_schema(tmp_path):
    source = tmp_path / "bad.jsonl"
    source.write_text(
        '{"id":"0","question":"Q?","golden_answers":["A"],"extra":1}\n',
        encoding="utf-8",
    )

    try:
        MODULE.load_rows(source)
    except ValueError as exc:
        assert "unexpected schema" in str(exc)
    else:
        raise AssertionError("unexpected Bamboogle fields should fail")
