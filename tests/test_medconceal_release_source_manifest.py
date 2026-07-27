from __future__ import annotations

import json

import pytest

from scripts.medconceal_release_source_manifest import (
    PARTITION_PREFIX,
    load_case_ids,
    partition_case_ids,
)


def test_partition_case_ids_is_complete_disjoint_and_deterministic() -> None:
    case_ids = [f"patient_{index}" for index in range(300)]

    first = partition_case_ids(case_ids)
    second = partition_case_ids(reversed(case_ids))

    assert first == second
    assert {name: len(ids) for name, ids in first.items()} == {
        "mechanics": 20,
        "development": 80,
        "confirmation": 100,
        "retained": 100,
    }
    flattened = [case_id for ids in first.values() for case_id in ids]
    assert len(flattened) == len(set(flattened)) == 300
    assert set(flattened) == set(case_ids)
    assert PARTITION_PREFIX == "medconceal-24421:"


def test_load_case_ids_rejects_duplicates(tmp_path) -> None:
    path = tmp_path / "cases.jsonl"
    rows = [{"case_id": f"patient_{index}"} for index in range(299)]
    rows.append({"case_id": "patient_0"})
    path.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="not unique"):
        load_case_ids(path)
