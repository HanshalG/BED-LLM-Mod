from __future__ import annotations

import importlib.util
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "hover_semantic_bed_manifest.py"
)
SPEC = importlib.util.spec_from_file_location(
    "hover_semantic_bed_manifest",
    SCRIPT_PATH,
)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _metadata_rows() -> list[dict[str, object]]:
    return [
        {"uid": f"h3-{index:04d}", "num_hops": 3}
        for index in range(240)
    ] + [
        {"uid": f"h4-{index:04d}", "num_hops": 4}
        for index in range(240)
    ]


def test_split_ids_is_deterministic_and_hop_stratified():
    first = MODULE.split_ids(
        _metadata_rows(),
        verify_frozen_hashes=False,
    )
    second = MODULE.split_ids(
        list(reversed(_metadata_rows())),
        verify_frozen_hashes=False,
    )

    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "mechanics": 6,
        "opportunity": 400,
        "development": 40,
        "holdout": 34,
    }
    assert len({task_id for values in first.values() for task_id in values}) == 480
    for name, size_per_hop in {
        "mechanics": 3,
        "opportunity": 200,
        "development": 20,
        "holdout": 17,
    }.items():
        assert sum(task_id.startswith("h3-") for task_id in first[name]) == size_per_hop
        assert sum(task_id.startswith("h4-") for task_id in first[name]) == size_per_hop


def test_validate_rows_rejects_unexpected_schema():
    rows = [
        {
            "uid": f"id-{index}",
            "num_hops": 3 if index < 1_835 else 4,
            "claim": "sealed",
            "hpqa_id": "sealed",
            "label": "sealed",
            "supporting_facts": [],
            "extra": "not allowed",
        }
        for index in range(4_000)
    ]

    try:
        MODULE.validate_rows(rows)
    except ValueError as exc:
        assert "schema changed" in str(exc)
    else:
        raise AssertionError("unexpected HoVer fields should fail")


def test_ordered_hash_is_order_sensitive():
    assert MODULE.ordered_hash(["a", "b"]) != MODULE.ordered_hash(["b", "a"])
