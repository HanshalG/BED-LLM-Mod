from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "symptomcheck_dynamic_support_source_audit.py"
SPEC = importlib.util.spec_from_file_location("symptomcheck_source", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_structurally_nonempty_accepts_optional_empty_siblings() -> None:
    assert MODULE.structurally_nonempty({"present": "fact", "optional": ""})
    assert not MODULE.structurally_nonempty({"optional": ""})


def test_normalized_is_order_stable() -> None:
    assert MODULE.normalized({"b": "Two", "a": " One "}) == MODULE.normalized({"a": "One", "b": "two"})


def test_split_is_deterministic_complete_and_disjoint() -> None:
    case_ids = [f"case-{index}" for index in range(400)]
    first = MODULE.split_ids(case_ids)
    assert first == MODULE.split_ids(list(reversed(case_ids)))
    assert {name: len(values) for name, values in first.items()} == {
        "mechanics": 6,
        "opportunity": 30,
        "development": 64,
        "confirmation": 96,
        "reserve": 204,
    }
    flat = [item for values in first.values() for item in values]
    assert len(flat) == len(set(flat)) == 400


def test_public_artifacts_do_not_expose_cases() -> None:
    output = ROOT / "results" / "nonmyopic" / "symptomcheck_dynamic_support_source"
    for path in (output / "MANIFEST.json", output / "SOURCE_AUDIT.json"):
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        privacy = value["privacy"]
        assert not privacy["individual_case_ids_serialized"]
        assert not privacy["source_values_serialized"]
        assert not privacy["diagnoses_serialized"]
        assert not privacy["dialogues_serialized"]
        assert not privacy["endpoints_opened"]
