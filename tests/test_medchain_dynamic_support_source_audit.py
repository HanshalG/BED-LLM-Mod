from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "medchain_dynamic_support_source_audit.py"
SPEC = importlib.util.spec_from_file_location("medchain_source", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_normalized_scalars_discards_empty_values() -> None:
    assert MODULE.normalized_scalars({"a": ["  Fever  ", ""], "b": None}) == ["fever"]


def test_eligible_shape_requires_private_history_and_two_exam_channels() -> None:
    row = {
        "\u3010\u75c5\u6848\u4ecb\u7ecd\u3011": {
            "\u4e3b\u8bc9": ["complaint"],
            "\u73b0\u75c5\u53f2": ["history"],
            "\u67e5\u4f53": {"\u4f53\u683c\u68c0\u67e5": "physical", "\u8f85\u52a9\u68c0\u67e5": "test"},
        }
    }
    assert MODULE.eligible_shape(row)
    row["\u3010\u75c5\u6848\u4ecb\u7ecd\u3011"]["\u67e5\u4f53"]["\u8f85\u52a9\u68c0\u67e5"] = ""
    assert not MODULE.eligible_shape(row)


def test_visible_private_separation_rejects_identical_history() -> None:
    row = {
        "\u3010\u75c5\u6848\u4ecb\u7ecd\u3011": {
            "\u4e3b\u8bc9": ["same"],
            "\u73b0\u75c5\u53f2": ["same"],
            "\u67e5\u4f53": {"\u4f53\u683c\u68c0\u67e5": "physical", "\u8f85\u52a9\u68c0\u67e5": "test"},
        },
        "tags": {"\u75c5\u79cd": ["diagnosis"]},
    }
    assert not MODULE.visible_private_separated(row)


def test_split_ids_is_deterministic_complete_and_disjoint() -> None:
    case_ids = [f"case-{index}" for index in range(250)]
    first = MODULE.split_ids(case_ids)
    second = MODULE.split_ids(list(reversed(case_ids)))
    assert first == second
    assert {name: len(values) for name, values in first.items()} == {
        "mechanics": 6,
        "opportunity": 30,
        "development": 64,
        "confirmation": 96,
        "reserve": 54,
    }
    flat = [item for values in first.values() for item in values]
    assert len(flat) == len(set(flat)) == len(case_ids)


def test_public_artifacts_cannot_contain_source_values() -> None:
    output = ROOT / "results" / "nonmyopic" / "medchain_dynamic_support_source"
    for path in (output / "MANIFEST.json", output / "SOURCE_AUDIT.json"):
        if not path.exists():
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        privacy = value["privacy"]
        assert not privacy["individual_case_ids_serialized"]
        assert not privacy["source_values_serialized"]
        assert not privacy["diagnoses_serialized"]
        assert not privacy["endpoints_opened"]
