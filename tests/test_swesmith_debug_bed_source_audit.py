from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "swesmith_debug_bed_source_audit.py"
SPEC = importlib.util.spec_from_file_location("swesmith_source", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_simple_yaml_lists_extracts_only_bound_lists(tmp_path: Path) -> None:
    path = tmp_path / "splits.yaml"
    path.write_text("excluded: [\n  'a', # note\n  'b',\n  ]\ntrain-789: [\n  'c', 'd'\n  ]\n", encoding="utf-8")
    assert MODULE.parse_simple_yaml_lists(path) == {"excluded": ["a", "b"], "train-789": ["c", "d"]}


def test_ordered_hash_is_order_sensitive_and_stable() -> None:
    assert MODULE.ordered_hash(["a", "b"]) == MODULE.ordered_hash(["a", "b"])
    assert MODULE.ordered_hash(["a", "b"]) != MODULE.ordered_hash(["b", "a"])


def test_all_shards_are_content_addressed() -> None:
    assert len(MODULE.SHARDS) == 11
    assert all(size > 0 and len(oid) == 64 for size, oid in MODULE.SHARDS.values())


def test_public_artifacts_keep_policy_sensitive_values_closed() -> None:
    output = ROOT / "results" / "nonmyopic" / "swesmith_debug_bed_source"
    for path in (output / "MANIFEST.json", output / "SOURCE_AUDIT.json"):
        if not path.exists():
            continue
        privacy = json.loads(path.read_text(encoding="utf-8"))["privacy"]
        assert not privacy["individual_instance_ids_serialized"]
        assert not privacy["problem_statements_opened"]
        assert not privacy["patches_opened"]
        assert not privacy["test_names_opened"]
        assert not privacy["test_outputs_opened"]
        assert not privacy["repository_source_opened"]
        assert not privacy["endpoints_opened"]
