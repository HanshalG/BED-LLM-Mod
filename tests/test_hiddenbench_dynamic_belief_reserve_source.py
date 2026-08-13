from __future__ import annotations

import json
from pathlib import Path

from scripts import hiddenbench_dynamic_belief_reserve_source as source


SOURCE_PATH = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")


def test_exact_source_audit_passes_and_manifest_is_private() -> None:
    manifest, result = source.audit(SOURCE_PATH)
    assert result["status"] == "source_pass"
    assert all(result["gates"].values())
    assert manifest["selection"]["cohort_count"] == 4
    assert manifest["selection"]["cohort_ordered_id_sha256"] == source.EXPECTED_PREFIX_SHA256
    def string_leaves(value):
        if isinstance(value, dict):
            for key, item in value.items():
                yield str(key)
                yield from string_leaves(item)
        elif isinstance(value, list):
            for item in value:
                yield from string_leaves(item)
        elif isinstance(value, str):
            yield value

    public_strings = set(string_leaves(manifest))
    rows = json.loads(SOURCE_PATH.read_text())
    semantic_values = []
    for row in rows:
        semantic_values.extend(
            [
                str(row["id"]),
                row["name"],
                row["description"],
                row["correct_answer"],
                *row["shared_information"],
                *row["hidden_information"],
                *row["possible_answers"],
            ]
        )
    assert not (public_strings & {value for value in semantic_values if value})


def test_source_tamper_fails(tmp_path: Path) -> None:
    target = tmp_path / "benchmark.json"
    target.write_bytes(SOURCE_PATH.read_bytes() + b" ")
    _, result = source.audit(target)
    assert result["status"] == "source_failed_closed"
    assert result["gates"]["exact_source_binding"] is False
