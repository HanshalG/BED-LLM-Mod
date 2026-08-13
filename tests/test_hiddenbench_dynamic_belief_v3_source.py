from __future__ import annotations

import json
from pathlib import Path

from scripts import hiddenbench_dynamic_belief_v3_source as source


SOURCE = Path("/tmp/bed-source-audits/hiddenbench/data/benchmark.json")


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


def test_v3_source_passes_and_is_value_private() -> None:
    manifest, result = source.audit(SOURCE)
    assert result["status"] == "source_pass"
    assert all(result["gates"].values())
    assert manifest["selection"]["cohort_ordered_id_sha256"] == source.EXPECTED_COHORT_SHA256
    public_strings = set(string_leaves(manifest))
    rows = json.loads(SOURCE.read_text())
    private_strings = set()
    for row in rows:
        private_strings.update(
            [str(row["id"]), row["name"], row["description"], row["correct_answer"], *row["shared_information"], *row["hidden_information"], *row["possible_answers"]]
        )
    assert not (public_strings & private_strings)


def test_v3_source_tamper_fails(tmp_path: Path) -> None:
    target = tmp_path / "benchmark.json"
    target.write_bytes(SOURCE.read_bytes() + b" ")
    _, result = source.audit(target)
    assert result["status"] == "source_failed_closed"
