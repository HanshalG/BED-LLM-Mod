from __future__ import annotations

from collections import Counter

import pytest

from scripts import ci_repair_bed_source_manifest as source


def synthetic_rows() -> list[dict[str, str]]:
    rows = []
    for repo_index in range(33):
        repo_name = f"repo-{repo_index:02d}"
        for state_index in range(4 + repo_index % 11):
            rows.append(
                {
                    "id": f"{repo_name}-state-{state_index:02d}",
                    "repo_name": repo_name,
                }
            )
    rows.extend(
        {"id": f"small-{index}", "repo_name": f"small-{index}"}
        for index in range(5)
    )
    return rows


def test_select_cohorts_is_repository_disjoint_and_capped(monkeypatch):
    monkeypatch.setattr(
        source,
        "EXPECTED_SELECTED_STATES",
        {},
    )
    rows = synthetic_rows()
    cohorts = source.select_cohorts(rows)
    assert {name: len({row["repo_name"] for row in cohort}) for name, cohort in cohorts.items()} == source.SPLIT_REPOSITORIES
    repo_sets = [set(row["repo_name"] for row in cohort) for cohort in cohorts.values()]
    assert sum(len(repo_names) for repo_names in repo_sets) == len(set().union(*repo_sets))
    available = Counter(row["repo_name"] for row in rows)
    for cohort in cohorts.values():
        selected = Counter(row["repo_name"] for row in cohort)
        assert all(4 <= available[name] for name in selected)
        assert all(count <= source.MAX_STATES for count in selected.values())


def test_selection_is_order_invariant():
    rows = synthetic_rows()
    assert source.select_cohorts(rows) == source.select_cohorts(list(reversed(rows)))


def test_public_manifest_never_serializes_ids_or_outcomes():
    manifest = source.public_manifest(synthetic_rows())
    text = source.canonical_json(manifest)
    assert "repo-00-state-00" not in text
    assert manifest["selected_ids_serialized"] is False
    assert manifest["source"]["projected_columns"] == ["id", "repo_name"]
    assert manifest["workflow_opened"] is False
    assert manifest["logs_opened"] is False
    assert manifest["diffs_opened"] is False
    assert manifest["changed_files_opened"] is False
    assert manifest["error_types_opened"] is False


@pytest.mark.parametrize("field", ["id", "repo_name"])
def test_empty_identifiers_fail_closed(field):
    rows = synthetic_rows()
    rows[0][field] = ""
    with pytest.raises(ValueError):
        source.select_cohorts(rows)


def test_duplicate_ids_fail_closed():
    rows = synthetic_rows()
    rows[1]["id"] = rows[0]["id"]
    with pytest.raises(ValueError):
        source.select_cohorts(rows)


def test_source_projection_is_metadata_only():
    assert source.PARQUET_COLUMNS == ("id", "repo_name")
    forbidden = {"workflow", "logs", "diff", "changed_files", "error_type"}
    assert not forbidden.intersection(source.PARQUET_COLUMNS)
