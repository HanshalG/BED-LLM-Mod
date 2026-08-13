from __future__ import annotations

import csv
from pathlib import Path

from scripts import ambig_swe_source_audit as audit


def write_csv(path: Path, ids: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["instance_id", "problem_statement", "endpoint"])
        for task_id in ids:
            writer.writerow([task_id, f"private text for {task_id}", "sealed"])


def make_csvs(root: Path, *, interaction_ids: list[str] | None = None) -> list[str]:
    ids = [f"task-{index:03d}" for index in range(220)]
    write_csv(root / audit.CSV_PATHS["fully_specified"], ids)
    write_csv(root / audit.CSV_PATHS["underspecified"], ids)
    write_csv(root / audit.CSV_PATHS["interaction"], interaction_ids or ids)
    return ids


def test_inventory_hashes_ids_without_serializing_task_text(tmp_path: Path) -> None:
    ids = make_csvs(tmp_path)
    result = audit.inventory_csvs(tmp_path)

    assert result["all_three_views_have_identical_id_sets"]
    assert result["aligned_task_count"] == len(ids)
    assert result["split_counts"] == {
        "mechanics": 6,
        "opportunity": 40,
        "development": 64,
        "confirmation": 96,
        "retained": 14,
    }
    assert "private text" not in audit.canonical_json(result)
    assert "task-000" not in audit.canonical_json(result)


def test_mismatched_interaction_population_fails_first_source_gate(tmp_path: Path) -> None:
    ids = [f"task-{index:03d}" for index in range(220)]
    make_csvs(tmp_path, interaction_ids=ids[:-3])
    result = audit.inventory_csvs(tmp_path)

    assert not result["all_three_views_have_identical_id_sets"]
    assert result["row_counts"] == {
        "fully_specified": 220,
        "underspecified": 220,
        "interaction": 217,
    }
    assert result["aligned_task_count"] == 217


def test_split_is_reproducible_and_disjoint(tmp_path: Path) -> None:
    ids = make_csvs(tmp_path)
    first = audit.split_ids(set(ids))
    second = audit.split_ids(set(reversed(ids)))

    assert first == second
    flattened = [task_id for split in first.values() for task_id in split]
    assert len(flattened) == len(set(flattened)) == len(ids)
