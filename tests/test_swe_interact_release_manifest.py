from __future__ import annotations

import pytest

from scripts.swe_interact_release_manifest import (
    FAMILY_PREFIXES,
    PARTITION_COUNTS,
    REQUIRED_RELATIVE_PATHS,
    parse_task_paths,
    partition_tasks,
)


def _paths() -> list[str]:
    paths: list[str] = []
    for family, prefix in FAMILY_PREFIXES.items():
        for index in range(25):
            task_id = f"{prefix}{family}-task-{index:02d}"
            base = f"data/multiturn/{task_id}"
            paths.extend(
                f"{base}/{relative}" for relative in REQUIRED_RELATIVE_PATHS
            )
            paths.append(f"data/singleturn/{task_id}/task.toml")
    return paths


def test_partition_is_stratified_disjoint_and_exhaustive() -> None:
    by_family = parse_task_paths(_paths())
    partitions = partition_tasks(by_family)

    assert {name: len(rows) for name, rows in partitions.items()} == {
        name: count * len(FAMILY_PREFIXES)
        for name, count in PARTITION_COUNTS.items()
    }
    task_ids = [
        row["task_id"] for rows in partitions.values() for row in rows
    ]
    assert len(task_ids) == len(set(task_ids)) == 75
    for partition, rows in partitions.items():
        family_counts = {
            family: sum(row["family"] == family for row in rows)
            for family in FAMILY_PREFIXES
        }
        assert family_counts == {
            family: PARTITION_COUNTS[partition]
            for family in FAMILY_PREFIXES
        }


def test_partition_is_reproducible_and_seed_sensitive() -> None:
    by_family = parse_task_paths(_paths())
    first = partition_tasks(by_family, seed=24423)
    second = partition_tasks(by_family, seed=24423)
    alternate = partition_tasks(by_family, seed=24424)

    assert first == second
    assert first != alternate


def test_parse_requires_complete_task_package() -> None:
    paths = _paths()
    missing = next(path for path in paths if path.endswith("persona.md"))
    paths.remove(missing)

    with pytest.raises(ValueError, match="missing required release paths"):
        parse_task_paths(paths)


def test_parse_requires_paired_singleturn_task() -> None:
    paths = _paths()
    missing = next(
        path
        for path in paths
        if path.startswith("data/singleturn/")
    )
    paths.remove(missing)

    with pytest.raises(ValueError, match="missing paired single-turn"):
        parse_task_paths(paths)
