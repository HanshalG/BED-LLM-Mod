from __future__ import annotations

import pytest

from scripts.pi_bench_release_source_manifest import (
    PERSONAS,
    parse_task_paths,
    partition_tasks,
)


def _task_paths() -> list[str]:
    paths = []
    for _, (directory, prefix) in PERSONAS.items():
        for position in range(1, 21):
            paths.append(
                f"data/{directory}/tasks/"
                f"{prefix}_task_{position:03d}/task.yaml"
            )
    return paths


def test_partition_tasks_preserves_episode_order_and_is_disjoint() -> None:
    tasks = parse_task_paths(_task_paths())
    partitions = partition_tasks(tasks)

    assert {name: len(rows) for name, rows in partitions.items()} == {
        "mechanics": 20,
        "development": 30,
        "confirmation": 30,
        "retained": 20,
    }
    flattened = [row["task_id"] for rows in partitions.values() for row in rows]
    assert len(flattened) == len(set(flattened)) == 100
    assert {
        row["session_position"] for row in partitions["mechanics"]
    } == {1, 2, 3, 4}
    assert {
        row["session_position"] for row in partitions["confirmation"]
    } == {11, 12, 13, 14, 15, 16}


def test_parse_task_paths_requires_all_twenty_sessions() -> None:
    paths = _task_paths()
    paths.remove("data/researcher/tasks/researcher_task_020/task.yaml")

    with pytest.raises(ValueError, match="Expected task positions 1-20"):
        parse_task_paths(paths)
