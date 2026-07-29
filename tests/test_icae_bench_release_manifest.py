from __future__ import annotations

import pytest

from scripts.icae_bench_release_manifest import (
    EXPECTED_LANGUAGES,
    PARTITION_COUNTS_PER_LANGUAGE,
    partition_aliases,
)


def _rows() -> list[dict[str, str]]:
    return [
        {"alias": f"{language}-task-{index:02d}", "language": language}
        for language in sorted(EXPECTED_LANGUAGES)
        for index in range(40)
    ]


def test_partition_is_language_stratified_disjoint_and_exhaustive() -> None:
    rows = _rows()
    partitions = partition_aliases(rows)

    assert {name: len(value) for name, value in partitions.items()} == {
        name: count * len(EXPECTED_LANGUAGES)
        for name, count in PARTITION_COUNTS_PER_LANGUAGE.items()
    }
    aliases = [
        row["alias"]
        for partition_rows in partitions.values()
        for row in partition_rows
    ]
    assert len(aliases) == len(set(aliases)) == 480
    for partition, partition_rows in partitions.items():
        assert {
            language: sum(row["language"] == language for row in partition_rows)
            for language in EXPECTED_LANGUAGES
        } == {
            language: PARTITION_COUNTS_PER_LANGUAGE[partition]
            for language in EXPECTED_LANGUAGES
        }


def test_partition_is_reproducible_and_seed_sensitive() -> None:
    rows = _rows()
    assert partition_aliases(rows, seed=50000) == partition_aliases(
        rows, seed=50000
    )
    assert partition_aliases(rows, seed=50000) != partition_aliases(
        rows, seed=50001
    )


def test_partition_rejects_bad_language_count() -> None:
    rows = _rows()
    rows.pop()
    with pytest.raises(ValueError, match="Expected 40 aliases per language"):
        partition_aliases(rows)
