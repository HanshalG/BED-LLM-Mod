from __future__ import annotations

import hashlib

from scripts import bongard_openworld_partition_integrity_audit as audit


def _row(uid: str, fingerprints: list[str]) -> dict:
    return {
        "uid": uid,
        "concept": f"concept {uid}",
        "caption": f"caption {uid}",
        "imageFiles": [f"{uid}-{index}" for index in range(len(fingerprints))],
        "fingerprints": fingerprints,
    }


def test_selection_rejects_internal_and_prior_image_reuse(monkeypatch) -> None:
    monkeypatch.setattr(
        audit,
        "PARTITION_SIZES",
        {"mechanics": 1, "development": 1, "confirmation": 1},
    )
    monkeypatch.setattr(
        audit.source_audit,
        "_selection_key",
        lambda row: (row["uid"], row["uid"]),
    )
    rows = [
        _row("0000", ["a", "b"]),
        _row("0001", ["c", "c"]),
        _row("0002", ["c", "d"]),
        _row("0003", ["a", "e"]),
        _row("0004", ["f", "g"]),
        _row("0005", ["h", "i"]),
    ]
    fingerprints = {
        path: value
        for row in rows
        for path, value in zip(row["imageFiles"], row["fingerprints"])
    }

    mechanics, development, confirmation, reserve, rejected = (
        audit.select_image_unique_partitions(rows, fingerprints)
    )

    assert [row["uid"] for row in mechanics] == ["0000"]
    assert [row["uid"] for row in development] == ["0002"]
    assert [row["uid"] for row in confirmation] == ["0004"]
    assert [row["uid"] for row in reserve] == ["0001", "0003", "0005"]
    assert [item["internal_duplicate_images"] for item in rejected] == [1, 0]
    assert [item["prior_partition_duplicate_images"] for item in rejected] == [0, 1]


def test_bound_clean_partition_hashes_and_no_exact_reuse() -> None:
    rows = audit.source_audit.load_rows("val")
    fingerprints = audit.image_fingerprints(rows)
    mechanics, development, confirmation, reserve, rejected = (
        audit.select_image_unique_partitions(rows, fingerprints)
    )
    parts = {
        "mechanics": mechanics,
        "development": development,
        "confirmation": confirmation,
        "reserve": reserve,
    }

    assert {name: len(part) for name, part in parts.items()} == {
        "mechanics": 4,
        "development": 32,
        "confirmation": 64,
        "reserve": 100,
    }
    assert {
        name: hashlib.sha256(
            "\n".join(sorted(row["uid"] for row in part)).encode()
        ).hexdigest()
        for name, part in parts.items()
    } == audit.EXPECTED_UID_SHA256
    assert len(rejected) == 7

    selected_values: set[str] = set()
    for part in (mechanics, development, confirmation):
        for row in part:
            values = [fingerprints[path] for path in row["imageFiles"]]
            assert len(values) == len(set(values))
            assert not (set(values) & selected_values)
            selected_values.update(values)
