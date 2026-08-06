from __future__ import annotations

from scripts import bongard_openworld_source_protocol_audit as audit


def _row(uid: str = "0123") -> dict:
    return {
        "uid": uid,
        "commonSense": "0",
        "concept": "hidden visual concept",
        "caption": "A hidden visual concept.",
        "imageFiles": [
            f"images/{uid}/pos__{index}__source-{index}.jpg"
            for index in range(7)
        ]
        + [
            f"images/{uid}/neg__{index}__source-{index}.jpg"
            for index in range(7)
        ],
    }


def test_row_errors_accepts_exact_balanced_task() -> None:
    assert audit.row_errors(_row()) == []


def test_row_errors_rejects_reordered_label_and_uid_leak() -> None:
    row = _row()
    row["imageFiles"][0], row["imageFiles"][7] = (
        row["imageFiles"][7],
        row["imageFiles"][0],
    )
    errors = audit.row_errors(row)
    assert "image_uid_mismatch" not in errors
    assert "image_label_order_mismatch" in errors

    wrong_uid = _row()
    wrong_uid["imageFiles"][3] = wrong_uid["imageFiles"][3].replace(
        "0123", "9999"
    )
    assert "image_uid_mismatch" in audit.row_errors(wrong_uid)


def test_task_protocol_is_deterministic_balanced_and_opaque() -> None:
    row = _row()
    first = audit.task_protocol(row)
    second = audit.task_protocol(row)
    assert first == second
    assert len(first["initial"]) == 4
    assert sorted(item["label"] for item in first["initial"]) == [
        "negative",
        "negative",
        "positive",
        "positive",
    ]
    assert len(first["candidates"]) == 8
    assert len(first["endpoints"]) == 2
    assert first["query_budget"] == 2
    assert all("label" not in item for item in first["candidates"])
    assert all("label" not in item for item in first["endpoints"])
    assert audit.hidden_state_boundary_holds(row)

    serialized = audit.canonical_json(first)
    assert row["uid"] not in serialized
    assert row["concept"] not in serialized
    assert row["caption"] not in serialized
    assert all(path not in serialized for path in row["imageFiles"])


def test_validation_split_is_deterministic_disjoint_and_exact(monkeypatch) -> None:
    monkeypatch.setattr(audit, "MECHANICS_TASKS", 2)
    monkeypatch.setattr(audit, "DEVELOPMENT_TASKS", 3)
    monkeypatch.setattr(audit, "CONFIRMATION_TASKS", 4)
    rows = [_row(f"{index:04d}") for index in range(12)]

    first = audit.split_validation_rows(rows)
    second = audit.split_validation_rows(list(reversed(rows)))
    assert [[row["uid"] for row in part] for part in first] == [
        [row["uid"] for row in part] for part in second
    ]
    assert [len(part) for part in first] == [2, 3, 4, 3]
    selected = [row["uid"] for part in first for row in part]
    assert len(selected) == len(set(selected))


def test_backup_range_assessment_requires_size_and_zip64_records() -> None:
    tail = b"prefix-PK\x06\x06-middle-PK\x06\x07-tail-PK\x05\x06"
    valid = audit.assess_backup_ranges(
        first_bytes=b"PK\x03\x04",
        tail_bytes=tail,
        total_size=audit.BACKUP_EXPECTED_SIZE,
    )
    assert all(
        value
        for key, value in valid.items()
        if key != "total_size_bytes"
    )
    invalid = audit.assess_backup_ranges(
        first_bytes=b"nope",
        tail_bytes=b"nope",
        total_size=1,
    )
    assert not invalid["expected_size_matches"]
    assert not invalid["zip_local_header_present"]
    assert not invalid["zip_end_record_present"]


def test_bound_official_source_and_partition() -> None:
    source = audit.verify_source()
    assert source["commit"] == audit.SOURCE_COMMIT
    splits = {
        split: audit.load_rows(split) for split in ("train", "val", "test")
    }
    assert {split: len(rows) for split, rows in splits.items()} == (
        audit.EXPECTED_SPLIT_SIZES
    )
    assert not any(audit.row_errors(row) for rows in splits.values() for row in rows)
