from __future__ import annotations

from scripts import guesswhat_semantic_bed_source_audit as audit


def _object(index: int, category: str = "person") -> dict:
    return {
        "category": category,
        "area": 100.0,
        "iscrowd": False,
        "object_id": index,
        "bbox": [float(index), 2.0, 10.0, 20.0],
        "category_id": 1,
        "segment": [],
    }


def _row(index: int = 1, picture_id: int | None = None) -> dict:
    objects = {
        str(object_index): _object(
            object_index,
            "person" if object_index <= 2 else f"category-{object_index}",
        )
        for object_index in range(1, 6)
    }
    return {
        "status": "success",
        "picture": {
            "file_name": f"COCO_val2014_{index:012d}.jpg",
            "flickr_url": "http://example.test/image.jpg",
            "width": 640,
            "height": 480,
            "coco_url": "http://mscoco.org/images/1",
        },
        "picture_id": picture_id if picture_id is not None else index,
        "qas": [
            {"q": f"question {qa_index}?", "a": "Yes", "id": qa_index}
            for qa_index in range(4)
        ],
        "questioner_id": 1,
        "timestamp": "2016-01-01 00:00:00",
        "object_id": 1,
        "dialogue_id": index,
        "objects": objects,
    }


def test_eligibility_accepts_ambiguous_successful_game() -> None:
    assert audit.eligibility_errors(_row()) == []


def test_eligibility_requires_same_category_ambiguity_and_target() -> None:
    row = _row()
    for index, obj in enumerate(row["objects"].values(), start=1):
        obj["category"] = f"unique-{index}"
    assert "no_same_category_ambiguity" in audit.eligibility_errors(row)

    missing = _row()
    missing["object_id"] = 999
    assert "target_missing" in audit.eligibility_errors(missing)


def test_split_is_deterministic_and_deduplicates_images(monkeypatch) -> None:
    monkeypatch.setattr(audit, "SERVING_COUNT", 2)
    monkeypatch.setattr(audit, "DEVELOPMENT_COUNT", 3)
    monkeypatch.setattr(audit, "HOLDOUT_COUNT", 4)
    rows = [_row(index) for index in range(1, 12)]
    rows.append(_row(100, picture_id=1))

    first = audit.split_rows(rows)
    second = audit.split_rows(list(reversed(rows)))
    assert [
        [audit.row_id(row) for row in split] for split in first
    ] == [
        [audit.row_id(row) for row in split] for split in second
    ]
    serving, development, holdout, unused = first
    assert [len(part) for part in first] == [2, 3, 4, 2]
    selected = serving + development + holdout
    assert len({audit.image_id(row) for row in selected}) == len(selected)
    assert len(unused) == 2


def test_candidate_payload_hides_target_answers_and_categories() -> None:
    row = _row()
    payload = audit.candidate_payload(row)
    serialized = audit.canonical_json(payload)
    assert audit.hidden_state_boundary_holds(row)
    assert '"qas":' not in serialized
    assert '"category":' not in serialized
    assert '"target_object_id":' not in serialized
    assert all(
        set(obj) == {"candidate_index", "object_id", "bbox"}
        for obj in payload["candidate_objects"]
    )


def test_public_row_contains_only_nonendpoint_metadata() -> None:
    public = audit.selected_public_row(_row())
    serialized = audit.canonical_json(public)
    assert set(public) == {
        "dialogue_id",
        "picture_id",
        "row_sha256",
        "image_file_name",
        "image_url",
        "object_count",
        "same_category_group_count",
        "human_question_count",
    }
    assert '"object_id":' not in serialized
    assert '"qas":' not in serialized


def test_bound_source_and_dataset() -> None:
    assert audit.verify_source()["data_sha256"] == audit.DATA_SHA256
    rows = audit.load_rows()
    assert len(rows) == audit.EXPECTED_ROWS
    assert all(
        str(row["object_id"]) in row["objects"]
        for row in rows
    )
