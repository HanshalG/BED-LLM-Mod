from __future__ import annotations

from io import BytesIO
from zipfile import ZipFile

from PIL import Image

from scripts import bongard_openworld_image_integrity_audit as audit
from scripts import bongard_openworld_source_protocol_audit as source_audit


def _image_bytes(image_format: str = "PNG") -> bytes:
    output = BytesIO()
    Image.new("RGB", (17, 13), color=(10, 20, 30)).save(
        output, format=image_format
    )
    return output.getvalue()


def _zip_infos(names: list[str]):
    output = BytesIO()
    with ZipFile(output, "w") as archive:
        archive.writestr("images/", b"")
        for name in names:
            archive.writestr(name, _image_bytes())
    output.seek(0)
    archive = ZipFile(output)
    return output, archive


def test_decode_image_verifies_and_loads_pixels() -> None:
    result = audit.decode_image(_image_bytes())
    assert result["format"] == "PNG"
    assert result["mode"] == "RGB"
    assert result["width"] == 17
    assert result["height"] == 13
    assert result["bytes"] > 0
    assert len(result["sha256"]) == 64


def test_member_inspection_detects_exact_and_extra_sets(monkeypatch) -> None:
    monkeypatch.setattr(audit, "EXPECTED_IMAGE_MEMBERS", 2)
    monkeypatch.setattr(audit, "EXPECTED_DIRECTORY_MEMBERS", 1)
    names = ["images/0001/a.jpg", "images/0001/b.jpg"]
    buffer, archive = _zip_infos(names)
    try:
        stats, gates = audit.inspect_members(archive.infolist(), set(names))
        assert stats["files"] == 2
        assert all(gates.values())
        _, extra_gates = audit.inspect_members(
            archive.infolist(), {"images/0001/a.jpg"}
        )
        assert not extra_gates[
            "archive_file_set_exactly_matches_bound_metadata"
        ]
    finally:
        archive.close()
        buffer.close()


def test_safe_member_name_rejects_traversal_and_backslashes() -> None:
    assert audit.safe_member_name("images/0001/image.jpg")
    assert not audit.safe_member_name("../image.jpg")
    assert not audit.safe_member_name("/absolute/image.jpg")
    assert not audit.safe_member_name("images\\image.jpg")


def test_public_payload_scan_rejects_truth_fields_paths_and_values() -> None:
    assert audit.public_payload_errors(
        {"task_id": "task-123", "images": [{"image_id": "image-01"}]},
        forbidden_values=["secret concept"],
    ) == []
    errors = audit.public_payload_errors(
        {
            "uid": "0001",
            "value": "secret concept",
            "other": "images/0001/pos__0__example.jpg",
        },
        forbidden_values=["secret concept"],
    )
    assert "forbidden_key:uid" in errors
    assert "semantic_truth_value" in errors
    assert "label_bearing_filename" in errors
    assert "source_image_path" in errors


def test_layout_refactor_preserves_frozen_protocol() -> None:
    rows = source_audit.load_rows("val")
    mechanics, _, _, _ = source_audit.split_validation_rows(rows)
    for row in mechanics:
        layout = source_audit._task_layout(row)
        payload = source_audit.task_protocol(row)
        assert payload["task_id"] == layout["task_id"]
        assert {
            item["image_id"] for item in payload["initial"]
        } == {
            layout["opaque_by_position"][position]
            for position in layout["initial_positions"]
        }
        assert source_audit.hidden_state_boundary_holds(row)


def test_bound_archive_is_present_with_first_complete_hash() -> None:
    assert audit.ARCHIVE_PATH.stat().st_size == audit.ARCHIVE_SIZE
    assert audit.sha256_file(audit.ARCHIVE_PATH) == audit.ARCHIVE_SHA256
