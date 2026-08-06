#!/usr/bin/env python3
"""Bind and verify the complete Bongard-OpenWorld image archive."""

from __future__ import annotations

import argparse
from collections import Counter
from io import BytesIO
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
from typing import Any, Iterable, Sequence
from zipfile import ZipFile, ZipInfo

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_source_protocol_audit as source_audit
from scripts.discoverphysics_oscillator_belief_smoke import checkpoint


SCHEMA_VERSION = 1
INTERFACE_VERSION = "bongard-openworld-image-integrity-audit-1"
ARCHIVE_PATH = source_audit.DATA_ROOT / "images.zip"
ARCHIVE_SIZE = 5_124_375_111
ARCHIVE_SHA256 = (
    "5ab838cffd8c1be6080e232b9fa4d9ca824d213371625abe59d984745f694d25"
)
SOURCE_PROTOCOL_MANIFEST = (
    REPO_ROOT
    / "results/nonmyopic/bongard_openworld_source_protocol_audit/"
    "bongard-openworld-source-protocol-audit-20260806/MANIFEST.json"
)
SOURCE_PROTOCOL_MANIFEST_SHA256 = (
    "7acd3cc9abd24fb60f7da98710aa2ed89b75d9c137ada46380f258d16380e763"
)
EXPECTED_IMAGE_MEMBERS = 14_140
EXPECTED_DIRECTORY_MEMBERS = 1_011
EXPECTED_MECHANICS_TASKS = 4
EXPECTED_MECHANICS_IMAGES = 56
ACCEPTED_IMAGE_FORMATS = {"JPEG", "PNG", "GIF", "WEBP", "BMP", "TIFF"}
FORBIDDEN_PUBLIC_KEYS = {
    "uid",
    "source_uid",
    "path",
    "member",
    "filename",
    "label",
    "concept",
    "caption",
    "source_position",
}


def canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_bound_sources() -> dict[str, Any]:
    source = source_audit.verify_source()
    manifest_sha256 = sha256_file(SOURCE_PROTOCOL_MANIFEST)
    if manifest_sha256 != SOURCE_PROTOCOL_MANIFEST_SHA256:
        raise ValueError(
            "source/protocol manifest changed: "
            f"expected {SOURCE_PROTOCOL_MANIFEST_SHA256}, "
            f"observed {manifest_sha256}"
        )
    manifest = json.loads(SOURCE_PROTOCOL_MANIFEST.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "source_protocol_pass"
        or manifest.get("all_gates_pass") is not True
        or manifest.get("authorizes_paid_calls") is not False
    ):
        raise ValueError("source/protocol manifest is not the frozen clean pass")
    return {
        "commit": source["commit"],
        "tree": source["tree"],
        "metadata_sha256": source["data_sha256"],
        "source_protocol_manifest_sha256": manifest_sha256,
    }


def expected_member_names() -> set[str]:
    rows = source_audit.load_rows("all")
    if any(source_audit.row_errors(row) for row in rows):
        raise ValueError("bound metadata no longer has the frozen image structure")
    return {path for row in rows for path in row["imageFiles"]}


def safe_member_name(name: str) -> bool:
    path = PurePosixPath(name)
    return (
        bool(name)
        and not name.startswith("/")
        and "\\" not in name
        and ".." not in path.parts
    )


def member_set_sha256(names: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(sorted(names)).encode()).hexdigest()


def inspect_members(
    infos: Sequence[ZipInfo], expected: set[str]
) -> tuple[dict[str, Any], dict[str, bool]]:
    names = [info.filename for info in infos]
    files = [info for info in infos if not info.is_dir()]
    file_names = {info.filename for info in files}
    missing = expected - file_names
    extra = file_names - expected
    stats = {
        "entries": len(infos),
        "files": len(files),
        "directories": len(infos) - len(files),
        "compressed_bytes": sum(info.compress_size for info in files),
        "uncompressed_bytes": sum(info.file_size for info in files),
        "member_set_sha256": member_set_sha256(file_names),
        "extension_counts": dict(
            sorted(
                Counter(
                    PurePosixPath(info.filename).suffix.lower()
                    for info in files
                ).items()
            )
        ),
        "missing_expected_members": len(missing),
        "extra_file_members": len(extra),
        "duplicate_entries": len(names) - len(set(names)),
        "zero_size_files": sum(info.file_size <= 0 for info in files),
    }
    gates = {
        "exact_expected_member_and_directory_counts": (
            len(files) == EXPECTED_IMAGE_MEMBERS
            and len(infos) - len(files) == EXPECTED_DIRECTORY_MEMBERS
        ),
        "member_names_are_unique_and_path_safe": (
            len(names) == len(set(names))
            and all(safe_member_name(name) for name in names)
        ),
        "archive_file_set_exactly_matches_bound_metadata": (
            not missing and not extra and len(expected) == EXPECTED_IMAGE_MEMBERS
        ),
        "all_expected_members_have_positive_sizes": all(
            info.file_size > 0 and info.compress_size > 0 for info in files
        ),
    }
    return stats, gates


def decode_image(data: bytes) -> dict[str, Any]:
    digest = hashlib.sha256(data).hexdigest()
    with Image.open(BytesIO(data)) as image:
        image.verify()
    with Image.open(BytesIO(data)) as image:
        image.load()
        width, height = image.size
        image_format = image.format
        mode = image.mode
    return {
        "bytes": len(data),
        "sha256": digest,
        "width": width,
        "height": height,
        "format": image_format,
        "mode": mode,
    }


def mechanics_rows() -> list[dict[str, Any]]:
    mechanics, _, _, _ = source_audit.split_validation_rows(
        source_audit.load_rows("val")
    )
    return mechanics


def decode_mechanics(archive: ZipFile) -> list[dict[str, Any]]:
    tasks = []
    for row in mechanics_rows():
        layout = source_audit._task_layout(row)
        role_by_position = {
            position: "initial" for position in layout["initial_positions"]
        }
        role_by_position.update(
            {
                position: "candidate"
                for position in layout["candidate_positions"]
            }
        )
        role_by_position.update(
            {
                position: "endpoint"
                for position in layout["endpoint_positions"]
            }
        )
        images = []
        for position, member_name in enumerate(row["imageFiles"]):
            record = decode_image(archive.read(member_name))
            record.update(
                {
                    "image_id": layout["opaque_by_position"][position],
                    "role": role_by_position[position],
                }
            )
            images.append(record)
        tasks.append(
            {
                "task_id": layout["task_id"],
                "images": sorted(images, key=lambda item: item["image_id"]),
            }
        )
    return sorted(tasks, key=lambda item: item["task_id"])


def public_payload_errors(
    value: Any, *, forbidden_values: Sequence[str]
) -> list[str]:
    errors: list[str] = []

    def visit(node: Any) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                if key in FORBIDDEN_PUBLIC_KEYS:
                    errors.append(f"forbidden_key:{key}")
                visit(child)
        elif isinstance(node, list):
            for child in node:
                visit(child)
        elif isinstance(node, str):
            lowered = node.lower()
            if "pos__" in lowered or "neg__" in lowered:
                errors.append("label_bearing_filename")
            if resemblessource_path(node):
                errors.append("source_image_path")
            for forbidden in forbidden_values:
                if forbidden and forbidden in node:
                    errors.append("semantic_truth_value")
                    break

    visit(value)
    return sorted(set(errors))


def resemblessource_path(value: str) -> bool:
    return bool(
        value.startswith("images/")
        or "/images/" in value
        or re_fullmatch_image_path(value)
    )


def re_fullmatch_image_path(value: str) -> bool:
    return source_audit.IMAGE_RE.fullmatch(value) is not None


def mechanics_summary(tasks: Sequence[dict[str, Any]]) -> dict[str, Any]:
    images = [image for task in tasks for image in task["images"]]
    return {
        "tasks": len(tasks),
        "images": len(images),
        "role_counts": dict(sorted(Counter(image["role"] for image in images).items())),
        "format_counts": dict(
            sorted(Counter(image["format"] for image in images).items())
        ),
        "mode_counts": dict(sorted(Counter(image["mode"] for image in images).items())),
        "total_bytes": sum(image["bytes"] for image in images),
        "min_width": min(image["width"] for image in images),
        "max_width": max(image["width"] for image in images),
        "min_height": min(image["height"] for image in images),
        "max_height": max(image["height"] for image in images),
    }


def run_audit(*, output_path: Path) -> dict[str, Any]:
    bound_source = verify_bound_sources()
    archive_size = ARCHIVE_PATH.stat().st_size
    archive_sha256 = sha256_file(ARCHIVE_PATH)
    expected = expected_member_names()

    with ZipFile(ARCHIVE_PATH) as archive:
        member_stats, member_gates = inspect_members(archive.infolist(), expected)
        bad_crc_member = archive.testzip()
        mechanics = decode_mechanics(archive)

    mechanics_stats = mechanics_summary(mechanics)
    source_rows = mechanics_rows()
    forbidden_values = [
        value
        for row in source_rows
        for value in (row["concept"], row["caption"], *row["imageFiles"])
    ]
    public_errors = public_payload_errors(
        {"mechanics": mechanics, "summary": mechanics_stats},
        forbidden_values=forbidden_values,
    )
    decoded_images = [
        image for task in mechanics for image in task["images"]
    ]
    mechanics_task_ids = [task["task_id"] for task in mechanics]
    mechanics_image_ids = [
        (task["task_id"], image["image_id"])
        for task in mechanics
        for image in task["images"]
    ]

    gates = {
        "bound_source_and_protocol_manifest_match": bool(bound_source),
        "archive_size_and_sha256_match_first_complete_binding": (
            archive_size == ARCHIVE_SIZE and archive_sha256 == ARCHIVE_SHA256
        ),
        **member_gates,
        "full_archive_crc_scan_passes": bad_crc_member is None,
        "mechanics_partition_has_exact_unique_task_and_image_counts": (
            len(mechanics) == EXPECTED_MECHANICS_TASKS
            and len(mechanics_task_ids) == len(set(mechanics_task_ids))
            and len(mechanics_image_ids) == EXPECTED_MECHANICS_IMAGES
            and len(mechanics_image_ids) == len(set(mechanics_image_ids))
        ),
        "all_mechanics_images_decode_with_positive_dimensions": (
            len(decoded_images) == EXPECTED_MECHANICS_IMAGES
            and all(
                image["bytes"] > 0
                and image["width"] > 0
                and image["height"] > 0
                for image in decoded_images
            )
        ),
        "all_mechanics_images_use_accepted_raster_formats": all(
            image["format"] in ACCEPTED_IMAGE_FORMATS
            for image in decoded_images
        ),
        "public_mechanics_payload_is_opaque_and_truth_free": not public_errors,
    }
    all_gates_pass = all(gates.values())
    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": {
            "interface_version": INTERFACE_VERSION,
            "preregistration": (
                "results/nonmyopic/"
                "BONGARD_OPENWORLD_IMAGE_INTEGRITY_AUDIT_PREREGISTRATION.md"
            ),
            "model_calls": 0,
            "cost_usd": 0.0,
        },
        "status": "image_integrity_pass" if all_gates_pass else "fail",
        "scientific_opportunity_status": "untested",
        "authorizes_paid_calls": False,
        "source": {
            **bound_source,
            "archive_size_bytes": archive_size,
            "archive_sha256": archive_sha256,
        },
        "archive": {
            **member_stats,
            "bad_crc_member": bad_crc_member,
        },
        "mechanics": {
            "summary": mechanics_stats,
            "tasks": mechanics,
            "public_payload_errors": public_errors,
        },
        "gates": gates,
        "all_gates_pass": all_gates_pass,
        "interpretation": (
            "A pass binds and verifies the complete image source and the four "
            "opaque mechanics tasks. It does not test planning quality, open "
            "a confirmation endpoint, or authorize model calls."
        ),
    }
    checkpoint(output_path, result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            REPO_ROOT
            / "results/nonmyopic/bongard_openworld_image_integrity_audit/"
            "bongard-openworld-image-integrity-audit-20260806/MANIFEST.json"
        ),
    )
    return parser.parse_args()


def main() -> None:
    result = run_audit(output_path=parse_args().output_path)
    print(
        json.dumps(
            {
                "status": result["status"],
                "scientific_opportunity_status": result[
                    "scientific_opportunity_status"
                ],
                "authorizes_paid_calls": result["authorizes_paid_calls"],
                "archive": result["archive"],
                "mechanics_summary": result["mechanics"]["summary"],
                "gates": result["gates"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
