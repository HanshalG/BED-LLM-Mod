#!/usr/bin/env python3
"""Verify the frozen Bongard Luna naive first-link protocol manifest."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


INTERFACE_VERSION = "bongard-openworld-luna-naive-first-link-manifest-1"
MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_naive_first_link/"
    "PROTOCOL_MANIFEST.json"
)
MANIFEST_SHA256 = (
    "dfd55b7ed3577fc69431e1dd514d69d158cd7eb01376fbde447064bc0bfdcc7f"
)
MAIN_MANIFEST = REPO_ROOT / (
    "results/nonmyopic/bongard_openworld_luna_vlm_development32/"
    "PROTOCOL_MANIFEST.json"
)
MAIN_MANIFEST_SHA256 = (
    "a0b70ff8bbe3e36eba56b357e563504f12e4792d92cedee237d4b261d15a7708"
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_protocol_manifest(path: Path = MANIFEST) -> dict[str, Any]:
    if sha256_file(path) != MANIFEST_SHA256:
        raise RuntimeError("naive first-link protocol manifest changed")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    expected_model = {
        "concurrency": 10,
        "id": "openai/gpt-5.6-luna",
        "max_tokens": 8192,
        "reasoning_effort": "medium",
        "reasoning_trace_excluded": True,
        "requests_per_development_block": 8,
        "smoke_requests": 10,
        "temperature": 0.0,
    }
    expected_schedule = {
        "development_blocks": {
            "a": "2026-08-11",
            "b": "2026-08-12",
            "c": "2026-08-13",
            "d": "2026-08-14",
        },
        "main_block_precedes_same_day_naive_block": True,
        "smoke_date": "2026-08-08",
    }
    expected_privacy = {
        "candidate_labels_materialized": False,
        "endpoint_images_exposed": False,
        "endpoint_labels_materialized": False,
        "source_metadata_exposed": False,
    }
    if (
        manifest.get("schema_version") != 1
        or manifest.get("interface_version") != INTERFACE_VERSION
        or manifest.get("frozen_before_model_responses") is not True
        or manifest.get("model") != expected_model
        or manifest.get("schedule") != expected_schedule
        or manifest.get("privacy") != expected_privacy
        or manifest.get("main_development_manifest_sha256")
        != MAIN_MANIFEST_SHA256
        or manifest.get("main_development_result_must_independently_replay")
        is not True
    ):
        raise RuntimeError("naive first-link protocol fields changed")
    if sha256_file(MAIN_MANIFEST) != MAIN_MANIFEST_SHA256:
        raise RuntimeError("main development manifest changed")
    file_hashes = manifest.get("files_sha256") or {}
    if not file_hashes:
        raise RuntimeError("naive first-link file bindings are absent")
    for relative, expected in file_hashes.items():
        if sha256_file(REPO_ROOT / relative) != expected:
            raise RuntimeError(f"naive first-link bound file changed: {relative}")
    return {
        "verified": True,
        "manifest_sha256": MANIFEST_SHA256,
        "main_development_manifest_sha256": MAIN_MANIFEST_SHA256,
        "bound_file_count": len(file_hashes),
    }


def main() -> int:
    print(json.dumps(verify_protocol_manifest(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
