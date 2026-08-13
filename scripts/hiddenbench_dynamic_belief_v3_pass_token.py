#!/usr/bin/env python3
"""Create and verify the label-free pass token for HiddenBench V3."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def file_digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object: {path}")
    return value


def create_token(
    *,
    execution_binding: Path,
    source_manifest: Path,
    source_audit: Path,
    raw_responses: Path,
    label_free_result: Path,
    verification: Path,
) -> dict[str, Any]:
    result = load_object(label_free_result)
    verified = load_object(verification)
    if (
        result.get("status") != "serving_pass"
        or result.get("authorizes") != "endpoint_only"
        or not result.get("gates")
        or not all(result["gates"].values())
        or result.get("registered_answers_opened") is not False
        or result.get("endpoint_scores_opened") is not False
    ):
        raise RuntimeError("label-free result does not authorize an endpoint")
    if (
        verified.get("status") != "verification_pass"
        or not verified.get("gates")
        or not all(verified["gates"].values())
        or verified.get("registered_answers_loaded") is not False
    ):
        raise RuntimeError("independent verification does not authorize an endpoint")
    if result.get("raw_response_sha256") != file_digest(raw_responses):
        raise RuntimeError("raw response hash does not match result")
    return {
        "schema_version": 1,
        "interface_version": "hiddenbench-dynamic-belief-v3-pass-token-v1",
        "status": "label_free_pass",
        "authorizes": "endpoint_extraction_once",
        "execution_binding_sha256": file_digest(execution_binding),
        "source_manifest_sha256": file_digest(source_manifest),
        "source_audit_sha256": file_digest(source_audit),
        "raw_response_sha256": file_digest(raw_responses),
        "label_free_result_sha256": file_digest(label_free_result),
        "verification_sha256": file_digest(verification),
        "registered_answers_opened": False,
        "endpoint_scores_opened": False,
    }


def verify_token(
    token_path: Path,
    *,
    execution_binding: Path,
    source_manifest: Path,
    source_audit: Path,
    raw_responses: Path,
    label_free_result: Path,
    verification: Path,
) -> dict[str, Any]:
    token = load_object(token_path)
    expected = create_token(
        execution_binding=execution_binding,
        source_manifest=source_manifest,
        source_audit=source_audit,
        raw_responses=raw_responses,
        label_free_result=label_free_result,
        verification=verification,
    )
    if token != expected:
        raise RuntimeError("label-free pass token does not match bound artifacts")
    return token
