from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class CheckResult:
    name: str
    ok: bool
    detail: str


def _json_pointer(document: Any, pointer: str) -> Any:
    if pointer == "":
        return document
    if not pointer.startswith("/"):
        raise ValueError("JSON pointer must be empty or start with '/'")

    current = document
    for raw_token in pointer[1:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if isinstance(current, list):
            try:
                index = int(token)
            except ValueError as exc:
                raise KeyError(f"list index is not an integer: {token!r}") from exc
            try:
                current = current[index]
            except IndexError as exc:
                raise KeyError(f"list index out of range: {index}") from exc
        elif isinstance(current, dict):
            if token not in current:
                raise KeyError(f"object key not found: {token!r}")
            current = current[token]
        else:
            raise KeyError(f"cannot descend through {type(current).__name__}")
    return current


def _values_match(actual: Any, expected: Any, tolerance: float) -> bool:
    if isinstance(actual, bool) or isinstance(expected, bool):
        return actual is expected
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return (
            math.isfinite(float(actual))
            and math.isfinite(float(expected))
            and math.isclose(
                float(actual),
                float(expected),
                rel_tol=0.0,
                abs_tol=tolerance,
            )
        )
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _values_match(actual_item, expected_item, tolerance)
            for actual_item, expected_item in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _values_match(actual[key], expected[key], tolerance)
            for key in actual
        )
    return actual == expected


def _safe_artifact_path(repo_root: Path, relative_path: str) -> Path:
    candidate = Path(relative_path)
    if candidate.is_absolute():
        raise ValueError("artifact path must be relative to the repository")
    resolved = (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise ValueError("artifact path escapes the repository") from exc
    return resolved


def validate_claim_manifest(manifest_path: Path, repo_root: Path) -> list[CheckResult]:
    repo_root = repo_root.resolve()
    manifest_path = manifest_path.resolve()
    if not manifest_path.exists():
        return [CheckResult("manifest_exists", False, f"missing {manifest_path}")]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [CheckResult("manifest_json", False, str(exc))]

    results = [CheckResult("manifest_exists", True, str(manifest_path))]
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1:
        results.append(CheckResult("manifest_schema", False, "schema_version must equal 1"))
        return results

    claims = manifest.get("claims")
    if not isinstance(claims, list) or not claims:
        results.append(CheckResult("manifest_schema", False, "claims must be a non-empty list"))
        return results

    claim_ids = [claim.get("id") for claim in claims if isinstance(claim, dict)]
    if (
        len(claim_ids) != len(claims)
        or any(not isinstance(claim_id, str) or not claim_id for claim_id in claim_ids)
        or len(set(claim_ids)) != len(claim_ids)
    ):
        results.append(
            CheckResult("manifest_schema", False, "claim ids must be unique non-empty strings")
        )
        return results
    results.append(CheckResult("manifest_schema", True, f"{len(claims)} claim bundle(s)"))

    for claim in claims:
        claim_id = claim["id"]
        try:
            artifact = claim["artifact"]
            relative_path = artifact["path"]
            expected_sha256 = artifact["sha256"]
            checks = claim["checks"]
            if not isinstance(relative_path, str) or not relative_path:
                raise ValueError("artifact.path must be a non-empty string")
            if not isinstance(expected_sha256, str) or not SHA256_PATTERN.fullmatch(
                expected_sha256
            ):
                raise ValueError("artifact.sha256 must be a lowercase SHA-256 digest")
            if not isinstance(checks, list) or not checks:
                raise ValueError("checks must be a non-empty list")

            artifact_path = _safe_artifact_path(repo_root, relative_path)
            artifact_bytes = artifact_path.read_bytes()
            actual_sha256 = hashlib.sha256(artifact_bytes).hexdigest()
            if actual_sha256 != expected_sha256:
                raise ValueError(
                    f"SHA-256 mismatch for {relative_path}: "
                    f"expected {expected_sha256}, got {actual_sha256}"
                )
            document = json.loads(artifact_bytes)

            for index, check in enumerate(checks):
                if not isinstance(check, dict):
                    raise ValueError(f"check {index} must be an object")
                pointer = check.get("pointer")
                if not isinstance(pointer, str) or "equals" not in check:
                    raise ValueError(f"check {index} requires string pointer and equals")
                tolerance = check.get("tolerance", 0.0)
                if (
                    isinstance(tolerance, bool)
                    or not isinstance(tolerance, (int, float))
                    or not math.isfinite(float(tolerance))
                    or tolerance < 0
                ):
                    raise ValueError(f"check {index} has invalid tolerance")
                actual = _json_pointer(document, pointer)
                expected = check["equals"]
                if not _values_match(actual, expected, float(tolerance)):
                    raise ValueError(
                        f"{pointer or '<root>'} mismatch: "
                        f"expected {expected!r}, got {actual!r}"
                    )
        except (KeyError, OSError, json.JSONDecodeError, TypeError, ValueError) as exc:
            results.append(CheckResult(f"claim:{claim_id}", False, str(exc)))
        else:
            results.append(
                CheckResult(
                    f"claim:{claim_id}",
                    True,
                    f"{relative_path}: SHA-256 and {len(checks)} value check(s)",
                )
            )
    return results


def summary_payload(results: list[CheckResult]) -> dict[str, Any]:
    return {
        "ok": all(result.ok for result in results),
        "checks": [
            {"name": result.name, "ok": result.ok, "detail": result.detail}
            for result in results
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate paper claims against hash-pinned public result artifacts."
    )
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--manifest", type=Path, default=Path("paper/claim_manifest.json"))
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    manifest_path = args.manifest
    if not manifest_path.is_absolute():
        manifest_path = repo_root / manifest_path
    payload = summary_payload(validate_claim_manifest(manifest_path, repo_root))
    if args.json:
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        for check in payload["checks"]:
            status = "ok" if check["ok"] else "failed"
            print(f"[{status}] {check['name']}: {check['detail']}")
    raise SystemExit(0 if payload["ok"] else 1)


if __name__ == "__main__":
    main()
