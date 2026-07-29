import hashlib
import json
from pathlib import Path

from scripts import validate_paper_claim_manifest as validator


def _write_fixture(tmp_path: Path, *, pointer: str = "/aggregate/value") -> tuple[Path, Path]:
    artifact = tmp_path / "results" / "result.json"
    artifact.parent.mkdir()
    artifact.write_text(
        json.dumps({"aggregate": {"value": 0.25, "interval": [0.1, 0.4]}}),
        encoding="utf-8",
    )
    digest = hashlib.sha256(artifact.read_bytes()).hexdigest()
    manifest = tmp_path / "paper" / "claim_manifest.json"
    manifest.parent.mkdir()
    manifest.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "claims": [
                    {
                        "id": "fixture",
                        "artifact": {
                            "path": "results/result.json",
                            "sha256": digest,
                        },
                        "checks": [
                            {
                                "pointer": pointer,
                                "equals": 0.2500000000001,
                                "tolerance": 1e-12,
                            },
                            {
                                "pointer": "/aggregate/interval",
                                "equals": [0.1, 0.4],
                            },
                        ],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    return manifest, artifact


def test_validate_claim_manifest_accepts_hash_and_values(tmp_path):
    manifest, _ = _write_fixture(tmp_path)

    payload = validator.summary_payload(
        validator.validate_claim_manifest(manifest, tmp_path)
    )

    assert payload["ok"] is True
    assert payload["checks"][-1]["name"] == "claim:fixture"
    assert "2 value check(s)" in payload["checks"][-1]["detail"]


def test_validate_claim_manifest_rejects_hash_mismatch(tmp_path):
    manifest, artifact = _write_fixture(tmp_path)
    artifact.write_text('{"aggregate":{"value":0.5}}', encoding="utf-8")

    payload = validator.summary_payload(
        validator.validate_claim_manifest(manifest, tmp_path)
    )

    assert payload["ok"] is False
    assert "SHA-256 mismatch" in payload["checks"][-1]["detail"]


def test_validate_claim_manifest_rejects_pointer_mismatch(tmp_path):
    manifest, _ = _write_fixture(tmp_path, pointer="/aggregate/missing")

    payload = validator.summary_payload(
        validator.validate_claim_manifest(manifest, tmp_path)
    )

    assert payload["ok"] is False
    assert "object key not found" in payload["checks"][-1]["detail"]


def test_validate_claim_manifest_rejects_escaping_artifact_path(tmp_path):
    manifest, _ = _write_fixture(tmp_path)
    data = json.loads(manifest.read_text(encoding="utf-8"))
    data["claims"][0]["artifact"]["path"] = "../outside.json"
    manifest.write_text(json.dumps(data), encoding="utf-8")

    payload = validator.summary_payload(
        validator.validate_claim_manifest(manifest, tmp_path)
    )

    assert payload["ok"] is False
    assert "escapes the repository" in payload["checks"][-1]["detail"]


def test_json_pointer_decodes_escaped_tokens():
    document = {"a/b": {"~key": 7}}

    assert validator._json_pointer(document, "/a~1b/~0key") == 7
