from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from scripts import sciconvbench_manifest as manifest


def _write_fixture(root: Path, *, rows_per_domain: int = 14) -> None:
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    subprocess.run(
        ["git", "-C", str(root), "config", "user.email", "test@example.com"],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(root), "config", "user.name", "Test"],
        check=True,
    )
    sources = {}
    for domain in manifest.DOMAINS:
        path = root / domain / f"disambiguation_{domain}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        rows = [
            {
                "id": f"case_{index:03d}",
                "incomplete_user_req": f"PRIVATE_INCOMPLETE_{domain}_{index}",
                "complete_user_req": f"PRIVATE_COMPLETE_{domain}_{index}",
                "missing_entities": [
                    f"PRIVATE_MISSING_{domain}_{index}_{part}" for part in range(3)
                ],
                "ontology_components": [
                    f"PRIVATE_ONTOLOGY_{domain}_{index}_{part}" for part in range(3)
                ],
            }
            for index in range(rows_per_domain)
        ]
        path.write_text(json.dumps(rows), encoding="utf-8")
        sources[domain] = {
            "path": str(path.relative_to(root)),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "rows": rows_per_domain,
        }
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    subprocess.run(
        ["git", "-C", str(root), "commit", "-qm", "fixture"], check=True
    )
    return sources


def test_manifest_is_deterministic_and_emits_no_semantic_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_fixture(tmp_path)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(manifest, "EXPECTED_SOURCES", sources)

    first = manifest.build_manifest(
        tmp_path, expected_commit=commit, enforce_frozen_hashes=False
    )
    second = manifest.build_manifest(
        tmp_path, expected_commit=commit, enforce_frozen_hashes=False
    )

    assert first == second
    assert first["access"]["content_emitted"] is False
    assert first["accounting"] == {
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "oatml_jobs": 0,
    }
    serialized = json.dumps(first)
    assert "PRIVATE_INCOMPLETE" not in serialized
    assert "PRIVATE_COMPLETE" not in serialized
    assert "PRIVATE_MISSING" not in serialized
    assert "PRIVATE_ONTOLOGY" not in serialized


def test_manifest_splits_each_domain_without_overlap(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_fixture(tmp_path)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    monkeypatch.setattr(manifest, "EXPECTED_SOURCES", sources)
    result = manifest.build_manifest(
        tmp_path, expected_commit=commit, enforce_frozen_hashes=False
    )

    for domain in manifest.DOMAINS:
        splits = result["selection"]["domain_splits"][domain]
        sets = [set(splits[name]) for name in splits]
        assert [len(splits[name]) for name in splits] == [1, 8, 4, 1]
        assert len(set.union(*sets)) == 14
        assert sum(len(values) for values in sets) == 14


def test_manifest_rejects_source_hash_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    sources = _write_fixture(tmp_path)
    commit = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    sources["fluids"]["sha256"] = "0" * 64
    monkeypatch.setattr(manifest, "EXPECTED_SOURCES", sources)

    with pytest.raises(ValueError, match="source hash mismatch"):
        manifest.build_manifest(
            tmp_path, expected_commit=commit, enforce_frozen_hashes=False
        )
