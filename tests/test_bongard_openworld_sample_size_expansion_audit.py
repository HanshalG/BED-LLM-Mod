from __future__ import annotations

import json

from scripts import bongard_openworld_sample_size_expansion_audit as expansion


def test_expanded_partition_preserves_existing_tasks_and_is_clean(tmp_path) -> None:
    output = tmp_path / "MANIFEST.json"
    result = expansion.run_audit(output_path=output)
    mechanics, development, confirmation, reserve = (
        expansion.expanded_validation_rows()
    )
    assert result["gates"]["all_pass"] is True
    assert (len(mechanics), len(development), len(confirmation), len(reserve)) == (
        4,
        64,
        96,
        36,
    )
    assert result["partition_uid_sha256"] == expansion.EXPECTED_UID_SHA256
    assert len(result["development_additions"]) == 32
    assert len(result["confirmation_additions"]) == 32
    assert result["development_semantic_duplicate_count"] == {
        "concept": 0,
        "caption": 0,
    }
    assert result["development_strict_perceptual_overlap"] == 0
    assert result["post_selection_semantic_overlap"] == {
        "concept": 0,
        "caption": 0,
    }
    assert result["post_selection_strict_perceptual_overlap"] == 0
    assert json.loads(output.read_text()) == result


def test_manifest_verifier_binds_hash_and_clean_status(tmp_path) -> None:
    output = tmp_path / "MANIFEST.json"
    expansion.run_audit(output_path=output)
    digest = expansion.sha256_file(output)
    assert expansion.verify_manifest(output, expected_sha256=digest)["verified"]
    value = json.loads(output.read_text())
    value["gates"]["all_pass"] = False
    output.write_text(json.dumps(value))
    try:
        expansion.verify_manifest(output, expected_sha256=digest)
    except ValueError:
        pass
    else:
        raise AssertionError("tampered expansion manifest should fail")
