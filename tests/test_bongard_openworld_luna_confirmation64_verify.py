from __future__ import annotations

import json

import pytest

from scripts import bongard_openworld_luna_confirmation64_verify as verify


def test_official_confirmation_freeze_independently_verifies() -> None:
    result = verify.verify_manifest()
    assert result["verified"] is True
    assert result["task_count"] == 64
    assert result["block_sizes"] == {"a": 16, "b": 16, "c": 16, "d": 16}
    assert result["maximum_requests_per_block"] == 688
    assert result["maximum_http_attempts_per_block"] == 702
    assert result["maximum_precharged_exposure_per_block_usd"] == pytest.approx(
        2.808
    )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("protocol", "model"), "wrong/model"),
        (("protocol", "blocks", "a", "maximum_requests"), 687),
        (("protocol", "blocks", "a", "maximum_http_attempts"), 701),
        (("development_precondition", "required_claim_tier"), "policy_only"),
        (("tasks", 0, "block_id"), "d"),
        (("gates", "model_calls_are_zero"), False),
    ],
)
def test_semantic_tampering_fails_without_hash_shortcut(
    tmp_path, path: tuple[object, ...], value: object
) -> None:
    manifest = json.loads(verify.MANIFEST.read_text(encoding="utf-8"))
    target = manifest
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    tampered = tmp_path / "MANIFEST.json"
    tampered.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        verify.verify_manifest(tampered, expected_sha256=None)
