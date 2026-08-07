from __future__ import annotations

import json
from pathlib import Path

from scripts import regretbench_deepseek_confirmation_protocol_verify as verify


def _copy_json(source: Path, destination: Path) -> dict:
    payload = json.loads(source.read_text(encoding="utf-8"))
    destination.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def test_frozen_confirmation_protocol_verifies_without_calls() -> None:
    result = verify.verify_protocol()

    assert result["status"] == "verified_frozen_protocol"
    assert result["gates"]["all_pass"] is True
    assert result["confirmation_task_count"] == 64
    assert result["confirmation_seed_count"] == 1345
    assert result["development_seed_overlap_count"] == 0
    assert result["model_calls_made"] == 0
    assert result["cost_usd"] == 0.0


def test_changed_science_threshold_fails_protocol_replay(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    payload = _copy_json(verify.PROTOCOL, path)
    payload["science_gates"]["dynamic_minus_myopic_brier_maximum"] = -0.01
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify.verify_protocol(
        protocol_path=path,
        require_frozen_protocol_hash=False,
    )

    assert result["status"] == "failed"
    assert result["gates"]["science_gates_unchanged"] is False


def test_confirmation_seed_overlap_with_development_fails(tmp_path: Path) -> None:
    path = tmp_path / "protocol.json"
    payload = _copy_json(verify.PROTOCOL, path)
    payload["seeds"]["initial_start"] = 202608089000
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify.verify_protocol(
        protocol_path=path,
        require_frozen_protocol_hash=False,
    )

    assert result["status"] == "failed"
    assert result["gates"]["confirmation_seeds_disjoint_from_development"] is False
    assert result["development_seed_overlap_count"] == 64


def test_reordered_confirmation_cohort_fails_source_replay(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    payload = _copy_json(verify.SOURCE_MANIFEST, path)
    ids = payload["splits"]["confirmation"]["ids"]
    ids[0], ids[1] = ids[1], ids[0]
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify.verify_protocol(source_manifest_path=path)

    assert result["status"] == "failed"
    assert result["gates"]["source_manifest_hash_matches"] is False
    assert result["gates"]["exact_source_split_hashes"] is False


def test_changed_implementation_binding_fails_protocol_replay(
    tmp_path: Path,
) -> None:
    path = tmp_path / "protocol.json"
    payload = _copy_json(verify.PROTOCOL, path)
    key = "scripts/regretbench_deepseek_dynamic_depth2_policy.py"
    payload["implementation_bindings_at_freeze"][key] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")

    result = verify.verify_protocol(
        protocol_path=path,
        require_frozen_protocol_hash=False,
    )

    assert result["status"] == "failed"
    assert result["gates"]["all_implementation_bindings_match"] is False
    assert result["implementation_bindings"][key] is False


def test_changed_daily_executor_binding_fails_protocol_replay(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "daily-binding.json"
    payload = _copy_json(verify.DAILY_EXECUTION_BINDING, path)
    key = "scripts/regretbench_deepseek_dynamic_depth2_confirmation_daily.py"
    payload["components"][key] = "0" * 64
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(verify, "DAILY_EXECUTION_BINDING", path)

    result = verify.verify_protocol()

    assert result["status"] == "failed"
    assert result["gates"]["daily_execution_binding_matches"] is False
    assert result["daily_component_bindings"][key] is False
