from __future__ import annotations

import json

import pytest

from scripts import tau2_mms_gemma26b_thinking_source_audit as audit


def test_documented_split_source_and_tool_contract_pass():
    result = audit.audit()
    assert result["status"] == "source_pass"
    assert result["gates"]["all_source_gates_pass"] is True
    assert result["public_tool_contract"]["root_tool_count"] == 8
    assert result["model_calls_made"] == 0
    assert result["repair_or_task_success_endpoints_opened"] is False


def test_documented_split_cohort_is_fresh_balanced_and_bound():
    episodes = audit.load_episodes()
    assert [row["family"] for row in episodes] == ["mms_abroad"] * 3 + ["mms_home"] * 3
    assert len({audit.episode_hash(row) for row in episodes}) == 6


def test_documented_split_manifest_position_tamper_fails(monkeypatch, tmp_path):
    payload = json.loads(audit.MANIFEST.read_text())
    payload["episodes"][0]["reserve_family_position"] = 11
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(payload))
    monkeypatch.setattr(audit, "MANIFEST", path)
    with pytest.raises(ValueError, match="cohort binding"):
        audit.load_episodes()


def test_documented_split_tool_source_tamper_fails(tmp_path):
    path = tmp_path / "user_tools.py"
    path.write_text(audit.USER_TOOLS.read_text().replace("Returns the name of all installed apps", "Returns selected apps", 1))
    with pytest.raises(ValueError, match="source binding"):
        audit.validate_public_tool_contract(path)
