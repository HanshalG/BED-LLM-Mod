from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "swesmith_debug_bed_mechanics_terminal_audit.py"
SPEC = importlib.util.spec_from_file_location("terminal_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def fixture() -> dict:
    return {
        "status": "infrastructure_failed_closed",
        "decision": "close_exact_swesmith_debug_bed_mechanics",
        "mechanics_task_count": 8,
        "structurally_eligible_count": 6,
        "structurally_invalid_count": 2,
        "attempted_eligible_tasks": 1,
        "fresh_arm_exit_codes": [133, 133],
        "fresh_arm_stdout_bytes": [0, 0],
        "mechanics_image_count": 7,
        "mechanics_image_platform_counts": {"linux/amd64": 7},
        "execution_host_platform": "linux/arm64",
        "test_matrix_opened": False,
        "pdb_handshake_opened": False,
        "planner_opened": False,
        "patch_endpoint_opened": False,
        "opportunity_opened": False,
        "development_opened": False,
        "confirmation_opened": False,
        "openrouter_calls": 0,
        "openrouter_cost_usd": 0.0,
        "cluster_use": 0,
    }


def test_terminal_fixture_passes(tmp_path: Path) -> None:
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(fixture()), encoding="utf-8")
    assert MODULE.audit(ROOT, path)["status"] == "audit_pass"


def test_scientific_stage_opening_fails(tmp_path: Path) -> None:
    value = fixture()
    value["planner_opened"] = True
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    result = MODULE.audit(ROOT, path)
    assert result["status"] == "audit_failed"
    assert not result["gates"]["scientific_stages_closed"]


def test_second_arm_mismatch_fails(tmp_path: Path) -> None:
    value = fixture()
    value["fresh_arm_exit_codes"] = [133, 0]
    path = tmp_path / "terminal.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    assert MODULE.audit(ROOT, path)["status"] == "audit_failed"
