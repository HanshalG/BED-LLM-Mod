from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/swesmith_debug_bed_native_amd64_preflight.py"
SPEC = importlib.util.spec_from_file_location("swesmith_amd64", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)
AUDIT_SCRIPT = ROOT / "scripts/swesmith_debug_bed_native_amd64_preflight_audit.py"
AUDIT_SPEC = importlib.util.spec_from_file_location("swesmith_amd64_audit", AUDIT_SCRIPT)
assert AUDIT_SPEC and AUDIT_SPEC.loader
AUDIT = importlib.util.module_from_spec(AUDIT_SPEC)
AUDIT_SPEC.loader.exec_module(AUDIT)
PROTOCOL = ROOT / "results/nonmyopic/SWESMITH_DEBUG_BED_NATIVE_AMD64_PREFLIGHT_PROTOCOL_20260813.md"
WORKFLOW = ROOT / ".github/workflows/swesmith_debug_bed_native_amd64_preflight.yml"
COMMIT = "a" * 40


def valid_result() -> dict:
    value = {
        "protocol_version": MOD.PROTOCOL_VERSION,
        "status": "preflight_pass",
        "decision": "fresh_native_amd64_protocol_may_be_frozen",
        "bindings": {
            "protocol_sha256": MOD.sha256_file(PROTOCOL),
            "workflow_sha256": MOD.sha256_file(WORKFLOW),
            "repository": MOD.EXPECTED_REPOSITORY,
            "branch": MOD.EXPECTED_BRANCH,
            "event": MOD.EXPECTED_EVENT,
            "commit": COMMIT,
        },
        "runner": {"host_machine": "x86_64", "docker_os": "linux", "docker_arch": "x86_64"},
        "controls": [
            {**control, "stdout_lines": [control["nonce"], "x86_64"], "exit_code": 0}
            for control in MOD.expected_controls()
        ],
        "privacy": {
            "swesmith_rows_read": 0,
            "instance_ids_read": 0,
            "task_payloads_read": 0,
            "private_files_read": 0,
            "endpoints_opened": 0,
        },
        "accounting": {"openrouter_calls": 0, "openrouter_cost_usd": 0.0, "oatml_cluster_use": 0},
        "authorizes": "freeze_new_protocol_only",
    }
    gates = MOD.validate(value, PROTOCOL, WORKFLOW, COMMIT)
    value["gates"] = gates
    return value


def test_valid_result_passes(tmp_path: Path) -> None:
    path = tmp_path / "result.json"
    path.write_text(json.dumps(valid_result()))
    assert all(MOD.verify(path, PROTOCOL, WORKFLOW, COMMIT).values())
    assert all(AUDIT.audit(valid_result(), PROTOCOL, WORKFLOW, COMMIT).values())


def test_tampered_cases_fail() -> None:
    cases = []
    for mutate in (
        lambda d: d["bindings"].update(branch="main"),
        lambda d: d["runner"].update(host_machine="arm64"),
        lambda d: d["controls"][0].update(stdout_lines=["x86_64"]),
        lambda d: d["controls"].pop(),
        lambda d: d["privacy"].update(swesmith_rows_read=1),
        lambda d: d["accounting"].update(openrouter_calls=1),
        lambda d: d.update(authorizes="run_mechanics"),
        lambda d: d.update(instance_id="secret"),
    ):
        value = copy.deepcopy(valid_result())
        mutate(value)
        cases.append(value)
    for value in cases:
        assert not all(MOD.validate(value, PROTOCOL, WORKFLOW, COMMIT).values())
        assert not all(AUDIT.audit(value, PROTOCOL, WORKFLOW, COMMIT).values())
