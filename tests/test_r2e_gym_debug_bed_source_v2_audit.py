from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r2e_gym_debug_bed_source_v2_audit.py"
SPEC = importlib.util.spec_from_file_location("r2e_source_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v2_audit_is_independently_fail_closed() -> None:
    source = SCRIPT.read_text()
    assert '== ["native_debugger_contract"]' in source
    assert '"v1_manifest_reproduced"' in source
    assert '"producer_source_pass"' in source
    assert '"zero_call_privacy_boundary"' in source


def test_v2_correction_does_not_define_new_selection() -> None:
    source = SCRIPT.read_text()
    assert "SALT =" not in source
    assert "eligible =" not in source
    assert "ordered =" not in source

