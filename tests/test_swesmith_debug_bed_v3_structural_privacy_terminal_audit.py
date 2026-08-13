from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/swesmith_debug_bed_v3_structural_privacy_terminal_audit.py"
SPEC = importlib.util.spec_from_file_location("swesmith_v3_privacy_terminal", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_terminal_closure_passes() -> None:
    assert all(MOD.audit(ROOT).values())
