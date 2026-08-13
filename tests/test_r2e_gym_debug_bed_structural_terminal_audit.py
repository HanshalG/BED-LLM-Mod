from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/r2e_gym_debug_bed_structural_terminal_audit.py"
SPEC = importlib.util.spec_from_file_location("r2e_terminal", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_terminal_audit_passes_banked_result() -> None:
    result = MODULE.audit(
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_structural_terminal/TERMINAL_RESULT.json",
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_structural_screen/STRUCTURAL_SCREEN.json",
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_source_v3/SOURCE_AUDIT.json",
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_source_v3/MANIFEST.json",
        ROOT / "results/nonmyopic/R2E_GYM_DEBUG_BED_STRUCTURAL_SCREEN_PROTOCOL_20260813.md",
    )
    assert result["status"] == "terminal_audit_pass"
    assert all(result["gates"].values())


def test_terminal_audit_rejects_open_downstream(tmp_path: Path) -> None:
    terminal = ROOT / "results/nonmyopic/r2e_gym_debug_bed_structural_terminal/TERMINAL_RESULT.json"
    value = __import__("json").loads(terminal.read_text())
    value["downstream"]["planner_opened"] = True
    tampered = tmp_path / "terminal.json"
    tampered.write_text(__import__("json").dumps(value))
    result = MODULE.audit(
        tampered,
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_structural_screen/STRUCTURAL_SCREEN.json",
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_source_v3/SOURCE_AUDIT.json",
        ROOT / "results/nonmyopic/r2e_gym_debug_bed_source_v3/MANIFEST.json",
        ROOT / "results/nonmyopic/R2E_GYM_DEBUG_BED_STRUCTURAL_SCREEN_PROTOCOL_20260813.md",
    )
    assert not result["gates"]["downstream_sealed"]
