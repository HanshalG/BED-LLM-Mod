from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/swesmith_debug_bed_source_v3_audit.py"
SPEC = importlib.util.spec_from_file_location("swesmith_v3", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_helpers_are_deterministic() -> None:
    assert MOD.digest(b"x") == MOD.digest(b"x")
    assert MOD.ordered_hash(["a", "b"]) != MOD.ordered_hash(["b", "a"])
    assert MOD.SALT != MOD.V1.SPLIT_SALT


def test_live_v3_audit_passes() -> None:
    debug_root = Path("/tmp/bed-source-audits/debug-gym")
    data_root = Path("/tmp/bed-source-audits/swesmith-data/repo")
    if not debug_root.exists() or not data_root.exists():
        return
    protocol = ROOT / "results/nonmyopic/SWESMITH_DEBUG_BED_SOURCE_V3_NATIVE_AMD64_PROTOCOL_20260813.md"
    manifest, result = MOD.audit(debug_root, data_root, protocol)
    assert result["status"] == "source_pass"
    assert all(result["gates"].values())
    assert manifest["selection"]["screening_count"] == 12
    text = str(manifest)
    assert "combine_file__" not in text
    assert "combine_module__" not in text
