from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/swesmith_debug_bed_v3_structural_screen.py"
SPEC = importlib.util.spec_from_file_location("swesmith_v3_structural", SCRIPT)
assert SPEC and SPEC.loader
MOD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MOD)


def test_hunk_count() -> None:
    patch = "diff --git a/a.py b/a.py\n@@ -1 +1 @@\n-a\n+b\n@@ -4 +4 @@\n-c\n+d\n"
    assert MOD.hunk_count(patch) == 2
    assert MOD.hunk_count("") == 0


def test_ordered_hash_is_order_sensitive() -> None:
    assert MOD.ordered_hash(["a", "b"]) != MOD.ordered_hash(["b", "a"])
