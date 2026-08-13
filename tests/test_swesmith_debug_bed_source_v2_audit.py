from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "swesmith_debug_bed_source_v2_audit.py"
SPEC = importlib.util.spec_from_file_location("swesmith_source_v2", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_v2_has_distinct_protocol_version() -> None:
    assert MODULE.PROTOCOL_VERSION.endswith("-v2")
    assert MODULE.PROTOCOL_VERSION != MODULE.V1.PROTOCOL_VERSION


def test_v2_public_artifacts_retain_privacy() -> None:
    output = ROOT / "results" / "nonmyopic" / "swesmith_debug_bed_source_v2"
    for path in (output / "MANIFEST.json", output / "SOURCE_AUDIT.json"):
        if not path.exists():
            continue
        privacy = json.loads(path.read_text(encoding="utf-8"))["privacy"]
        assert all(value is False for value in privacy.values())
