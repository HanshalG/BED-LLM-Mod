#!/usr/bin/env python3
"""Verify the frozen Bongard confirmation64 execution implementation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import bongard_openworld_luna_confirmation64_verify as protocol_verify


INTERFACE_VERSION = "bongard-openworld-luna-confirmation64-execute-verify-1"
CORE = REPO_ROOT / "scripts/bongard_openworld_luna_confirmation64.py"
DAILY = REPO_ROOT / (
    "scripts/bongard_openworld_luna_confirmation64_daily_execute.py"
)
AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_CONFIRMATION64_EXECUTION_AMENDMENT.md"
)
CORE_SHA256 = "4b35322bda8e7a270501aa90e690a444764b7b7bbfefb999dfd38c0e65e5b548"
DAILY_SHA256 = "0a6355e2239799f29e0fab2bdb0600285051b1217cbe7be97606c2dbdc4e77ee"
AMENDMENT_SHA256 = (
    "7d826d811980bf406756dcd91727b7fb2ba7a3722a1723d769dd23690f9f4b59"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution_bindings() -> dict[str, Any]:
    expected = {
        CORE: CORE_SHA256,
        DAILY: DAILY_SHA256,
        AMENDMENT: AMENDMENT_SHA256,
    }
    changed = [
        str(path)
        for path, digest in expected.items()
        if not path.is_file() or _sha256(path) != digest
    ]
    if changed:
        raise ValueError(f"confirmation execution binding changed: {changed}")
    protocol = protocol_verify.verify_manifest(
        require_unopened_predecessors=False
    )
    return {
        "verified": True,
        "interface_version": INTERFACE_VERSION,
        "core_sha256": CORE_SHA256,
        "daily_sha256": DAILY_SHA256,
        "execution_amendment_sha256": AMENDMENT_SHA256,
        "protocol_manifest_sha256": protocol["manifest_sha256"],
    }


def main() -> int:
    print(json.dumps(verify_execution_bindings(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
