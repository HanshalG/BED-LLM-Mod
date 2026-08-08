#!/usr/bin/env python3
"""Verify the frozen Bongard confirmation96 execution implementation."""

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


INTERFACE_VERSION = "bongard-openworld-luna-confirmation96-execute-verify-6"
CORE = REPO_ROOT / "scripts/bongard_openworld_luna_confirmation64.py"
DAILY = REPO_ROOT / (
    "scripts/bongard_openworld_luna_confirmation64_daily_execute.py"
)
AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_CONFIRMATION64_EXECUTION_AMENDMENT.md"
)
TERMINAL_OBEDIENCE_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_TERMINAL_OBEDIENCE_AMENDMENT.md"
)
ENDPOINT_UTILITY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/BONGARD_OPENWORLD_ENDPOINT_PREDICTIVE_UTILITY_AMENDMENT.md"
)
CORE_SHA256 = "80dda8508e1378e48593190968086f3b7db451732f5091606c5fa0a75e30b18e"
DAILY_SHA256 = "627eb36ee0af272c1f6f5541c9d45d5605cceb74d526e988169f9e3311296c5d"
AMENDMENT_SHA256 = (
    "0f284c475e04c58f546d21baf9d2f0b41975832c0f11b1fc54317d3de124a351"
)
TERMINAL_OBEDIENCE_AMENDMENT_SHA256 = (
    "0706ff63310ee3b2c7ca603cadf43308d0e427868a11ae99959d81d126219daf"
)
ENDPOINT_UTILITY_AMENDMENT_SHA256 = (
    "2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_execution_bindings() -> dict[str, Any]:
    expected = {
        CORE: CORE_SHA256,
        DAILY: DAILY_SHA256,
        AMENDMENT: AMENDMENT_SHA256,
        TERMINAL_OBEDIENCE_AMENDMENT: TERMINAL_OBEDIENCE_AMENDMENT_SHA256,
        ENDPOINT_UTILITY_AMENDMENT: ENDPOINT_UTILITY_AMENDMENT_SHA256,
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
        "terminal_obedience_amendment_sha256": (
            TERMINAL_OBEDIENCE_AMENDMENT_SHA256
        ),
        "endpoint_predictive_utility_amendment_sha256": (
            ENDPOINT_UTILITY_AMENDMENT_SHA256
        ),
        "protocol_manifest_sha256": protocol["manifest_sha256"],
    }


def main() -> int:
    print(json.dumps(verify_execution_bindings(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
