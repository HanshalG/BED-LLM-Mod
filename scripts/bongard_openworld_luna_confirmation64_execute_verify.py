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


INTERFACE_VERSION = "bongard-openworld-luna-confirmation96-execute-verify-8"
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
MATCHED_REALIZED_UPDATER_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_LUNA_MATCHED_REALIZED_UPDATER_AMENDMENT_20260808.md"
)
MATCHED_UPDATER_INTEGRITY_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_MATCHED_UPDATER_INTEGRITY_AMENDMENT_20260808.md"
)
BUDGET_CHAIN_AMENDMENT = REPO_ROOT / (
    "results/nonmyopic/"
    "BONGARD_OPENWORLD_CONFIRMATION_ACCOUNT_WIDE_BUDGET_CHAIN_CORRECTION_20260809.md"
)
CORE_SHA256 = "d20aca14b8b254851caf785cac3dcf4a82a2e0b41c327506991e77ad676ad794"
DAILY_SHA256 = "7b313f65786fbe5e20c65d226fb519a3cb97241b18f224f9d5d7d0903ee2a050"
AMENDMENT_SHA256 = (
    "0f284c475e04c58f546d21baf9d2f0b41975832c0f11b1fc54317d3de124a351"
)
TERMINAL_OBEDIENCE_AMENDMENT_SHA256 = (
    "0706ff63310ee3b2c7ca603cadf43308d0e427868a11ae99959d81d126219daf"
)
ENDPOINT_UTILITY_AMENDMENT_SHA256 = (
    "2fce4b66696d5b635f8d32c5f968a917aac20ab64305cab2728bc5f9973a55b8"
)
MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256 = (
    "dfa981153687004c8fb2c1195879d0774a281ca6c231c55d85495f2ac622178b"
)
MATCHED_UPDATER_INTEGRITY_AMENDMENT_SHA256 = (
    "1f0da098fbed4968a3594761194f661a0ebf49b12c477383d7c90bfa4989abf9"
)
BUDGET_CHAIN_AMENDMENT_SHA256 = (
    "40889c8d1dd3768f5ba06d454c0ee13250bcbfc95046caab3075fa318b41ebbf"
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
        MATCHED_REALIZED_UPDATER_AMENDMENT: (
            MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256
        ),
        MATCHED_UPDATER_INTEGRITY_AMENDMENT: (
            MATCHED_UPDATER_INTEGRITY_AMENDMENT_SHA256
        ),
        BUDGET_CHAIN_AMENDMENT: BUDGET_CHAIN_AMENDMENT_SHA256,
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
        "matched_realized_updater_amendment_sha256": (
            MATCHED_REALIZED_UPDATER_AMENDMENT_SHA256
        ),
        "matched_updater_integrity_amendment_sha256": (
            MATCHED_UPDATER_INTEGRITY_AMENDMENT_SHA256
        ),
        "budget_chain_amendment_sha256": BUDGET_CHAIN_AMENDMENT_SHA256,
        "protocol_manifest_sha256": protocol["manifest_sha256"],
    }


def main() -> int:
    print(json.dumps(verify_execution_bindings(), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
