from __future__ import annotations

import subprocess
import sys

from scripts.chembench_mopen_mechanics import (
    AMENDMENT_PATH,
    AMENDMENT_SHA256,
    ARCHITECTURE_PATH,
    ARCHITECTURE_SHA256,
    PROTOCOL_PATH,
    PROTOCOL_SHA256,
    V1_TERMINAL_PATH,
    V1_TERMINAL_SHA256,
    V2_PROTOCOL_PATH,
    V2_PROTOCOL_SHA256,
    _sha256,
    verify_protocol_bindings,
)
from scripts.chembench_mopen_mechanics_verify import _compare


def test_mechanics_protocol_hashes_are_exact() -> None:
    verify_protocol_bindings()
    assert _sha256(PROTOCOL_PATH) == PROTOCOL_SHA256
    assert _sha256(AMENDMENT_PATH) == AMENDMENT_SHA256
    assert _sha256(ARCHITECTURE_PATH) == ARCHITECTURE_SHA256
    assert _sha256(V1_TERMINAL_PATH) == V1_TERMINAL_SHA256
    assert _sha256(V2_PROTOCOL_PATH) == V2_PROTOCOL_SHA256


def test_independent_comparison_uses_frozen_practical_tolerance() -> None:
    comparison = _compare([1.0, 1.0, 1.0], [0.9, 1.0 + 1e-8, 1.1])
    assert comparison["wins"] == 1
    assert comparison["ties"] == 1
    assert comparison["losses"] == 1
    assert comparison["tolerance"] == 1e-6


def test_mechanics_scripts_run_directly_without_pythonpath() -> None:
    for script in (
        "scripts/chembench_mopen_mechanics.py",
        "scripts/chembench_mopen_mechanics_verify.py",
    ):
        completed = subprocess.run(
            [sys.executable, script, "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
