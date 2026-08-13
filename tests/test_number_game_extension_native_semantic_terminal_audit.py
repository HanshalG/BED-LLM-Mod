from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.number_game_extension_native_semantic_terminal_audit import audit


ROOT = Path("results/nonmyopic/number_game_extension_native_semantic_gate")


def test_banked_terminal_failure_replays():
    result = audit(ROOT)
    assert result["status"] == "terminal_audit_pass"
    assert result["transport"]["accepted_requests"] == 10
    assert result["transport"]["audit_requests"] == 0


def test_audit_rejects_open_authority(tmp_path: Path):
    target = tmp_path / "gate"
    target.mkdir()
    for source in ROOT.rglob("*"):
        if source.is_file():
            destination = target / source.relative_to(ROOT)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
    ledger_source = ROOT.parent / "openrouter_daily_budget/2026-08-13-number-game-extension-native-semantic.json"
    ledger_target = target.parent / "openrouter_daily_budget/2026-08-13-number-game-extension-native-semantic.json"
    ledger_target.parent.mkdir(parents=True)
    ledger_target.write_bytes(ledger_source.read_bytes())
    failure = target / "DAILY_FAILURE_20260813.json"
    value = json.loads(failure.read_text())
    value["authorizes"] = "endpoint"
    failure.write_text(json.dumps(value))
    with pytest.raises(RuntimeError):
        audit(target)
