from pathlib import Path

from scripts.number_game_atomic_particle_terminal_audit import audit


def test_atomic_particle_zero_call_failure_replays(tmp_path: Path):
    result = audit(output=tmp_path / "AUDIT.json")
    assert result["status"] == "terminal_audit_pass"
    assert result["model_calls_made"] == 0
    assert result["policy_endpoints_opened"] is False
