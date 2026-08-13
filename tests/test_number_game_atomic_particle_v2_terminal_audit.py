from pathlib import Path

from scripts.number_game_atomic_particle_v2_terminal_audit import audit


def test_v2_initial_diversity_null_replays(tmp_path: Path):
    result = audit(output=tmp_path / "AUDIT.json")
    assert result["status"] == "terminal_audit_pass"
    assert result["diagnostic"]["valid_particles"] == 53
    assert result["diagnostic"]["unique_extensions"] == 15
    assert result["policy_endpoints_opened"] is False
