from pathlib import Path

from scripts.number_game_overgenerated_factorized_terminal_audit import audit


def test_overgenerated_factorized_terminal_replays():
    result = audit(Path("results/nonmyopic/number_game_overgenerated_factorized_gate"))
    assert result["status"] == "terminal_audit_pass"
    assert result["proposal"]["valid_items"] == 302
    assert result["proposal"]["minimum_shard_valid"] == 0
