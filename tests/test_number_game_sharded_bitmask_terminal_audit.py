from pathlib import Path
from scripts.number_game_sharded_bitmask_terminal_audit import audit
def test_banked_sharded_terminal_replays():
    result=audit(Path("results/nonmyopic/number_game_sharded_bitmask_semantic_gate")); assert result["status"]=="terminal_audit_pass"; assert result["mechanics"]["history_contradicting_hypotheses"]==163
