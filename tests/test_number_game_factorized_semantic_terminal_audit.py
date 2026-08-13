from pathlib import Path
from scripts.number_game_factorized_semantic_terminal_audit import audit
def test_factorized_terminal_replays():
    result=audit(Path("results/nonmyopic/number_game_factorized_semantic_gate"));assert result["status"]=="terminal_audit_pass";assert result["proposal"]["valid_items"]==231
