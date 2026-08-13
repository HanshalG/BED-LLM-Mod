from __future__ import annotations

import json
from pathlib import Path

from scripts import revengebench_execution_opportunity_terminal_audit as audit


RESULT = Path("results/nonmyopic/revengebench_execution_opportunity/TERMINAL_RESULT.json")


def test_banked_terminal_result_passes() -> None:
    result = audit.audit(RESULT)

    assert result["status"] == "pass"
    assert all(result["gates"].values())


def test_positive_margin_tamper_is_rejected(tmp_path: Path) -> None:
    value = json.loads(RESULT.read_text())
    for beta in value["arenas"]["battlesnake"]["beta_results"].values():
        beta["depth_two_margin_nats"] = 0.02
        beta["changed_first_action"] = True
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value))

    assert audit.audit(path)["status"] == "fail"


def test_three_decision_halite_tamper_is_rejected(tmp_path: Path) -> None:
    value = json.loads(RESULT.read_text())
    value["arenas"]["halite"]["failed_cell"]["decision_count_per_arm"] = [3, 3]
    path = tmp_path / "result.json"
    path.write_text(json.dumps(value))

    assert audit.audit(path)["status"] == "fail"
