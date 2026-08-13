from __future__ import annotations

import json
from pathlib import Path

from scripts import revengebench_counterfactual_interface_audit as audit


def test_audit_excludes_java_target_without_python_surrogate(tmp_path: Path) -> None:
    source = tmp_path / "source"
    inverse = source / "src/revenge_bench/tournaments/inverse_strategy.py"
    inverse.parent.mkdir(parents=True)
    inverse.write_text(
        "\n".join(
            (
                "def _process_battlesnake_traces():",
                " self._query_learner(target_state)",
                " extract_state_action_pairs(sim_file, target_name)",
                "def _process_halite_traces():",
                " query_compiled_bot(",
                " extract_state_action_pairs(",
                "def _process_huskybench_traces():",
                " bot.get_action(round_state, remaining_chips)",
                " extract_state_action_pairs(sim_file, target_name)",
                "def _process_robocode_traces():",
                " self._query_learner(target_state)",
                " fallback = (",
                ') / "main.py"',
            )
        ),
        encoding="utf-8",
    )
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "arenas": {
                    "battlesnake": {"entrypoint": "main.py"},
                    "halite": {"entrypoint": "main.c"},
                    "huskybench": {"entrypoint": "player.py"},
                    "robocode": {"entrypoint": "MyTank.java"},
                }
            }
        ),
        encoding="utf-8",
    )

    result = audit.audit(source, manifest)

    assert result["status"] == "infrastructure_inconclusive"
    assert result["callable_arenas"] == ["battlesnake", "halite", "huskybench"]
    assert result["arenas"]["robocode"]["native_counterfactual_callable"] is False
    assert result["privacy"]["trajectory_or_outcome_opened"] is False
