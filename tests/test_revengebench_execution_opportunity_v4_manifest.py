from __future__ import annotations

import json
from pathlib import Path

from scripts import revengebench_execution_opportunity_v4_manifest as manifest


def test_v4_preserves_v3_and_binds_feeder(tmp_path: Path) -> None:
    v3 = tmp_path / "v3.json"
    runner = tmp_path / "runner.py"
    v3.write_text(
        json.dumps({"protocol_version": "revengebench-execution-opportunity-v3",
                    "predecessor_bindings": {"v2": "abc"}, "arenas": {"halite": {}}}),
        encoding="utf-8",
    )
    runner.write_text("frame i before move i\n", encoding="utf-8")

    result = manifest.build(v3, runner)

    assert result["protocol_version"] == "revengebench-execution-opportunity-v4"
    assert result["arenas"] == {"halite": {}}
    assert result["halite_feeder_repair"]["first_getframe_contains_frame_zero"] is True
    assert result["halite_feeder_repair"]["v3_trajectories_reused"] is False
    assert result["privacy"]["corrected_halite_trajectory_opened"] is False
