from __future__ import annotations

import json
from pathlib import Path

from scripts import revengebench_execution_opportunity_v3_manifest as manifest


def test_v3_preserves_v2_and_binds_exact_patch(tmp_path: Path) -> None:
    v2 = tmp_path / "v2.json"
    patch = tmp_path / "repair.patch"
    v2.write_text(
        json.dumps(
            {
                "protocol_version": "revengebench-execution-opportunity-v2",
                "predecessor_bindings": {"v1": "abc"},
                "arenas": {"battlesnake": {}, "halite": {}, "huskybench": {}},
                "betas": [0.5, 1.0, 2.0],
            }
        ),
        encoding="utf-8",
    )
    patch.write_text("frozen patch\n", encoding="utf-8")

    result = manifest.build(v2, patch)

    assert result["protocol_version"] == "revengebench-execution-opportunity-v3"
    assert result["arenas"] == {"battlesnake": {}, "halite": {}, "huskybench": {}}
    assert result["betas"] == [0.5, 1.0, 2.0]
    assert result["mechanics_repair"]["minimum_fresh_arm_preflight"] == 3
    assert result["mechanics_repair"]["v2_trajectories_reused"] is False
    assert result["privacy"]["repaired_trajectory_opened"] is False
