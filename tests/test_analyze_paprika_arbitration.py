from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_paprika_arbitration import analyze


def _write_run(root: Path, method: str, resolution_turns: list[int | None]) -> None:
    item_dir = root / "items" / f"000_{method}"
    item_dir.mkdir(parents=True)
    records = []
    for index, resolution in enumerate(resolution_turns):
        turns = [
            {"mapped_cleanly": True, "goal_reached": resolution == turn}
            for turn in range(1, (resolution or 2) + 1)
        ]
        records.append(
            {"task_id": f"customer_service:eval:{index:04d}", "turns": turns}
        )
    artifact = item_dir / "paprika_smoke.json"
    artifact.write_text(json.dumps(records))
    metrics = {
        "structured_parse_failures": [0],
        "simulator_faithfulness_observations": [10],
        "simulator_faithfulness_checks": [10],
        "simulator_faithfulness_raw_contradictions": [0],
        "simulator_faithfulness_repairs": [0],
        "simulator_faithfulness_failures": [0],
        "simulator_faithfulness_final_inconsistency_rate": [0],
        "simulator_terminal_claims": [1],
        "simulator_terminal_checks": [1],
        "simulator_terminal_rejections": [0],
    }
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "method": method,
                        "artifacts": {"paprika_smoke": str(artifact.relative_to(root))},
                        "metrics": metrics,
                    }
                ]
            }
        )
    )


def test_arbitration_analyzer_applies_frozen_primary_gate(tmp_path: Path) -> None:
    arbitration = tmp_path / "arbitration"
    thinking = tmp_path / "thinking"
    nonthinking = tmp_path / "nonthinking"
    _write_run(
        arbitration,
        "NaivePrimaryArbitration",
        [1, 1, 2, 2, 2, 2, None, None, None, None],
    )
    _write_run(thinking, "naive", [2, 2, None, None, None, None, None, None, None, None])
    _write_run(nonthinking, "naive", [1, 1, None, None, None, None, None, None, None, None])
    result = analyze(arbitration, nonthinking, thinking, round_budget=2)
    assert result["status"] == "arbitration_pass_stop_and_discuss"
    assert result["endpoint_valid"] is True
    assert result["primary_arbitration_vs_naive_thinking"]["gate_pass"] is True
