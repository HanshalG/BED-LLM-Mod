from __future__ import annotations

import json
from pathlib import Path

from scripts.analyze_paprika_step1_rescue import analyze


def _write_run(root: Path, method: str, turns: list[int | None]) -> None:
    item = root / "items" / f"000_{method}"
    item.mkdir(parents=True)
    records = []
    for index, resolution in enumerate(turns):
        transcript = [
            {"mapped_cleanly": True, "goal_reached": resolution == turn}
            for turn in range(1, (resolution or 2) + 1)
        ]
        records.append(
            {"task_id": f"customer_service:eval:{index:04d}", "turns": transcript}
        )
    artifact = item / "paprika_smoke.json"
    artifact.write_text(json.dumps(records))
    (root / "metrics.json").write_text(
        json.dumps(
            {
                "items": [
                    {
                        "method": method,
                        "artifacts": {"paprika_smoke": str(artifact.relative_to(root))},
                        "metrics": {},
                    }
                ]
            }
        )
    )


def test_rescue_analyzer_reuses_frozen_matched_gate(tmp_path: Path) -> None:
    rescue = tmp_path / "rescue"
    matched = tmp_path / "matched"
    adversarial = tmp_path / "adversarial"
    _write_run(rescue, "EIG", [1, 1, 1, 1, 1, 1, None, None, None, None])
    _write_run(matched, "naive", [2, 2, None, None, None, None, None, None, None, None])
    _write_run(adversarial, "naive", [1, 1, 2, 2, None, None, None, None, None, None])
    result = analyze(rescue, matched, adversarial, round_budget=2)
    assert result["status"] == "rescue_pass_continue_claim1_transfer"
    assert result["claim1_matched_rescue_vs_naive_nonthinking"]["gate_pass"] is True
