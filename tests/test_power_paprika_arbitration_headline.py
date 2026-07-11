from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.power_paprika_arbitration_headline import power_plan


def test_power_plan_freezes_maximum_for_half_pilot_effect(tmp_path: Path) -> None:
    report = tmp_path / "pilot.json"
    deltas = [-5, -1, 0, 0, -2, -2, 0, -1, 0, -1]
    report.write_text(
        json.dumps(
            {
                "primary_arbitration_vs_naive_thinking": {
                    "per_task": [
                        {"censored_turn_delta": value} for value in deltas
                    ]
                }
            }
        )
    )
    result = power_plan(report)
    assert result["pilot_mean_censored_turn_delta"] == pytest.approx(-1.2)
    assert result["assumed_absolute_effect"] == pytest.approx(0.6)
    assert result["normal_approx_required_tasks"] == 53
    assert result["selected_tasks"] == 50
    assert result["normal_approx_power_at_selected_n"] == pytest.approx(0.7827, abs=1e-3)
    assert result["task_range"]["end_offset_inclusive"] == 59
