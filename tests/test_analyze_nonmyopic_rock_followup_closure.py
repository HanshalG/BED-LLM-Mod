import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_nonmyopic_rock_followup_closure import analyze, render_report


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results/nonmyopic/rocksample_7_8_scale_20260721/L1.json"
SOURCE_11 = (
    ROOT
    / "results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json"
)


def test_followup_closure_screen_uses_all_bank_h2_states() -> None:
    summary = analyze(json.loads(SOURCE.read_text()))

    assert summary["no_llm_calls"]
    assert summary["strategy_eig"]["num_states"] == 270
    assert summary["random_strategy"]["num_states"] == 270
    assert summary["strategy_eig"]["closed_worsens_states"] == 0


def test_followup_closure_screen_rejects_wrong_source_run() -> None:
    payload = copy.deepcopy(json.loads(SOURCE.read_text()))
    payload["run_id"] = "wrong"

    with pytest.raises(AssertionError):
        analyze(payload)


def test_followup_closure_screen_supports_registered_eleven_rock_run() -> None:
    summary = analyze(json.loads(SOURCE_11.read_text()))

    assert summary["map_name"] == "11-11"
    assert summary["strategy_eig"]["num_states"] == 330
    assert summary["random_strategy"]["num_states"] == 330
    assert len(summary["strategy_eig"]["round_closed_fraction_mean"]) == 11
    assert render_report(summary).startswith(
        "# RockSample[11,11] Exact Follow-Up Closure Screen"
    )
