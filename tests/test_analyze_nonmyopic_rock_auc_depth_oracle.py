import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_nonmyopic_rock_auc_depth_oracle import analyze


ROOT = Path(__file__).resolve().parents[1]
ALIGNED = ROOT / "results/nonmyopic/rocksample_7_8_auc_depth3_oracle_20260721/REPORT.json"
TERMINAL = ROOT / "results/nonmyopic/rocksample_7_8_depth3_oracle_20260721/REPORT.json"
REFERENCE = ROOT / "results/nonmyopic/rocksample_7_8_scale_20260721/L1.json"


@pytest.mark.skipif(not ALIGNED.exists(), reason="formal AUC-aligned artifact is not packaged yet")
def test_auc_depth_analyzer_triangulates_reference_results() -> None:
    audit = analyze(
        json.loads(ALIGNED.read_text()),
        json.loads(TERMINAL.read_text()),
        json.loads(REFERENCE.read_text()),
    )

    assert audit["audit_passed"]
    assert audit["same_utility_depth_monotonicity_passed"]
    assert not audit["strict_gain_over_prior_best_d2"]
    assert audit["mechanism"]["aligned_d3_matches_external_d2_curve"]


@pytest.mark.skipif(not ALIGNED.exists(), reason="formal AUC-aligned artifact is not packaged yet")
def test_auc_depth_analyzer_rejects_failed_mechanics() -> None:
    payload = copy.deepcopy(json.loads(ALIGNED.read_text()))
    payload["mechanics"]["all_selected_actions_legal"] = False

    with pytest.raises(AssertionError):
        analyze(
            payload,
            json.loads(TERMINAL.read_text()),
            json.loads(REFERENCE.read_text()),
        )
