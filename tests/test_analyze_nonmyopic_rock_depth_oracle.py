import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_nonmyopic_rock_depth_oracle import analyze


ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "results/nonmyopic/rocksample_7_8_depth3_oracle_20260721/REPORT.json"
REFERENCE = ROOT / "results/nonmyopic/rocksample_7_8_scale_20260721/L1.json"


@pytest.mark.skipif(not RESULT.exists(), reason="formal depth-three artifact is not packaged yet")
def test_depth_oracle_analyzer_audits_formal_artifact() -> None:
    audit = analyze(json.loads(RESULT.read_text()), json.loads(REFERENCE.read_text()))

    assert audit["audit_passed"]
    assert not audit["primary_gate_passed"]
    assert audit["prior_d2_reference_matched"]
    assert audit["mechanism"]["first_divergence_round"] == 6


@pytest.mark.skipif(not RESULT.exists(), reason="formal depth-three artifact is not packaged yet")
def test_depth_oracle_analyzer_rejects_unpaired_truth() -> None:
    payload = json.loads(RESULT.read_text())
    payload = copy.deepcopy(payload)
    payload["traces"]["3"][4]["truth_index"] = 999

    with pytest.raises(AssertionError):
        analyze(payload)
