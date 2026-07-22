import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_nonmyopic_rocksample_15_15_exact import analyze


@pytest.fixture(scope="module")
def result_payload() -> dict:
    path = Path(
        "results/nonmyopic/rocksample_15_15_exact_qualification_20260722/REPORT.json"
    )
    if not path.exists():
        pytest.skip("formal frozen RockSample[15,15] qualification artifact is absent")
    return json.loads(path.read_text(encoding="utf-8"))


def test_exact_15_15_auditor_reconstructs_formal_result(result_payload: dict) -> None:
    audit = analyze(result_payload)

    assert audit["audit_passed"]
    assert audit["primary_gate_passed"]
    assert audit["truth_log_corroboration_passed"]
    assert audit["comparison"]["wins_ties_losses"] == [100, 0, 0]
    assert audit["depth_summaries"]["1"]["move_count"] == 0
    assert audit["depth_summaries"]["2"]["move_count"] == 500


def test_exact_15_15_auditor_rejects_trace_metric_mismatch(result_payload: dict) -> None:
    tampered = copy.deepcopy(result_payload)
    tampered["traces"]["2"][0]["entropy_auc"] += 0.1

    with pytest.raises(AssertionError):
        analyze(tampered)
