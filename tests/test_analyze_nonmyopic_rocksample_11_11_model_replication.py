import copy
import json
from pathlib import Path

import pytest

from scripts.analyze_nonmyopic_rocksample_11_11_model_replication import analyze


ROOT = Path(__file__).resolve().parents[1]
GEMMA = (
    ROOT
    / "results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json"
)
GPT = (
    ROOT
    / "results/nonmyopic/rocksample_11_11_gpt54mini_slot_replication_20260721/L1.json"
)


def _payloads() -> tuple[dict, dict]:
    return json.loads(GEMMA.read_text()), json.loads(GPT.read_text())


def test_cross_model_audit_reconstructs_both_runs() -> None:
    audit = analyze(*_payloads())

    assert audit["all_primary_gates_passed"]
    assert audit["all_truth_log_gates_passed"]
    assert audit["runs"]["gpt54_mini"]["comparisons"]["shared_d1"][
        "entropy_auc_gain"
    ] == pytest.approx(0.581070858516133)
    assert audit["runs"]["gpt54_mini"]["resume"]["accepted_cells_reused"] == 1038


def test_cross_model_audit_rejects_trace_mismatch() -> None:
    gemma, gpt = _payloads()
    gpt = copy.deepcopy(gpt)
    gpt["traces"]["11-11"]["random_strategy"][2]["steps"][1][
        "truth_log_probability"
    ] += 0.2

    with pytest.raises(AssertionError):
        analyze(gemma, gpt)
