import copy
import json
from pathlib import Path

from scripts.analyze_nonmyopic_rocksample_11_11_llm import EXPECTED_RUNS
from scripts.analyze_nonmyopic_rocksample_11_11_multiseed import RUN_KEYS, analyze


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json"
)


def _payloads() -> list[dict]:
    reference = json.loads(REFERENCE.read_text())
    payloads = []
    for run_key in RUN_KEYS:
        expected = EXPECTED_RUNS[run_key]
        payload = copy.deepcopy(reference)
        payload["run_id"] = expected["run_id"]
        payload["config"]["seed"] = expected["seed"]
        payloads.append(payload)
    return payloads


def test_multiseed_auditor_reconstructs_all_runs_and_pools() -> None:
    audit = analyze(_payloads())

    assert audit["all_18_intervals_passed"]
    assert set(audit["runs"]) == set(RUN_KEYS)
    assert audit["pooled_fresh_90_pair_comparisons"]["shared_d1"][
        "entropy_auc_wins_ties_losses"
    ] == [90, 0, 0]


def test_multiseed_auditor_reports_a_failed_interval() -> None:
    payloads = _payloads()
    paired = payloads[1]["maps"]["11-11"]["paired"]
    paired["strategy_eig_minus_shared_d1"]["entropy_auc_gain_ci95"][0] = -0.1
    payloads[1]["gate_passed"] = False
    payloads[1]["maps"]["11-11"]["gate_passed"] = False

    audit = analyze(payloads)

    assert not audit["all_per_seed_primary_gates_passed"]
    assert not audit["all_18_intervals_passed"]
