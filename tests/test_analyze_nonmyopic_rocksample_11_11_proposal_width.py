import copy
import json
from pathlib import Path

from scripts.analyze_nonmyopic_rocksample_11_11_llm import EXPECTED_RUNS
from scripts.analyze_nonmyopic_rocksample_11_11_proposal_width import (
    RUN_KEYS,
    analyze,
)


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT
    / "results/nonmyopic/rocksample_11_11_gemma_slot_confirmation_20260721/L1.json"
)
RESULTS = tuple(
    ROOT
    / f"results/nonmyopic/rocksample_11_11_width_k{width}_seed_24085_20260721/L1.json"
    for width in (2, 4, 6)
)


def _payloads() -> list[dict]:
    reference = json.loads(REFERENCE.read_text())
    payloads = []
    for run_key in RUN_KEYS:
        expected = EXPECTED_RUNS[run_key]
        width = expected["num_strategies"]
        payload = copy.deepcopy(reference)
        payload["run_id"] = expected["run_id"]
        payload["config"]["seed"] = expected["seed"]
        payload["config"]["num_strategies"] = width
        for arm in ("strategy_eig", "shared_d1", "random_strategy"):
            for trace in payload["traces"]["11-11"][arm]:
                for step in trace["steps"]:
                    step["candidate_strategies"] = step["candidate_strategies"][:width]
        payloads.append(payload)
    return payloads


def test_width_auditor_reconstructs_all_runs() -> None:
    audit = analyze(_payloads())

    assert audit["all_18_intervals_passed"]
    assert set(audit["runs"]) == set(RUN_KEYS)
    assert audit["secondary_cross_width_comparisons"][
        "width_k6_minus_width_k2"
    ]["entropy_auc_gain"] == 0.0


def test_width_auditor_rejects_wrong_candidate_count() -> None:
    payloads = _payloads()
    payloads[0]["traces"]["11-11"]["strategy_eig"][0]["steps"][0][
        "candidate_strategies"
    ].append("extra")

    try:
        analyze(payloads)
    except AssertionError:
        pass
    else:
        raise AssertionError("auditor accepted a K2 step with three strategies")


def test_width_auditor_reconstructs_registered_artifacts() -> None:
    audit = analyze([json.loads(path.read_text()) for path in RESULTS])

    assert audit["all_18_intervals_passed"]
    assert audit["total_requests"] == 3157
    assert audit["secondary_cross_width_comparisons"][
        "width_k4_minus_width_k2"
    ]["entropy_auc_gain"] == 0.5655142301465437
