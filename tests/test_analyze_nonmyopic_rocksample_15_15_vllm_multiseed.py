import json
from pathlib import Path

import scripts.analyze_nonmyopic_rocksample_15_15_vllm_multiseed as multiseed


ROOT = Path(__file__).resolve().parents[1]
RESULTS = (
    ROOT / "results/nonmyopic/rocksample_15_15_vllm_replication_20260722/L1.json",
    ROOT / "results/nonmyopic/rocksample_15_15_vllm_seed_24102_20260722/L1.json",
    ROOT / "results/nonmyopic/rocksample_15_15_vllm_seed_24103_20260722/L1.json",
)


def _payloads() -> list[dict]:
    return [json.loads(path.read_text()) for path in RESULTS]


def test_multiseed_auditor_reconstructs_all_runs_and_pools() -> None:
    audit = multiseed.analyze(_payloads())

    assert audit["all_12_fresh_seed_intervals_passed"]
    assert audit["all_three_direct_vllm_seeds_passed"]
    assert set(audit["runs"]) == set(multiseed.RUN_KEYS)
    assert audit["pooled_90_pair_comparisons"]["shared_d1"][
        "entropy_auc_wins_ties_losses"
    ] == [90, 0, 0]


def test_multiseed_auditor_reports_a_failed_fresh_seed(monkeypatch) -> None:
    payloads = _payloads()
    original_analyze_run = multiseed.analyze_run

    def fail_seed_24102(payload: dict, run_key: str) -> dict:
        run = original_analyze_run(payload, run_key)
        if run_key == "vllm_seed_24102":
            run["primary_gate_passed"] = False
        return run

    monkeypatch.setattr(multiseed, "analyze_run", fail_seed_24102)

    audit = multiseed.analyze(payloads)

    assert not audit["all_fresh_seed_primary_gates_passed"]
    assert not audit["all_12_fresh_seed_intervals_passed"]


def test_multiseed_auditor_reconstructs_registered_artifacts() -> None:
    audit = multiseed.analyze(_payloads())

    assert audit["all_12_fresh_seed_intervals_passed"]
    assert audit["all_three_direct_vllm_seeds_passed"]
    assert audit["total_requests"] == 3964
