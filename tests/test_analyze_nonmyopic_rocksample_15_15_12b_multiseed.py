from __future__ import annotations

import scripts.analyze_nonmyopic_rocksample_15_15_12b_multiseed as twelve_b
import scripts.analyze_nonmyopic_rocksample_15_15_vllm_multiseed as base
from scripts.analyze_nonmyopic_rock_branch_result import ARMS, BASELINES


def _payload(gain: float) -> dict:
    traces = {
        arm: [
            {
                "steps": [
                    {
                        "entropy_after": 4.0 if arm == "strategy_eig" else 4.0 + gain,
                        "truth_log_probability": -4.0
                        if arm == "strategy_eig"
                        else -4.0 - gain,
                    }
                ]
            }
            for _ in range(30)
        ]
        for arm in ARMS
    }
    return {"traces": {"15-15": traces}}


def test_12b_multiseed_auditor_uses_frozen_run_set_and_pooling(
    monkeypatch,
) -> None:
    def fake_analyze_run(_payload, run_key):
        return {
            "label": run_key,
            "primary_gate_passed": True,
            "truth_log_corroboration_passed": True,
            "usage": {"requests": 10, "run_cost_usd": 0.0},
        }

    monkeypatch.setattr(base, "analyze_run", fake_analyze_run)
    audit = twelve_b.analyze([_payload(0.4), _payload(0.5), _payload(0.6)])

    assert twelve_b.RUN_KEYS == (
        "12b_vllm",
        "12b_vllm_seed_24107",
        "12b_vllm_seed_24108",
    )
    assert audit["all_12_fresh_seed_intervals_passed"]
    assert audit["all_three_direct_vllm_seeds_passed"]
    assert audit["bootstrap"]["base_seed"] == 24109
    assert audit["pooled_90_pair_comparisons"]["shared_d1"][
        "entropy_auc_gain"
    ] == 0.5
    assert audit["total_requests"] == 30


def test_12b_multiseed_requires_each_fresh_seed_to_pass(monkeypatch) -> None:
    def fake_analyze_run(_payload, run_key):
        return {
            "label": run_key,
            "primary_gate_passed": run_key != "12b_vllm_seed_24108",
            "truth_log_corroboration_passed": True,
            "usage": {"requests": 10, "run_cost_usd": 0.0},
        }

    monkeypatch.setattr(base, "analyze_run", fake_analyze_run)
    audit = twelve_b.analyze([_payload(0.4), _payload(0.5), _payload(0.6)])

    assert not audit["all_fresh_seed_primary_gates_passed"]
    assert not audit["all_12_fresh_seed_intervals_passed"]
