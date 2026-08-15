from __future__ import annotations

from copy import deepcopy

from scripts.chembench_policy_ladder_mechanics_verify import recompute_gate


def _slice_result(offset: int) -> dict:
    levels = {}
    for level, value in ((1, 1.0), (2, 0.8), (3, 0.6)):
        levels[f"d{level}"] = {
            "planned_value": value,
            "expected_terminal_mse": value,
            "root_action_index": offset + level,
            "truth_losses": [value, value],
        }
    return {
        "policy_levels": levels,
        "call_matched_d1": {
            "matches_primary_d1": True,
            "cache_misses_before": 3,
            "cache_misses_after": 3,
        },
    }


def test_recomputed_policy_ladder_gate_passes_calibrated_monotonic_result() -> None:
    result = recompute_gate([_slice_result(index) for index in range(3)], {"ok": True})
    assert result["passed"]
    assert result["conditions"]["planned_truth_replay_calibrated"]
    assert result["conditions"]["per_slice_model_risk_nonincreasing"]


def test_recomputed_policy_ladder_gate_rejects_calibration_mismatch() -> None:
    slices = [_slice_result(index) for index in range(3)]
    tampered = deepcopy(slices)
    tampered[0]["policy_levels"]["d3"]["planned_value"] += 0.01
    result = recompute_gate(tampered, {"ok": True})
    assert not result["passed"]
    assert not result["conditions"]["planned_truth_replay_calibrated"]
