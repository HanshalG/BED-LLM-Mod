from __future__ import annotations

from scripts.chembench_adaptive_smc_v3 import _health_pass


def test_v3_health_gate_requires_acceptance_and_rung_bounds() -> None:
    summary = {
        "smc_banks": {
            "bank_1": {
                "max_num_rungs": 20,
                "aggregate_acceptance_rate": 0.3,
                "nonzero_acceptance_fraction": 1.0,
            },
            "bank_2": {
                "max_num_rungs": 21,
                "aggregate_acceptance_rate": 0.25,
                "nonzero_acceptance_fraction": 1.0,
            },
        }
    }
    assert _health_pass((summary,))
    summary["smc_banks"]["bank_2"]["max_num_rungs"] = 81
    assert not _health_pass((summary,))
