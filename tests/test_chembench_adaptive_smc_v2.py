from __future__ import annotations

from scripts.chembench_adaptive_smc_v2 import _bank_ratio_pass


def test_bank_ratio_gate_handles_small_and_material_mse() -> None:
    assert _bank_ratio_pass({"mse": {"smc_1": 1e-8, "smc_2": 9e-7}})
    assert _bank_ratio_pass({"mse": {"smc_1": 0.1, "smc_2": 0.15}})
    assert not _bank_ratio_pass({"mse": {"smc_1": 0.1, "smc_2": 0.151}})
