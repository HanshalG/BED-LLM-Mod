from helpers import load_config
from scripts.nonmyopic_strategy_successor_smoke import (
    SuccessorSmokeConfig,
    _RoutingDeterministicModel,
    run_successor_smoke,
)


def test_successor_smoke_requires_ten_strict_parseable_cells() -> None:
    result = run_successor_smoke(_RoutingDeterministicModel(), SuccessorSmokeConfig())

    assert result["status"] == "passed"
    assert result["checks"] == {
        "successful_generation_cells": 10,
        "l1_successful_cells": 5,
        "l3_successful_cells": 5,
        "zero_invalid_or_repaired_cells": True,
        "zero_forced_exits": True,
    }
    assert len(result["l1_requests"]) == 5
    assert len(result["l3_requests"]) == 5


def test_successor_qwen_configs_preserve_thinking_and_hard_caps() -> None:
    smoke = load_config("configs/config_nonmyopic_strategy_successor_smoke_qwen397_thinking_openrouter.yaml")
    formal = load_config("configs/config_nonmyopic_copex_strategy_l3_successor_qwen397_thinking_openrouter.yaml")

    assert smoke.model_pairs[0].questioner.model == "qwen/qwen3.5-397b-a17b"
    assert smoke.model_pairs[0].questioner.thinking is True
    assert smoke.model_pairs[0].questioner.thinking_max_new_tokens == 512
    assert smoke.model_pairs[0].questioner.thinking_final_max_new_tokens == 256
    assert smoke.openrouter_run_budget_usd == 0.05
    assert formal.openrouter_run_budget_usd == 3.0
    assert formal.openrouter_projected_cost_usd == 2.75
