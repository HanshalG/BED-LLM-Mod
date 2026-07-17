import pytest

from helpers import load_config
from scripts.nonmyopic_rock_strategy_prior import StrategyProposalError
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


def test_successor_deepseek_config_uses_openrouter_reasoning_effort() -> None:
    smoke = load_config("configs/config_nonmyopic_strategy_successor_smoke_deepseek_v4_pro_openrouter.yaml")
    formal = load_config("configs/config_nonmyopic_copex_strategy_l3_successor_deepseek_v4_pro_openrouter.yaml")

    assert smoke.model_pairs[0].questioner.reasoning_effort == "medium"
    assert smoke.openrouter_max_output_tokens == 2048
    assert formal.model_pairs[0].questioner.model == "deepseek/deepseek-v4-pro"
    assert formal.openrouter_run_budget_usd == 3.0


def test_successor_smoke_exposes_partial_invalid_cells_on_failure() -> None:
    class InvalidModel:
        def chat_complete(self, messages, temperature, num_responses=1):
            del messages, temperature, num_responses
            return ["not json"]

        def usage_snapshot(self):
            return {"forced_exits": 0}

    with pytest.raises(StrategyProposalError) as raised:
        run_successor_smoke(InvalidModel(), SuccessorSmokeConfig())

    assert len(raised.value.l1_invalid_responses) == 1
    assert raised.value.l1_invalid_responses[0]["raw_response"] == "not json"
