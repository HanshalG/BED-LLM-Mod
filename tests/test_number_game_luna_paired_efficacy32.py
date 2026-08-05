from scripts import number_game_deepseek_v4_flash_paired_efficacy32 as base
from scripts import number_game_luna_paired_efficacy32 as luna


def test_luna_smoke_is_hash_bound_and_passed() -> None:
    with luna.configured_base():
        result = base.validate_smoke_result(luna.SMOKE_RESULT)

    assert result["status"] == "passed"
    assert result["protocol"]["model"] == luna.MODEL_ID
    assert result["usage"]["adapter_reasoning_tokens"] == 0


def test_luna_configuration_is_scoped() -> None:
    original_model = base.MODEL_ID
    original_budget = base.RUN_BUDGET_USD
    original_bootstrap = base.BOOTSTRAP_SEED

    with luna.configured_base():
        assert base.MODEL_ID == luna.MODEL_ID
        assert base.RUN_BUDGET_USD == luna.RUN_BUDGET_USD
        assert base.BOOTSTRAP_SEED == luna.BOOTSTRAP_SEED

    assert base.MODEL_ID == original_model
    assert base.RUN_BUDGET_USD == original_budget
    assert base.BOOTSTRAP_SEED == original_bootstrap
