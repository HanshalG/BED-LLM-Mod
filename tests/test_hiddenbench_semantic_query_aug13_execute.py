from __future__ import annotations

import math

import pytest

from scripts import hiddenbench_semantic_query_aug13_execute as execute


def live(usage: float = 220.20) -> dict[str, float]:
    return {
        "total_credits_usd": 245.0,
        "total_usage_usd": usage,
        "balance_usd": 245.0 - usage,
    }


def catalog(
    prompt: float = 0.09 / 1_000_000,
    completion: float = 0.18 / 1_000_000,
) -> dict:
    return {
        "data": [
            {
                "id": "deepseek/deepseek-v4-flash-0731",
                "architecture": {
                    "input_modalities": ["text"],
                    "output_modalities": ["text"],
                },
                "supported_parameters": ["seed", "response_format"],
                "pricing": {
                    "prompt": str(prompt),
                    "completion": str(completion),
                },
            }
        ]
    }


def test_account_values_use_frozen_aug13_boundary() -> None:
    assert execute.prior_spend(live()) == pytest.approx(
        220.20 - execute.OPENING_USAGE_USD
    )
    with pytest.raises(RuntimeError, match="invalid"):
        execute.validate_live(live(execute.OPENING_USAGE_USD - 0.01))
    malformed = live()
    malformed["balance_usd"] += 1
    with pytest.raises(RuntimeError, match="invalid"):
        execute.validate_live(malformed)


def test_catalog_accepts_only_frozen_nonincreased_price() -> None:
    checked = execute.validate_catalog(catalog())
    assert checked["id"] == "deepseek/deepseek-v4-flash-0731"
    assert checked["covered_prompt_tokens_at_live_price"] >= 8_000
    with pytest.raises(RuntimeError, match="price increased"):
        execute.validate_catalog(catalog(prompt=0.091 / 1_000_000))
    with pytest.raises(RuntimeError, match="price increased"):
        execute.validate_catalog(catalog(completion=0.181 / 1_000_000))


def test_reconciliation_uses_maximum_of_posted_and_local_spend() -> None:
    ledger = {
        "recorded_actual_spend_usd": 0.05,
        "execution_opening_total_usage_usd": 220.20,
        "stage": {"status": "authorized_pending"},
    }
    reconciled = execute.reconcile(
        ledger,
        status="serving_pass",
        local_cost=0.01,
        live=live(220.25),
    )
    assert math.isclose(
        reconciled["recorded_actual_spend_usd"],
        220.25 - execute.OPENING_USAGE_USD,
    )
