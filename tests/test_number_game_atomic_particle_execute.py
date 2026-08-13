from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_atomic_particle_aug13_execute as execute


def catalog(*, prompt: float = .32e-6, completion: float = 1.28e-6):
    return {
        "data": [
            {
                "id": execute.MODEL_ID,
                "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
                "supported_parameters": ["seed", "response_format"],
                "pricing": {"prompt": str(prompt), "completion": str(completion)},
            }
        ]
    }


def live(usage: float = 220.334124806):
    return {
        "total_credits_usd": 245.0,
        "total_usage_usd": usage,
        "balance_usd": 245.0 - usage,
    }


def test_catalog_and_exposure_are_exact_and_price_fail_closed():
    prices = execute.validate_catalog(catalog())
    block = execute.exposure(execute.initial_block(), prices)
    assert block["request_count"] == execute.mechanics.BLOCK_SIZE
    assert 0 < block["exact_block_exposure_usd"] < execute.STAGE_CAP_USD
    assert execute.maximum_request_exposure(prices) >= block["maximum_request_exposure_usd"]
    with pytest.raises(RuntimeError, match="capability or price"):
        execute.validate_catalog(catalog(prompt=.33e-6))


def test_live_account_validation_rejects_negative_or_inconsistent_values():
    assert execute.prior_spend(live()) == pytest.approx(0.199995926)
    bad = live()
    bad["balance_usd"] += 1
    with pytest.raises(RuntimeError, match="account values invalid"):
        execute.validate_live(bad)


def test_preflight_is_zero_call_and_requires_pristine_paths(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": "a" * 64})
    monkeypatch.setattr(execute, "RESULT", tmp_path / "RESULT.json")
    monkeypatch.setattr(execute, "FAILURE", tmp_path / "FAILURE.json")
    monkeypatch.setattr(execute, "RUN", tmp_path / "run")
    monkeypatch.setattr(execute, "LEDGER", tmp_path / "ledger.json")
    now = datetime(2026, 8, 13, 12, tzinfo=ZoneInfo("Europe/London"))
    result = execute.preflight(now=now, live_reader=live, catalog_reader=catalog)
    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == result["files_written"] == 0
    (tmp_path / "RESULT.json").write_text("{}")
    with pytest.raises(RuntimeError, match="not pristine"):
        execute.preflight(now=now, live_reader=live, catalog_reader=catalog)


def test_preflight_counts_unrelated_account_spend(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": "a" * 64})
    monkeypatch.setattr(execute, "RESULT", tmp_path / "RESULT.json")
    monkeypatch.setattr(execute, "FAILURE", tmp_path / "FAILURE.json")
    monkeypatch.setattr(execute, "RUN", tmp_path / "run")
    monkeypatch.setattr(execute, "LEDGER", tmp_path / "ledger.json")
    now = datetime(2026, 8, 13, 12, tzinfo=ZoneInfo("Europe/London"))
    with pytest.raises(RuntimeError, match="allowance unavailable"):
        execute.preflight(
            now=now,
            live_reader=lambda: live(execute.OPENING_USAGE_USD + 4.99),
            catalog_reader=catalog,
        )
