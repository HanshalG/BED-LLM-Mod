from __future__ import annotations

from datetime import datetime
import threading
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_deepseek_stratified_semantic_v4_aug15_execute as v4
from scripts import number_game_deepseek_stratified_semantic_v5_router_aug15_execute as v5
from scripts.number_game_bitmask_semantic_gate import SeededAdapter
from tests.test_number_game_deepseek_stratified_semantic_v4 import SyntheticAdapter


NOW = datetime(2026, 8, 15, 16, tzinfo=ZoneInfo("Europe/London"))


def catalog(*, expensive_eligible: bool = False):
    rows = [
        {
            "provider_name": "cheap",
            "status": 0,
            "supported_parameters": ["seed", "reasoning", "response_format"],
            "pricing": {"prompt": "0.00000007", "completion": "0.00000014"},
        },
        {
            "provider_name": "ceiling",
            "status": 0,
            "supported_parameters": ["seed", "reasoning", "structured_outputs"],
            "pricing": {
                "prompt": "0.00000020",
                "completion": "0.00000051" if expensive_eligible else "0.00000050",
            },
        },
        {
            "provider_name": "ineligible-expensive",
            "status": 0,
            "supported_parameters": ["reasoning", "response_format"],
            "pricing": {"prompt": "0.1", "completion": "0.1"},
        },
        {
            "provider_name": "inactive-expensive",
            "status": 1,
            "supported_parameters": ["seed", "reasoning", "response_format"],
            "pricing": {"prompt": "0.1", "completion": "0.1"},
        },
    ]
    return {"data": {"id": v5.MODEL_ID, "endpoints": rows}}


def live(usage: float = 220.346566406):
    return {
        "total_credits_usd": 245.0,
        "total_usage_usd": usage,
        "balance_usd": 245.0 - usage,
    }


def relocate(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "v5"
    run = root / "run"
    monkeypatch.setattr(v5, "ROOT", root)
    monkeypatch.setattr(v5, "RUN", run)
    monkeypatch.setattr(v5, "RESULT", root / "RESULT.json")
    monkeypatch.setattr(v5, "FAILURE", root / "FAILURE.json")
    monkeypatch.setattr(v5, "LEDGER", root / "LEDGER.json")
    monkeypatch.setattr(v5, "RAW", run / "private/RAW_RESPONSES.json")
    monkeypatch.setattr(v5, "LABEL", run / "LABEL_FREE_RESULT.json")
    monkeypatch.setattr(v5, "VERIFY", run / "VERIFICATION.json")
    monkeypatch.setattr(
        v5,
        "validate_bindings",
        lambda: {"execution_binding_sha256": "e" * 64},
    )


def test_catalog_uses_componentwise_maximum_over_only_eligible_rows() -> None:
    prices = v5.validate_catalog(catalog())
    assert prices["provider"] == "parameter_constrained_router"
    assert prices["eligible_providers"] == ["ceiling", "cheap"]
    assert prices["prompt_price_usd_per_token"] == pytest.approx(0.20e-6)
    assert prices["completion_price_usd_per_token"] == pytest.approx(0.50e-6)
    with pytest.raises(RuntimeError, match="price increased"):
        v5.validate_catalog(catalog(expensive_eligible=True))


def test_router_payload_has_no_provider_pin(monkeypatch, tmp_path: Path) -> None:
    relocate(monkeypatch, tmp_path)
    monkeypatch.setenv("OPENROUTER_API_KEY", "unit-test-only")
    adapter = v5.build_adapter(request_cap=0.01, authorize=None)
    assert v5.RouterAdapter._payload is SeededAdapter._payload
    adapter._seed = threading.local()
    adapter._seed.value = 202608210000
    payload = adapter._payload(
        [{"role": "user", "content": "test"}],
        0.6,
        1,
        20,
        disable_reasoning=True,
        response_format={"type": "json_object"},
    )
    assert payload["provider"] == {"require_parameters": True}
    assert "order" not in payload["provider"]
    assert "allow_fallbacks" not in payload["provider"]
    assert payload["seed"] == 202608210000


def test_preflight_restores_closed_v4_executor_globals(monkeypatch, tmp_path: Path) -> None:
    relocate(monkeypatch, tmp_path)
    original = {
        "ROOT": v4.ROOT,
        "PROVIDER": v4.PROVIDER,
        "validate_catalog": v4.validate_catalog,
        "build_adapter": v4.build_adapter,
    }
    ready = v5.preflight(now=NOW, live_reader=live, catalog_reader=catalog)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(0.00729728)
    assert v4.ROOT == original["ROOT"]
    assert v4.PROVIDER == original["PROVIDER"]
    assert v4.validate_catalog is original["validate_catalog"]
    assert v4.build_adapter is original["build_adapter"]


def test_default_catalog_reader_uses_immutable_parent_function(monkeypatch) -> None:
    monkeypatch.setattr(v5, "PARENT_READ_CATALOG", catalog)
    with v5.configured_parent():
        assert v4.read_catalog is v5.read_catalog
        assert v5.read_catalog() == catalog()


def test_full_fake_execute_preserves_v4_science_and_writes_v5_terminal(
    monkeypatch, tmp_path: Path
) -> None:
    relocate(monkeypatch, tmp_path)
    created = {}

    def build_adapter(*, request_cap, authorize):
        created["adapter"] = SyntheticAdapter(authorize=authorize)
        return created["adapter"]

    monkeypatch.setattr(v5, "build_adapter", build_adapter)
    terminal = v5.execute(now=NOW, live_reader=live, catalog_reader=catalog)
    assert terminal["status"] == "semantic_pass"
    assert terminal["authorizes"] == "fresh_mechanics_protocol_only"
    assert created["adapter"].requests == 192
    assert v5.RESULT.exists() and v5.LABEL.exists() and v5.VERIFY.exists()
    assert not v5.FAILURE.exists()
    assert v4.ROOT != v5.ROOT


def test_execute_banks_failed_closed_prefix_and_restores_parent(
    monkeypatch, tmp_path: Path
) -> None:
    relocate(monkeypatch, tmp_path)
    original_root = v4.ROOT
    monkeypatch.setattr(
        v5,
        "build_adapter",
        lambda *, request_cap, authorize: SyntheticAdapter(authorize=authorize),
    )
    calls = {"catalog": 0}

    def catalog_then_fail():
        calls["catalog"] += 1
        if calls["catalog"] >= 2:
            raise RuntimeError("injected router block loss")
        return catalog()

    with pytest.raises(RuntimeError, match="injected router"):
        v5.execute(now=NOW, live_reader=live, catalog_reader=catalog_then_fail)
    failure = v5.load(v5.FAILURE)
    assert failure["status"] == "failed_closed"
    assert failure["authorizes"] == "nothing"
    assert failure["raw_exists"] is False
    assert v4.ROOT == original_root
