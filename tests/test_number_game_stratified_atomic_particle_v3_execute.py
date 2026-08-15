from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_stratified_atomic_particle_v3_aug15_execute as execute
from scripts import number_game_stratified_atomic_particle_v3_mechanics as mechanics
from tests.test_number_game_atomic_particle_execute import catalog, live
from tests.test_number_game_atomic_particle_mechanics import synthetic_rules
from tests.test_number_game_stratified_atomic_particle_v3 import StratifiedSyntheticAdapter


NOW = datetime(2026, 8, 15, 12, tzinfo=ZoneInfo("Europe/London"))


def v3_live(usage: float = execute.OPENING_USAGE_USD):
    return live(usage)


def relocate(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "v3"
    run = root / "run"
    monkeypatch.setattr(execute, "ROOT", root)
    monkeypatch.setattr(execute, "RUN", run)
    monkeypatch.setattr(execute, "RESULT", root / "RESULT.json")
    monkeypatch.setattr(execute, "FAILURE", root / "FAILURE.json")
    monkeypatch.setattr(execute, "LEDGER", root / "LEDGER.json")
    monkeypatch.setattr(execute, "RAW", run / "private/RAW_RESPONSES.json")
    monkeypatch.setattr(execute, "TOPOLOGY", run / "private/TOPOLOGY.json")
    monkeypatch.setattr(execute, "VERIFY", run / "LABEL_FREE_VERIFICATION.json")
    monkeypatch.setattr(execute, "SCIENCE", run / "SCIENTIFIC_RESULT.json")
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": "c" * 64})


def test_full_fake_v3_execute_writes_terminal(monkeypatch, tmp_path: Path):
    relocate(monkeypatch, tmp_path)
    created = {}

    def build_adapter(*, request_cap, authorize):
        created["adapter"] = StratifiedSyntheticAdapter(authorize=authorize)
        return created["adapter"]

    monkeypatch.setattr(execute, "build_adapter", build_adapter)
    targets = synthetic_rules()[:33]
    monkeypatch.setattr(mechanics, "canonical_targets", lambda: targets)
    monkeypatch.setattr(
        mechanics,
        "build_classical_grammar_bank",
        lambda: (set(), {"sha256": mechanics.GRAMMAR_SHA256, "unique_nonconstant_extension_count": 0}),
    )
    result = execute.execute(now=NOW, live_reader=v3_live, catalog_reader=catalog)
    assert result["status"] in {"atomic_particle_mechanics_pass", "atomic_particle_mechanics_null"}
    assert created["adapter"].requests == 6400
    assert execute.RESULT.exists() and execute.SCIENCE.exists() and execute.VERIFY.exists()
    assert not execute.FAILURE.exists()
    ledger = execute.load(execute.LEDGER)
    assert ledger["stage"]["status"] == result["status"]
    assert len(ledger["block_authorizations"]) > 1


def test_v3_execute_banks_failed_closed_prefix(monkeypatch, tmp_path: Path):
    relocate(monkeypatch, tmp_path)
    monkeypatch.setattr(execute, "build_adapter", lambda *, request_cap, authorize: StratifiedSyntheticAdapter(authorize=authorize))
    calls = {"catalog": 0}

    def catalog_then_fail():
        calls["catalog"] += 1
        if calls["catalog"] >= 2:
            raise RuntimeError("injected V3 block loss")
        return catalog()

    with pytest.raises(RuntimeError, match="injected V3"):
        execute.execute(now=NOW, live_reader=v3_live, catalog_reader=catalog_then_fail)
    failure = execute.load(execute.FAILURE)
    assert failure["status"] == "failed_closed"
    assert failure["raw_exists"] is False
    assert failure["scientific_result_exists"] is False
    assert not execute.RESULT.exists()
