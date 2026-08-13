from __future__ import annotations

from pathlib import Path

import pytest

from scripts import number_game_atomic_particle_v2_aug13_execute as execute
from scripts import number_game_atomic_particle_v2_mechanics as mechanics
from tests.test_number_game_atomic_particle_execute import catalog, live
from tests.test_number_game_atomic_particle_mechanics import SyntheticAdapter, synthetic_rules


class AuthorizedSyntheticAdapter(SyntheticAdapter):
    def __init__(self, authorize) -> None:
        super().__init__()
        self.authorize = authorize

    def complete(self, messages, seeds, *, response_format, max_tokens):
        for _ in seeds:
            self.authorize()
        return super().complete(messages, seeds, response_format=response_format, max_tokens=max_tokens)


def relocate(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "v2"
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
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": "b" * 64})


def test_full_fake_execute_traverses_every_request_and_writes_terminal(monkeypatch, tmp_path: Path):
    relocate(monkeypatch, tmp_path)
    created = {}

    def build_adapter(*, request_cap, authorize):
        created["adapter"] = AuthorizedSyntheticAdapter(authorize)
        return created["adapter"]

    monkeypatch.setattr(execute, "build_adapter", build_adapter)
    targets = synthetic_rules()[:33]
    monkeypatch.setattr(mechanics, "canonical_targets", lambda: targets)
    monkeypatch.setattr(
        mechanics,
        "build_classical_grammar_bank",
        lambda: (set(), {"sha256": mechanics.GRAMMAR_SHA256, "unique_nonconstant_extension_count": 0}),
    )
    result = execute.execute(live_reader=live, catalog_reader=catalog)
    assert result["status"] in {"atomic_particle_mechanics_pass", "atomic_particle_mechanics_null"}
    assert created["adapter"].requests == 6400
    assert execute.RESULT.exists() and execute.SCIENCE.exists() and execute.VERIFY.exists()
    assert not execute.FAILURE.exists()
    ledger = execute.load(execute.LEDGER)
    assert ledger["stage"]["status"] == result["status"]
    assert len(ledger["block_authorizations"]) > 1


def test_execute_banks_one_failed_closed_prefix_on_block_loss(monkeypatch, tmp_path: Path):
    relocate(monkeypatch, tmp_path)
    monkeypatch.setattr(execute, "build_adapter", lambda *, request_cap, authorize: AuthorizedSyntheticAdapter(authorize))
    calls = {"catalog": 0}

    def catalog_then_fail():
        calls["catalog"] += 1
        if calls["catalog"] >= 2:
            raise RuntimeError("injected block authorization loss")
        return catalog()

    with pytest.raises(RuntimeError, match="injected block"):
        execute.execute(live_reader=live, catalog_reader=catalog_then_fail)
    failure = execute.load(execute.FAILURE)
    assert failure["status"] == "failed_closed"
    assert failure["authorizes"] == "nothing"
    assert failure["raw_exists"] is False
    assert failure["scientific_result_exists"] is False
    assert not execute.RESULT.exists()
