from __future__ import annotations

from datetime import datetime
import json
import hashlib
from pathlib import Path
import random
from zoneinfo import ZoneInfo

import pytest

from scripts import number_game_deepseek_stratified_semantic_v4 as gate
from scripts import number_game_deepseek_stratified_semantic_v4_verify as verifier
from scripts import number_game_deepseek_stratified_semantic_v4_aug15_execute as execute
from tests.test_number_game_atomic_particle_mechanics import synthetic_rules


NOW = datetime(2026, 8, 15, 15, tzinfo=ZoneInfo("Europe/London"))


class SyntheticAdapter:
    def __init__(self, authorize=None, *, compliant: bool = True) -> None:
        self.rules = synthetic_rules()
        self.authorize = authorize
        self.compliant = compliant
        self.requests = 0
        self.rows = []

    def complete(self, messages, seeds, *, response_format, max_tokens):
        assert response_format == gate.response_format()
        assert max_tokens == gate.MAX_TOKENS
        output = []
        for prompt, seed in zip(messages, seeds, strict=True):
            if self.authorize:
                self.authorize()
            if self.compliant:
                payload = json.loads(prompt[1]["content"])
                history = tuple(
                    (row["number"], row["answer"] == "YES")
                    for row in payload["observations"]
                )
                anchors = tuple(row["number"] for row in payload["target_stratum"])
                signature = tuple(
                    row["required_membership"] == "YES"
                    for row in payload["target_stratum"]
                )
                compatible = [
                    rule
                    for rule in self.rules
                    if all(
                        rule.extension[number] is answer
                        for number, answer in (*history, *zip(anchors, signature, strict=True))
                    )
                ]
                rule = compatible[random.Random(int(seed)).randrange(len(compatible))]
                value = json.dumps({"name": rule.name, "expression": rule.expression})
            else:
                value = json.dumps({"name": "even", "expression": "divisible(n, 2)"})
            output.append(value)
            self.rows.append({
                "seed": int(seed),
                "model_requested": gate.MODEL_ID,
                "model_returned": gate.MODEL_ID,
                "finish_reasons": ["stop"],
                "prompt_sha256": hashlib.sha256(
                    json.dumps(prompt, sort_keys=True, separators=(",", ":")).encode()
                ).hexdigest(),
            })
            self.requests += 1
        return output

    def usage_snapshot(self):
        return {
            "adapter_requests": self.requests,
            "http_attempts": self.requests,
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "adapter_cost_usd": 0.01,
        }

    def records(self):
        return list(self.rows)


def catalog():
    return {
        "data": {
            "id": gate.MODEL_ID,
            "endpoints": [{
                "provider_name": execute.PROVIDER,
                "status": 0,
                "supported_parameters": ["seed", "reasoning", "response_format"],
                "pricing": {"prompt": "0.00000007", "completion": "0.00000014"},
            }],
        }
    }


def live(usage: float = 220.346566406):
    return {
        "total_credits_usd": 245.0,
        "total_usage_usd": usage,
        "balance_usd": 245.0 - usage,
    }


def relocate(monkeypatch, tmp_path: Path) -> None:
    root = tmp_path / "v4"
    run = root / "run"
    monkeypatch.setattr(execute, "ROOT", root)
    monkeypatch.setattr(execute, "RUN", run)
    monkeypatch.setattr(execute, "RESULT", root / "RESULT.json")
    monkeypatch.setattr(execute, "FAILURE", root / "FAILURE.json")
    monkeypatch.setattr(execute, "LEDGER", root / "LEDGER.json")
    monkeypatch.setattr(execute, "RAW", run / "private/RAW_RESPONSES.json")
    monkeypatch.setattr(execute, "LABEL", run / "LABEL_FREE_RESULT.json")
    monkeypatch.setattr(execute, "VERIFY", run / "VERIFICATION.json")
    monkeypatch.setattr(
        execute,
        "validate_bindings",
        lambda: {"execution_binding_sha256": "d" * 64},
    )


def test_request_schedule_covers_every_signature_at_each_history() -> None:
    assert gate.MODEL_SEEDS == tuple(range(202608210000, 202608210192))
    for group_index, (_, history) in enumerate(gate.GROUPS):
        rows = gate.group_requests(group_index)
        assert len(rows) == 64
        assert all(row["observations"] == [list(item) for item in history] for row in rows)
        counts = {signature: 0 for signature in {row["signature"] for row in rows}}
        for row in rows:
            counts[row["signature"]] += 1
            assert not set(row["protected"]) & set(row["anchors"])
        assert len(counts) == 32 and set(counts.values()) == {2}


def test_complete_synthetic_gate_passes_and_replays(tmp_path: Path) -> None:
    result = gate.produce(output_dir=tmp_path, adapter=SyntheticAdapter())
    assert result["status"] == "semantic_pass"
    assert result["accepted_requests"] == 192
    assert all(group["passed"] for group in result["groups"])
    assert verifier.verify(tmp_path)["status"] == "verification_pass"


def test_semantic_failure_stops_after_first_group(tmp_path: Path) -> None:
    result = gate.produce(
        output_dir=tmp_path,
        adapter=SyntheticAdapter(compliant=False),
    )
    assert result["status"] == "semantic_null"
    assert result["accepted_requests"] == 64
    assert result["authorizes"] == "nothing"
    assert verifier.verify(tmp_path)["status"] == "verification_pass"


def test_verifier_rejects_identity_tampering(tmp_path: Path) -> None:
    gate.produce(output_dir=tmp_path, adapter=SyntheticAdapter())
    raw_path = tmp_path / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    raw["responses"][0]["signature"] = "11111"
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="replay failed"):
        verifier.verify(tmp_path)


def test_catalog_and_preflight_are_fail_closed(monkeypatch, tmp_path: Path) -> None:
    prices = execute.validate_catalog(catalog())
    assert prices["provider"] == execute.PROVIDER
    assert execute.maximum_request_exposure(prices) >= execute.exposure(
        execute.initial_block(), prices
    )["maximum_request_exposure_usd"]
    expensive = catalog()
    expensive["data"]["endpoints"][0]["pricing"]["completion"] = "0.00000015"
    with pytest.raises(RuntimeError, match="capability or price"):
        execute.validate_catalog(expensive)
    relocate(monkeypatch, tmp_path)
    ready = execute.preflight(now=NOW, live_reader=live, catalog_reader=catalog)
    assert ready["status"] == "ready_without_paid_calls"
    assert ready["budget"]["prior_account_spend_usd"] == pytest.approx(0.00729728)


def test_full_fake_execute_writes_terminal(monkeypatch, tmp_path: Path) -> None:
    relocate(monkeypatch, tmp_path)
    created = {}

    def build_adapter(*, request_cap, authorize):
        created["adapter"] = SyntheticAdapter(authorize=authorize)
        return created["adapter"]

    monkeypatch.setattr(execute, "build_adapter", build_adapter)
    terminal = execute.execute(now=NOW, live_reader=live, catalog_reader=catalog)
    assert terminal["status"] == "semantic_pass"
    assert terminal["authorizes"] == "fresh_mechanics_protocol_only"
    assert created["adapter"].requests == 192
    assert execute.RESULT.exists() and execute.LABEL.exists() and execute.VERIFY.exists()
    assert not execute.FAILURE.exists()


def test_execute_banks_failed_closed_prefix(monkeypatch, tmp_path: Path) -> None:
    relocate(monkeypatch, tmp_path)
    monkeypatch.setattr(
        execute,
        "build_adapter",
        lambda *, request_cap, authorize: SyntheticAdapter(authorize=authorize),
    )
    calls = {"catalog": 0}

    def catalog_then_fail():
        calls["catalog"] += 1
        if calls["catalog"] >= 2:
            raise RuntimeError("injected DeepSeek block loss")
        return catalog()

    with pytest.raises(RuntimeError, match="injected DeepSeek"):
        execute.execute(now=NOW, live_reader=live, catalog_reader=catalog_then_fail)
    failure = execute.load(execute.FAILURE)
    assert failure["status"] == "failed_closed"
    assert failure["authorizes"] == "nothing"
    assert failure["raw_exists"] is False
