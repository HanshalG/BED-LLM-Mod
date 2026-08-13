from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import Any, Sequence

import pytest
from datetime import datetime, timezone

from scripts import number_game_extension_native_semantic_codec as codec
from scripts import number_game_extension_native_semantic_gate as gate
from scripts import number_game_extension_native_semantic_verify as verifier
from scripts import number_game_extension_native_semantic_aug13_execute as execute


def extension(draw_index: int, hypothesis_index: int) -> list[int]:
    values = {
        number
        for number in range(101)
        if hashlib.sha256(
            f"synthetic|{draw_index}|{hypothesis_index}|{number}".encode()
        ).digest()[0]
        < 128
    }
    for number, answer in codec.HISTORIES[draw_index // 2]:
        values.add(number) if answer else values.discard(number)
    if not values:
        values.add((draw_index + hypothesis_index + 1) % 101)
    if len(values) == 101:
        values.discard(100)
    return sorted(values)


def proposal_raw(draw_index: int) -> str:
    return json.dumps({"hypotheses": [{"name": f"Family {draw_index} {index}", "description": f"Numbers satisfying semantic family {draw_index} variant {index}", "members": extension(draw_index, index)} for index in range(24)]})


def audit_raw(draw_index: int) -> str:
    rows = []
    proposals = codec.parse_proposal(proposal_raw(draw_index), codec.HISTORIES[draw_index // 2])
    for index, proposal in enumerate(proposals):
        members = set(proposal["members"])
        probes = codec.probes_for(codec.PROPOSAL_SEEDS[draw_index], index, codec.HISTORIES[draw_index // 2])
        rows.append({"hypothesis_index": index, "memberships": [probe in members for probe in probes]})
    return json.dumps({"judgments": rows})


class FakeAdapter:
    def __init__(self, model: str, seeds: Sequence[int], responses: Sequence[str], cost: float = 0.001) -> None:
        self.model = model
        self.seeds = tuple(seeds)
        self.responses = list(responses)
        self.cost = cost

    def complete_seeded(self, batch_messages, seeds, *, response_format, max_new_tokens):
        assert len(batch_messages) == len(seeds) == 10
        assert tuple(seeds) == self.seeds
        assert response_format["type"] == "json_schema"
        assert max_new_tokens in (gate.PROPOSAL_MAX_TOKENS, gate.AUDIT_MAX_TOKENS)
        return list(self.responses)

    def usage_snapshot(self) -> dict[str, Any]:
        return {"adapter_requests": 10, "http_attempts": 10, "retry_count": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "forced_final_requests": 0, "adapter_prompt_tokens": 100, "adapter_completion_tokens": 100, "adapter_cost_usd": self.cost}

    def request_records(self) -> list[dict[str, Any]]:
        return [{"seed": seed, "model_requested": self.model, "model_returned": self.model, "prompt_sha256": f"p{seed}", "payload_sha256": f"x{seed}", "finish_reasons": ["stop"], "provider_error": False} for seed in self.seeds]


def test_probe_schedule_is_deterministic_unique_and_contains_observations():
    history = codec.HISTORIES[2]
    first = codec.probes_for(codec.PROPOSAL_SEEDS[4], 7, history)
    assert first == codec.probes_for(codec.PROPOSAL_SEEDS[4], 7, history)
    assert len(first) == len(set(first)) == 8
    assert first[-2:] == (8, 16)


@pytest.mark.parametrize("field,value", [("description", "members are 1, 2, 3, 4"), ("members", [3, 2, 1])])
def test_proposal_parser_rejects_lexical_and_extension_failures(field, value):
    payload = json.loads(proposal_raw(0))
    payload["hypotheses"][0][field] = value
    with pytest.raises(ValueError):
        codec.parse_proposal(json.dumps(payload), ())


def test_proposal_parser_rejects_history_contradiction():
    payload = json.loads(proposal_raw(2))
    payload["hypotheses"][0]["members"] = [number for number in payload["hypotheses"][0]["members"] if number != 8]
    with pytest.raises(ValueError, match="contradicts"):
        codec.parse_proposal(json.dumps(payload), codec.HISTORIES[1])


def test_audit_parser_rejects_missing_row():
    payload = json.loads(audit_raw(0))
    payload["judgments"].pop()
    with pytest.raises(ValueError):
        codec.parse_audit(json.dumps(payload))


def test_complete_synthetic_gate_and_independent_replay(tmp_path: Path, monkeypatch):
    amendment = tmp_path / "amendment.md"
    amendment.write_text("bound\n")
    protocol = tmp_path / "protocol.md"
    protocol.write_text("bound\n")
    monkeypatch.setattr(gate, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(gate, "AMENDMENT_PATH", amendment)
    monkeypatch.setattr(gate, "PROTOCOL_SHA256", gate.file_digest(protocol))
    monkeypatch.setattr(gate, "AMENDMENT_SHA256", gate.file_digest(amendment))
    proposal_adapter = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, [proposal_raw(i) for i in range(10)])
    audit_adapter = FakeAdapter(gate.AUDIT_MODEL_ID, codec.AUDIT_SEEDS, [audit_raw(i) for i in range(10)])
    result = gate.run_gate(output_dir=tmp_path / "run", proposal_adapter=proposal_adapter, audit_adapter_factory=lambda _messages: audit_adapter)
    assert result["status"] == "mechanics_pass"
    assert result["semantic"]["agreement_rate"] == 1.0
    replay = verifier.verify(tmp_path / "run", output_path=tmp_path / "run/VERIFICATION.json")
    assert replay["status"] == "verification_pass"


def test_semantic_disagreement_fails_gate():
    proposals = [codec.parse_proposal(proposal_raw(i), codec.HISTORIES[i // 2]) for i in range(10)]
    audits = [codec.parse_audit(audit_raw(i)) for i in range(10)]
    audits[0] = [tuple(not value for value in row) for row in audits[0]]
    audits[1] = [tuple(not value for value in row) for row in audits[1]]
    _, gates = codec.semantic_diagnostics(proposals, audits)
    assert gates["pooled_auditor_agreement_at_least_90_percent"] is False
    assert gates["every_draw_has_at_least_18_semantic_valid"] is False


def catalog(prompt: float = 0.32e-6, completion: float = 1.28e-6) -> dict[str, Any]:
    def row(model_id, p, c):
        return {"id": model_id, "architecture": {"input_modalities": ["text", "image"], "output_modalities": ["text"]}, "supported_parameters": ["seed", "response_format"], "pricing": {"prompt": str(p), "completion": str(c)}}
    return {"data": [row(gate.PROPOSAL_MODEL_ID, prompt, completion), row(gate.AUDIT_MODEL_ID, 0.10e-6, 0.60e-6)]}


def live(usage: float = execute.OPENING_USAGE_USD) -> dict[str, float]:
    return {"total_credits_usd": 245.0, "total_usage_usd": usage, "balance_usd": 245.0 - usage}


def test_catalog_and_exposure_fail_closed():
    with pytest.raises(RuntimeError, match="price increased"):
        execute.validate_catalog(catalog(completion=1.29e-6), gate.PROPOSAL_MODEL_ID)
    prices = execute.validate_catalog(catalog(), gate.PROPOSAL_MODEL_ID)
    messages = [codec.proposal_messages(()) for _ in range(10)]
    exposure = execute.phase_exposure(messages, max_tokens=gate.PROPOSAL_MAX_TOKENS, prices=prices)
    assert exposure["tracker_reservation_usd"] < execute.STAGE_CAP_USD
    assert exposure["tracker_reservation_usd"] >= exposure["exact_phase_exposure_usd"]


def test_authorize_phase_rejects_negative_boundary_and_daily_race():
    messages = [codec.proposal_messages(()) for _ in range(10)]
    with pytest.raises(RuntimeError, match="account values"):
        execute.authorize_phase(model_id=gate.PROPOSAL_MODEL_ID, messages=messages, max_tokens=gate.PROPOSAL_MAX_TOKENS, accepted_cost_usd=0.0, live_reader=lambda: live(execute.OPENING_USAGE_USD - 0.01), catalog_reader=catalog)
    with pytest.raises(RuntimeError, match="account allowance"):
        execute.authorize_phase(model_id=gate.PROPOSAL_MODEL_ID, messages=messages, max_tokens=gate.PROPOSAL_MAX_TOKENS, accepted_cost_usd=0.0, live_reader=lambda: live(execute.OPENING_USAGE_USD + 4.99), catalog_reader=catalog)


def test_preflight_is_zero_call_and_writes_nothing(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.md"; protocol.write_text("p\n")
    amendment = tmp_path / "amendment.md"; amendment.write_text("a\n")
    codec_path = tmp_path / "codec.py"; codec_path.write_text("c\n")
    producer = tmp_path / "producer.py"; producer.write_text("d\n")
    verify_path = tmp_path / "verify.py"; verify_path.write_text("e\n")
    wrapper = tmp_path / "wrapper.py"; wrapper.write_text("f\n")
    tests = tmp_path / "tests.py"; tests.write_text("g\n")
    paths = {"protocol": protocol, "budget_amendment": amendment, "codec": codec_path, "producer": producer, "verifier": verify_path, "wrapper": wrapper, "tests": tests}
    binding = {name: {"path": path.name, "sha256": execute.sha256_file(path)} for name, path in paths.items()}
    binding.update({"date": execute.DATE, "opening_total_usage_usd": execute.OPENING_USAGE_USD, "daily_cap_usd": execute.DAILY_CAP_USD, "stage_cap_usd": execute.STAGE_CAP_USD, "proposal_requests": 10, "audit_requests": 10, "maximum_http_attempts": 20, "maximum_retries": 0, "number_game_targets_authorized": False, "policy_endpoints_authorized": False})
    binding_path = tmp_path / "binding.json"; binding_path.write_text(json.dumps(binding))
    monkeypatch.setattr(execute, "EXECUTION_BINDING", binding_path)
    monkeypatch.setattr(execute, "RUN_DIR", tmp_path / "run")
    monkeypatch.setattr(execute, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(execute, "DAILY_RESULT", tmp_path / "result.json")
    monkeypatch.setattr(execute, "DAILY_FAILURE", tmp_path / "failure.json")
    monkeypatch.setattr(execute.serving, "PROTOCOL_PATH", protocol)
    monkeypatch.setattr(execute.serving, "AMENDMENT_PATH", amendment)
    monkeypatch.setattr(execute, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": execute.sha256_file(binding_path)})
    result = execute.preflight(now=datetime(2026, 8, 13, 12, tzinfo=timezone.utc), live_reader=live, catalog_reader=catalog)
    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == result["files_written"] == 0
    assert not (tmp_path / "run").exists()
