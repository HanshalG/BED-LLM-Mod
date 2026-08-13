from __future__ import annotations

import hashlib
import json

import pytest

from scripts import number_game_factorized_semantic_codec as codec
from scripts import number_game_factorized_semantic_gate as gate
from scripts import number_game_factorized_semantic_verify as verifier
from scripts import number_game_factorized_semantic_aug13_execute as execute
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


def proposal_raw(draw: int, shard: int) -> str:
    return json.dumps({"hypotheses": [{"hypothesis_id": f"H{i + 1}", "name": f"Rule {draw} {shard} {i}", "description": f"Semantic family {draw} shard {shard} variant {i}"} for i in range(8)]})


def mask(draw: int, shard: int, item: int) -> str:
    bits = ["1" if hashlib.sha256(f"factor|{draw}|{shard}|{item}|{n}".encode()).digest()[0] < 128 else "0" for n in range(101)]
    for number, answer in codec.HISTORIES[draw // 2]:
        bits[number] = "1" if answer else "0"
    return "".join(bits)


def translation_raw(draw: int, shard: int) -> str:
    return json.dumps({"extensions": [{"hypothesis_id": f"H{i + 1}", "membership_mask": mask(draw, shard, i)} for i in range(8)]})


def make_draws():
    draws = []
    for draw in range(10):
        shards = []
        for shard in range(3):
            proposals = codec.parse_proposal(proposal_raw(draw, shard))
            shards.append(codec.parse_translation(translation_raw(draw, shard), proposals))
        merged, diagnostic = codec.merge_draw(shards, codec.HISTORIES[draw // 2])
        assert diagnostic == {"translated_unique_count": 24, "history_consistent_count": 24}
        draws.append(merged)
    return draws


def make_audits(draws):
    return [[tuple(row["mask"][probe] == "1" for probe in codec.probes_for(draw_index, index, codec.HISTORIES[draw_index // 2])) for index, row in enumerate(draw)] for draw_index, draw in enumerate(draws)]


def test_structural_blindness_and_injection_rejection():
    proposals = codec.parse_proposal(proposal_raw(0, 0))
    translation = codec.translation_messages(proposals)
    codec.assert_translation_blind(translation)
    payload = json.loads(translation[1]["content"])
    payload["observations"] = [{"number": 11, "answer": "YES"}]
    translation[1]["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="leaks"):
        codec.assert_translation_blind(translation)

    draw = make_draws()[0]
    audit = codec.audit_messages(draw, 0, codec.HISTORIES[0])
    codec.assert_audit_blind(audit)
    payload = json.loads(audit[1]["content"])
    payload["rules"][0]["membership_mask"] = draw[0]["mask"]
    audit[1]["content"] = json.dumps(payload)
    with pytest.raises(ValueError, match="leaks"):
        codec.assert_audit_blind(audit)


def test_proposal_parser_rejects_observed_answer_language():
    value = json.loads(proposal_raw(0, 0))
    value["hypotheses"][0]["description"] = "Numbers for which the answer is YES"
    with pytest.raises(ValueError, match="description"):
        codec.parse_proposal(json.dumps(value))


def test_translation_parser_rejects_bad_id_and_mask():
    proposals = codec.parse_proposal(proposal_raw(0, 0))
    value = json.loads(translation_raw(0, 0))
    value["extensions"][0]["hypothesis_id"] = "H2"
    with pytest.raises(ValueError, match="coverage"):
        codec.parse_translation(json.dumps(value), proposals)
    value = json.loads(translation_raw(0, 0))
    value["extensions"][0]["membership_mask"] = "0" * 101
    with pytest.raises(ValueError, match="mask"):
        codec.parse_translation(json.dumps(value), proposals)


def test_history_filter_is_not_repair():
    proposals = codec.parse_proposal(proposal_raw(0, 0))
    translated = codec.parse_translation(translation_raw(0, 0), proposals)
    bad = list(translated[0]["mask"])
    bad[11] = "0"
    translated[0] = {**translated[0], "mask": "".join(bad), "extension_hash": codec.sha256_text("".join(bad))}
    other_shards = [codec.parse_translation(translation_raw(0, shard), codec.parse_proposal(proposal_raw(0, shard))) for shard in (1, 2)]
    merged, diagnostic = codec.merge_draw([translated, *other_shards], codec.HISTORIES[0])
    assert diagnostic["history_consistent_count"] == 23
    assert merged[0]["history_consistent"] is False
    assert merged[0]["mask"][11] == "0"


def test_full_synthetic_factorized_pass_and_semantic_failure():
    draws = make_draws()
    audits = make_audits(draws)
    summary, gates = codec.diagnostics(draws, audits)
    assert summary["agreement_rate"] == 1.0
    assert all(gates.values())
    audits[0] = [tuple(not value for value in row) for row in audits[0]]
    audits[1] = [tuple(not value for value in row) for row in audits[1]]
    _, failed = codec.diagnostics(draws, audits)
    assert failed["pooled_agreement_at_least_90_percent"] is False
    assert failed["every_draw_at_least_18_semantic_valid_consistent"] is False


def audit_raw(draws):
    audits = make_audits(draws)
    return [json.dumps({"judgments": [{"hypothesis_index": index, "memberships": list(values)} for index, values in enumerate(draw)]}) for draw in audits]


class FakeAdapter:
    def __init__(self, model, seeds, responses): self.model, self.seeds, self.responses = model, tuple(seeds), list(responses); self.called = False
    def complete(self, messages, seeds, *, response_format, max_tokens):
        self.called = True; assert tuple(seeds) == self.seeds; assert len(messages) == len(self.responses); return list(self.responses)
    def usage_snapshot(self) -> dict[str, Any]:
        return {"adapter_requests": len(self.responses), "http_attempts": len(self.responses), "retry_count": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "forced_final_requests": 0, "adapter_cost_usd": 0.001}
    def records(self):
        return [{"seed": seed, "model_requested": self.model, "model_returned": self.model, "finish_reasons": ["stop"], "prompt_sha256": str(seed), "payload_sha256": str(seed)} for seed in self.seeds]


def test_full_producer_replay_and_public_privacy(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.md"; protocol.write_text("frozen\n")
    monkeypatch.setattr(gate, "PROTOCOL", protocol); monkeypatch.setattr(gate, "PROTOCOL_SHA256", gate.digest(protocol))
    proposal_responses = [proposal_raw(draw, shard) for draw in range(10) for shard in range(3)]
    translation_responses = [translation_raw(draw, shard) for draw in range(10) for shard in range(3)]
    draws = make_draws()
    p = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, proposal_responses)
    t = FakeAdapter(gate.TRANSLATION_MODEL_ID, codec.TRANSLATION_SEEDS, translation_responses)
    a = FakeAdapter(gate.AUDIT_MODEL_ID, codec.AUDIT_SEEDS, audit_raw(draws))
    result = gate.run_gate(output_dir=tmp_path / "run", proposal_adapter=p, translation_factory=lambda messages: t, audit_factory=lambda messages: a)
    assert result["status"] == "mechanics_pass"
    assert verifier.verify(tmp_path / "run")["status"] == "verification_pass"
    public = (tmp_path / "run/LABEL_FREE_RESULT.json").read_text().casefold()
    assert '"description"' not in public and '"membership_mask"' not in public


def test_fail_closed_phase_ordering(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.md"; protocol.write_text("frozen\n")
    monkeypatch.setattr(gate, "PROTOCOL", protocol); monkeypatch.setattr(gate, "PROTOCOL_SHA256", gate.digest(protocol))
    malformed = [proposal_raw(draw, shard) for draw in range(10) for shard in range(3)]; malformed[4] = "{"
    p = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, malformed)
    called = {"translation": False, "audit": False}
    with pytest.raises(json.JSONDecodeError):
        gate.run_gate(output_dir=tmp_path / "proposal-fail", proposal_adapter=p, translation_factory=lambda _: called.__setitem__("translation", True), audit_factory=lambda _: called.__setitem__("audit", True))
    assert called == {"translation": False, "audit": False}

    p = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, [proposal_raw(d, s) for d in range(10) for s in range(3)])
    malformed_t = [translation_raw(d, s) for d in range(10) for s in range(3)]; malformed_t[8] = "{"
    t = FakeAdapter(gate.TRANSLATION_MODEL_ID, codec.TRANSLATION_SEEDS, malformed_t)
    with pytest.raises(json.JSONDecodeError):
        gate.run_gate(output_dir=tmp_path / "translation-fail", proposal_adapter=p, translation_factory=lambda _: t, audit_factory=lambda _: called.__setitem__("audit", True))
    assert called["audit"] is False


def catalog():
    def row(model, prompt, completion): return {"id": model, "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]}, "supported_parameters": ["seed", "response_format"], "pricing": {"prompt": str(prompt), "completion": str(completion)}}
    return {"data": [row(gate.PROPOSAL_MODEL_ID, .32e-6, 1.28e-6), row(gate.TRANSLATION_MODEL_ID, .08e-6, .18e-6), row(gate.AUDIT_MODEL_ID, .10e-6, .60e-6)]}


def live(usage=execute.OPENING): return {"total_credits_usd": 245., "total_usage_usd": usage, "balance_usd": 245.-usage}


def test_phase_authorization_counts_prior_cost_and_account_race():
    messages = [codec.translation_messages(codec.parse_proposal(proposal_raw(0, 0))) for _ in range(30)]
    authorization = execute.authorize(gate.TRANSLATION_MODEL_ID, messages, gate.TRANSLATION_MAX_TOKENS, .04, live, catalog)
    assert .04 + authorization["exposure"]["tracker_reservation_usd"] <= execute.STAGE_CAP
    with pytest.raises(RuntimeError, match="stage cap"):
        execute.authorize(gate.TRANSLATION_MODEL_ID, messages, gate.TRANSLATION_MAX_TOKENS, .079, live, catalog)
    with pytest.raises(RuntimeError, match="account allowance"):
        execute.authorize(gate.TRANSLATION_MODEL_ID, messages, gate.TRANSLATION_MAX_TOKENS, 0., lambda: live(execute.OPENING + 4.999), catalog)


def test_preflight_is_zero_call_and_zero_write(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": "x" * 64})
    monkeypatch.setattr(execute, "RUN", tmp_path / "run"); monkeypatch.setattr(execute, "LEDGER", tmp_path / "ledger.json"); monkeypatch.setattr(execute, "RESULT", tmp_path / "result.json"); monkeypatch.setattr(execute, "FAILURE", tmp_path / "failure.json")
    result = execute.preflight(now=datetime(2026, 8, 13, 12, tzinfo=timezone.utc), live_reader=live, catalog_reader=catalog)
    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == result["files_written"] == 0
    assert not (tmp_path / "run").exists() and not (tmp_path / "ledger.json").exists()
