from __future__ import annotations

import hashlib
import json

import pytest
from pathlib import Path
from typing import Any, Sequence

from scripts import number_game_bitmask_semantic_codec as codec
from scripts import number_game_bitmask_semantic_gate as gate
from scripts import number_game_bitmask_semantic_verify as verifier


def mask(draw: int, index: int) -> str:
    bits = ["1" if hashlib.sha256(f"mask|{draw}|{index}|{number}".encode()).digest()[0] < 128 else "0" for number in range(101)]
    for number, answer in codec.HISTORIES[draw // 2]:
        bits[number] = "1" if answer else "0"
    return "".join(bits)


def proposal(draw: int) -> str:
    return json.dumps({"hypotheses": [{"name": f"Rule {draw} {index}", "description": f"Numbers in semantic family {draw} variant {index}", "membership_mask": mask(draw, index)} for index in range(24)]})


def audits(draw: int) -> list[tuple[bool, ...]]:
    rows = codec.parse_proposal(proposal(draw), codec.HISTORIES[draw // 2])
    return [tuple(row["mask"][probe] == "1" for probe in codec.probes_for(codec.PROPOSAL_SEEDS[draw], index, codec.HISTORIES[draw // 2])) for index, row in enumerate(rows)]


def audit_raw(draw: int) -> str:
    return json.dumps({"judgments": [{"hypothesis_index": index, "memberships": list(values)} for index, values in enumerate(audits(draw))]})


class FakeAdapter:
    def __init__(self, model: str, seeds: Sequence[int], responses: Sequence[str]) -> None:
        self.model, self.seeds, self.responses = model, tuple(seeds), list(responses)

    def complete(self, messages, seeds, *, response_format, max_tokens):
        assert len(messages) == len(seeds) == 10
        assert tuple(seeds) == self.seeds
        assert response_format["type"] == "json_schema"
        return list(self.responses)

    def usage_snapshot(self) -> dict[str, Any]:
        return {"adapter_requests": 10, "http_attempts": 10, "retry_count": 0, "adapter_reasoning_tokens": 0, "forced_exits": 0, "forced_final_requests": 0, "adapter_cost_usd": 0.001}

    def records(self):
        return [{"seed": seed, "model_requested": self.model, "model_returned": self.model, "prompt_sha256": str(seed), "payload_sha256": str(seed), "finish_reasons": ["stop"]} for seed in self.seeds]


def test_parse_and_probe_obedience():
    rows = codec.parse_proposal(proposal(3), codec.HISTORIES[1])
    assert len(rows) == len({row["mask_hash"] for row in rows}) == 24
    probes = codec.probes_for(codec.PROPOSAL_SEEDS[3], 4, codec.HISTORIES[1])
    assert probes == codec.probes_for(codec.PROPOSAL_SEEDS[3], 4, codec.HISTORIES[1])
    assert len(probes) == len(set(probes)) == 8
    assert probes[-2:] == (9, 18)


@pytest.mark.parametrize("bad", ["0" * 101, "1" * 101, "01", "x" * 101])
def test_parser_rejects_invalid_masks(bad):
    value = json.loads(proposal(0))
    value["hypotheses"][0]["membership_mask"] = bad
    with pytest.raises(ValueError):
        codec.parse_proposal(json.dumps(value), codec.HISTORIES[0])


def test_parser_rejects_history_contradiction_and_mask_reference():
    value = json.loads(proposal(0))
    current = list(value["hypotheses"][0]["membership_mask"])
    current[9] = "0"
    value["hypotheses"][0]["membership_mask"] = "".join(current)
    with pytest.raises(ValueError, match="contradicts"):
        codec.parse_proposal(json.dumps(value), codec.HISTORIES[0])
    value = json.loads(proposal(0))
    value["hypotheses"][0]["description"] = "Use the mask as a lookup"
    with pytest.raises(ValueError, match="lexical"):
        codec.parse_proposal(json.dumps(value), codec.HISTORIES[0])


def test_full_synthetic_semantic_pass():
    proposals = [codec.parse_proposal(proposal(draw), codec.HISTORIES[draw // 2]) for draw in range(10)]
    summary, gates = codec.diagnostics(proposals, [audits(draw) for draw in range(10)])
    assert summary["agreement_rate"] == 1.0
    assert all(gates.values())


def test_semantic_flip_fails_calibration():
    proposals = [codec.parse_proposal(proposal(draw), codec.HISTORIES[draw // 2]) for draw in range(10)]
    all_audits = [audits(draw) for draw in range(10)]
    all_audits[0] = [tuple(not value for value in row) for row in all_audits[0]]
    all_audits[1] = [tuple(not value for value in row) for row in all_audits[1]]
    _, gates = codec.diagnostics(proposals, all_audits)
    assert gates["pooled_agreement_at_least_90_percent"] is False
    assert gates["every_draw_at_least_18_semantic_valid"] is False


def test_full_producer_and_independent_replay(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.md"; protocol.write_text("frozen\n")
    monkeypatch.setattr(gate, "PROTOCOL", protocol)
    monkeypatch.setattr(gate, "PROTOCOL_SHA256", gate.file_digest(protocol))
    proposals = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, [proposal(draw) for draw in range(10)])
    audit_adapter = FakeAdapter(gate.AUDIT_MODEL_ID, codec.AUDIT_SEEDS, [audit_raw(draw) for draw in range(10)])
    result = gate.run_gate(output_dir=tmp_path / "run", proposal_adapter=proposals, audit_factory=lambda _messages: audit_adapter)
    assert result["status"] == "mechanics_pass"
    replay = verifier.verify(tmp_path / "run", output=tmp_path / "run/VERIFICATION.json")
    assert replay["status"] == "verification_pass"
    raw = json.loads((tmp_path / "run/private/RAW_RESPONSES.json").read_text())
    assert set(raw) == {"proposal", "audit"}
