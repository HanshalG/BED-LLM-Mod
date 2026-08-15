from __future__ import annotations

import json
from pathlib import Path
import random

import pytest

from scripts import number_game_stratified_atomic_particle_v3_codec as codec
from scripts import number_game_stratified_atomic_particle_v3_mechanics as mechanics
from scripts import number_game_stratified_atomic_particle_v3_verify as verifier
from tests.test_number_game_atomic_particle_mechanics import synthetic_rules


class StratifiedSyntheticAdapter:
    def __init__(self, authorize=None) -> None:
        self.rules = synthetic_rules()
        self.requests = 0
        self.rows: list[dict] = []
        self.authorize = authorize

    def complete(self, messages, seeds, *, response_format, max_tokens):
        assert response_format == mechanics.response_format()
        output = []
        for prompt, seed in zip(messages, seeds, strict=True):
            if self.authorize:
                self.authorize()
            payload = json.loads(prompt[1]["content"])
            history = tuple((row["number"], row["answer"] == "YES") for row in payload["observations"])
            anchors = tuple(row["number"] for row in payload["target_stratum"])
            signature = tuple(row["required_membership"] == "YES" for row in payload["target_stratum"])
            compatible = [
                rule for rule in self.rules
                if all(rule.extension[number] is answer for number, answer in (*history, *zip(anchors, signature, strict=True)))
            ]
            if not compatible:
                raise RuntimeError("synthetic stratum has no compatible rule")
            rule = compatible[random.Random(int(seed)).randrange(len(compatible))]
            output.append(json.dumps({"name": rule.name, "expression": rule.expression}))
            self.rows.append({
                "seed": int(seed),
                "model_requested": mechanics.MODEL_ID,
                "model_returned": mechanics.MODEL_ID,
                "finish_reasons": ["stop"],
                "prompt_sha256": mechanics.hashlib.sha256(
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
            "adapter_cost_usd": 0.25,
        }

    def records(self):
        return list(self.rows)


def test_signature_schedule_and_anchor_protection_are_exact():
    assert codec.MODEL_SEEDS == tuple(range(202608180000, 202608186400))
    assert codec.TREE_SEEDS == tuple(range(202608187000, 202608187004))
    anchors = codec.anchors_for((7, 25, 81))
    assert anchors == (12, 42, 5, 16, 33)
    assert not set(anchors) & {7, 25, 81}
    assert len({codec.signature_for_slot(slot, 32) for slot in range(32)}) == 32
    counts = {signature: 0 for signature in codec.SIGNATURES}
    for slot in range(64):
        counts[codec.signature_for_slot(slot, 64)] += 1
    assert set(counts.values()) == {2}


def test_prompt_serializes_signature_and_parser_rejects_noncompliance():
    anchors = codec.anchors_for(())
    signature = codec.SIGNATURES[0]
    messages = codec.particle_messages((), anchors=anchors, signature=signature)
    payload = json.loads(messages[1]["content"])
    assert [row["number"] for row in payload["target_stratum"]] == list(anchors)
    assert {row["required_membership"] for row in payload["target_stratum"]} == {"NO"}
    bad = codec.parse_stratified(json.dumps({"name": "even", "expression": "divisible(n, 2)"}), (), anchors, signature)
    assert bad.rejection == "signature"


def test_different_signatures_imply_different_extensions():
    anchors = codec.anchors_for(())
    rules = synthetic_rules()
    selected = {}
    for rule in rules:
        selected.setdefault(codec.particle_signature(rule, anchors), rule)
    assert len(selected) == 32
    assert len({rule.extension for rule in selected.values()}) == 32


def test_full_stratified_bank_replays_independently(tmp_path: Path):
    result = mechanics.produce_bank(output_dir=tmp_path, adapter=StratifiedSyntheticAdapter())
    assert len(result["raw"]["responses"]) == 6400
    assert result["topology"]["request_count"] == 6400
    assert all(result["topology"]["transport"]["gates"].values())
    assert all(bank["diagnostics"]["initial"]["unique_signatures"] == 32 for bank in result["banks"])
    assert all(not set(bank["first_blind_anchors"]) & set(bank["roots"]) for bank in result["banks"])
    verified = verifier.verify(tmp_path, output=tmp_path / "VERIFICATION.json")
    assert verified["status"] == "verification_pass"


def test_replay_rejects_signature_or_anchor_tampering(tmp_path: Path):
    mechanics.produce_bank(output_dir=tmp_path, adapter=StratifiedSyntheticAdapter())
    raw_path = tmp_path / "private/RAW_RESPONSES.json"
    raw = json.loads(raw_path.read_text())
    raw["responses"][0]["signature"] = "11111"
    raw_path.write_text(json.dumps(raw))
    with pytest.raises(RuntimeError, match="identity changed"):
        verifier.verify(tmp_path)
