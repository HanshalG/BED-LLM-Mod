from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from scripts import number_game_overgenerated_factorized_aug13_execute as execute
from scripts import number_game_overgenerated_factorized_codec as codec
from scripts import number_game_overgenerated_factorized_gate as gate
from scripts import number_game_overgenerated_factorized_verify as verifier


def proposal_value(draw: int, shard: int) -> dict[str, Any]:
    return {
        "hypotheses": [
            {
                "hypothesis_id": f"H{i + 1}",
                "name": f"Rule {draw} {shard} {i}",
                "description": f"Semantic family {draw} shard {shard} variant {i}",
            }
            for i in range(8)
        ]
    }


def proposal_raw(draw: int, shard: int, invalid: tuple[int, str] | None = None) -> str:
    value = proposal_value(draw, shard)
    if invalid:
        index, kind = invalid
        if kind == "observed":
            value["hypotheses"][index]["description"] = "Numbers whose observed answer is YES"
        elif kind == "lexical":
            value["hypotheses"][index]["description"] = "lambda n: n % 2 == 0"
        elif kind == "duplicate_id":
            value["hypotheses"][index]["hypothesis_id"] = "H1"
        else:
            raise AssertionError(kind)
    return json.dumps(value)


def proposal_responses(*, one_invalid_per_shard: bool = False) -> list[str]:
    return [
        proposal_raw(draw, shard, (7, "observed") if one_invalid_per_shard else None)
        for draw in range(10)
        for shard in range(4)
    ]


def mask(draw: int, chunk: int, item: int) -> str:
    bits = [
        "1" if hashlib.sha256(f"overgenerated|{draw}|{chunk}|{item}|{number}".encode()).digest()[0] < 128 else "0"
        for number in range(101)
    ]
    for number, answer in codec.HISTORIES[draw // 2]:
        bits[number] = "1" if answer else "0"
    return "".join(bits)


def translation_raw(draw: int, chunk: int) -> str:
    return json.dumps({
        "extensions": [
            {"hypothesis_id": f"H{i + 1}", "membership_mask": mask(draw, chunk, i)}
            for i in range(8)
        ]
    })


def make_draws(proposals: list[str] | None = None):
    selected, proposal_diagnostics = codec.parse_and_select_proposals(proposals or proposal_responses())
    chunks = [chunk for draw in selected for chunk in codec.translation_chunks(draw)]
    translated = [
        codec.parse_translation(translation_raw(draw, chunk), chunks[3 * draw + chunk])
        for draw in range(10)
        for chunk in range(3)
    ]
    draws = [codec.merge_draw(translated[3 * draw : 3 * draw + 3], codec.HISTORIES[draw // 2])[0] for draw in range(10)]
    return draws, proposal_diagnostics


def make_audits(draws):
    return [
        [
            tuple(row["mask"][probe] == "1" for probe in codec.probes_for(draw_index, index, codec.HISTORIES[draw_index // 2]))
            for index, row in enumerate(draw)
        ]
        for draw_index, draw in enumerate(draws)
    ]


def audit_raw(draws):
    return [
        json.dumps({
            "judgments": [
                {"hypothesis_index": index, "memberships": list(values)}
                for index, values in enumerate(draw)
            ]
        })
        for draw in make_audits(draws)
    ]


def test_item_isolated_filter_and_deterministic_first_24_selection():
    parsed, diagnostic = codec.parse_proposal_shard(proposal_raw(0, 0, (7, "observed")), shard_index=0)
    assert len(parsed) == diagnostic["valid_count"] == 7
    assert diagnostic["rejection_counts"] == {"observed_answer_language": 1}

    selected, draw_diagnostics = codec.parse_and_select_proposals(proposal_responses(one_invalid_per_shard=True))
    assert all(row["valid_count"] == 28 for row in draw_diagnostics)
    assert all(row["proposal_shard_valid_counts"] == [7, 7, 7, 7] for row in [
        {
            "proposal_shard_valid_counts": [shard["valid_count"] for shard in draw["shards"]]
        }
        for draw in draw_diagnostics
    ])
    expected = [f"S{shard}H{item}" for shard in range(1, 4) for item in range(1, 8)] + ["S4H1", "S4H2", "S4H3"]
    assert [row["source_hypothesis_id"] for row in selected[0]] == expected


def test_shard_and_draw_thresholds_fail_closed():
    value = proposal_value(0, 0)
    value["hypotheses"][6]["description"] = "The observed answer is YES"
    value["hypotheses"][7]["description"] = "lambda n: n % 2 == 0"
    with pytest.raises(ValueError, match="fewer than seven"):
        codec.parse_proposal_shard(json.dumps(value), shard_index=0)

    shards = []
    diagnostics = []
    for shard in range(4):
        parsed, diagnostic = codec.parse_proposal_shard(proposal_raw(0, shard, (7, "observed")), shard_index=shard)
        shards.append(parsed)
        diagnostics.append(diagnostic)
    shards[-1] = shards[-1][:-1]
    with pytest.raises(ValueError, match="fewer than 28"):
        codec.select_draw(shards, diagnostics)


def test_duplicate_uniqueness_is_symmetric_not_first_wins():
    value = proposal_value(0, 0)
    value["hypotheses"][7]["name"] = value["hypotheses"][6]["name"]
    with pytest.raises(ValueError, match="fewer than seven"):
        codec.parse_proposal_shard(json.dumps(value), shard_index=0)


def test_structural_blindness_and_private_selection_metadata():
    selected, _ = codec.parse_and_select_proposals(proposal_responses(one_invalid_per_shard=True))
    translation_batch, chunks = gate.build_translation_batch(selected)
    assert len(translation_batch) == len(chunks) == 30
    for messages in translation_batch:
        codec.assert_translation_blind(messages)
        payload = json.loads(messages[1]["content"])
        assert set(payload) == {"domain", "rules"}
        assert all(set(row) == {"hypothesis_id", "name", "description"} for row in payload["rules"])
        serialized = messages[1]["content"].casefold()
        assert "observations" not in serialized and "answer" not in serialized and "source_hypothesis_id" not in serialized

    draws, _ = make_draws(proposal_responses(one_invalid_per_shard=True))
    for draw_index, draw in enumerate(draws):
        messages = codec.audit_messages(draw, draw_index, codec.HISTORIES[draw_index // 2])
        codec.assert_audit_blind(messages)
        serialized = messages[1]["content"].casefold()
        assert "membership_mask" not in serialized and "observations" not in serialized and "answer" not in serialized


def test_full_synthetic_pass_and_semantic_failure():
    draws, proposal_diagnostics = make_draws(proposal_responses(one_invalid_per_shard=True))
    audits = make_audits(draws)
    summary, gates = codec.diagnostics(draws, audits, proposal_diagnostics)
    assert summary["agreement_rate"] == 1.0
    assert all(gates.values())
    audits[0] = [tuple(not value for value in row) for row in audits[0]]
    audits[1] = [tuple(not value for value in row) for row in audits[1]]
    _, failed = codec.diagnostics(draws, audits, proposal_diagnostics)
    assert failed["pooled_agreement_at_least_90_percent"] is False
    assert failed["every_draw_at_least_18_semantic_valid_consistent"] is False


class FakeAdapter:
    def __init__(self, model, seeds, responses):
        self.model, self.seeds, self.responses = model, tuple(seeds), list(responses)
        self.called = False

    def complete(self, messages, seeds, *, response_format, max_tokens):
        self.called = True
        assert tuple(seeds) == self.seeds
        assert len(messages) == len(self.responses)
        return list(self.responses)

    def usage_snapshot(self) -> dict[str, Any]:
        return {
            "adapter_requests": len(self.responses),
            "http_attempts": len(self.responses),
            "retry_count": 0,
            "adapter_reasoning_tokens": 0,
            "forced_exits": 0,
            "forced_final_requests": 0,
            "adapter_cost_usd": 0.001,
        }

    def records(self):
        return [
            {
                "seed": seed,
                "model_requested": self.model,
                "model_returned": self.model,
                "finish_reasons": ["stop"],
                "prompt_sha256": str(seed),
                "payload_sha256": str(seed),
            }
            for seed in self.seeds
        ]


def test_full_producer_replay_and_public_privacy(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen\n")
    monkeypatch.setattr(gate, "PROTOCOL", protocol)
    monkeypatch.setattr(gate, "PROTOCOL_SHA256", gate.digest(protocol))
    proposals = proposal_responses(one_invalid_per_shard=True)
    draws, _ = make_draws(proposals)
    proposal_adapter = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, proposals)
    translation_adapter = FakeAdapter(
        gate.TRANSLATION_MODEL_ID,
        codec.TRANSLATION_SEEDS,
        [translation_raw(draw, chunk) for draw in range(10) for chunk in range(3)],
    )
    audit_adapter = FakeAdapter(gate.AUDIT_MODEL_ID, codec.AUDIT_SEEDS, audit_raw(draws))
    result = gate.run_gate(
        output_dir=tmp_path / "run",
        proposal_adapter=proposal_adapter,
        translation_factory=lambda messages: translation_adapter,
        audit_factory=lambda messages: audit_adapter,
    )
    assert result["status"] == "mechanics_pass"
    assert verifier.verify(tmp_path / "run")["status"] == "verification_pass"
    public = (tmp_path / "run/LABEL_FREE_RESULT.json").read_text().casefold()
    assert '"description"' not in public and '"membership_mask"' not in public and '"observations"' not in public


def test_fail_closed_phase_ordering(tmp_path: Path, monkeypatch):
    protocol = tmp_path / "protocol.md"
    protocol.write_text("frozen\n")
    monkeypatch.setattr(gate, "PROTOCOL", protocol)
    monkeypatch.setattr(gate, "PROTOCOL_SHA256", gate.digest(protocol))
    malformed = proposal_responses()
    malformed[4] = "{"
    proposal_adapter = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, malformed)
    called = {"translation": False, "audit": False}

    def translation_factory(_):
        called["translation"] = True
        raise AssertionError("translation must remain closed")

    def audit_factory(_):
        called["audit"] = True
        raise AssertionError("audit must remain closed")

    with pytest.raises(json.JSONDecodeError):
        gate.run_gate(
            output_dir=tmp_path / "proposal-fail",
            proposal_adapter=proposal_adapter,
            translation_factory=translation_factory,
            audit_factory=audit_factory,
        )
    assert called == {"translation": False, "audit": False}

    proposals = proposal_responses()
    proposal_adapter = FakeAdapter(gate.PROPOSAL_MODEL_ID, codec.PROPOSAL_SEEDS, proposals)
    malformed_translation = [translation_raw(draw, chunk) for draw in range(10) for chunk in range(3)]
    malformed_translation[8] = "{"
    translation_adapter = FakeAdapter(gate.TRANSLATION_MODEL_ID, codec.TRANSLATION_SEEDS, malformed_translation)
    with pytest.raises(json.JSONDecodeError):
        gate.run_gate(
            output_dir=tmp_path / "translation-fail",
            proposal_adapter=proposal_adapter,
            translation_factory=lambda _: translation_adapter,
            audit_factory=audit_factory,
        )
    assert called["audit"] is False


def catalog():
    def row(model, prompt, completion):
        return {
            "id": model,
            "architecture": {"input_modalities": ["text"], "output_modalities": ["text"]},
            "supported_parameters": ["seed", "response_format"],
            "pricing": {"prompt": str(prompt), "completion": str(completion)},
        }

    return {"data": [
        row(gate.PROPOSAL_MODEL_ID, .32e-6, 1.28e-6),
        row(gate.TRANSLATION_MODEL_ID, .08e-6, .18e-6),
        row(gate.AUDIT_MODEL_ID, .10e-6, .60e-6),
    ]}


def live(usage=execute.OPENING):
    return {"total_credits_usd": 245.0, "total_usage_usd": usage, "balance_usd": 245.0 - usage}


def test_phase_authorization_counts_prior_cost_and_account_race():
    selected, _ = codec.parse_and_select_proposals(proposal_responses())
    messages, _ = gate.build_translation_batch(selected)
    authorization = execute.authorize(gate.TRANSLATION_MODEL_ID, messages, gate.TRANSLATION_MAX_TOKENS, .04, live, catalog)
    assert .04 + authorization["exposure"]["tracker_reservation_usd"] <= execute.STAGE_CAP
    with pytest.raises(RuntimeError, match="stage cap"):
        execute.authorize(gate.TRANSLATION_MODEL_ID, messages, gate.TRANSLATION_MAX_TOKENS, .079, live, catalog)
    with pytest.raises(RuntimeError, match="account allowance"):
        execute.authorize(gate.TRANSLATION_MODEL_ID, messages, gate.TRANSLATION_MAX_TOKENS, 0.0, lambda: live(execute.OPENING + 4.999), catalog)


def test_preflight_is_zero_call_and_zero_write(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(execute, "validate_bindings", lambda: {"execution_binding_sha256": "x" * 64})
    monkeypatch.setattr(execute, "RUN", tmp_path / "run")
    monkeypatch.setattr(execute, "LEDGER", tmp_path / "ledger.json")
    monkeypatch.setattr(execute, "RESULT", tmp_path / "result.json")
    monkeypatch.setattr(execute, "FAILURE", tmp_path / "failure.json")
    result = execute.preflight(
        now=datetime(2026, 8, 13, 12, tzinfo=timezone.utc),
        live_reader=live,
        catalog_reader=catalog,
    )
    assert result["status"] == "ready_without_paid_calls"
    assert result["model_calls_made"] == result["files_written"] == 0
    assert not (tmp_path / "run").exists() and not (tmp_path / "ledger.json").exists()
