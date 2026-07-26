from __future__ import annotations

import json

import pytest

from scripts import infoquest_refresh_continuation_gate as gate
from scripts import infoquest_support_causal_link_gate as base


def _fixtures() -> list[base.WorldFixture]:
    fixtures = []
    for record_id in base.MECHANICS_IDS:
        for world in (1, 2):
            fixtures.append(
                base.WorldFixture(
                    fixture_id=f"I{record_id}W{world}",
                    record_id=record_id,
                    world=world,
                    seed_message=f"Ambiguous request {record_id}",
                    simulator_system="Answer one specific question.",
                    truth_packet={},
                    checklist=tuple(
                        f"Checklist {index}" for index in range(5)
                    ),
                )
            )
    return fixtures


def _models(
    checklist_model=None,
) -> gate.ModelBundle:
    return gate.ModelBundle(
        generator=gate.DeterministicGenerator("generator"),
        simulator=base.DeterministicFixtureModel("simulator"),
        checklist_judge=(
            checklist_model
            or base.DeterministicFixtureModel("checklist_judge")
        ),
    )


def _initial() -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(
            f"Concrete hidden context {index} has goal and constraint."
            for index in range(1, 9)
        ),
        roots=tuple(
            f"What is hidden detail {index}?" for index in range(1, 6)
        ),
    )


def test_fixed_support_parser_requires_exact_hypothesis_copy():
    initial = _initial()
    value = {
        **{
            f"h{index}": initial.hypotheses[index - 1]
            for index in range(1, 9)
        },
        "followup": "What is the next detail?",
    }
    parsed = gate.parse_fixed_support(json.dumps(value), initial)
    assert parsed.hypotheses == initial.hypotheses

    value["h4"] = "Changed support."
    with pytest.raises(ValueError, match="changed the initial support"):
        gate.parse_fixed_support(json.dumps(value), initial)


def test_fixed_and_dynamic_outputs_are_compute_matched():
    messages = gate.fixed_support_messages(
        "Ambiguous request",
        _initial(),
        "What is hidden detail 1?",
        "The hidden value is one.",
    )
    assert "h1..h8 and followup" in messages[0]["content"]
    request = json.loads(messages[-1]["content"])
    assert len(request["fixed_initial_hypotheses"]) == 8


def test_dry_serving_is_exactly_ten_calls(tmp_path):
    result = gate.run_serving_gate(
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 10
    assert result["usage"]["http_attempts"] == 10
    assert result["gates"]["all_pass"] is True


def test_dry_mechanics_is_exactly_159_calls_and_passes(tmp_path):
    result = gate.run_mechanics_gate(
        _fixtures(),
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["usage"]["physical_requests"] == 159
    assert result["usage"]["http_attempts"] == 159
    assert result["gates"]["all_pass"] is True
    assert result["metrics"]["mean_dynamic_minus_fixed_checklist"] > 0


class BadChecklistModel(base.DeterministicFixtureModel):
    def chat_complete_messages_batched(
        self,
        batch_messages,
        temperature,
        block_size,
        max_new_tokens=None,
    ):
        del temperature, block_size, max_new_tokens
        self.requests += len(batch_messages)
        return ["malformed"] * len(batch_messages)


def test_serving_checkpoints_malformed_response_before_parse(tmp_path):
    raw_path = tmp_path / "raw.json"
    with pytest.raises(gate.GateExecutionError):
        gate.run_serving_gate(
            _models(BadChecklistModel("checklist_judge")),
            raw_path=raw_path,
        )
    raw = json.loads(raw_path.read_text())
    assert raw["checklist_judgment"] == ["malformed"]
