from __future__ import annotations

import json

import pytest

from scripts import infoquest_discrete_action_causal_gate as gate
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


def _models(checklist_model=None) -> gate.ModelBundle:
    return gate.ModelBundle(
        generator=gate.DeterministicGenerator("generator"),
        simulator=base.DeterministicFixtureModel("simulator"),
        checklist_judge=(
            checklist_model
            or base.DeterministicFixtureModel("checklist_judge")
        ),
    )


def _choice_response(
    initial: base.InitialPolicy,
    *,
    choice: str = "B",
    changed: bool = False,
) -> str:
    hypotheses = (
        tuple(f"Changed context {index}." for index in range(1, 9))
        if changed
        else initial.hypotheses
    )
    return json.dumps(
        {
            **{
                f"h{index}": hypotheses[index - 1]
                for index in range(1, 9)
            },
            "choice": choice,
        }
    )


def test_candidate_bank_excludes_asked_root_and_preserves_order():
    initial = _initial()
    assert gate.candidate_bank(initial, 2) == (
        initial.roots[0],
        initial.roots[1],
        initial.roots[3],
        initial.roots[4],
    )


def test_choice_parser_uses_shared_bank_and_exact_choice_code():
    initial = _initial()
    policy = gate.parse_choice_policy(
        _choice_response(initial, choice="B", changed=True),
        initial,
        2,
        require_fixed_support=False,
    )
    assert policy.followup == initial.roots[1]

    with pytest.raises(ValueError, match="exactly one"):
        gate.parse_choice_policy(
            _choice_response(initial, choice="B because it is best"),
            initial,
            2,
            require_fixed_support=False,
        )


def test_fixed_choice_requires_verbatim_support_copy():
    initial = _initial()
    with pytest.raises(ValueError, match="changed the initial support"):
        gate.parse_choice_policy(
            _choice_response(initial, changed=True),
            initial,
            0,
            require_fixed_support=True,
        )


def test_dry_serving_is_exactly_seven_calls(tmp_path):
    result = gate.run_serving_gate(
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 7
    assert result["usage"]["http_attempts"] == 7
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
    assert result["metrics"]["dynamic_choice_differs_from_fixed_cells"] == 24
    assert (
        result["metrics"]["fixtures_with_at_least_2_dynamic_choice_labels"]
        == 6
    )


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
