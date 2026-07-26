from __future__ import annotations

import json
import math

import pytest

from scripts import infoquest_partition_eig_causal_gate as gate
from scripts import infoquest_support_causal_link_gate as base


def _fixtures() -> list[base.WorldFixture]:
    return [
        base.WorldFixture(
            fixture_id=f"I{record_id}W{world}",
            record_id=record_id,
            world=world,
            seed_message=f"Ambiguous request {record_id}",
            simulator_system="Answer one specific question.",
            truth_packet={},
            checklist=tuple(f"Checklist {index}" for index in range(5)),
        )
        for record_id in base.MECHANICS_IDS
        for world in (1, 2)
    ]


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


def _partition_response(
    initial: base.InitialPolicy,
    *,
    changed: bool = False,
) -> str:
    hypotheses = (
        tuple(f"Changed context {index}." for index in range(1, 9))
        if changed
        else initial.hypotheses
    )
    value = {
        **{
            f"h{index}": hypotheses[index - 1]
            for index in range(1, 9)
        },
        **{f"w{index}": 10 for index in range(1, 9)},
    }
    profiles = {
        "a": (0, 0, 1, 1, 2, 2, 3, 3),
        "b": (0, 0, 0, 0, 1, 1, 1, 1),
        "c": (0,) * 8,
        "d": (0,) * 8,
    }
    for action, labels in profiles.items():
        for index, label in enumerate(labels, start=1):
            value[f"{action}{index}"] = label
    return json.dumps(value)


def test_partition_eig_matches_entropy_of_cluster_mass():
    score = gate.partition_eig(
        (1,) * 8,
        (0, 0, 1, 1, 2, 2, 3, 3),
    )
    assert score == pytest.approx(math.log(4))


def test_parser_uses_exact_eig_and_frozen_tie_break():
    initial = _initial()
    belief = gate.parse_partition_belief(
        _partition_response(initial, changed=True),
        initial,
        2,
        require_fixed_support=False,
    )
    assert belief.selected_action_index == 0
    assert belief.selected_question == initial.roots[0]
    assert belief.eig_scores[0] == pytest.approx(math.log(4))


def test_parser_rejects_changed_fixed_support_and_noninteger_profile():
    initial = _initial()
    with pytest.raises(ValueError, match="changed the initial support"):
        gate.parse_partition_belief(
            _partition_response(initial, changed=True),
            initial,
            0,
            require_fixed_support=True,
        )

    value = json.loads(_partition_response(initial))
    value["a1"] = "0"
    with pytest.raises(ValueError, match="not an integer"):
        gate.parse_partition_belief(
            json.dumps(value),
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
        result["metrics"][
            "dynamic_cells_with_at_least_2_informative_actions"
        ]
        == 30
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
