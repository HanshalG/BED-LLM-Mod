from __future__ import annotations

import json

import pytest

from scripts import infoquest_support_causal_link_gate as gate


def _initial_response() -> str:
    value = {
        **{
            f"h{index}": (
                f"Concrete context {index} with goal {index}, "
                f"obstacle {index}, and constraint {index}."
            )
            for index in range(1, gate.SUPPORT_SIZE + 1)
        },
        **{
            f"q{index}": f"What is hidden detail {index}?"
            for index in range(1, gate.ROOT_COUNT + 1)
        },
    }
    return json.dumps(value)


def _fixtures() -> list[gate.WorldFixture]:
    fixtures = []
    for record_id in gate.MECHANICS_IDS:
        for world in (1, 2):
            fixtures.append(
                gate.WorldFixture(
                    fixture_id=f"I{record_id}W{world}",
                    record_id=record_id,
                    world=world,
                    seed_message=f"Ambiguous request {record_id}",
                    simulator_system="Answer one specific question.",
                    truth_packet={
                        "description": "Hidden context",
                        "goal": "Goal",
                        "obstacle": "Obstacle",
                        "constraints": ["Constraint"],
                        "solution": "Solution",
                        "persona": "Persona",
                    },
                    checklist=tuple(
                        f"Checklist {index}" for index in range(5)
                    ),
                )
            )
    return fixtures


def _models() -> gate.ModelBundle:
    return gate.ModelBundle(
        generator=gate.DeterministicFixtureModel("generator"),
        simulator=gate.DeterministicFixtureModel("simulator"),
        support_judge=gate.DeterministicFixtureModel("support_judge"),
        checklist_judge=gate.DeterministicFixtureModel("checklist_judge"),
    )


def test_initial_parser_accepts_single_fence_and_rejects_nonatomic_root():
    response = f"```json\n{_initial_response()}\n```"
    parsed = gate.parse_initial(response)
    assert len(parsed.hypotheses) == 8
    assert len(parsed.roots) == 5

    value = json.loads(_initial_response())
    value["q3"] = "What is the budget and schedule?"
    with pytest.raises(ValueError, match="not one atomic"):
        gate.parse_initial(json.dumps(value))


def test_support_and_checklist_line_parsers_are_strict():
    support = gate.parse_support_judgment(
        "\n".join(
            (
                "I|00|50",
                "Q1|00|60",
                "Q2|01|80",
                "Q3|01|90",
                "Q4|01|70",
                "Q5|00|40",
            )
        )
    )
    assert support.scores == (50, 60, 80, 90, 70, 40)

    checklist = gate.parse_checklist_judgment(
        "\n".join(
            (
                "Q1|00000|11000|10000",
                "Q2|00000|11100|10000",
                "Q3|00000|11110|11000",
                "Q4|00000|11000|10000",
                "Q5|00000|10000|00000",
            )
        )
    )
    assert sum(checklist.dynamic[2]) == 4
    with pytest.raises(ValueError, match="loses immediate"):
        gate.parse_checklist_judgment(
            "\n".join(
                (
                    "Q1|10000|00000|10000",
                    "Q2|10000|11100|10000",
                    "Q3|10000|11110|11000",
                    "Q4|00000|11000|10000",
                    "Q5|00000|10000|00000",
                )
            )
        )


def test_initial_prompt_is_target_blind():
    messages = gate.initial_messages("Ambiguous request")
    payload = json.loads(messages[-1]["content"])
    assert set(payload) == {"ambiguous_seed_message"}
    serialized = json.dumps(messages)
    assert "hidden_context" not in serialized
    assert "checklist" not in serialized


def test_dry_serving_is_exactly_ten_calls(tmp_path):
    result = gate.run_serving_gate(
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 10
    assert result["usage"]["http_attempts"] == 10
    assert result["gates"]["all_pass"] is True


def test_dry_mechanics_is_exactly_165_calls_and_passes(tmp_path):
    result = gate.run_mechanics_gate(
        _fixtures(),
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["usage"]["physical_requests"] == 165
    assert result["usage"]["http_attempts"] == 165
    assert result["gates"]["all_pass"] is True
    assert result["metrics"]["dynamic_minus_myopic_selected_mean"] > 0
    assert (
        result["metrics"]["dynamic_minus_fixed_continuation_mean_all_roots"]
        > 0
    )
