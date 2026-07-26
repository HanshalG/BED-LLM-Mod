from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from scripts import infoquest_information_need_gate as gate
from scripts import infoquest_support_causal_link_gate as base


IMMEDIATE = (
    (1, 0, 0, 0, 0),
    (1, 1, 0, 0, 0),
    (1, 1, 1, 0, 0),
    (1, 1, 1, 1, 0),
    (1, 1, 1, 1, 1),
)


def _fixtures() -> list[base.WorldFixture]:
    return [
        base.WorldFixture(
            fixture_id=f"I{record_id}W{world}",
            record_id=record_id,
            world=world,
            seed_message=f"Ambiguous request {record_id} world {world}",
            simulator_system="Unused.",
            truth_packet={},
            checklist=tuple(f"Hidden checklist {index}" for index in range(5)),
        )
        for record_id in base.MECHANICS_IDS
        for world in (1, 2)
    ]


def _initial(record_id: int) -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(f"Unused hypothesis {index}" for index in range(8)),
        roots=tuple(
            f"What is record {record_id} detail {index}?"
            for index in range(5)
        ),
    )


def _judgments() -> list[base.ChecklistJudgment]:
    return [
        base.ChecklistJudgment(IMMEDIATE, IMMEDIATE, IMMEDIATE)
        for _ in range(6)
    ]


def _models(selected_actions=()) -> gate.ModelBundle:
    return gate.ModelBundle(
        generator=gate.DeterministicNeedGenerator(
            "generator",
            selected_actions=selected_actions,
        ),
        simulator=base.DeterministicFixtureModel("simulator"),
        checklist_judge=base.DeterministicFixtureModel("checklist_judge"),
    )


def _response(selected: int = 2) -> str:
    profiles = []
    for action_index in range(4):
        profiles.append([90 if action_index == selected else 10] * 5)
    return json.dumps(
        {
            "n": [f"Missing value {index}" for index in range(5)],
            "w": [100, 80, 60, 40, 20],
            "a": profiles[0],
            "b": profiles[1],
            "c": profiles[2],
            "d": profiles[3],
        }
    )


def test_expected_resolved_mass_matches_weighted_probability():
    assert gate.expected_resolved_mass(
        [100, 50, 25, 25, 50],
        [100, 50, 0, 100, 50],
    ) == pytest.approx(0.7)

    with pytest.raises(ValueError, match="five values"):
        gate.expected_resolved_mass([1], [1])


def test_prompt_payload_is_target_blind_and_excludes_scenario_hypotheses():
    initial = _initial(0)
    messages = gate.information_need_messages(
        "Please help me decide.",
        initial,
        1,
        "I need it tomorrow.",
    )
    payload = json.loads(messages[-1]["content"])
    assert set(payload) == {
        "ambiguous_request",
        "observed_question",
        "observed_answer",
        "candidate_questions",
    }
    assert "checklist" not in messages[-1]["content"].lower()
    assert all(
        hypothesis not in messages[-1]["content"]
        for hypothesis in initial.hypotheses
    )


def test_parser_selects_highest_expected_resolved_mass_and_rejects_duplicates():
    initial = _initial(0)
    belief = gate.parse_information_need_belief(_response(2), initial, 0)
    assert belief.selected_action_index == 2
    assert belief.selected_question == initial.roots[3]
    assert belief.scores[2] == pytest.approx(0.9)

    malformed = json.loads(_response())
    malformed["n"][1] = malformed["n"][0]
    with pytest.raises(ValueError, match="not distinct"):
        gate.parse_information_need_belief(
            json.dumps(malformed),
            initial,
            0,
        )


def test_serving_gate_has_exact_ten_calls_and_no_scientific_endpoint(tmp_path):
    result = gate.run_serving_gate(
        _models([index % 4 for index in range(10)]),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 10
    assert result["protocol"]["scientific_endpoint_evaluated"] is False
    assert result["protocol"]["target_or_checklist_content_in_prompt"] is False


def test_target_metrics_recover_known_target_ordering():
    fixtures = _fixtures()
    initials = {
        record_id: _initial(record_id) for record_id in base.MECHANICS_IDS
    }
    judgments = _judgments()
    beliefs = []
    dynamic = []
    fixed = []
    for fixture_index, fixture in enumerate(fixtures):
        initial = initials[fixture.record_id]
        belief_row = []
        dynamic_row = []
        fixed_row = []
        for root_index in range(base.ROOT_COUNT):
            candidate_indices = [
                index for index in range(base.ROOT_COUNT) if index != root_index
            ]
            gains = [
                gate.alignment.additive_gain(
                    IMMEDIATE[root_index],
                    IMMEDIATE[index],
                )
                for index in candidate_indices
            ]
            profiles = tuple(
                tuple([gain * 25] * 5) for gain in gains
            )
            scores = tuple(
                gate.expected_resolved_mass([1] * 5, profile)
                for profile in profiles
            )
            selected = max(range(4), key=scores.__getitem__)
            belief_row.append(
                gate.InformationNeedBelief(
                    needs=tuple(f"Need {index}" for index in range(5)),
                    weights=(1, 1, 1, 1, 1),
                    resolution_probabilities=profiles,
                    scores=scores,
                    selected_action_index=selected,
                    selected_question=initial.roots[candidate_indices[selected]],
                )
            )
            baseline_question = initial.roots[candidate_indices[0]]
            dynamic_row.append(
                SimpleNamespace(selected_question=baseline_question)
            )
            fixed_row.append(SimpleNamespace(selected_question=baseline_question))
        beliefs.append(belief_row)
        dynamic.append(dynamic_row)
        fixed.append(fixed_row)

    metrics, fixture_metrics, gates = gate._target_metrics(
        fixtures,
        initials,
        beliefs,
        dynamic,
        fixed,
        judgments,
    )
    assert all(gates.values())
    assert metrics["mean_information_need_target_spearman"] == pytest.approx(1)
    assert metrics["information_need_target_optimal_cells"] == 30
    assert len(fixture_metrics) == 6


def test_mechanics_gate_uses_only_thirty_compiler_calls(tmp_path):
    fixtures = _fixtures()
    initials = {
        record_id: _initial(record_id) for record_id in base.MECHANICS_IDS
    }
    judgments = _judgments()
    selected_actions = gate._oracle_actions(judgments)
    root_answers = [
        [f"Answer {fixture_index} {root_index}" for root_index in range(5)]
        for fixture_index in range(6)
    ]
    fixed = []
    for fixture in fixtures:
        initial = initials[fixture.record_id]
        fixed.append(
            [
                SimpleNamespace(
                    selected_question=next(
                        root
                        for index, root in enumerate(initial.roots)
                        if index != root_index
                    )
                )
                for root_index in range(5)
            ]
        )
    result = gate.run_mechanics_gate(
        fixtures,
        initials,
        root_answers,
        fixed,
        fixed,
        judgments,
        [fixture.fixture_id for fixture in fixtures],
        _models(selected_actions),
        raw_path=tmp_path / "raw.json",
    )
    assert result["status"] == "passed"
    assert result["usage"]["physical_requests"] == 30
    assert result["protocol"]["simulator_calls"] == 0
    assert result["protocol"]["checklist_judge_calls"] == 0
