from __future__ import annotations

from pathlib import Path

from scripts import infoquest_cached_answer_ranking_diagnostic as diagnostic
from scripts import infoquest_cached_partition_eig_gate as cached
from scripts import infoquest_partition_eig_causal_gate as partition
from scripts import infoquest_support_causal_link_gate as base


def _initial() -> base.InitialPolicy:
    return base.InitialPolicy(
        hypotheses=tuple(f"Hypothesis {index}" for index in range(8)),
        roots=tuple(f"What is detail {index}?" for index in range(5)),
    )


def _belief(initial: base.InitialPolicy, selected: int):
    return partition.PartitionBelief(
        hypotheses=initial.hypotheses,
        weights=(10,) * 8,
        profiles=((0,) * 8,) * 4,
        eig_scores=(0.0,) * 4,
        selected_action_index=selected,
        selected_question=initial.roots[selected],
    )


def _models():
    return diagnostic.ModelBundle(
        generator=base.DeterministicFixtureModel("generator"),
        simulator=base.DeterministicFixtureModel("simulator"),
        checklist_judge=diagnostic.DeterministicChecklistJudge(
            "checklist_judge"
        ),
    )


def test_cached_answers_reuse_the_selected_root_answer():
    initial = _initial()
    fixture = base.WorldFixture(
        fixture_id="I0W1",
        record_id=0,
        world=1,
        seed_message="Seed",
        simulator_system="Unused",
        truth_packet={},
        checklist=("A", "B", "C", "D", "E"),
    )
    answers = ["A0", "A1", "A2", "A3", "A4"]
    selected = [_belief(initial, index) for index in range(5)]
    result = diagnostic.cached_answers_for_choices(
        [fixture],
        {0: initial},
        [answers],
        [selected],
    )
    assert result == [answers]


def test_dry_serving_uses_exactly_one_judge_call(tmp_path: Path):
    result = diagnostic.run_serving_gate(
        _models(),
        raw_path=tmp_path / "raw.json",
    )
    assert result["gates"]["all_pass"] is True
    assert result["usage"]["physical_requests"] == 1
    assert result["usage"]["http_attempts"] == 1
