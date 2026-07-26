from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts import infoquest_cross_family_partition_gate as gate
from scripts import infoquest_support_causal_link_gate as base


def test_cross_family_parser_is_exact_and_selects_a_root():
    initial = base.InitialPolicy(
        hypotheses=tuple(f"Hypothesis {index}" for index in range(8)),
        roots=tuple(f"What is detail {index}?" for index in range(5)),
    )
    response = json.dumps(
        {
            "w": [10] * 8,
            "a": [0] * 8,
            "b": [0, 0, 1, 1, 2, 2, 3, 3],
            "c": [0] * 8,
            "d": [0] * 8,
        }
    )
    belief = gate.parse_scored_belief(
        response,
        initial.hypotheses,
        initial,
        0,
    )
    assert belief.selected_action_index == 1

    malformed = json.loads(response)
    malformed["b"].pop()
    with pytest.raises(ValueError, match="b array does not have eight"):
        gate.parse_scored_belief(
            json.dumps(malformed),
            initial.hypotheses,
            initial,
            0,
        )


def test_dry_serving_is_exactly_three_calls(tmp_path: Path):
    models = gate.ModelBundle(
        generator=base.DeterministicFixtureModel("generator"),
        simulator=gate.DeterministicCrossFamilyScorer("simulator"),
        checklist_judge=gate.diagnostic.DeterministicChecklistJudge(
            "checklist_judge"
        ),
    )
    result = gate.run_serving_gate(
        models,
        raw_path=tmp_path / "raw.json",
    )
    assert result["gates"]["all_pass"] is True
    assert result["usage"]["physical_requests"] == 3
    assert result["usage"]["http_attempts"] == 3
