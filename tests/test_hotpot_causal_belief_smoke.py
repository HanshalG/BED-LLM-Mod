from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.hotpot_causal_belief_smoke import (
    parse_final,
    parse_initial,
    parse_refresh,
    parse_scorer,
    policy_selections,
    support_coverage,
    token_f1,
)


def _hypotheses() -> dict[str, str]:
    return {
        f"hypothesis_{index}": f"distinct hypothesis {index}"
        for index in range(1, 9)
    }


def test_initial_and_refresh_parsers_accept_frozen_score_representations() -> None:
    payload = {
        **_hypotheses(),
        "root_1_score": 90,
        "root_2_score": "80",
        "root_3_score": 30,
        "root_4_score": "0",
    }
    parsed = parse_initial(json.dumps(payload))
    assert parsed["root_scores"] == [90, 80, 30, 0]
    assert len(parse_refresh(json.dumps(_hypotheses()))) == 8


def test_initial_parser_rejects_noncanonical_score() -> None:
    payload = {
        **_hypotheses(),
        "root_1_score": "090",
        "root_2_score": 80,
        "root_3_score": 30,
        "root_4_score": 0,
    }
    with pytest.raises(ValueError, match="canonical"):
        parse_initial(json.dumps(payload))


def test_scorer_parser_requires_all_three_state_vectors() -> None:
    payload = {
        f"state_{state}_title_{index}_score": index
        for state in ("a", "b", "c")
        for index in range(1, 10)
    }
    parsed = parse_scorer(json.dumps(payload), 9)
    assert parsed["a"] == list(range(1, 10))
    del payload["state_c_title_9_score"]
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_scorer(json.dumps(payload), 9)


def test_policy_selection_uses_each_frozen_state() -> None:
    rows = []
    for root in range(4):
        aligned = [0] * 9
        shuffled = [0] * 9
        initial = [0] * 9
        aligned[root] = 80 + root
        shuffled[(root + 1) % 9] = 70 + root
        initial[(root + 2) % 9] = 60 + root
        rows.append(
            {
                "aligned_scores": aligned,
                "shuffled_scores": shuffled,
                "initial_scores": initial,
            }
        )
    selected = policy_selections(
        immediate_scores=[100, 10, 10, 10],
        continuation_rows=rows,
    )
    assert selected["myopic"]["root_index"] == 0
    assert selected["model_aware"]["followup_candidate_index"] == 0
    assert selected["fixed"]["followup_candidate_index"] == 2
    assert selected["shuffled"]["followup_candidate_index"] == 1


def test_support_coverage_counts_unique_support_titles() -> None:
    titles = [f"title {index}" for index in range(10)]
    coverage, chosen = support_coverage(
        selection={"root_index": 0, "followup_candidate_index": 2},
        titles=titles,
        support_titles={"title 2", "title 3"},
    )
    assert chosen == ["title 2", "title 3"]
    assert coverage == 2


def test_token_f1_and_final_parser() -> None:
    assert token_f1("The 1901 founding", "1901") == pytest.approx(0.5)
    assert token_f1("unrelated", "1901") == 0.0
    assert parse_final('{"answer":"1901","confidence":"90"}') == {
        "answer": "1901",
        "confidence": 90,
    }


def test_config_has_frozen_budget_and_model() -> None:
    config = load_config(
        "configs/config_hotpot_causal_belief_smoke_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_concurrency == 4
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.15)
