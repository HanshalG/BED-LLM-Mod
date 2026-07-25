from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.musique_branching_causal_smoke import (
    FOLLOWUP_COUNT,
    ROOT_COUNT,
    connected_prefix_length,
    parse_initial,
    parse_refresh,
    parse_scorer,
    policy_selections,
    scorer_keys,
)


def _hypotheses() -> dict[str, str]:
    return {
        f"hypothesis_{index}": f"distinct dependency hypothesis {index}"
        for index in range(1, 9)
    }


def test_initial_and_refresh_parsers_accept_frozen_schema() -> None:
    initial = {
        **_hypotheses(),
        **{
            f"root_{index}_query": f"search query {index}"
            for index in range(1, ROOT_COUNT + 1)
        },
        **{
            f"root_{index}_score": index * 10
            for index in range(1, ROOT_COUNT + 1)
        },
    }
    parsed = parse_initial(json.dumps(initial))
    assert len(parsed["hypotheses"]) == 8
    assert len(parsed["root_queries"]) == ROOT_COUNT
    assert parsed["root_scores"][-1] == 60

    refresh = {
        **_hypotheses(),
        **{
            f"followup_{index}_query": f"followup query {index}"
            for index in range(1, FOLLOWUP_COUNT + 1)
        },
    }
    assert len(parse_refresh(json.dumps(refresh))["followup_queries"]) == 4


def test_initial_parser_rejects_duplicate_queries() -> None:
    payload = {
        **_hypotheses(),
        **{
            f"root_{index}_query": "same query"
            for index in range(1, ROOT_COUNT + 1)
        },
        **{
            f"root_{index}_score": index
            for index in range(1, ROOT_COUNT + 1)
        },
    }
    with pytest.raises(ValueError, match="unique"):
        parse_initial(json.dumps(payload))


def test_scorer_parser_requires_every_blinded_state_score() -> None:
    payload = {key: 10 for key in scorer_keys()}
    parsed = parse_scorer(json.dumps(payload))
    assert len(parsed) == ROOT_COUNT
    assert parsed[0]["a"] == [10] * FOLLOWUP_COUNT
    del payload[scorer_keys()[-1]]
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_scorer(json.dumps(payload))


def test_policy_selection_uses_aligned_fixed_and_shuffled_states() -> None:
    rows = []
    for root in range(ROOT_COUNT):
        aligned = [0] * FOLLOWUP_COUNT
        shuffled = [0] * FOLLOWUP_COUNT
        initial = [0] * FOLLOWUP_COUNT
        aligned[root % FOLLOWUP_COUNT] = 80 + root
        shuffled[(root + 1) % FOLLOWUP_COUNT] = 70 + root
        initial[(root + 2) % FOLLOWUP_COUNT] = 60 + root
        rows.append(
            {
                "aligned_scores": aligned,
                "shuffled_scores": shuffled,
                "initial_scores": initial,
            }
        )
    selected = policy_selections(
        immediate_scores=[100, 10, 10, 10, 10, 10],
        continuation_rows=rows,
        task_index=0,
    )
    assert selected["myopic"]["root_index"] == 0
    assert selected["model_aware"]["followup_index"] == 0
    assert selected["fixed"]["followup_index"] == 2
    assert selected["shuffled"]["followup_index"] == 1


def test_connected_prefix_requires_ordered_deep_pair() -> None:
    supports = [2, 5, 7, 11]
    assert (
        connected_prefix_length(
            first_paragraph_index=2,
            second_paragraph_index=5,
            support_indices=supports,
        )
        == 2
    )
    assert (
        connected_prefix_length(
            first_paragraph_index=7,
            second_paragraph_index=2,
            support_indices=supports,
        )
        == 1
    )
    assert (
        connected_prefix_length(
            first_paragraph_index=0,
            second_paragraph_index=1,
            support_indices=supports,
        )
        == 0
    )


def test_config_has_frozen_model_concurrency_and_budget() -> None:
    config = load_config(
        "configs/config_musique_branching_causal_smoke_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_concurrency == 24
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.20)
