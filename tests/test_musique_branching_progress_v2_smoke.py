from __future__ import annotations

import json

import pytest

from helpers import load_config
from scripts.musique_branching_progress_v2_smoke import (
    FOLLOWUP_COUNT,
    ROOT_COUNT,
    parse_initial,
    parse_scorer,
    policy_selections,
    scorer_keys,
    spearman,
)


def _hypotheses() -> dict[str, str]:
    return {
        f"hypothesis_{index}": f"distinct chain hypothesis {index}"
        for index in range(1, 9)
    }


def test_v2_initial_parser_has_no_utility_scores() -> None:
    payload = {
        **_hypotheses(),
        **{
            f"root_{index}_query": f"search query {index}"
            for index in range(1, ROOT_COUNT + 1)
        },
    }
    parsed = parse_initial(json.dumps(payload))
    assert set(parsed) == {"hypotheses", "root_queries"}
    payload["root_1_score"] = 90
    with pytest.raises(ValueError, match="unexpected keys"):
        parse_initial(json.dumps(payload))


def test_progress_scorer_accepts_only_bands_zero_through_four() -> None:
    payload = {key: 2 for key in scorer_keys()}
    parsed = parse_scorer(json.dumps(payload))
    assert parsed[0]["a"]["current"] == 2
    assert parsed[0]["a"]["terminal"] == [2] * FOLLOWUP_COUNT
    payload[scorer_keys()[0]] = 5
    with pytest.raises(ValueError, match="outside"):
        parse_scorer(json.dumps(payload))


def test_progress_policy_compares_current_to_terminal_on_same_scale() -> None:
    rows = []
    for root in range(ROOT_COUNT):
        rows.append(
            {
                "aligned": {
                    "current": 2 if root == 0 else 1,
                    "terminal": [3 if root == 1 else 2, 1, 1, 1],
                },
                "shuffled": {
                    "current": 0,
                    "terminal": [4 if root == 2 else 1, 1, 1, 1],
                },
                "initial": {
                    "current": 0,
                    "terminal": [4 if root == 3 else 1, 1, 1, 1],
                },
            }
        )
    selected = policy_selections(progress_rows=rows, task_index=0)
    assert selected["myopic"]["root_index"] == 0
    assert selected["model_aware"]["root_index"] == 1
    assert selected["shuffled"]["root_index"] == 2
    assert selected["fixed"]["root_index"] == 3


def test_spearman_handles_ties_and_detects_order() -> None:
    assert spearman([1, 2, 3], [1, 2, 3]) == pytest.approx(1.0)
    assert spearman([3, 2, 1], [1, 2, 3]) == pytest.approx(-1.0)
    assert spearman([1, 1, 2, 2], [0, 0, 1, 1]) == pytest.approx(1.0)


def test_v2_config_has_frozen_budget_and_concurrency() -> None:
    config = load_config(
        "configs/config_musique_branching_progress_v2_smoke_openrouter.yaml"
    )
    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4"
    assert config.openrouter_concurrency == 24
    assert config.openrouter_run_budget_usd == pytest.approx(0.50)
    assert config.openrouter_projected_cost_usd == pytest.approx(0.20)
