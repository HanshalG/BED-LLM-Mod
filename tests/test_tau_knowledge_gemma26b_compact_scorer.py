from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.tau_knowledge_gemma26b_compact_scorer import (
    compact_focused_messages,
    compact_root_messages,
    parse_compact_focused_scores,
    parse_compact_root_scores,
)


ROOT = Path(__file__).resolve().parents[1]


def test_compact_root_parser_accepts_exact_array_schema() -> None:
    myopic = parse_compact_root_scores(
        '{"scores":[0,"20",40,60,100]}',
        include_followups=False,
    )
    assert myopic["scores"] == [0, 20, 40, 60, 100]
    assert myopic["best_followup_indices"] == []

    nonmyopic = parse_compact_root_scores(
        '{"scores":[10,20,30,40,50],"best_followups":[1,"02",3,4,1]}',
        include_followups=True,
    )
    assert nonmyopic["best_followup_indices"] == [0, 1, 2, 3, 0]


def test_compact_root_parser_rejects_extra_keys_and_wrong_lengths() -> None:
    with pytest.raises(ValueError):
        parse_compact_root_scores(
            '{"scores":[1,2,3,4,5],"rationale":"extra"}',
            include_followups=False,
        )
    with pytest.raises(ValueError):
        parse_compact_root_scores(
            '{"scores":[1,2,3,4]}',
            include_followups=False,
        )


def test_compact_focused_parser_enforces_count_bands() -> None:
    parsed = parse_compact_focused_scores(
        '{"scores":[0,"31",60,99]}'
    )
    assert parsed["scores"] == [0, 31, 60, 99]
    with pytest.raises(ValueError):
        parse_compact_focused_scores('{"scores":[0,20,60,99]}')


def test_compact_messages_request_no_rationales() -> None:
    record = {
        "opening": "I need help comparing two account policies.",
        "initial_information_need_hypotheses": ["eligibility", "fees"],
        "first_branches": [
            {
                "query": f"query {root}",
                "first_results": [
                    {
                        "id": f"d{root}",
                        "title": f"Title {root}",
                        "content": "Relevant policy text.",
                    }
                ],
                "refreshed_information_need_hypotheses": ["details"],
                "followups": [
                    {
                        "query": f"hidden {followup}",
                        "information_need_hypotheses": ["details"],
                        "results": [
                            {
                                "id": f"d{root}-{followup}",
                                "title": f"Followup {followup}",
                                "content": "More policy text.",
                            }
                        ],
                    }
                    for followup in range(4)
                ],
            }
            for root in range(5)
        ],
    }
    root_prompt = compact_root_messages(
        record,
        include_followups=True,
    )[-1]["content"]
    focused_prompt = compact_focused_messages(record, 0)[-1]["content"]
    assert '"scores" and "best_followups"' in root_prompt
    assert "rationale" not in root_prompt.lower()
    assert '"scores"' in focused_prompt
    assert "rationale" not in focused_prompt.lower()


def test_compact_gemma_config_is_frozen() -> None:
    config = load_config(
        str(
            ROOT
            / "configs/"
            "config_tau_knowledge_gemma26b_compact_scorer_openrouter.yaml"
        )
    )
    spec = config.model_pairs[0].questioner
    assert spec.model == "google/gemma-4-26b-a4b-it"
    assert spec.thinking is True
    assert spec.thinking_max_new_tokens == 8192
    assert spec.thinking_final_max_new_tokens == 512
    assert config.openrouter_concurrency == 256
    assert config.mediq_seed == 24358
