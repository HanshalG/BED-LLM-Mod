from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts.tau_knowledge_cross_model_scorer import (
    coerce_json_int,
    parse_focused_scores,
    parse_root_scores,
    summarize_cross_model,
)


ROOT = Path(__file__).resolve().parents[1]


def test_coerce_json_int_accepts_frozen_numeric_representations() -> None:
    assert coerce_json_int(2, minimum=0, maximum=4, max_digits=2) == 2
    assert coerce_json_int("02", minimum=0, maximum=4, max_digits=2) == 2
    assert coerce_json_int("100", minimum=0, maximum=100, max_digits=3) == 100
    with pytest.raises(ValueError):
        coerce_json_int(True, minimum=0, maximum=4, max_digits=2)
    with pytest.raises(ValueError):
        coerce_json_int("002", minimum=0, maximum=4, max_digits=2)


def test_root_parser_accepts_json_numbers_and_strings() -> None:
    payload = {}
    for index in range(1, 6):
        payload[f"root_{index}_score"] = index * 10 if index % 2 else str(index * 10)
        payload[f"root_{index}_best_followup"] = index % 4 + 1
        payload[f"root_{index}_rationale"] = f"reason {index}"
    parsed = parse_root_scores(
        json.dumps(payload),
        include_followups=True,
    )
    assert parsed["scores"] == [10, 20, 30, 40, 50]
    assert parsed["best_followup_indices"] == [1, 2, 3, 0, 1]


def test_focused_parser_accepts_v3_1_zero_padding() -> None:
    parsed = parse_focused_scores(
        json.dumps(
            {
                "followup_1_score": "02",
                "followup_2_score": 31,
                "followup_3_score": "60",
                "followup_4_score": 99,
            }
        )
    )
    assert parsed["scores"] == [2, 31, 60, 99]


def test_cross_model_summary_replaces_only_request_gate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fake_summary(*args, **kwargs):
        return {
            "gates": {
                "exact_physical_request_count": True,
                "zero_reasoning_tokens": True,
                "all_pass": True,
            }
        }

    monkeypatch.setattr(
        "scripts.tau_knowledge_cross_model_scorer.summarize",
        fake_summary,
    )
    summary = summarize_cross_model(
        [],
        [],
        {"physical_requests": 13, "reasoning_tokens": 0},
        stage="serving_smoke",
        myopic_scores=[],
        nonmyopic_scores=[],
    )
    assert summary["gates"]["exact_physical_request_count"] is False
    assert summary["gates"]["zero_reasoning_tokens"] is True
    assert summary["gates"]["all_pass"] is False


def test_gemini_replication_config_is_frozen() -> None:
    config = load_config(
        str(
            ROOT
            / "configs/"
            "config_tau_knowledge_gemini31pro_scorer_replication_openrouter.yaml"
        )
    )

    assert (
        config.model_pairs[0].questioner.model
        == "google/gemini-3.1-pro-preview"
    )
    assert config.openrouter_max_output_tokens == 4096
    assert config.mediq_seed == 24344


def test_gpt54mini_replication_config_is_frozen() -> None:
    config = load_config(
        str(
            ROOT
            / "configs/"
            "config_tau_knowledge_gpt54mini_scorer_replication_openrouter.yaml"
        )
    )

    assert config.model_pairs[0].questioner.model == "openai/gpt-5.4-mini"
    assert config.openrouter_max_output_tokens == 4096
    assert config.mediq_seed == 24345
