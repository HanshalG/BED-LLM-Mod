from __future__ import annotations

import json
from pathlib import Path

import pytest

from helpers import load_config
from scripts import tau_knowledge_belief_bottleneck_smoke as bottleneck


ROOT = Path(__file__).resolve().parents[1]
SMOKE = (
    ROOT
    / "results/nonmyopic/tau_knowledge_receding_continuation_v3_smoke"
    / "tau-knowledge-receding-v3-smoke-20260725T023000Z"
    / "SERVING_SMOKE.json"
)


def _records():
    return bottleneck.load_records(SMOKE, stage="serving_smoke")


def test_blinded_label_schedule_is_reproducible_and_balanced() -> None:
    first = bottleneck.aligned_label_schedule()
    second = bottleneck.aligned_label_schedule()

    assert first == second
    assert first == [
        True,
        False,
        False,
        False,
        True,
        True,
        False,
        False,
        True,
        True,
    ]
    assert sum(first) == 5


def test_payload_hides_every_frozen_bypass_channel() -> None:
    record = _records()[0]
    root_index = 0
    messages = bottleneck.scorer_messages(
        record,
        root_index,
        aligned_is_a=True,
    )
    encoded = json.dumps(messages)
    branch = record["first_branches"][root_index]

    assert record["opening"] not in encoded
    assert branch["query"] not in encoded
    assert all(
        followup["query"] not in encoded for followup in branch["followups"]
    )
    assert all(
        result["id"] not in encoded
        and str(result["bm25_score"]) not in encoded
        for followup in branch["followups"]
        for result in followup["results"]
    )
    assert branch["refreshed_information_need_hypotheses"][0] in encoded
    assert branch["followups"][0]["results"][0]["title"] in encoded
    assert "excerpt" in encoded


def test_paired_parser_is_strict_and_accepts_count_bands() -> None:
    payload = {
        key: str(value)
        for key, value in zip(
            bottleneck.score_schema(),
            [0, 31, 62, 93, 9, 39, 69, 99],
            strict=True,
        )
    }
    assert bottleneck.parse_paired_scores(json.dumps(payload)) == {
        "a": [0, 31, 62, 93],
        "b": [9, 39, 69, 99],
    }

    bad_numeric = dict(payload)
    bad_numeric["state_a_followup_1_score"] = 0
    with pytest.raises(ValueError):
        bottleneck.parse_paired_scores(json.dumps(bad_numeric))

    bad_band = dict(payload)
    bad_band["state_a_followup_1_score"] = "10"
    with pytest.raises(ValueError):
        bottleneck.parse_paired_scores(json.dumps(bad_band))


def test_synthetic_aligned_advantage_passes_efficacy_gates() -> None:
    rows = []
    for index in range(10):
        rows.append(
            {
                "aligned_label": "A" if index < 5 else "B",
                "state_pair_distinct": True,
                "aligned_scores_vary": True,
                "pairwise_comparable_count": 6,
                "aligned_pairwise_points": 6.0,
                "shuffled_pairwise_points": 3.0,
                "aligned_is_oracle_optimal": True,
                "shuffled_is_oracle_optimal": index >= 4,
                "aligned_selected_exact_value": 2,
                "shuffled_selected_exact_value": 1,
            }
        )
    usage = {
        "physical_requests": 10,
        "reasoning_tokens": 0,
        "adapter_cost_usd": 0.1,
        "generator": {
            "http_attempts": 10,
            "retry_count": 0,
            "forced_exits": 0,
        },
    }

    summary = bottleneck.summarize_rows(rows, usage)

    assert summary["aligned_pairwise_accuracy"] == 1.0
    assert summary["shuffled_pairwise_accuracy"] == 0.5
    assert summary["aligned_optimal_count_gain"] == 4
    assert summary["aligned_selected_exact_total_gain"] == 10
    assert summary["gates"]["all_pass"]


def test_config_freezes_model_seed_and_cost_cap() -> None:
    config = load_config(
        str(
            ROOT
            / "configs/"
            "config_tau_knowledge_belief_bottleneck_smoke_openrouter.yaml"
        )
    )

    assert config.model_pairs[0].questioner.model == bottleneck.MODEL_ID
    assert config.mediq_seed == bottleneck.SELECTION_SEED
    assert config.openrouter_run_budget_usd == bottleneck.MAX_COST_USD
