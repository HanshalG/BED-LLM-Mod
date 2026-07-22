from __future__ import annotations

import pytest

from scripts.analyze_nonmyopic_gated_sensor_choice_mechanics import analyze
from scripts.nonmyopic_gated_sensor_strategy_prior_v2 import (
    DeterministicIndexedModel,
    IndexedStrategyConfig,
    IndexedStrategyProvider,
    run_experiment_v2,
)


def test_choice_mechanics_scores_semantic_branch_order_and_ignores_padding() -> None:
    config = IndexedStrategyConfig(
        num_trials=2,
        num_rounds=3,
        num_strategies=4,
        bootstrap_replicates=100,
        trial_concurrency=1,
    )
    provider = IndexedStrategyProvider(DeterministicIndexedModel(), config)
    payload = run_experiment_v2(provider, config)
    payload["run_id"] = "test-gated-choice-mechanics"

    summary = analyze(payload)

    reached = summary["strategy_reached"]
    assert summary["no_llm_calls"]
    assert reached["num_requests"] == 4
    assert reached["num_branch_choices"] == 22
    assert reached["index_zero_rate"] == 1.0
    assert reached["measurement_selected_index_counts"] == {"0": 12}
    assert all(row["branch"] in {"none", "positive", "negative"} for row in summary["rows"])
    assert all(row["selected_index"] == 0 for row in summary["rows"])


def test_choice_mechanics_detects_repeated_and_zero_information_choices() -> None:
    config = IndexedStrategyConfig(
        num_trials=1,
        num_rounds=2,
        num_strategies=4,
        bootstrap_replicates=100,
        trial_concurrency=1,
    )
    provider = IndexedStrategyProvider(DeterministicIndexedModel(), config)
    payload = run_experiment_v2(provider, config)
    payload["run_id"] = "test-gated-choice-pathologies"
    request = next(
        request
        for request in payload["candidate_requests"]
        if request["active_panel"] is None and request["trial_index"] == 0
    )
    request["raw_response"] = '{"choices":[[0,0],[0,0],[0,0],[3,0]]}'

    summary = analyze(payload)
    measurement_rows = [
        row
        for row in summary["rows"]
        if row["strategy_reached"] and row["root_action"] == "screen:bit-0"
    ]

    assert [row["branch"] for row in measurement_rows] == ["positive", "negative"]
    assert measurement_rows[0]["selected_action"] == "screen:bit-0"
    assert measurement_rows[0]["repeats_root_predicate"]
    assert measurement_rows[1]["selected_action"] == "activate:A"
    assert measurement_rows[1]["zero_eig"]
    assert summary["strategy_reached"]["measurement_repeat_root_rate"] == pytest.approx(0.5)
    assert summary["strategy_reached"]["measurement_activation_rate"] == pytest.approx(0.5)
