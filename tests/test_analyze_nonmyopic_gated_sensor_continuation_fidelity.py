from __future__ import annotations

import pytest

from scripts.analyze_nonmyopic_gated_sensor_continuation_fidelity import analyze
from scripts.nonmyopic_gated_sensor_strategy_prior_v2 import (
    DeterministicIndexedModel,
    IndexedStrategyConfig,
    IndexedStrategyProvider,
    run_experiment_v2,
)


def test_continuation_fidelity_decomposes_proposal_and_root_quality() -> None:
    config = IndexedStrategyConfig(
        num_trials=3,
        num_rounds=4,
        num_strategies=4,
        bootstrap_replicates=100,
        trial_concurrency=1,
    )
    provider = IndexedStrategyProvider(DeterministicIndexedModel(), config)
    payload = run_experiment_v2(provider, config)
    payload["run_id"] = "test-gated-continuation-fidelity"

    summary = analyze(payload)

    assert summary["no_llm_calls"]
    for arm in ("strategy_eig", "shared_state_random", "random_strategy"):
        arm_summary = summary["arms"][arm]
        assert arm_summary["num_h2_states"] == 9
        assert 0.0 <= arm_summary["mean_continuation_efficiency"] <= 1.0
        assert 0.0 <= arm_summary["mean_root_coverage"] <= 1.0
    for rows in summary["rows"].values():
        for row in rows:
            assert row["proposed_value"] <= row["same_root_closed_value"] + 1e-12
            assert row["proposal_exhaustive_fraction"] == pytest.approx(
                row["continuation_efficiency"] * row["root_coverage"]
            )
    assert [
        (row["trial_index"], row["round"], row["active_panel"])
        for row in summary["rows"]["strategy_eig"]
    ] == [
        (row["trial_index"], row["round"], row["active_panel"])
        for row in summary["rows"]["shared_state_random"]
    ]
