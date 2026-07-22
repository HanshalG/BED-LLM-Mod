import json
import math

from scripts.nonmyopic_gated_sensor_strategy_prior import (
    DeterministicStrategyModel,
    GatedStrategyProvider,
    StrategyConfig,
    _repair_terminal_roots,
    run_experiment,
)
from environments.gated_sensor import GatedSensorModel, SensorState


def test_dry_run_exercises_all_paired_arms_without_rollout_llm_calls() -> None:
    config = StrategyConfig(
        num_trials=4,
        num_rounds=4,
        num_strategies=6,
        bootstrap_replicates=100,
        trial_concurrency=2,
    )
    provider = GatedStrategyProvider(DeterministicStrategyModel(), config)
    summary = run_experiment(provider, config)

    assert summary["mechanics"]["paired_trials_and_truths"]
    assert summary["mechanics"]["all_selected_actions_legal"]
    assert summary["mechanics"]["strategy_initial_cells_cover_all_activation_roots"]
    assert summary["mechanics"]["rollout_scoring_llm_calls"] == 0
    assert summary["provider"]["physical_requests"] > 0
    assert set(summary["traces"]) == {
        "strategy_eig",
        "shared_d1",
        "exhaustive_d1",
        "random_strategy",
        "exhaustive_d2",
    }
    assert all(
        math.isfinite(trace["entropy_auc"])
        for traces in summary["traces"].values()
        for trace in traces
    )


def test_terminal_root_repair_is_deterministic_and_preserves_unique_roots() -> None:
    model = GatedSensorModel()
    response = (
        '{"strategies":['
        '{"name":"a","description":"valid description","root_action":"screen:bit-2","followups":{}},'
        '{"name":"b","description":"valid description","root_action":"screen:bit-2","followups":{}},'
        '{"name":"c","description":"valid description","root_action":"activate:A","followups":{}}]}'
    )
    repaired, count = _repair_terminal_roots(response, model=model, state=SensorState())

    assert count == 2
    assert '"root_action":"screen:bit-2"' in repaired
    assert '"root_action":"screen:bit-0"' in repaired
    assert '"root_action":"screen:bit-1"' in repaired


def test_failure_cache_revalidates_accepted_cells(tmp_path) -> None:
    config = StrategyConfig(
        num_trials=1,
        num_rounds=2,
        num_strategies=6,
        bootstrap_replicates=10,
        trial_concurrency=1,
    )
    model = GatedSensorModel()
    source = GatedStrategyProvider(DeterministicStrategyModel(), config)
    source.propose(
        model,
        trial_index=0,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )
    artifact = tmp_path / "FAILURE.json"
    artifact.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": "failed_closed",
                "config": config.__dict__,
                "candidate_requests": source.physical_requests,
                "invalid_responses": [],
            }
        )
    )
    resumed = GatedStrategyProvider(DeterministicStrategyModel(), config)
    info = resumed.load_failure_cache(artifact, model)
    cell = resumed.propose(
        model,
        trial_index=0,
        state=model.initial_state,
        belief=model.initial_belief,
        history=(),
        horizon=2,
    )

    assert info["accepted_cells_reused"] == 1
    assert cell.cache_hit
    assert resumed.cache_hits == 1
