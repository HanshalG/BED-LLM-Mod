from dataclasses import asdict

from scripts.copex_grid_sensitivity import run_sensitivity
from scripts.nonmyopic_copex_strategy_prior import (
    ContinuousStrategyProvider,
    DeterministicContinuousModel,
    L3Config,
    run_l3,
)


def test_grid_sensitivity_reuses_formal_schedule_without_llm_calls() -> None:
    config = L3Config(
        num_trials=2,
        num_rounds=3,
        num_particles=12,
        num_strategies=3,
        planning_horizon=3,
        rollout_samples=8,
        grid_resolution=4,
        bootstrap_replicates=30,
        trial_concurrency=2,
    )
    formal = run_l3(
        ContinuousStrategyProvider(DeterministicContinuousModel(), config), config
    )
    formal["run_id"] = "dry"
    formal["config"] = asdict(config)

    summary = run_sensitivity(formal, (4, 8))

    assert summary["zero_llm_calls"]
    assert summary["resolutions"]["4"]["full_depth2_sequence_count"] == 16
    assert summary["resolutions"]["8"]["full_depth2_sequence_count"] == 64
