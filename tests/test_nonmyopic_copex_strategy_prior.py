import math

from helpers import load_config
from scripts.nonmyopic_copex_strategy_prior import (
    ContinuousStrategyProvider,
    DeterministicContinuousModel,
    L3Config,
    run_l3,
)


def test_small_dry_l3_preserves_controls_and_constraints() -> None:
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
    summary = run_l3(
        ContinuousStrategyProvider(DeterministicContinuousModel(), config), config
    )

    mechanics = summary["mechanics"]
    assert mechanics["all_actions_legal"]
    assert mechanics["initial_strategy_cells_shared_with_d1"]
    assert mechanics["width_scorer_units_match_strategy_eig"]
    assert mechanics["grid_d2_scorer_units_match_strategy_eig"]
    assert mechanics["rollout_scoring_llm_calls"] == 0
    assert all(math.isfinite(summary["summary"][arm]["final_entropy_mean"]) for arm in summary["summary"])


def test_l3_openrouter_config_is_bounded_and_nonthinking() -> None:
    config = load_config("configs/config_nonmyopic_copex_strategy_l3_openrouter.yaml")
    assert config.model_pairs[0].questioner.model == "google/gemma-4-26b-a4b-it"
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_run_budget_usd == 1.5
    assert config.openrouter_concurrency == 128
