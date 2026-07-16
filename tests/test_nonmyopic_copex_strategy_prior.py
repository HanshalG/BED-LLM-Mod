import math

import numpy as np

from helpers import load_config
from scripts.nonmyopic_copex_strategy_prior import (
    ContinuousStrategyProvider,
    DeterministicContinuousModel,
    L3Config,
    _parse_width_cell,
    run_l3,
)


def test_width_angle_cell_compiles_distinct_legal_vectors() -> None:
    steps = _parse_width_cell(
        '{"angles_deg":[0,90,180,270]}', expected_count=4, max_step=0.1
    )
    assert len(steps) == 4
    assert len({(round(step.dx or 0.0, 8), round(step.dy or 0.0, 8)) for step in steps}) == 4


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


def test_strategy_prompt_displays_only_addressable_particle_ranks() -> None:
    config = L3Config()
    provider = ContinuousStrategyProvider(DeterministicContinuousModel(), config)
    lines = provider._belief_lines(
        particles=np.zeros((6, 2)),
        probabilities=np.asarray([0.3, 0.25, 0.2, 0.15, 0.06, 0.04]),
    )
    assert len(lines) == 4
    assert lines[-1].startswith("rank 3:")
