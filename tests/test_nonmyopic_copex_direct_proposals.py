import math

import numpy as np
import pytest

from helpers import load_config
from scripts.nonmyopic_copex_direct_proposals import (
    DeterministicDirectProposalModel,
    DirectProposalConfig,
    DirectProposalError,
    DirectProposalProvider,
    _parse_angles,
    _grid_actions,
    _immediate_eig,
    run_factorial,
)


def test_direct_angle_parser_deduplicates_boundary_endpoints_without_padding() -> None:
    parsed = _parse_angles(
        '{"angles_deg":[0.0,270.0]}',
        expected_count=2,
        position=np.asarray([0.5, 0.5]),
        max_step=0.1,
    )
    assert np.allclose(parsed, ((0.6, 0.5), (0.5, 0.4)))
    parsed_boundary = _parse_angles(
        '{"angles_deg":[90.0,135.0,225.0]}',
        expected_count=3,
        position=np.asarray([0.0, 0.0]),
        max_step=0.1,
    )
    assert len(parsed_boundary) == 1
    with pytest.raises(DirectProposalError, match="distinct"):
        _parse_angles(
            '{"angles_deg":[0.0,0.0]}',
            expected_count=2,
            position=np.asarray([0.5, 0.5]),
            max_step=0.1,
        )


def test_grid_candidates_cover_the_circle_before_filling_boundary_gaps() -> None:
    config = DirectProposalConfig(
        num_trials=1,
        num_rounds=2,
        num_particles=2,
        candidate_width=3,
        outer_rollouts=2,
        child_rollouts=2,
        grid_resolution=12,
        bootstrap_replicates=2,
        trial_concurrency=1,
    )
    position = np.asarray([0.5, 0.5])
    deltas = np.asarray(_grid_actions(position, config)) - position
    assert np.any(deltas[:, 0] > 0.0)
    assert np.any(deltas[:, 0] < 0.0)
    assert np.any(deltas[:, 1] < 0.0)


def test_small_dry_factorial_has_paired_controls() -> None:
    config = DirectProposalConfig(
        num_trials=2,
        num_rounds=3,
        num_particles=12,
        candidate_width=2,
        outer_rollouts=2,
        child_rollouts=3,
        grid_resolution=6,
        bootstrap_replicates=30,
        trial_concurrency=2,
    )
    summary = run_factorial(
        DirectProposalProvider(DeterministicDirectProposalModel(), config), config
    )
    mechanics = summary["mechanics"]
    assert mechanics["all_actions_legal"]
    assert mechanics["initial_root_cell_shared"]
    assert mechanics["width_call_allocation_matches_virtual_depth_two"]
    assert mechanics["inner_llm_calls_used_only_for_action_proposals"]
    assert all(math.isfinite(row["entropy_auc_mean"]) for row in summary["summary"].values())
    assert len(summary["comparisons"]["llm_d2_minus_llm_d1"]["paired_values"]) == 2
    for trial in summary["trials"]:
        assert trial["traces"]["llm_d2"][-1]["logical_llm_calls"] == 1
        assert trial["traces"]["llm_width"][-1]["logical_llm_calls"] == 1


def test_direct_proposal_openrouter_config_is_bounded_and_nonthinking() -> None:
    config = load_config("configs/config_nonmyopic_copex_direct_proposals_openrouter.yaml")
    assert config.model_pairs[0].questioner.model == "google/gemma-4-26b-a4b-it"
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_run_budget_usd == 2.5
    assert config.openrouter_concurrency == 128


def test_quadrature_one_step_score_is_independent_of_monte_carlo_draws() -> None:
    config = DirectProposalConfig(
        num_trials=1,
        num_rounds=2,
        num_particles=2,
        candidate_width=2,
        outer_rollouts=2,
        child_rollouts=2,
        grid_resolution=3,
        bootstrap_replicates=2,
        trial_concurrency=1,
        one_step_scoring="quadrature",
    )
    particles = np.asarray([[0.2, 0.2], [0.8, 0.8], [0.5, 0.5]])
    probabilities = np.asarray([0.2, 0.3, 0.5])
    first = _immediate_eig(
        (0.6, 0.5), position=np.asarray([0.5, 0.5]), particles=particles,
        probabilities=probabilities, uniforms=np.asarray([0.1]), noise_zs=np.asarray([10.0]), config=config,
    )
    second = _immediate_eig(
        (0.6, 0.5), position=np.asarray([0.5, 0.5]), particles=particles,
        probabilities=probabilities, uniforms=np.asarray([0.9, 0.8]), noise_zs=np.asarray([-5.0, 5.0]), config=config,
    )
    assert math.isfinite(first)
    assert first == second
