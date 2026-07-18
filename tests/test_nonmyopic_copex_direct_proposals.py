import math

import numpy as np
import pytest

from helpers import load_config
from scripts.nonmyopic_copex_direct_proposals import (
    DeterministicDirectProposalModel,
    DirectProposalConfig,
    DirectProposalError,
    DirectProposalProvider,
    _parse_moves,
    run_factorial,
)


def test_direct_move_parser_requires_distinct_legal_endpoints() -> None:
    parsed = _parse_moves(
        '{"moves":[{"dx":0.1,"dy":0.0},{"dx":0.0,"dy":-0.1}]}',
        expected_count=2,
        position=np.asarray([0.5, 0.5]),
        max_step=0.1,
    )
    assert parsed == ((0.6, 0.5), (0.5, 0.4))
    with pytest.raises(DirectProposalError, match="exits"):
        _parse_moves(
            '{"moves":[{"dx":0.1,"dy":0.0},{"dx":0.0,"dy":-0.1}]}',
            expected_count=2,
            position=np.asarray([0.95, 0.5]),
            max_step=0.1,
        )


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


def test_direct_proposal_openrouter_config_is_bounded_and_nonthinking() -> None:
    config = load_config("configs/config_nonmyopic_copex_direct_proposals_openrouter.yaml")
    assert config.model_pairs[0].questioner.model == "google/gemma-4-26b-a4b-it"
    assert config.model_pairs[0].questioner.thinking is False
    assert config.openrouter_run_budget_usd == 2.5
    assert config.openrouter_concurrency == 128
