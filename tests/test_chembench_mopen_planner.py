from __future__ import annotations

import json

import numpy as np
import pytest

from environments.chembench_mopen.mechanics import (
    BankedProposer,
    DynamicPlanner,
    FixedProposer,
    HistoryBlindProposer,
    ModelBank,
    OracleProposer,
    PolicyLadderPlanner,
    ProposalCache,
    SpeculativePlanner,
)
from scripts.chembench_mopen_mechanics import comparison_at_tolerance


def _bank() -> ModelBank:
    likelihoods = np.array(
        [
            [[0.90, 0.09, 0.01], [0.80, 0.19, 0.01], [0.34, 0.33, 0.33]],
            [[0.09, 0.90, 0.01], [0.80, 0.19, 0.01], [0.33, 0.34, 0.33]],
            [[0.01, 0.09, 0.90], [0.01, 0.19, 0.80], [0.33, 0.33, 0.34]],
            [[0.09, 0.01, 0.90], [0.01, 0.80, 0.19], [0.34, 0.32, 0.34]],
        ],
        dtype=float,
    )
    features = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
        ]
    )
    return ModelBank(
        likelihoods,
        features,
        model_names=("m0", "m1", "m2", "m3"),
        action_names=("a0", "a1", "a2"),
        action_groups=("x", "y", "z"),
        initial_support=(0, 1),
        live_cap=3,
        reserve_cap=1,
    )


def test_dynamic_state_is_normalized_private_and_immutable() -> None:
    bank = _bank()
    parent = bank.initial_state()
    before = parent
    child = bank.transition(parent, 0, 2, (2, 2, -1, 99, 3))
    assert parent == before
    assert child.history == ((0, 2),)
    assert set(child.discovered) == {0, 1, 2, 3}
    assert sum(child.represented_mass) + child.outside_mass == pytest.approx(1.0)
    assert "truth" not in json.dumps(child.public_key()).lower()


def test_proposal_cache_reuses_exact_branch_and_replays_from_bank() -> None:
    bank = _bank()
    state = bank.initial_state()
    cache = ProposalCache(OracleProposer(bank))
    first = cache.get(state, 0, 2, 7)
    second = cache.get(state, 0, 2, 7)
    assert first == second
    assert cache.misses == 1
    assert cache.hits == 1

    replay = ProposalCache(BankedProposer(cache.source_mode, cache.records), source_mode=cache.source_mode)
    assert replay.get(state, 0, 2, 7) == first
    with pytest.raises(KeyError, match="absent"):
        replay.get(state, 1, 2, 7)


def test_fixed_and_history_blind_proposers_do_not_use_branch_answer() -> None:
    bank = _bank()
    state = bank.initial_state()
    fixed = FixedProposer()
    assert fixed.propose(state, 0, 0, 1) == ()
    blind = HistoryBlindProposer((2, 3))
    assert blind.propose(state, 0, 0, 11) == blind.propose(state, 2, 2, 99)


def test_dynamic_planner_runs_all_horizons_with_finite_losses() -> None:
    bank = _bank()
    cache = ProposalCache(OracleProposer(bank))
    planner = DynamicPlanner(bank, cache, seed=123)
    results = [planner.evaluate_horizon(depth, execution_budget=2) for depth in (1, 2, 3)]
    for result in results:
        assert np.isfinite(result["expected_terminal_mse"])
        assert len(result["truth_losses"]) == bank.num_models
        assert result["root_action_index"] in range(bank.num_actions)
    assert cache.misses > 0

    misses = cache.misses
    replay_planner = DynamicPlanner(bank, cache, seed=123)
    replay = replay_planner.evaluate_horizon(1, execution_budget=2)
    assert replay == results[0]
    assert cache.misses == misses
    assert cache.hits > 0


def test_practical_truth_cell_comparison_keeps_aggregate_and_ties_tiny_changes() -> None:
    result = comparison_at_tolerance(
        [1.0, 1.0, 1.0, 1.0],
        [0.9, 1.1, 1.0 + 1e-8, 1.0 - 1e-8],
        1e-6,
    )
    assert result["wins"] == 1
    assert result["losses"] == 1
    assert result["ties"] == 2
    assert result["left_mean"] == pytest.approx(1.0)
    assert result["right_mean"] == pytest.approx(1.0)


def test_speculative_leaf_value_matches_truth_conditional_replay() -> None:
    bank = _bank()
    cache = ProposalCache(OracleProposer(bank))
    planner = SpeculativePlanner(bank, cache, (2, 3), seed=456)
    state = planner.initial_state()
    available = tuple(range(bank.num_actions))
    action = 0
    planned = planner.action_value(state, available, action, 1, 0)

    manual = 0.0
    prior = state.weights()
    for particle_position, truth in enumerate(planner.particle_indices):
        for outcome, probability in enumerate(bank.likelihoods[truth, action, :]):
            if probability <= 1e-14:
                continue
            child = planner.transition(state, action, outcome)
            forecast = planner.forecast(child)
            truth_loss = np.mean((forecast - bank.target_features[truth]) ** 2)
            manual += float(prior[particle_position] * probability * truth_loss)
    assert planned == pytest.approx(manual, abs=1e-12)


def test_speculative_planner_runs_all_horizons_and_reuses_proposals() -> None:
    bank = _bank()
    cache = ProposalCache(OracleProposer(bank))
    results = []
    for depth in (3, 2, 1):
        planner = SpeculativePlanner(bank, cache, (2, 3), seed=456)
        results.append(planner.evaluate_horizon(depth, execution_budget=2))
    assert all(np.isfinite(item["expected_terminal_mse"]) for item in results)
    misses = cache.misses
    replay = SpeculativePlanner(bank, cache, (2, 3), seed=456).evaluate_horizon(
        1, execution_budget=2
    )
    assert replay == results[-1]
    assert cache.misses == misses


def test_policy_ladder_is_calibrated_and_nonworsening() -> None:
    bank = _bank()
    cache = ProposalCache(OracleProposer(bank))
    planner = PolicyLadderPlanner(bank, cache, (2, 3), seed=789)
    results = [planner.evaluate_policy_level(level, execution_budget=3) for level in (1, 2, 3)]

    for result in results:
        assert result["planned_value"] == pytest.approx(
            result["expected_terminal_mse"], abs=1e-12
        )
    assert results[1]["planned_value"] <= results[0]["planned_value"] + 1e-12
    assert results[2]["planned_value"] <= results[1]["planned_value"] + 1e-12


def test_policy_ladder_reuses_banked_proposals_exactly() -> None:
    bank = _bank()
    cache = ProposalCache(OracleProposer(bank))
    planner = PolicyLadderPlanner(bank, cache, (2, 3), seed=789)
    expected = planner.evaluate_policy_level(3, execution_budget=3)
    records = cache.records

    replay_cache = ProposalCache(
        BankedProposer(cache.source_mode, records),
        source_mode=cache.source_mode,
    )
    replay = PolicyLadderPlanner(bank, replay_cache, (2, 3), seed=789)
    assert replay.evaluate_policy_level(3, execution_budget=3) == expected
