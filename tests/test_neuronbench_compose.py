from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from environments.neuronbench_compose.mechanics import (
    CompositionalBank,
    CompositionalPlanner,
    candidate_masks,
    compare_losses,
    truth_masks,
)
from scripts.neuronbench_compose_opportunity import (
    build_query_battery,
    canonical_segments,
    mechanism_kwargs,
    verify_source,
)


def synthetic_bank() -> CompositionalBank:
    masks = candidate_masks()
    action_counts = np.asarray(
        [
            [
                int(bool(mask & 1)) * 4,
                int(bool(mask & 2)) * 4,
                int(bool(mask & 4)) * 4,
                mask.bit_count(),
            ]
            for mask in masks
        ],
        dtype=float,
    )
    query_counts = np.asarray(
        [
            [
                sum((index + 1) * bool(mask & (1 << index)) for index in range(6)),
                mask.bit_count(),
            ]
            for mask in masks
        ],
        dtype=float,
    )
    return CompositionalBank(
        action_counts,
        query_counts,
        ("a", "b", "c", "count"),
        ("q0", "q1"),
    )


def test_candidate_and_truth_universes_are_frozen() -> None:
    candidates = candidate_masks()
    truths = truth_masks()
    assert len(candidates) == 22
    assert len(set(candidates)) == 22
    assert candidates[0] == 0
    assert len(truths) == 15
    assert all(mask.bit_count() == 2 for mask in truths)


def test_oracle_proposal_adds_exactly_one_valid_edit() -> None:
    bank = synthetic_bank()
    root = bank.initial_state()
    child = bank.transition(root, 0, 4, proposal_mode="oracle")
    assert len(child.support) == 2
    proposal = next(mask for mask in child.support if mask != 0)
    assert proposal.bit_count() == 1
    grandchild = bank.transition(child, 1, 4, proposal_mode="oracle")
    assert len(grandchild.support) == 3
    added = next(mask for mask in grandchild.support if mask not in child.support)
    assert any((base | added) == added and added.bit_count() == base.bit_count() + 1 for base in child.support)
    assert all(mask in candidate_masks() for mask in grandchild.support)


def test_truth_and_inference_weights_are_finite_and_normalized() -> None:
    bank = synthetic_bank()
    state = bank.transition(bank.initial_state(), 0, 4, proposal_mode="oracle")
    assert np.isclose(state.weights().sum(), 1.0)
    assert np.isclose(bank.inference_weights(state).sum(), 1.0)
    assert np.isfinite(bank.forecast(state)).all()
    assert np.isclose(sum(probability for _, probability in bank.branches(state, 1)), 1.0)


def test_policy_branch_plan_matches_explicit_truth_replay() -> None:
    bank = synthetic_bank()
    planner = CompositionalPlanner(bank)
    result = planner.evaluate(2, execution_budget=2)
    assert result["root_action_index"] in range(bank.num_actions)
    assert result["planned_value"] == pytest.approx(
        result["expected_terminal_mse"], abs=1e-12
    )
    assert len(result["truth_losses"]) == 15
    assert all(len(actions) == 2 for actions in result["truth_actions"])


def test_full_support_forecast_uses_all_models_without_proposals() -> None:
    bank = synthetic_bank()
    planner = CompositionalPlanner(bank, proposal_mode="fixed")
    result = planner.evaluate(1, execution_budget=1, initial_support=bank.masks)
    assert all(len(support) == 22 for support in result["final_supports"])


def test_query_battery_dedupes_and_excludes_design_actions() -> None:
    pool = (("design", [(10, 1)]),)
    worlds = SimpleNamespace(
        POOL=pool,
        WORLDS={
            "first": {"test": [("duplicate design", [(10, 1)]), ("new", [(20, 2)])]},
            "second": {"test": [("duplicate new", [(20, 2)]), ("newer", [(30, -1), (5, 3)])]},
        },
    )
    assert build_query_battery(worlds) == (
        canonical_segments([(20, 2)]),
        canonical_segments([(30, -1), (5, 3)]),
    )


def test_mechanism_kwargs_compose_multiple_channels_and_slow_na() -> None:
    worlds = SimpleNamespace(Z="z", IH="ih", MT="t", ID="d", MC="m")
    kwargs = mechanism_kwargs(worlds, (1 << 0) | (1 << 2))
    assert kwargs == {"extra": ["z"], "slow_na": True}
    kwargs = mechanism_kwargs(worlds, (1 << 1) | (1 << 5))
    assert kwargs == {"extra": ["ih", "m"], "slow_na": False}


def test_source_binding_accepts_pinned_checkout_and_rejects_other_root(tmp_path) -> None:
    binding = verify_source(__import__("pathlib").Path("/private/tmp/neuronbench-c354622"))
    assert binding["commit"].startswith("c354622")
    with pytest.raises(Exception):
        verify_source(tmp_path)


def test_loss_comparison_is_paired_and_directional() -> None:
    result = compare_losses([4.0, 2.0, 1.0], [2.0, 2.0, 3.0])
    assert result["wins"] == 1
    assert result["ties"] == 1
    assert result["losses"] == 1
    assert result["relative_reduction"] == pytest.approx(0.0)
