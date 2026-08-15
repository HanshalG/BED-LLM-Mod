from __future__ import annotations

import numpy as np

from environments.chembench_mopen.compositional import (
    AtomicStructureOracleProposer,
    CompoundSignature,
    CompositionalEditCompiler,
    StructureParticleIndex,
)
from environments.chembench_mopen.costed import CostedCompositionalPolicyPlanner
from environments.chembench_mopen.mechanics import ModelBank, ProposalCache
from scripts.chembench_costed_repeat_corridor import (
    BASE_ASSAY_NAMES,
    WELL_BUDGET,
    costed_likelihoods,
)


def test_costed_thresholds_ignore_heldout_truth_rows() -> None:
    inference = np.linspace(0.1, 8.0, 60).reshape(6, 10)
    truth = np.full((2, 10), 2.0)
    first_likelihoods, first_thresholds, actions = costed_likelihoods(
        np.vstack((inference, truth)), num_inference_particles=6
    )
    changed_likelihoods, changed_thresholds, changed_actions = costed_likelihoods(
        np.vstack((inference, truth * 100.0)), num_inference_particles=6
    )
    assert np.array_equal(first_thresholds, changed_thresholds)
    assert np.array_equal(first_likelihoods[:6], changed_likelihoods[:6])
    assert not np.array_equal(first_likelihoods[6:], changed_likelihoods[6:])
    assert actions == changed_actions
    assert len(actions) == len(BASE_ASSAY_NAMES) * 3


def _planner() -> CostedCompositionalPolicyPlanner:
    base_means = np.asarray(
        [np.linspace(0.5 + index, 2.0 + index, 10) for index in range(10)]
    )
    likelihoods, _, actions = costed_likelihoods(
        base_means, num_inference_particles=9
    )
    bank = ModelBank(
        likelihoods,
        np.arange(30, dtype=float).reshape(10, 3),
        tuple(f"particle-{index}" for index in range(10)),
        tuple(action.name for action in actions),
        tuple(action.base_name for action in actions),
        (0, 1, 2),
        live_cap=10,
        reserve_cap=0,
    )
    compiler = CompositionalEditCompiler(
        {
            "base": CompoundSignature("hill"),
            "warm": CompoundSignature("hill", temperature="arrhenius"),
            "warm_ph": CompoundSignature(
                "hill", temperature="arrhenius", ph_dependence="bell_curve"
            ),
        }
    )
    particles = StructureParticleIndex(
        ("base",) * 3 + ("warm",) * 3 + ("warm_ph",) * 4,
        inference_particles={
            "base": (0, 1, 2),
            "warm": (3, 4, 5),
            "warm_ph": (6, 7, 8),
        },
        truth_particles=(9,),
    )
    return CostedCompositionalPolicyPlanner(
        bank,
        ProposalCache(AtomicStructureOracleProposer(bank, compiler, particles)),
        (9,),
        compiler=compiler,
        particles=particles,
        actions=actions,
        well_budget=WELL_BUDGET,
        seed=23,
    )


def test_costed_planner_removes_repeat_variants_and_respects_budget() -> None:
    planner = _planner()
    available = tuple(range(planner.bank.num_actions))
    for action in planner.feasible_actions(available, WELL_BUDGET):
        remainder = planner.remaining_actions(available, action)
        chosen_base = planner.actions[action].base_index
        assert all(planner.actions[item].base_index != chosen_base for item in remainder)
    assert not planner.feasible_actions(available, 0)
    assert all(planner.actions[item].cost <= 2 for item in planner.feasible_actions(available, 2))

    scenarios = np.random.default_rng(11).random((1, 4, WELL_BUDGET))
    result = planner.evaluate_policy_level(2, scenario_uniforms=scenarios)
    assert np.isfinite(result["expected_terminal_mse"])
    assert result["root_repeat_count"] in {1, 2, 4}
    assert result["execution_audit"]["within_budget_no_base_reuse"]
    assert result["execution_audit"]["maximum_wells_spent"] <= WELL_BUDGET


def test_cost_blind_planning_uses_unit_future_cost_but_actual_execution_is_bounded() -> None:
    planner = _planner()
    planner.cost_aware = False
    assert all(planner.planning_cost(action) == 1 for action in range(planner.bank.num_actions))
    scenarios = np.random.default_rng(12).random((1, 4, WELL_BUDGET))
    result = planner.evaluate_policy_level(3, scenario_uniforms=scenarios)
    assert np.isfinite(result["expected_terminal_mse"])
    assert result["execution_audit"]["within_budget_no_base_reuse"]
