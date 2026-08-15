from __future__ import annotations

import numpy as np

from environments.chembench_mopen.empirical import (
    EmpiricalOracleProposer,
    EmpiricalParameterBank,
    EmpiricalPolicyLadderPlanner,
)
from environments.chembench_mopen.mechanics import ProposalCache


def _bank() -> EmpiricalParameterBank:
    means = (
        np.array([[1.0, 1.0, 2.0], [1.1, 0.9, 2.1]]),
        np.array([[2.0, 1.0, 1.0], [2.2, 1.1, 0.9]]),
        np.array([[1.0, 3.0, 1.0], [0.9, 3.2, 1.1]]),
        np.array([[3.0, 1.0, 3.0], [3.2, 0.9, 2.8]]),
    )
    targets = tuple(
        np.stack((np.log1p(item[:, 0]), np.log1p(item[:, 1])), axis=1)
        for item in means
    )
    return EmpiricalParameterBank(
        means,
        targets,
        model_names=("m0", "m1", "m2", "m3"),
        action_names=("C_A=1", "C_I=1", "T=300"),
        action_groups=("C_A", "C_I", "T"),
        action_inputs=np.array(
            [
                [1.0, 0.0, 1.0, 0.0, 1.0, 310.0, 7.0],
                [1.0, 1.0, 1.0, 0.0, 1.0, 310.0, 7.0],
                [1.0, 0.0, 1.0, 0.0, 1.0, 300.0, 7.0],
            ]
        ),
        initial_support=(0, 1),
        live_cap=3,
        reserve_cap=1,
    )


def test_empirical_belief_updates_structure_and_parameter_weights() -> None:
    bank = _bank()
    state = bank.initial_state()
    child = bank.transition(state, 1, 2, (2, 3))
    assert set(child.discovered) == {0, 1, 2, 3}
    assert np.isclose(sum(child.represented_mass) + child.outside_mass, 1.0)
    assert all(np.isclose(sum(weights), 1.0) for weights in child.parameter_weight)
    assert np.isfinite(bank.forecast(child)).all()
    assert all((likelihoods > 0).all() for likelihoods in bank.particle_likelihoods)


def test_empirical_residual_report_contains_signed_log_rate_signal() -> None:
    bank = _bank()
    child = bank.transition(bank.initial_state(), 1, 2, ())
    report = bank.residual_report(child)
    assert len(report["observations"]) == 1
    assert np.isfinite(report["latest_signed_residual"])
    assert report["observations"][0]["action"] == "C_I=1"


def test_empirical_policy_ladder_model_risk_is_nonincreasing() -> None:
    bank = _bank()
    cache = ProposalCache(EmpiricalOracleProposer(bank))
    truth_means = np.array([[1.05, 3.1, 1.05], [3.1, 0.95, 2.9]])
    truth_targets = np.array(
        [[np.log1p(1.05), np.log1p(3.1)], [np.log1p(3.1), np.log1p(0.95)]]
    )
    planner = EmpiricalPolicyLadderPlanner(
        bank,
        cache,
        (2, 3),
        truth_means,
        truth_targets,
        seed=41,
    )
    results = [planner.evaluate_policy_level(level, execution_budget=3) for level in (1, 2, 3)]
    planned = [item["planned_particle_risk"] for item in results]
    assert planned[1] <= planned[0] + 1e-12
    assert planned[2] <= planned[1] + 1e-12
    assert all(np.isfinite(item["expected_truth_mse"]) for item in results)
    assert cache.misses > 0
