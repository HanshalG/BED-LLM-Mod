from __future__ import annotations

import hashlib
import json
import math

import numpy as np

from environments.chembench_mopen.continuous import (
    ContinuousOracleProposer,
    ContinuousParameterBank,
    ContinuousPolicyLadderPlanner,
    ScenarioPolicyLadderPlanner,
    weighted_quantile_branches,
)
from environments.chembench_mopen.mechanics import ProposalCache, proposal_key


def _bank() -> ContinuousParameterBank:
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
    return ContinuousParameterBank(
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


def _planner(*, num_branches: int = 12) -> ContinuousPolicyLadderPlanner:
    bank = _bank()
    truth_means = np.array([[1.05, 3.1, 1.05], [3.1, 0.95, 2.9]])
    truth_targets = np.array(
        [[np.log1p(1.05), np.log1p(3.1)], [np.log1p(3.1), np.log1p(0.95)]]
    )
    return ContinuousPolicyLadderPlanner(
        bank,
        ProposalCache(ContinuousOracleProposer(bank)),
        (2, 3),
        truth_means,
        truth_targets,
        num_branches=num_branches,
        seed=41,
    )


def _scenario_planner(**overrides: object) -> ScenarioPolicyLadderPlanner:
    base = _planner()
    kwargs: dict[str, object] = {
        "bank": base.bank,
        "proposal_cache": ProposalCache(ContinuousOracleProposer(base.bank)),
        "speculative_models": (2, 3),
        "truth_observation_means": base.truth_observation_means,
        "truth_target_features": base.truth_target_features,
        "scenario_counts_by_remaining": (3, 4, 6),
        "action_widths_by_remaining": (2, 3, 3),
        "seed": 73,
    }
    kwargs.update(overrides)
    return ScenarioPolicyLadderPlanner(**kwargs)


def test_weighted_quantile_branches_preserve_mass_and_support() -> None:
    values = np.array([1.0, 2.0, 10.0])
    weights = np.array([0.2, 0.3, 0.5])
    branches = weighted_quantile_branches(values, weights, 5)
    assert len(branches) == 3
    assert np.isclose(sum(item.probability for item in branches), 1.0)
    assert set(item.observation for item in branches).issubset(set(values))
    assert [item.observation for item in branches] == sorted(
        item.observation for item in branches
    )


def test_weighted_quantile_branches_are_invariant_to_weight_scale() -> None:
    values = np.array([1.0, 2.0, 10.0])
    normalized = weighted_quantile_branches(values, np.array([0.2, 0.3, 0.5]), 5)
    scaled = weighted_quantile_branches(values, np.array([2e20, 3e20, 5e20]), 5)
    assert scaled == normalized


def test_outside_likelihood_uses_same_log_rate_density_convention() -> None:
    bank = _bank()
    action = 1
    observation = 3.05
    transformed = math.log1p(observation)
    sigma = float(bank.outside_log_sigma[action])
    residual = (transformed - float(bank.outside_log_mean[action])) / sigma
    expected = -0.5 * (residual**2 + math.log(2.0 * math.pi)) - math.log(sigma)
    assert np.isclose(bank.outside_log_likelihood(((action, observation),)), expected)


def test_continuous_observation_updates_parameter_and_structure_weights() -> None:
    bank = _bank()
    state = bank.initial_state()
    child = bank.transition(state, 1, 3.05, (2, 3))
    assert set(child.discovered) == {0, 1, 2, 3}
    assert np.isclose(sum(child.represented_mass) + child.outside_mass, 1.0)
    assert all(np.isclose(sum(weights), 1.0) for weights in child.parameter_weight)
    top_model = child.represented_models[int(np.argmax(child.represented_weight))]
    assert top_model == 2
    assert np.isfinite(bank.forecast(child)).all()
    report = bank.residual_report(child)
    assert report["observations"][0]["observed_rate"] == 3.05
    assert np.isfinite(report["latest_signed_log_rate_residual"])


def test_integer_proposal_keys_remain_backward_compatible() -> None:
    state = _bank().initial_state()
    payload = {
        "mode": "fixture",
        "seed": 11,
        "state": state.public_key(),
        "action": 2,
        "outcome": 1,
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert proposal_key("fixture", state, 2, 1, 11) == expected
    assert proposal_key("fixture", state, 2, 1.25, 11) != expected


def test_quantile_one_step_values_match_monte_carlo_ranking() -> None:
    planner = _planner(num_branches=12)
    state = planner.initial_state()
    available = tuple(range(planner.bank.num_actions))
    quantile_values = [
        planner.policy_action_value(state, available, 1, 1, action)
        for action in available
    ]
    monte_carlo = [
        planner.monte_carlo_one_step_value(
            state,
            action,
            num_samples=1_000,
            seed=100 + action,
        )
        for action in available
    ]
    monte_carlo_values = [item[0] for item in monte_carlo]
    assert int(np.argmin(quantile_values)) == int(np.argmin(monte_carlo_values))
    for approximation, (reference, standard_error) in zip(
        quantile_values, monte_carlo, strict=True
    ):
        assert abs(approximation - reference) <= max(0.01, 5.0 * standard_error)


def test_continuous_policy_ladder_planned_risk_is_nonincreasing() -> None:
    planner = _planner()
    results = [planner.evaluate_policy_level(level, execution_budget=3) for level in (1, 2, 3)]
    planned = [item["planned_particle_risk"] for item in results]
    assert planned[1] <= planned[0] + 1e-12
    assert planned[2] <= planned[1] + 1e-12
    assert all(np.isfinite(item["expected_truth_mse"]) for item in results)


def test_parameter_kernel_expands_sparse_particle_likelihood_support() -> None:
    exact = _bank()
    means = exact.particle_observation_means
    targets = exact.particle_target_features
    kernel = ContinuousParameterBank(
        means,
        targets,
        model_names=exact.model_names,
        action_names=exact.action_names,
        action_groups=exact.action_groups,
        action_inputs=exact.action_inputs,
        initial_support=exact.initial_support,
        parameter_kernel_scale=1.0,
    )
    for exact_sigma, kernel_sigma in zip(
        exact.particle_log_sigmas,
        kernel.particle_log_sigmas,
        strict=True,
    ):
        assert np.all(kernel_sigma >= exact_sigma)
    assert any(
        np.any(kernel_sigma > exact_sigma)
        for exact_sigma, kernel_sigma in zip(
            exact.particle_log_sigmas,
            kernel.particle_log_sigmas,
            strict=True,
        )
    )


def test_scenario_policy_ladder_is_deterministic_and_finite() -> None:
    first = _scenario_planner()
    second = _scenario_planner()
    first_results = [
        first.evaluate_policy_level(level, execution_budget=3) for level in (1, 2)
    ]
    second_results = [
        second.evaluate_policy_level(level, execution_budget=3) for level in (1, 2)
    ]
    assert first_results == second_results
    assert all(np.isfinite(item["planned_particle_risk"]) for item in first_results)
    assert all(np.isfinite(item["expected_truth_mse"]) for item in first_results)
    assert first_results[-1]["policy_abstraction"]["signature_mode"] == "predictive"


def test_scenario_planner_uses_independent_speculative_particles() -> None:
    base = _planner()
    speculative_means = tuple(item * 1.25 for item in base.bank.particle_observation_means)
    speculative_targets = tuple(item + 0.2 for item in base.bank.particle_target_features)
    planner = _scenario_planner(
        speculative_particle_observation_means=speculative_means,
        speculative_particle_target_features=speculative_targets,
    )
    assert np.allclose(
        planner._pair_raw_means[0],
        speculative_means[2][0],
    )
    assert not np.allclose(
        planner._pair_raw_means[0],
        base.bank.particle_observation_means[2][0],
    )
    assert np.allclose(planner._pair_targets[0], speculative_targets[2][0])


def test_predictive_policy_abstraction_reuses_and_bounds_actions() -> None:
    planner = _scenario_planner(policy_table_max_size=2)
    state = planner.initial_state()
    available = tuple(range(planner.bank.num_actions))
    first = planner.policy_action(state, available, 2, 1)
    misses = planner.abstract_policy_misses
    planner.clear_runtime_caches(clear_policy_table=False)
    second = planner.policy_action(state, available, 2, 1)
    assert second == first
    assert planner.abstract_policy_hits == 1
    assert planner.abstract_policy_misses == misses

    planner._remember_abstract_policy_action(("extra", 1), 0)
    planner._remember_abstract_policy_action(("extra", 2), 1)
    planner._remember_abstract_policy_action(("extra", 3), 2)
    diagnostics = planner.policy_diagnostics()
    assert diagnostics["table_size"] == 2
    assert diagnostics["evictions"] >= 2


def test_frozen_policy_table_falls_back_to_predecessor_on_uncovered_state() -> None:
    planner = _scenario_planner(freeze_predecessor_on_miss=True)
    state = planner.initial_state()
    available = tuple(range(planner.bank.num_actions))
    predecessor = planner.policy_action(state, available, 2, 1)
    planner.clear_runtime_caches(clear_policy_table=False)
    planner._frozen_policy_level = 2
    selected = planner.policy_action(state, available, 2, 2)
    assert selected == predecessor
    assert planner.abstract_policy_predecessor_fallbacks == 1


def test_conservative_policy_improvement_rejects_small_or_inconsistent_gain() -> None:
    def selected_for(values: tuple[float, float, float]) -> int:
        planner = _scenario_planner(use_policy_abstraction=False)
        planner.candidate_actions = lambda state, available, width: (0, 1)  # type: ignore[method-assign]

        def action_value(
            state: object,
            available: tuple[int, ...],
            remaining: int,
            level: int,
            action: int,
            replicate: int,
        ) -> float:
            del state, available, remaining
            if level == 1:
                return 1.0 if action == 0 else 2.0
            return 1.0 if action == 0 else values[replicate]

        planner.policy_action_value_replicate = action_value  # type: ignore[method-assign]
        return planner.policy_action(planner.initial_state(), (0, 1, 2), 2, 2)

    assert selected_for((0.96, 0.96, 0.96)) == 0
    assert selected_for((0.8, 0.8, 1.1)) == 0
    assert selected_for((0.8, 0.8, 0.8)) == 1
