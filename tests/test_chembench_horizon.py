from __future__ import annotations

from itertools import permutations, product
import json

import numpy as np
import pytest

from environments.chembench_mopen.horizon import (
    BeliefBranch,
    FiniteBeliefModel,
    HorizonPlanner,
    SearchLimitExceeded,
    SearchLimits,
)
from scripts.chembench_horizon_reference import (
    adaptive_model,
    build_report,
    receding_reference_tree,
    xor_model,
)


def all_trees(menu, depth, outcomes=2):
    if depth == 0:
        return [None]
    trees = []
    for action in menu:
        children = all_trees(tuple(a for a in menu if a != action), depth - 1, outcomes)
        trees.extend(
            (action, branches) for branches in product(children, repeat=outcomes)
        )
    return trees


def direct_tree_loss(model, tree, masses):
    """Independent world/observation enumeration, without model Bayes methods."""
    if masses.sum() == 0:
        return 0.0
    if tree is None:
        mean = masses @ model.targets / masses.sum()
        return float(masses @ ((model.targets - mean) ** 2 @ model.target_weights))
    action, children = tree
    return sum(
        direct_tree_loss(model, child, masses * model.likelihoods[:, action, outcome])
        for outcome, child in enumerate(children)
    )


def sequence_tree(sequence, outcomes=2):
    if not sequence:
        return None
    return sequence[0], (sequence_tree(sequence[1:], outcomes),) * outcomes


def plan_tree(node, outcomes=2):
    if node.action is None:
        return None
    children = [None] * outcomes
    for edge in node.branches:
        children[edge.observation] = plan_tree(edge.child, outcomes)
    return node.action, tuple(children)


@pytest.mark.parametrize("seed", range(4))
@pytest.mark.parametrize("horizon", (1, 2, 3))
def test_adaptive_matches_independent_exhaustive_policy_enumeration(seed, horizon):
    rng = np.random.default_rng(seed)
    model = FiniteBeliefModel(
        rng.dirichlet([1, 1], size=(4, 3)),
        rng.normal(size=(4, 2)),
        rng.dirichlet(np.ones(4)),
        target_weights=np.array([0.2, 0.8]),
    )
    planner = HorizonPlanner(model)
    plan = planner.plan(model.initial_state, horizon)
    oracle = min(
        direct_tree_loss(model, tree, np.array(model.initial_state))
        for tree in all_trees((0, 1, 2), horizon)
    )
    assert plan.root.expected_risk == pytest.approx(oracle, abs=1e-12)
    assert direct_tree_loss(
        model, plan_tree(plan.root), np.array(model.initial_state)
    ) == pytest.approx(oracle)
    open_loop = planner.plan(model.initial_state, horizon, mode="open_loop")
    fixed_oracle = min(
        direct_tree_loss(model, sequence_tree(sequence), np.array(model.initial_state))
        for sequence in permutations((0, 1, 2), horizon)
    )
    assert open_loop.root.expected_risk == pytest.approx(fixed_oracle)
    assert plan.root.expected_risk <= open_loop.root.expected_risk + 1e-12
    assert dict(plan.root_action_values)[plan.root.action] == pytest.approx(oracle)


def test_complementarity_is_not_screened_out_and_depth_three_plateaus():
    model = xor_model()
    planner = HorizonPlanner(model)
    plans = [planner.plan(model.initial_state, h) for h in (1, 2, 3)]
    assert plans[0].root.action == 2
    assert dict(plans[0].root_action_values) == pytest.approx(
        {0: 0.25, 1: 0.25, 2: 0.1875}
    )
    assert [p.root.expected_risk for p in plans] == pytest.approx([0.1875, 0, 0])
    assert plans[1].root.action == 0


def test_real_contingency_and_open_loop_nonanticipativity():
    model = adaptive_model()
    planner = HorizonPlanner(model)
    adaptive = planner.plan(model.initial_state, 2)
    assert adaptive.root.action == 0
    assert {edge.observation: edge.child.action for edge in adaptive.root.branches} == {
        0: 1,
        1: 2,
    }
    assert adaptive.root.expected_risk == pytest.approx(0)
    fixed = planner.plan(model.initial_state, 2, mode="open_loop")
    assert fixed.root.expected_risk == pytest.approx(0.125)
    assert {edge.child.action for edge in fixed.root.branches} == {
        fixed.fixed_sequence[1]
    }


def test_receding_execution_has_same_measurement_budget_and_clamps_horizon():
    model = adaptive_model()
    planner = HorizonPlanner(model)
    total_by_horizon = []
    for horizon in (1, 2, 3):
        observed_depths = []

        def execute(state, available, remaining):
            if not remaining:
                return model.risk(state)
            plan = planner.plan(state, min(horizon, remaining), available=available)
            observed_depths.append((remaining, plan.effective_horizon))
            action = plan.root.action
            return sum(
                row.probability
                * execute(
                    row.state, tuple(a for a in available if a != action), remaining - 1
                )
                for row in model.branches(state, action)
            )

        total_by_horizon.append(execute(model.initial_state, (0, 1, 2), 2))
        assert all(
            depth == min(horizon, remaining) for remaining, depth in observed_depths
        )
    assert total_by_horizon == pytest.approx([0.125, 0, 0])


def test_linear_gaussian_known_noise_has_no_adaptivity_gap():
    class LinearGaussian:
        num_actions = 3
        designs = np.array([[1, 0], [0, 1], [1, 1]], dtype=float)

        def risk(self, state):
            return float(np.trace(np.array(state).reshape(2, 2)))

        def branches(self, state, action):
            covariance = np.array(state).reshape(2, 2)
            design = self.designs[action]
            updated = np.linalg.inv(
                np.linalg.inv(covariance) + np.outer(design, design)
            )
            # Integrate observations analytically: covariance is outcome-independent.
            return (BeliefBranch(None, 1.0, tuple(updated.ravel())),)

    model = LinearGaussian()
    planner = HorizonPlanner(model)
    state = (1.0, 0.0, 0.0, 1.0)
    for h in (1, 2, 3):
        expected = min(
            np.trace(
                np.linalg.inv(
                    np.eye(2)
                    + sum(
                        np.outer(model.designs[a], model.designs[a]) for a in sequence
                    )
                )
            )
            for sequence in permutations(range(3), h)
        )
        for mode in ("adaptive", "open_loop"):
            assert planner.plan(
                state, h, mode=mode
            ).root.expected_risk == pytest.approx(expected)


def test_open_loop_planning_gap_is_not_automatically_a_deployment_gap():
    model = adaptive_model()
    fixed = HorizonPlanner(model).plan(model.initial_state, 2, mode="open_loop")
    assert fixed.root.expected_risk == pytest.approx(0.125)
    replay = receding_reference_tree(
        model, model.initial_state, horizon=2, mode="open_loop"
    )
    assert replay["terminal_expected_risk"] == pytest.approx(0)
    for row in replay["branches"]:
        assert row["child"]["optimized_horizon"] == 1
        assert all(
            child["child"]["remaining_measurements"] == 0
            for child in row["child"]["branches"]
        )


def test_small_cache_changes_effort_not_the_policy():
    model = adaptive_model()
    small = HorizonPlanner(model, limits=SearchLimits(cache_size=1)).plan(
        model.initial_state, 3
    )
    usual = HorizonPlanner(model).plan(model.initial_state, 3)
    assert small.root == usual.root
    assert small.root_action_values == usual.root_action_values


@pytest.mark.parametrize(
    "bad_prior", ([0.5, 0.5], [0, 0, 0, 0], [1, 1, 1, 1], [float("nan"), 0, 0, 1])
)
def test_invalid_model_priors_fail_closed(bad_prior):
    model = xor_model()
    with pytest.raises(ValueError, match="weights"):
        FiniteBeliefModel(model.likelihoods, model.targets, np.array(bad_prior))


def test_zero_probability_branches_and_bayes_update_are_explicit():
    model = xor_model()
    child = model.condition(model.initial_state, 0, 0)
    assert child == (0.5, 0.5, 0, 0)
    rows = model.branches(child, 0)
    assert len(rows) == 1
    assert rows[0].probability == 1
    with pytest.raises(ValueError, match="zero predictive"):
        model.condition(child, 0, 1)
    with pytest.raises(ValueError, match="weights"):
        model.risk((1, 1, 1, 1))


def test_inputs_are_copied_and_target_weights_stay_fixed():
    likelihoods = np.full((2, 1, 2), 0.5)
    targets = np.array([[0, 0], [2, 4]], dtype=float)
    model = FiniteBeliefModel(
        likelihoods, targets, np.array([0.5, 0.5]), target_weights=np.array([1, 0])
    )
    targets[:] = 100
    likelihoods[:] = 0
    assert model.risk(model.initial_state) == pytest.approx(1)
    assert model.risk(model.condition(model.initial_state, 0, 0)) == pytest.approx(1)
    with pytest.raises(ValueError):
        model.targets[0, 0] = 3


@pytest.mark.parametrize("mode", ("adaptive", "open_loop"))
def test_repeats_exhaustion_zero_horizon_and_stable_ties(mode):
    model = xor_model()
    planner = HorizonPlanner(model)
    assert planner.plan(
        model.initial_state, 0, mode=mode
    ).root.expected_risk == pytest.approx(0.25)
    assert (
        planner.plan(model.initial_state, 3, available=(), mode=mode).root.action
        is None
    )
    clamped = planner.plan(model.initial_state, 4, mode=mode)
    assert clamped.requested_horizon == 4
    assert clamped.effective_horizon == 3
    repeated = planner.plan(
        model.initial_state, 2, available=(2,), allow_repeats=True, mode=mode
    )
    oracle = direct_tree_loss(
        model, sequence_tree((2, 2)), np.array(model.initial_state)
    )
    assert repeated.root.expected_risk == pytest.approx(oracle)
    assert {edge.child.action for edge in repeated.root.branches} == {2}
    assert (
        planner.plan(model.initial_state, 2, available=(1, 0), mode=mode).root.action
        == 0
    )


@pytest.mark.parametrize("horizon", (-1, 1.5, True))
def test_invalid_horizon_rejected(horizon):
    model = xor_model()
    with pytest.raises(ValueError, match="horizon"):
        HorizonPlanner(model).plan(model.initial_state, horizon)


@pytest.mark.parametrize("available", ((0, 0), (-1,), (3,), (True,)))
def test_invalid_actions_rejected(available):
    model = xor_model()
    with pytest.raises(ValueError):
        HorizonPlanner(model).plan(model.initial_state, 1, available=available)


@pytest.mark.parametrize("mode", ("adaptive", "open_loop"))
def test_limits_fail_without_fallback_and_do_not_poison_next_plan(mode):
    model = xor_model()
    planner = HorizonPlanner(model, limits=SearchLimits(max_nodes=2))
    with pytest.raises(SearchLimitExceeded, match="max_nodes"):
        planner.plan(model.initial_state, 3, mode=mode)
    assert planner.plan(model.initial_state, 0, mode=mode).root.action is None
    with pytest.raises(SearchLimitExceeded, match="max_depth"):
        HorizonPlanner(model, limits=SearchLimits(max_depth=1)).plan(
            model.initial_state, 2, mode=mode
        )


def test_time_limit_and_invalid_provider_fail_closed(monkeypatch):
    from environments.chembench_mopen import horizon

    ticks = iter((0.0, 100.0))
    monkeypatch.setattr(horizon, "monotonic", lambda: next(ticks))
    model = xor_model()
    with pytest.raises(SearchLimitExceeded, match="max_seconds"):
        HorizonPlanner(model).plan(model.initial_state, 1)


@pytest.mark.parametrize("probabilities", ((0.5,), (float("nan"),), (1.0, 1.0)))
def test_invalid_branch_provider_rejected(probabilities):
    class Invalid:
        num_actions = 1

        def risk(self, state):
            return 0.0

        def branches(self, state, action):
            return tuple(BeliefBranch(i, p, state) for i, p in enumerate(probabilities))

    with pytest.raises(ValueError, match="branches"):
        HorizonPlanner(Invalid()).plan((), 1)


def test_reference_report_is_constructed_and_has_no_paid_authority():
    report = build_report()
    assert report["status"] == "reference_checks_passed"
    assert all(report["gates"].values())
    assert report["model_calls"] == report["paid_cost_usd"] == 0
    assert not report["chemistry_endpoints_opened"]


def test_runner_writes_atomic_report_and_refuses_overwrite(tmp_path, monkeypatch):
    from scripts import chembench_horizon_reference as runner

    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["reference", "--output-dir", str(output)])
    runner.main()
    result = output / "RESULT.json"
    original = result.read_bytes()
    assert json.loads(original)["status"] == "reference_checks_passed"
    assert not list(output.glob("*.tmp"))
    with pytest.raises(FileExistsError):
        runner.main()
    assert result.read_bytes() == original


def test_runner_banks_execution_failure_without_success_artifact(tmp_path, monkeypatch):
    from scripts import chembench_horizon_reference as runner

    def fail():
        raise SearchLimitExceeded("test limit")

    output = tmp_path / "run"
    monkeypatch.setattr("sys.argv", ["reference", "--output-dir", str(output)])
    monkeypatch.setattr(runner, "build_report", fail)
    with pytest.raises(SystemExit) as raised:
        runner.main()
    assert raised.value.code == 1
    assert (
        json.loads((output / "FAILURE.json").read_text())["status"]
        == "reference_execution_failed"
    )
    assert not (output / "RESULT.json").exists()
