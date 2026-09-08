from fractions import Fraction

import numpy as np
import pytest

from core.prequential import PrequentialMixture
from environments.chembench_mopen.horizon import FiniteBeliefModel, HorizonPlanner
from environments.chembench_mopen.predictive_mixture import PredictivePoolMixture
from tests.test_chembench_horizon import all_trees, direct_tree_loss


def pool(rows, targets, prior=None):
    rows = np.asarray(rows)
    return FiniteBeliefModel(
        np.stack([1 - rows, rows], axis=-1),
        np.asarray(targets).reshape(len(rows), -1),
        np.full(len(rows), 1 / len(rows)) if prior is None else np.asarray(prior),
    )


def test_model_count_cannot_inflate_group_mass_or_change_plan():
    a = pool([[0, 0], [1, 0]], [0, 1])
    b = pool([[0, 0], [0, 1]], [0, 1])
    expanded = pool([[0, 0]] * 50 + [[0, 1]] * 50, [0] * 50 + [1] * 50)
    plans = []
    for second in (b, expanded):
        m = PredictivePoolMixture({"a": a, "b": second}, {"a": 0.25, "b": 0.75})
        assert m.group_masses(m.initial_state) == pytest.approx({"a": 0.25, "b": 0.75})
        p = HorizonPlanner(m).plan(m.initial_state, 1)
        plans.append(p)
        assert p.root.action == 1
    assert plans[0].root.expected_risk == pytest.approx(plans[1].root.expected_risk)
    reverse = PredictivePoolMixture({"a": a, "b": b}, {"a": 0.75, "b": 0.25})
    assert HorizonPlanner(reverse).plan(reverse.initial_state, 1).root.action == 0


def test_frozen_group_conditioning_matches_prequential_credit_without_mutating_it():
    a = pool([[0, 0], [1, 1]], [0, 1], [0.75, 0.25])
    b = pool([[0, 0], [1, 1]], [0, 1], [0.25, 0.75])
    learner = PrequentialMixture(["a", "b"], [Fraction(1, 2)] * 2)
    m = PredictivePoolMixture(
        {"a": a, "b": b}, dict(zip(learner.expert_ids, learner.weights))
    )
    forecasts = m.group_forecasts(m.initial_state, 0)
    sealed = learner.forecast(
        "real-0", {k: tuple(Fraction(p) for p in v) for k, v in forecasts.items()}
    )
    imagined = m.condition(m.initial_state, 0, 1)
    HorizonPlanner(m).plan(m.initial_state, 2)
    assert learner.weights == (Fraction(1, 2), Fraction(1, 2))
    receipt = learner.observe(sealed, 1)
    assert m.group_masses(imagined) == pytest.approx(
        dict(zip(learner.expert_ids, map(float, receipt.posterior_weights)))
    )


@pytest.mark.parametrize("horizon", [1, 2, 3])
def test_horizon_matches_independent_persistent_world_enumeration(horizon):
    a = pool([[0, 0, 1], [1, 0, 0]], [0, 1], [0.8, 0.2])
    b = pool([[0, 1, 0], [1, 1, 1]], [0.3, 0.7], [0.3, 0.7])
    m = PredictivePoolMixture({"a": a, "b": b}, {"a": 0.4, "b": 0.6})
    plan = HorizonPlanner(m).plan(m.initial_state, horizon)
    expected = min(
        direct_tree_loss(m, t, np.array(m.initial_state))
        for t in all_trees((0, 1, 2), horizon)
    )
    assert plan.root.expected_risk == pytest.approx(expected, abs=1e-12)


def test_eliminated_pool_and_snapshot_copy():
    a = pool([[0]], [0])
    b = pool([[1]], [1])
    m = PredictivePoolMixture({"a": a, "b": b}, {"a": 0.5, "b": 0.5})
    a.initial_state = (0.0,)
    assert m.initial_state == (0.5, 0.5)
    child = m.condition(m.initial_state, 0, 1)
    assert m.group_forecasts(child, 0) == {"a": None, "b": (0.0, 1.0)}
    with pytest.raises(ValueError, match="zero predictive"):
        m.condition(child, 0, 0)


def test_post_observation_refresh_keeps_earned_group_weight():
    learner = PrequentialMixture(["a", "b"], [Fraction(1, 2)] * 2)
    sealed = learner.forecast(
        "before-refresh",
        {"a": [Fraction(3, 4), Fraction(1, 4)], "b": [Fraction(1, 4), Fraction(3, 4)]},
    )
    learner.observe(sealed, 1)
    # Fresh pools already incorporate the observed history. Their count or
    # retrospective fit cannot earn a second weight update for that outcome.
    refreshed = PredictivePoolMixture(
        {"a": pool([[1, 0]], [0]), "b": pool([[1, 1]] * 20, [1] * 20)},
        dict(zip(learner.expert_ids, learner.weights)),
    )
    assert refreshed.group_masses(refreshed.initial_state) == pytest.approx(
        {"a": 0.25, "b": 0.75}
    )
    with pytest.raises(ValueError, match="outstanding"):
        learner.observe(sealed, 1)


def test_invalid_pool_contracts():
    a = pool([[0]], [0])
    with pytest.raises(ValueError, match="keys"):
        PredictivePoolMixture({"a": a}, {"b": 1})
    with pytest.raises(ValueError, match="normalized"):
        PredictivePoolMixture({"a": a}, {"a": float("nan")})
    with pytest.raises(ValueError, match="axes"):
        PredictivePoolMixture({"a": a, "b": pool([[0, 1]], [1])}, {"a": 0.5, "b": 0.5})
