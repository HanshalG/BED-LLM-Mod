import numpy as np
import pytest
from scipy.special import ndtr

from environments.chembench_mopen.horizon import HorizonPlanner, SearchLimits
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from environments.chembench_mopen.raw_integration import predictive_expectation


def test_quantile_inversion_and_raw_conditioning():
    model = QuantileGaussianModel([[-2], [3]], [[0.3], [2]], [[0], [1]], [0.3, 0.7])
    branches = model.branches(model.initial_state, 0)
    assert len(branches) == 16
    for b, u in zip(branches, model._quantiles):
        actual = np.array([0.3, 0.7]) @ ndtr(
            (b.observation - np.array([-2, 3])) / [0.3, 2]
        )
        assert actual == pytest.approx(u, abs=1e-8)
        assert b.state == model.condition(model.initial_state, 0, b.observation)
    assert sum(b.probability for b in branches) == pytest.approx(1)


def test_multistep_reference_without_nested_integration_cost():
    model = QuantileGaussianModel(
        [[-0.3] * 3, [0.3] * 3], 1, [[0], [1]], [0.5, 0.5], branch_count=16
    )
    reference = QuantileGaussianModel(
        [[-0.3], [0.3]], 1 / np.sqrt(3), [[0], [1]], [0.5, 0.5]
    )
    expected = predictive_expectation(
        reference, reference.initial_state, 0, reference.risk, value_bound=0.25
    )
    plan = HorizonPlanner(
        model, limits=SearchLimits(max_nodes=100_000, max_seconds=20)
    ).plan(model.initial_state, 3)
    assert abs(plan.root.expected_risk - expected.value) < 0.001
    assert plan.effective_horizon == 3
    assert plan.root.branches[0].child.remaining_depth == 2


@pytest.mark.parametrize("count", [1, True, 129, 3.5])
def test_invalid_count(count):
    with pytest.raises(ValueError):
        QuantileGaussianModel([[0]], 1, [[0]], [1], branch_count=count)


def test_materialization_reuses_scored_root():
    class CountingModel(QuantileGaussianModel):
        def branches(self, state, action):
            if state == self.initial_state:
                self.root_calls[action] += 1
            return super().branches(state, action)

    model = CountingModel([[-1, -0.1], [1, 0.1]], 1, [[0], [1]], [0.5, 0.5])
    model.root_calls = [0, 0]
    plan = HorizonPlanner(model, limits=SearchLimits(cache_size=1)).plan(
        model.initial_state, 1
    )
    assert plan.root.action == 0
    assert model.root_calls == [2, 1]
