from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("Cython")

from environments.chembench_mopen.batch_horizon import BatchPlan
from environments.chembench_mopen.native_belief import NativeEnvelopeGaussianModel
from environments.chembench_mopen.policy_value import MyopicPolicyValue
from scripts import chembench_calibrated_context_opportunity as runner


def test_all_prespecified_contexts_reported_without_hidden_worlds(
    monkeypatch, tmp_path
):
    from environments.chembench_mopen import pilot_data

    def bomb(*args):
        pytest.fail("hidden world opened")

    monkeypatch.setattr(pilot_data, "build_hidden_worlds", bomb)
    monkeypatch.setattr(
        runner, "preflight", lambda *args: {"status": "public_preflight_passed"}
    )
    monkeypatch.setattr(runner, "load_source", lambda *args: object())
    m = NativeEnvelopeGaussianModel(
        [[-0.2] * 4, [0.2] * 4], 0.15, [[0], [1]], [0.5, 0.5]
    )
    monkeypatch.setattr(
        runner,
        "build_public_pilot",
        lambda *args, **kwargs: SimpleNamespace(model=m, candidate_parameters=()),
    )
    monkeypatch.setattr(runner, "predict", lambda *args: np.array([[-0.2], [0.2]]))
    calls = []

    def myopic(model, state, budget, **kwargs):
        calls.append((model.branch_count, budget))
        return MyopicPolicyValue(1.0, budget, 0, (), 1, 0.01)

    def planned(model, state, horizon, **kwargs):
        return BatchPlan(
            1 if horizon == 2 else 2,
            0.6,
            ((0, 0.9), (1, 0.8), (2, 0.6), (3, 0.7)),
            horizon,
            horizon,
            "adaptive",
            None,
            1,
            0.01,
        )

    monkeypatch.setattr(runner, "evaluate_myopic_policy", myopic)
    monkeypatch.setattr(runner, "plan_batched", planned)
    report = runner.execute(tmp_path, tmp_path)
    assert report["status"] == "conditional_opportunity_pass"
    assert len(report["rows"]) == 8
    assert calls == [(32, 3), (64, 3)] * 4
    np.testing.assert_allclose(
        report["aggregate_full_budget_values"]["64"], [1, 0.8, 0.6]
    )


def test_failed_predecessor_stops_before_source_load(monkeypatch, tmp_path):
    monkeypatch.setattr(runner, "preflight", lambda *args: {"status": "failed"})
    monkeypatch.setattr(
        runner, "load_source", lambda *args: pytest.fail("source loaded")
    )
    with pytest.raises(ValueError, match="predecessor"):
        runner.execute(tmp_path, tmp_path)
