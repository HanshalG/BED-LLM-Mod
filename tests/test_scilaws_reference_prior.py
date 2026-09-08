import json

import numpy as np
import pytest

from environments.scilaws.reference_prior import (
    FAMILIES,
    ResponseScale,
    features,
    initialize,
    make_model,
)


def designs():
    return json.load(
        open("results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json")
    )["tasks"]


def test_full_geometry_prior_builds_without_outcomes():
    for d in designs():
        m = make_model(d, quadrature_order=4)
        assert m.num_actions == 8
        assert len(m.components) == 4
        assert m.include_observation_noise
        assert np.isfinite(m.risk(m.initial_state))
        dim = len(d["axes"])
        assert [len(c.mean) for c in m.components] == [
            1,
            1 + dim,
            1 + dim + dim * (dim + 1) // 2,
            1 + 2 * dim,
        ]


def test_scale_and_initial_replay_are_shared_and_unit_invariant():
    d = designs()[0]
    y = np.arange(1, 2 * len(d["initial_points"]) + 1).reshape(-1, 2)
    a, state, scale = initialize(d, y, quadrature_order=4)
    b, other, other_scale = initialize(d, y * 100, quadrature_order=4)
    np.testing.assert_allclose(a.forecast(state), b.forecast(other), atol=1e-12)
    assert scale.value * 100 == other_scale.value
    assert state.components[0].shape == 3 + y.size / 2
    with pytest.raises(ValueError):
        initialize(d, y[:-1], quadrature_order=4)


def test_asinh_transform_stable_zero_negative_and_extreme():
    y = np.array([-100.0, -1.0, 0.0, 2.0, 1000.0])
    np.testing.assert_allclose(
        ResponseScale(2.0).transform(y), np.arcsinh(y / 2), atol=1e-15
    )
    assert np.isfinite(ResponseScale(1e-300).transform([1e300])).all()
    assert ResponseScale.from_initial([0.0, 0.0]).value == 1
    with pytest.raises(ValueError):
        ResponseScale.from_initial([float("nan")])


def test_feature_values_and_no_arbitrary_expressions():
    axes = [dict(name="x", bounds=[1.0, 100.0], transform="log")]
    p = [dict(x=10.0)]
    np.testing.assert_allclose(
        features(p, axes, "quadratic"), [[1.0, 0.0, 0.0]], atol=1e-15
    )
    with pytest.raises(ValueError):
        features(p, axes, "__import__")
    with pytest.raises(ValueError):
        features([dict(x=101)], axes, FAMILIES[0])
