import json

import numpy as np
import pytest

from environments.scilaws.regression_belief import _feature_solve
from environments.scilaws.reference_prior import make_model


def test_gram_risk_matches_full_target_moments_all_development_dimensions():
    tasks = json.load(
        open("results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json")
    )["tasks"]
    for task in tasks:
        m = make_model(task, quadrature_order=4)
        state = m.initial_state
        for action, y in ((0, -0.7), (7, 3.0), (0, 0.1), (3, -2.0)):
            assert m.risk(state) == pytest.approx(
                float(m.moments(state)[1] @ m.target_weights), rel=1e-12, abs=1e-12
            )
            state = m.condition(state, action, y)
        assert m.risk(state) == pytest.approx(
            float(m.moments(state)[1] @ m.target_weights), rel=1e-12, abs=1e-12
        )
        assert m._target_leverage.cache_info().maxsize == 256


def test_cached_solve_matches_direct_solve_and_is_bounded_readonly():
    _feature_solve.cache_clear()
    p = ((2.0, 0.5), (0.5, 3.0))
    x = (1.0, -2.0)
    actual = _feature_solve(p, x)
    np.testing.assert_array_equal(actual, np.linalg.solve(p, x))
    assert _feature_solve(p, x) is actual
    assert _feature_solve.cache_info().hits == 1
    assert _feature_solve.cache_info().maxsize == 1024
    with pytest.raises(ValueError):
        actual[0] = 0
