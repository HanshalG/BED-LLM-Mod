import json

import numpy as np
import pytest

from environments.scilaws.initialized_reference import initialize_corrected
from environments.scilaws.reference_prior import initialize


DESIGNS = json.load(open('results/nonmyopic/SCILAWS_MEASUREMENT_DESIGN_20260908.json'))['tasks']


@pytest.mark.parametrize('design', DESIGNS, ids=lambda d: d['task_id'])
def test_full_geometry_posterior_is_not_reset(design):
    shape = (len(design['initial_points']), design['initial_replicates'])
    observations = np.linspace(-1, 1, np.prod(shape)).reshape(shape)
    base, posterior, scale = initialize(design, observations, quadrature_order=4)
    model, state, actual_scale = initialize_corrected(design, observations, quadrature_order=4)
    assert state == model.initial_state
    assert state.components == posterior.components
    np.testing.assert_allclose(state.log_weights, posterior.log_weights, atol=1e-14)
    assert scale == actual_scale
    assert state.components[0].shape == 3+observations.size/2
    assert model.num_actions == 8
    assert len(model.target_weights) == 64
    np.testing.assert_allclose(model.forecast(state), base.forecast(posterior), atol=1e-12)
    assert model.risk(state) == pytest.approx(base.risk(posterior), abs=1e-12)
    child = model.condition(state, 0, .2)
    other = base.condition(posterior, 0, .2)
    assert child.components == other.components
    np.testing.assert_allclose(child.log_weights, other.log_weights, atol=1e-14)
