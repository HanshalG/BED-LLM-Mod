import sys

import numpy as np

from scripts.parameter_smc_reference_audit import fixture, multimodal_reference


def test_four_mode_reference_symmetry_and_convergence():
    a, b = multimodal_reference(1024), multimodal_reference(2048)
    for key in ('mean', 'variance', 'log_evidence'):
        np.testing.assert_allclose(a[key], b[key], rtol=0, atol=1e-9)
    assert a['mode_mass'] == [.25]*4
    assert a['mean'][:3] == [0.]*3
    assert all(v > 0 for v in a['variance'])


def test_fixtures_shapes_and_prior_coordinates(monkeypatch):
    monkeypatch.delitem(sys.modules, 'torch', raising=False)
    for name, d in (('one_parameter', 1), ('four_modes', 2)):
        prior, likelihood, predict, _ = fixture(name)
        points = prior.sample(np.random.default_rng(3), 10)
        assert points.shape == (10, d)
        assert likelihood(points).shape == (10,)
        assert predict(points).shape == (10, 4)
        assert np.isfinite(likelihood(points)).all()
