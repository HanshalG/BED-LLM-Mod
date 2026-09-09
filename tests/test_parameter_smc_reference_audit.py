import sys
import hashlib
import json
from pathlib import Path

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


def test_banked_complete_coverage_and_independent_decisions():
    from scripts.parameter_smc_reference_audit import OUTPUT, PROTOCOL
    data = json.loads(OUTPUT.read_text())
    assert data['status'] == 'complete'
    assert data['model_calls'] == data['cost_usd'] == 0
    assert not data['paid_authorized']
    assert data['protocol_sha256'] == hashlib.sha256(PROTOCOL.read_bytes()).hexdigest()
    assert data['smc_sha256'] == hashlib.sha256(Path('environments/chembench_mopen/smc.py').read_bytes()).hexdigest()
    assert len(data['rows']) == 32
    assert {(r['fixture'], r['particles'], r['seed']) for r in data['rows']} == {
        (f, n, seed) for f in ('one_parameter', 'four_modes')
        for n in (512, 2048) for seed in range(8)}
    for row in data['rows']:
        ref = data['references'][row['fixture']]
        error = np.abs(np.array(row['mean']) - ref['mean'])
        relative_variance_error = np.max(np.abs(np.array(row['variance']) / ref['variance'] - 1))
        np.testing.assert_allclose(row['max_mean_error'], error.max())
        np.testing.assert_allclose(row['max_relative_variance_error'], relative_variance_error)
        passed = row['absolute_log_evidence_error'] <= .1 and relative_variance_error <= .2
        if row['fixture'] == 'one_parameter':
            passed = passed and error.max() <= .01
        else:
            masses = np.array(row['mode_masses'])
            np.testing.assert_allclose(masses.sum(), 1)
            assert np.all(masses > 0)
            passed = passed and np.max(error / np.sqrt(ref['variance'])) <= .15
            passed = passed and np.max(np.abs(masses - .25)) <= .1
        assert bool(passed) == row['numerical_qualified']
        assert 0 < row['likelihood_rows'] <= 600000
        assert 0 < row['seconds'] <= 30
