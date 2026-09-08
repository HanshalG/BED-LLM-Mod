import numpy as np
import pytest

from environments.scilaws.value_surrogate import interpolation_check


def test_preserves_separate_actions_and_no_extrapolation():
    x = np.linspace(-2, 2, 65)
    checks = (x[:-1]+x[1:])/2
    training = np.stack((2+x, 2-x), axis=1)
    actual = np.stack((2+checks, 2-checks), axis=1)
    model, error = interpolation_check(x, training, checks, actual)
    assert error < 1e-12
    assert np.argmin(model(-1)) == 0
    assert np.argmin(model(1)) == 1
    assert np.isnan(model(3)).all()


def test_check_values_are_not_training_values():
    x = np.linspace(-1, 1, 65)
    checks = (x[:-1]+x[1:])/2
    _, error = interpolation_check(x, np.zeros((65, 2)), checks, np.ones((64, 2)))
    assert error == pytest.approx(1.0)


def test_failed_validation_returns_no_score(monkeypatch):
    from environments.scilaws.adaptive_reference import AdaptiveReference
    from environments.scilaws import value_surrogate
    from scripts.scilaws_mixed_refinement_audit import fixture
    m, state = fixture(8, ())
    ref = AdaptiveReference(m, predictive_coordinates=True)
    monkeypatch.setattr(ref, 'terminal', lambda *args: (1.0, 0.0))
    monkeypatch.setattr(value_surrogate, 'interpolation_check', lambda *args: (None, 1.0))
    result = value_surrogate.approximate_root(ref, state, 0)
    assert result['status'] == 'validation_failed'
    assert 'interior_estimate' not in result
