import numpy as np
import pytest
from scipy.special import gammaln
from scipy.stats import t

from environments.scilaws.regression_belief import RegressionBelief


def prior():
    return RegressionBelief([0.0, 0.0], [[2.0, 0.2], [0.2, 1.0]], 3.0, 2.0)


def test_predictive_matches_independent_student_t():
    b = prior()
    x = np.array([1.0, 2.0])
    df, location, scale2 = b.predictive(x)
    assert df == 6
    for y in (-12.0, -0.1, 0.0, 4.0, 30.0):
        assert b.log_predictive(x, y) == pytest.approx(
            t.logpdf(y, df, loc=location, scale=np.sqrt(scale2))
        )


def test_batch_posterior_evidence_and_order():
    b = prior()
    x = np.array([[1.0, -1.0], [1.0, 0.5], [1.0, 2.0], [1.0, 0.3]])
    y = np.array([-0.3, 1.1, 2.3, 0.6])
    state, logz = b, 0.0
    for row, value in zip(x, y):
        state, increment = state.condition(row, value)
        logz += increment
    precision = np.asarray(b.precision) + x.T @ x
    mean = np.linalg.solve(precision, np.asarray(b.precision) @ b.mean + x.T @ y)
    shape = b.shape + len(y) / 2
    scale = b.scale + 0.5 * (
        y @ y
        + np.asarray(b.mean) @ np.asarray(b.precision) @ b.mean
        - mean @ precision @ mean
    )
    expected = (
        gammaln(shape)
        - gammaln(b.shape)
        + b.shape * np.log(b.scale)
        - shape * np.log(scale)
        + 0.5 * (np.linalg.slogdet(b.precision)[1] - np.linalg.slogdet(precision)[1])
        - len(y) / 2 * np.log(2 * np.pi)
    )
    np.testing.assert_allclose(state.mean, mean)
    np.testing.assert_allclose(state.precision, precision)
    assert state.scale == pytest.approx(scale)
    assert logz == pytest.approx(expected)
    reverse = b
    for row, value in zip(x[::-1], y[::-1]):
        reverse, _ = reverse.condition(row, value)
    np.testing.assert_allclose(reverse.mean, state.mean)
    assert reverse.scale == pytest.approx(state.scale)
    assert b == prior()
    hash(state)


def test_noise_and_parameter_uncertainty_are_distinct():
    b = prior()
    x = np.array([[1.0, 0.0], [1.0, 2.0]])
    _, latent_variance = b.target_moments(x)
    for row, v in zip(x, latent_variance):
        df, _, scale2 = b.predictive(row)
        assert scale2 * df / (df - 2) == pytest.approx(v + b.noise_variance)
    updated, _ = b.condition([1.0, 0.0], 20.0)
    assert updated.noise_variance > b.noise_variance


@pytest.mark.parametrize(
    "precision,shape,scale",
    [([[0.0]], 3, 2), ([[-1.0]], 3, 2), ([[1.0]], 1, 2), ([[1.0]], 3, -1)],
)
def test_invalid_priors_fail(precision, shape, scale):
    with pytest.raises(ValueError):
        RegressionBelief([0.0], precision, shape, scale)


def test_invalid_observations_features_fail():
    for y in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError):
            prior().condition([1.0, 0.0], y)
    with pytest.raises(ValueError):
        prior().predictive([1.0])
