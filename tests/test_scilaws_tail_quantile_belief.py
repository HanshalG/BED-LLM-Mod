import numpy as np
import pytest

from environments.chembench_mopen.batch_horizon import posterior_branches_many
from environments.scilaws.tail_quantile_belief import TailQuantileGaussianModel


@pytest.mark.parametrize('count', [16, 32])
def test_probability_mass_polynomials_and_full_updates(count):
    m = TailQuantileGaussianModel([[-2.], [1.]], [[.3], [1.1]], [[0.], [1.]], [.2, .8],
                                  branch_count=count)
    for degree in range(2*(count//8)):
        assert m._quadrature_weights @ m._quantiles**degree == pytest.approx(1/(degree+1), abs=1e-14)
    ys, logs, masses = posterior_branches_many(m, np.asarray(m.initial_state)[None, :], 0, return_weights=True)
    assert ys.shape == (1, count)
    assert masses.sum() == pytest.approx(1.)
    for y, state in zip(ys[0], logs[0]):
        np.testing.assert_allclose(state, m.condition(m.initial_state, 0, y), atol=1e-12)
    scalar = m.branches(m.initial_state, 0)
    np.testing.assert_allclose([b.observation for b in scalar], ys[0], atol=1e-10)


@pytest.mark.parametrize('count', [True, 4, 24, 64])
def test_unsupported_count(count):
    with pytest.raises(ValueError):
        TailQuantileGaussianModel([[0.]], 1., [[0.]], [1.], branch_count=count)
