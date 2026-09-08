import numpy as np
import pytest

from environments.chembench_mopen.centered_risk import CenteredTargetRisk
from environments.chembench_mopen.quantile_belief import QuantileGaussianModel
from scripts.scilaws_particle_kernel_profile import density_risk


@pytest.mark.parametrize('joint', [[-2., -3.], [-1000., -1001.], [0., -10000.]])
def test_max_shift_equivalence(joint):
    m = QuantileGaussianModel([[0.], [1.]], 1., [[1e10], [1e10+2]], [.3, .7],
                             target_conditional_variances=[[.2], [.8]])
    risk = CenteredTargetRisk(m)
    joint = np.asarray(joint)
    assert density_risk(joint, risk, shifted=True) == pytest.approx(density_risk(joint, risk), abs=1e-12)
