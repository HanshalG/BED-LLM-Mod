"""Fixed probability-space subdivisions; tails retain all their probability mass."""
import numpy as np
from numpy.polynomial.legendre import leggauss

from environments.chembench_mopen.quantile_belief import QuantileGaussianModel


class TailQuantileGaussianModel(QuantileGaussianModel):
    def __init__(self, *args, branch_count=16, **kwargs):
        if isinstance(branch_count, bool) or branch_count not in (16, 32):
            raise ValueError('tail rule requires 16 or 32 total branches')
        super().__init__(*args, branch_count=branch_count, **kwargs)
        boundaries = np.array([0., 1e-5, 1e-3, .05, .5, .95, .999, .99999, 1.])
        nodes, weights = leggauss(branch_count // 8)
        widths = np.diff(boundaries)
        self._quantiles = (boundaries[:-1, None] + widths[:, None]*(nodes+1)/2).ravel()
        self._quadrature_weights = (widths[:, None]*weights/2).ravel()
        if (not np.all(np.diff(self._quantiles) > 0) or self._quantiles[0] <= 0
                or self._quantiles[-1] >= 1 or np.any(self._quadrature_weights <= 0)
                or abs(self._quadrature_weights.sum()-1) > 1e-14):
            raise ValueError('invalid composite probability rule')
        self._quantiles.setflags(write=False)
        self._quadrature_weights.setflags(write=False)
