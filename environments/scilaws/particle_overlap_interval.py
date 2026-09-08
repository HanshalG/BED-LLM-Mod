"""Streamed Gaussian-overlap lower bound for finite-particle one-step risk."""
from time import monotonic
import math

import numpy as np

from environments.chembench_mopen.horizon import SearchLimitExceeded
from .particle_risk_interval import ParticleRiskIntervals


class ParticleOverlapIntervals(ParticleRiskIntervals):
    def actions(self, state, *, max_seconds=5., max_workspace_bytes=64*1024*1024):
        started = monotonic()
        if not math.isfinite(max_seconds) or max_seconds <= 0:
            raise ValueError('invalid overlap time budget')
        m = self.model
        p, t = m.targets.shape
        # Reserve direct target differences, squares, and all pairwise scalar work.
        fixed = self.risk.centered.nbytes+self.risk.second.nbytes
        row_bytes = 8*p*(4*t+24)
        batch = min(32, (max_workspace_bytes-fixed)//row_bytes)
        if batch < 1:
            raise SearchLimitExceeded('overlap workspace too small')
        rows = super().actions(state)
        weights = np.exp(m._logs(state))
        totals = np.zeros(m.num_actions)
        pair_terms = 0
        for start in range(0, p, batch):
            stop = min(p, start+batch)
            differences = self.risk.centered[start:stop, None, :]-self.risk.centered[None, :, :]
            distances = np.sum(differences**2, axis=2)
            del differences
            if not np.isfinite(distances).all():
                raise ValueError('unrepresentable pair distances')
            weighted_distances = distances * weights[start:stop, None] * weights[None, :]
            for a in range(m.num_actions):
                if monotonic()-started > max_seconds:
                    raise SearchLimitExceeded('overlap shared time budget exceeded')
                si, sj = m.sigmas[start:stop, a, None], m.sigmas[None, :, a]
                scale = np.maximum(si, sj)
                ri, rj = si/scale, sj/scale
                denominator = ri**2+rj**2
                with np.errstate(over='ignore'):
                    delta = (m.means[start:stop, a, None]-m.means[None, :, a])/scale
                    overlap = (2*ri*rj/denominator)*np.exp(-delta**2/(2*denominator))
                if not np.isfinite(overlap).all():
                    raise ValueError('invalid Gaussian overlap')
                totals[a] += .5*float(np.sum(weighted_distances*overlap))
                pair_terms += (stop-start)*p
        for a, row in enumerate(rows):
            lower = row['noise_floor']+totals[a]-row['float_padding']
            if lower > row['upper']+row['float_padding']:
                raise ValueError('overlap lower bound exceeds linear upper bound')
            row['lower'] = max(row['lower'], float(lower))
        if monotonic()-started > max_seconds:
            raise SearchLimitExceeded('overlap shared time budget exceeded')
        return dict(intervals=rows, seconds=monotonic()-started, pair_terms=pair_terms,
                    block_rows=batch, workspace_allowance_bytes=int(fixed+batch*row_bytes))
