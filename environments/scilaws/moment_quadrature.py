"""Experimental positive quadrature conserving component predictive moments.

Only integration weights change; posterior densities and observation nodes do
not. Failure to satisfy constraints is terminal, never a fallback to old weights.
Moment conservation alone does not certify integration of nonlinear utilities.
"""

import numpy as np
from scipy.optimize import linprog
from scipy.sparse import bmat, csr_matrix, eye
from scipy.special import logsumexp
from scipy.stats import t

from .regression_mixture import RegressionMixture


def match_moments(nodes, masses, component_weights, parameters):
    y, q, w = (np.asarray(a, dtype=float) for a in (nodes, masses, component_weights))
    pars = np.asarray(parameters, dtype=float)
    if (
        y.ndim != 1
        or not 2 <= len(y) <= 2048
        or q.shape != y.shape
        or w.ndim != 1
        or not 1 <= len(w) <= 16
        or pars.shape != (len(w), 3)
        or any(not np.isfinite(a).all() for a in (y, q, w, pars))
        or np.any(q < 0)
        or np.any(w < 0)
        or abs(q.sum() - 1) > 1e-12
        or abs(w.sum() - 1) > 1e-12
        or np.any(pars[:, 0] <= 2)
        or np.any(pars[:, 2] <= 0)
    ):
        raise ValueError("invalid moment matching inputs")
    df, loc, scale2 = pars.T
    variance = scale2 * df / (df - 2)
    centre = w @ loc
    normalization = np.sqrt(w @ (variance + (loc - centre) ** 2))
    z = (y - centre) / normalization
    with np.errstate(divide="ignore"):
        log_joint = t.logpdf(y[:, None], df, loc=loc, scale=np.sqrt(scale2)) + np.log(w)
    responsibilities = np.exp(log_joint - logsumexp(log_joint, axis=1)[:, None])
    matrix = np.concatenate([responsibilities.T * z**power for power in (0, 1, 2)])
    target = np.concatenate(
        [
            w,
            w * (loc - centre) / normalization,
            w * (variance + (loc - centre) ** 2) / normalization**2,
        ]
    )
    if not np.isfinite(matrix).all() or not np.isfinite(target).all():
        raise ValueError("moment constraint overflow")
    n = len(y)
    identity = eye(n, format="csr")
    # Minimize L1 departure from the existing quadrature subject to exact moments.
    inequalities = bmat([[identity, -identity], [-identity, -identity]], format="csr")
    equalities = bmat(
        [[csr_matrix(matrix), csr_matrix((len(target), n))]], format="csr"
    )
    result = linprog(
        np.r_[np.zeros(n), np.ones(n)],
        A_ub=inequalities,
        b_ub=np.r_[q, -q],
        A_eq=equalities,
        b_eq=target,
        bounds=(0, None),
        method="highs",
        options=dict(
            time_limit=1.0,
            maxiter=10000,
            primal_feasibility_tolerance=1e-9,
            dual_feasibility_tolerance=1e-9,
        ),
    )
    if not result.success:
        raise ValueError(f"moment matching failed (solver status {result.status})")
    matched = result.x[:n]
    if not np.isfinite(matched).all() or np.any(matched < 0) or matched.sum() <= 0:
        raise ValueError("invalid matched weights")
    matched = matched / matched.sum()
    if np.max(np.abs(matrix @ matched - target)) > 5e-9:
        raise ValueError("moment matching residual exceeds tolerance")
    return matched


class MomentMatchedMixture(RegressionMixture):
    def _quadrature(self, state, action):
        rows = super()._quadrature(state, action)
        nodes, masses = np.asarray(rows).T
        weights = np.exp(self._state(state))
        parameters = [
            b.predictive(x[action])
            for b, x in zip(state.components, self.action_features, strict=True)
        ]
        matched = match_moments(nodes, masses, weights, parameters)
        return tuple(
            (float(y), float(p)) for y, p in zip(nodes, matched, strict=True) if p > 0
        )
