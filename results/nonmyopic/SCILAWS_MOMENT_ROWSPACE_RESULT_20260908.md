# Row-space audit: stop tuning fixed order-eight weights

Previous turn implemented a positive moment-matching candidate. This turn changes
the linear system representation, not the original residual acceptance test.
An SVD whitens the represented independent equations; numerical rank uses machine
epsilon times the largest singular value times maximum matrix dimension. Target
projection outside that row space by more than1e-10 fails. Every returned solution
is still checked against ALL original constraints at the existing5e-9 tolerance.
The physical model, observation nodes and nonnegative-weight requirement remain.

The infeasible two-node unit fixture is now rejected earlier by row-space
consistency rather than the LP, so its expected error message was updated. The
audit ran after that message-only test failure and before correcting the test;
it opened no source outcomes. This result was not a prospectively pushed new
efficacy protocol. Existing numerical thresholds and case definitions were reused
unchanged, not selected after seeing the new result.

## Result

Artifact SCILAWS_MOMENT_AUDIT_WHITENED_20260908.json SHA256
18b9efff35ea2adfcae484cb4bf0ca54454cacc8c26e76aaeb6b08531a9f2e19.

- Order8: 0/16 complete rows; each stops at a solver status2 (infeasible) result.
- Order16: 16/16 rows pass the unchanged numerical thresholds.
- Combined audit remains false. No pruning or scientific execution authorization.

Status2 under this better-conditioned representation is stronger diagnostic
evidence than the earlier status4 failures, but not a formal exact-arithmetic
infeasibility certificate. Whitening also makes weak directions numerically
visible: the two former low-order passes should not be treated as independent
confirmations that exact moment matching was feasible. Higher-order passes do
not rescue low-order deployment or the original h3 node-count failure.

110 SciLaws tests pass in5.51s, scoped lint passes. New tests retain a weak but
independent constraint and reject an inconsistent redundant system. No source
measurements, LLM calls or paid costs; all processes exited.

## Next implementation, not another solver-parameter search

Stop tuning this fixed order8 positive-weight system. Under the conjugate mixture,
the last-step expected within-family target variance is known analytically:

    sum_k w_k E[sigma_k^2] *
        (trace(Lambda_{k,after_action}^{-1} X_k' W X_k) + noisy_indicator).

Only the expected between-family prediction variance requires integration there.
This suggests using the analytic within-family term as a control variate for
chance-node risk, instead of forcing all moments onto a small fixed node set.
For a deeper continuation V and within-family potential U, the identity is

    E[V(s_next)] = E[U(s_next)] + E[V(s_next)-U(s_next)].

The first term is analytic for a prescribed current action; the second remains
numerical. The identity holds regardless of subsequent adaptive decisions because
V is evaluated at the next belief. This is NOT a proof that finite quadrature of
the second term is accurate, nor does it remove the need for calibration and
deep refinement. It also changes numerical chance-node evaluation, so displayed
tree risks must explicitly include any correction rather than silently disagree
with sums of child risks. Retain a scalar independent path and compare full
action vectors and tree reconstruction before a new bounded check.

The node-count obstacle remains separate. No runtime-cap increase, uncharged
virtual leaves or source-support reduction is authorized by this calculation.
The original empirical-Bayes reference is still classical, and the LLM-native
proposal/refresh and anticipated-discovery goals remain unfinished. Account and
London ledger are unchanged; automation remains paused.
