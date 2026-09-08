# Retrospective refresh direction versus size diagnostic

Before evaluating this decomposition: use all32 trees/all256 roots/all5744 cases
of the fixed parent first-refresh result f0704ed9. Retain its target/root/tree
weighting and fixed101-number target domain. No new response, target, policy or
model call. Parent result, bank, compiler and extractor hashes must match.

Let p0 be initial-filtered prediction, p1 the retained-refresh prediction,
d=p1-p0. For a GLOBAL descriptive interpolation t, p(t)=p0+t*d:

    MSE(p(t), y) - MSE(p0,y) = t*A + t^2*B
    A = mean[2*(p0-y)*d], B = mean[d^2] >= 0.

Compute A and B at each saved root and aggregate with the parent's exact rational
weights. Require A+B equals every parent's saved refreshed-minus-initial loss.
Report all trees, positive/nonnegative direction counts, and useful-direction but
overshot counts. Do not optimize/select t, choose a subset, fit on target labels,
turn this into a calibration method, or reopen old gates.

If aggregate A>=0, every positive global scalar in this direction is non-improving
on this bank; shrinking the common step alone cannot yield aggregate improvement.
If A<0 but A+B>0, the observed step overshoots an initially beneficial direction.
Neither case rules out history-conditioned weighting or genuinely better proposals;
neither identifies a deployment rule from held-out labels. This is a descriptive
diagnosis with no paid or efficacy authority. Old failures remain closed.
