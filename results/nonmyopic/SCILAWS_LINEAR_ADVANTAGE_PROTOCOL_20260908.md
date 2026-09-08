# Linear-advantage representation check

Freeze before workload results. Squared-error orthogonality gives risk(linear)-
risk(Bayes)=E[(linear predictor-conditional mean)^2], with fixed target weights.
Integrate this nonnegative residual over the same full predictive real line, then
subtract from analytic best-linear risk. No likelihood, quadrature tolerance,
posterior or action change. Default reference unchanged, explicit subclass only.

First frozen task, first action's first order4 branch, zero/affine/quadratic initial
tables, all8 terminal actions. Both formulations share5second/100000evaluation caps
within each scenario/method, tolerance1e-8, one BLAS thread. Require max difference
<=1e-7 and all reported errors<=1e-7. Report both evaluation counts regardless of
direction. This is a workload diagnostic, not complete-case or full-plan accuracy.

Unit tests independently compare all four original two-family histories and the
single-family analytic case. No source/model calls or deployment authorization.
