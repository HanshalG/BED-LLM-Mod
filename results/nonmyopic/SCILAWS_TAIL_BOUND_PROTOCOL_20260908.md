# Tail envelope before value-function interpolation

Analytically integrate the conditional target second moment over the omitted
observation tails, using unnormalized Student-t moments0/1/2. Predicting zero
is a feasible terminal decision; hence this quantity bounds the contribution
of optimal future Bayes risk over those tails for the fixed continuous working
model. It does not bound arbitrary approximate model updates or source truth.

No probability renormalization or unreported tail dropping. If a future estimator
integrates only the interior, the omitted contribution is an interval[0,B], not
zero. Interior interpolation error must be controlled separately.

Before interpolator responses, evaluate analytic bounds on all four existing
synthetic histories and both actions at radii4/8/16/32/64 predictive standard
deviations about the predictive mean. Record the first radius with B<=1e-6,
or null if absent. This is a fixed numerical tail allowance within the earlier
1e-4 total error scale, not a change to that acceptance threshold. No new source
measurements, modelcalls or source-execution authorization. Tests independently
integrate Student-t moments and full conditioned target second moments.
