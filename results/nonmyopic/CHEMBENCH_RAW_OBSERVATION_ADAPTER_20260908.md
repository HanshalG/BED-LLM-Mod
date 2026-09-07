# Raw-observation reference adapter and integration diagnostic

## Scope

Added `GaussianParticleModel`, an isolated numerical provider for the existing
ordinary-horizon planner. Measurements are scalar Gaussian values, not categorical
bins. Fixed particle predictions and a supplied prior are its only inputs; it
never receives the realized world index. Fixed target-weighted squared prediction
error is shared by planning and forecasting. Log-weight updates preserve tiny
nonzero support so later evidence can reverse a confident belief.

This is not yet a chemistry observation specification: Gaussian raw measurements
are not silently interchangeable with multiplicative or log-normal rate noise.
The chemistry transformation, noise law, particle prior and panel remain to be
frozen before source response generation.

## Numerical Result

The independently specified eight-case synthetic audit compares order-nine
componentwise Gauss-Hermite prediction against adaptive density integration.
It covers separations 0.2, 1, 3 and 8, equal/unequal noise, and a 0.3/0.7 prior.
Absolute risk error and selected-action regret caps were both 0.001; reference
integration error cap was 1e-8. Constants were set before executing this audit.

**Result: one_step_reference_failed.** Two of eight cases exceed the absolute
error cap. Maximum error is 0.0253258354, in the unequal-noise separation-0.2 case
(reference risk 0.1082240549, approximation 0.1335498903). Selected-action regret
is zero on all eight cases. The candidate gaps here are wide: this does not
establish close-decision accuracy or multi-step fidelity. The failure is an
integration-rule limitation, not an LLM failure or evidence against planning.

Saved immutable result:
`chembench_raw_integration/20260908-v1/RESULT.json`
SHA256 `3a3352b8087fb3d95be99fa03eda625e7684b7979df95a77549d7e87120dab23`.
The result binds exact adapter and audit source hashes.

## Verification And Limits

Focused raw-adapter plus horizon tests: 58/58. Including adjacent planner, IR,
source, continuous, empirical and costed-repeat regressions: 105/105 in 140.98s.
Scoped lint and whitespace checks pass.

Tests check Bayes updates, history-order invariance, recovery after floating-point
probability underflow, zero prior support, mixture moments, duplicate nodes,
independent one-step density integration, invalid inputs and planner limits.
The test environment's lightweight torch stub conflicts with SciPy array-API
dispatch; production uses NumPy's stable logaddexp reduction and the independent
test uses the closed-form Gaussian density with SciPy quadrature.

Each action expands at most particles times quadrature order observations.
The existing planner's time/node/depth limits remain active. This reference
adapter is not a claim of scalable deep planning. Increasing order blindly would
compound the tree-size problem encountered in the old project.

## Next Dependency

Do not deploy order nine on chemistry, relax this saved diagnostic, or infer a
multi-step pass from the successful unit tests. Develop a separate, bounded
integration method with an explicit approximation-error check, then test it on
close action gaps and two-/three-step reference problems. Only then freeze the
eight-world chemistry engineering pilot and its actual noise/target semantics.
The synthetic diagnostic is not that pilot and has no chemistry efficacy status.

No model calls, paid spend, chemistry outcomes, old endpoint reruns, cluster work,
cleanup or automation changes occurred. The overall research goal is unfinished.
