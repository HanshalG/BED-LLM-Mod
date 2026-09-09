# Public residual/domain feedback mechanics

The last turn decomposed the paid failure. This turn implements a separate
prospective interface, leaving the closed experiment and all its forecasts intact.

## Numerical scale

For each proposed positive response shape f, model observed log response as
log(f(x))+b+epsilon, b~Normal(0,4), epsilon~Normal(0,.05²). Integrate b exactly with
the Gaussian rank-one covariance, rather than point-fitting an offset then treating
it as certain. Candidate evidence includes the normalization determinant. The new
predictor uses those evidences over deduplicated AST candidates and retains the
posterior variance of b. A dense covariance calculation independently matches the
analytic evidence to1e-9 in a focused test. This is exact for the declared scalar
model, not a calibrated full posterior over physical mechanisms or source functions.
The fixed scale prior is syntax/parameterization dependent; no invariance claim.

## Public-only feedback

The interface accepts only proposed expressions, named inputs, public context,
descriptions and supplied observations. It computes calibrated residuals and checks
positivity/finite execution on public-domain corners, center and every integer N
at the center. Duplicate points are removed; N remains integer. Feedback includes
the first four failing inputs, anonymously remapped consistently with prompt names.
The guard grid is an inexpensive check, not proof of global validity. No true
equation, hidden state, evaluation output, benchmark ID or endpoint callback exists
in this API. Invalid candidates cannot regain support through scale calibration.

Running this contract on the saved four-task proposals reads public.json and
forecasts.json only, never outcomes.json. It finds:

- Hard457 original semantic: maximum absolute residual12.50 observation-noise units
  even after Bayesian scale adjustment, without any domain-grid failure.
- Hard457 refreshed:9invalid guard points, maximum residual4.30noise units on real
  history. The public domain check exposes its extrapolation error without labels.
- Semantic103/458/653: maximum residual1.40/.70/1.47noise units, no guard failures.

These residuals are descriptive, not posterior-predictive p-values: they condition
on the same fitted history and are not independent standardized tests. No efficacy
score or gate is recomputed. PHYSICS_PUBLIC_FEEDBACK_AUDIT_20260909.json banks all
candidate diagnostics and implementation hash. Existing paid-run files unchanged.

## Next experiment boundary

Before calls, freeze a new paired repair protocol using identical initial proposals
and identical public-domain checking in both arms. Only one proposer gets a newly
observed response; both numerical final predictors use all real observations.
Separate feedback benefit from extra-observation benefit. Require a new-data effect,
not simply easier positivity or more candidate computation.

Do not blindly rerun the same four-task test: three tasks were near saturation, so
the previous two-task-win criterion had little remaining room. That does not permit
relaxing the old gate. A new prospective cohort or a narrowly labelled hard-case
development repair diagnostic must be declared honestly, with no positive BED or
monotonic-depth claim. Ordinary sequential planning still requires joint-transition
calibration and demonstrable headroom. No paid runner or call authorized here.

Twenty-two feedback/proposal/interpreter tests pass(.28s), including dense-normal
evidence, unsafe-expression rejection, history-only dependency, integer-domain
failure detection, duplicate handling, scale uncertainty and empty support. Lint
passes. Account245/221.223686739/23.776313261 unchanged, modelcalls0/cost0; daily
remaining4.19459174 includes old.04uncertainty. Automation/cluster/protected runtime
untouched. Previous turnprogress, currenttestedmechanics+newpublicdiagnostic evidence;
full goal remains active/unachieved.
