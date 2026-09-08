# Mixed-family refinement: lower-order deployment not qualified

Protocol and runner frozen at pushed78f50292 before the single audit. All20
plans complete within unchanged five-second/100000-node caps. Three focused
gate/fixture tests pass in.86s; scoped E4/E7/E9/F lint passes. No production
planner changes in this step.

Artifact SCILAWS_MIXED_REFINEMENT_AUDIT_20260908.json SHA256:
087ccfafedd41c05aa1887b5f7b5636f4946b06319d20096e82b7d74b3d83058.

| History | 32-vs-64 root delta | Reference valid | Order4 error | Order8 error | Order16 error |
| --- | ---: | --- | ---: | ---: | ---: |
| Empty | 1.279e-5 | No | .00492847 | .00040757 | .00015062 |
| +.7 at action0 | 7.531e-6 | Yes | .00052356 | .00013924 | .00002419 |
| -.7 at action0 | 2.366e-6 | Yes | .00031297 | .00002322 | .00003018 |
| +.7 then -.7 | 3.367e-5 | No | .00182180 | .00019219 | .00003309 |

Errors are maximum across both exact numerical root values, relative to order64.
Rows with an invalid reference are diagnostic discrepancies, not certified true
errors. Required reference agreement is1e-5, candidate error/regret1e-4.
Order4 passes0/4, order8 passes1/4, order16 passes2/4. No order qualifies across
the full panel. Empty-history order4 also selects a different root with .00180411
regret relative to order64, but the reference there itself remains unconverged.

## Interpretation

This disproves the practical shortcut proposed for investigation: we cannot use
four integration nodes per component merely to make depth3 tractable. Even at
depth2 with two actions, mixed-family residual integration is not generally
resolved by the analytic within-family/horizon correction. The exact single-
family tests and the valid search-pruning induction do not establish mixed-
family integration accuracy. Eight public zero-mean priors were too weak as an
accuracy stress check; full numerical root values now expose the gap.

The positive-history case directly fails the current order8 threshold against a
reference that meets the frozen convergence test. Negative history passes it.
Do not select histories, change likelihood/noise or relax tolerance after seeing
this distinction. Order16's two passes do not license source execution either.

## Next dependency

An independent mixed-family integration reference is needed before further
solver acceleration or an integration-rule replacement. Inspect the residual
integrand (between-family uncertainty and the continuation action minimum), and
test a deterministic adaptive integration reference with an explicit evaluation
limit and reported integration error on these already-opened synthetic cases.
This is numerical debugging, not new source evidence. Preserve all outcomes,
actions and prior parameters; leave any new scientific cases sealed. No further
unchanged full-public runtime reruns, paid proposer calls, threshold rescues or
automatic lower-order deployment are authorized by this result.

This is still a classical numerical instrument, not the intended LLM-native
research result. Useful proposals, source calibration, paired receding-horizon
endpoints, compute-matched controls and anticipated discovery remain unproven.
No source observations or model calls, $0 spend. Authenticated account remains
245/220.376693994/24.623306006, Sept8 ledger spend0. Process exited; automation
paused and full goal unfinished.
