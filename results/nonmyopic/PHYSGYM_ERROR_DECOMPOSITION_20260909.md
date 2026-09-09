# Banked PhysGym error decomposition

Previous turn completed the paid16-call study. This zero-call audit replays it first,
then diagnoses every saved candidate on the already opened observations and targets.
No new proposals/outcomes, modified gates, or changed frozen predictions.

## Missing models versus weighting

Hard task457's initial semantic and final refreshed arms each have exactly one
valid formula, with target MSE1.127190. Reweighting that support cannot improve it.
The two valid formulas in the redraw union give best-single MSE.640294, already
attained by the saved likelihood-weighted predictor. No claim about a convex-mixture
optimum is made for multi-formula pools. Blind457 has some weighting error: saved
MSE.216062 versus hindsight best-single.130771, while the best-fitting initial
training formula has target MSE.463196. Neither hindsight selection nor additional
weight fitting can be presented as a fresh predictive result.

All tasks/arms and every formula are recorded in
PHYSGYM_PROPOSAL_ERROR_AUDIT_20260909.json, including invalid target indices. Scores
on the valid subset of an invalid formula are explicitly not comparable to the
full target endpoint. In particular, dropping the four N=3 cases is not permitted.

## Numerical scale versus structural error

A separate retrospective diagnostic fits one additive log-output offset to the
four training observations only, equivalent to a positive multiplicative scale.
It uses no target labels for fitting and does not recompute the original gate.

| Hard-case formula | Raw training MSE (4 points) | Scale-fit training MSE | Scale-fit full target MSE |
|---|---:|---:|---:|
| Initial semantic | .918678 | .202742 | .318253 |
| Refreshed | .523727 | .017949 | Invalid on4inputs |
| Redraw | .544017 | .510976 | .655682 |

Noise variance is.0025. The initial semantic formula is inconsistent with observed
data at the assumed noise scale, before examining test outcomes. Scale fitting
helps, but substantial structural error remains. The refreshed expression is a
better training fit after scaling, yet cannot yield valid predictions over the
declared input domain. Thus a missing domain check is not the only problem, and
numeric constants cannot be left entirely to the LLM while treating its output as
an adequate world model. These results do not establish how an unseen repair prompt
would behave or whether any query sequence has a non-myopic advantage.

## Next supported change

A genuinely new proposer interface should receive numerical residuals and
domain-validity diagnostics for its initial proposals, with scalar calibration
owned by numerical code. It should generate globally valid positive response
structures rather than interpolate a handful of observed values without domain
constraints. The matched control must receive the same validation machinery and
compute allowance but no additional observation during proposal generation; both
final numerical predictors still use all real observations.

Before another paid block, implement/test that public-only feedback contract and
freeze a new development protocol. Do not rerun the old interface on new seeds or
drop the hard task. Retain the easy tasks to expose saturation, and require actual
new-observation benefit beyond feedback alone. Even a successful repair gate would
open only joint transition calibration, not a monotonic-depth claim. Under the
current evidence, a depth sweep would optimize a misspecified proposal process.

Two artifacts bank raw candidate and scale diagnostics. Three focused tests pass
(.15s), covering invalid-output retention, squared loss, array validation and
training-only scale fitting; lint passes. Separate original-run replay remains
valid. No paid calls. Authenticated posted usage has now caught up:
credits245,usage221.223686739,balance23.776313261; conservative daily spend.80540826
and remaining4.19459174 include the retained older.04 uncertainty. No cluster,
automation or historical-runtime action. Goal remains active/unachieved; this turn
adds measured evidence that distinguishes support, scale and weighting errors.
