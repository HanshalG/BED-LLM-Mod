# Deep control-variate check: order8 still unqualified

Previous turn passed one-step fixtures. This turn freezes and runs a deeper
diagnostic at pushed6608baf0, without model changes or source observations.
Every plan retains the five-second/100000-node cap.

Artifact SHA256:
5deb9192ceb135e9328792f5ff8ded949328a24c962e97119b7b4b49a7a22f3b.

## Analytic depth fixture

One uncertain coefficient, precision1, InverseGamma(3,.2), action features1 and2,
target features1 and3 weighted .25/.75, future-observation squared risk. The
optimal action is always2 (index1), and the exact depth-d risk is
.1*(1+7/(1+4*d)). This is a known-family, non-adaptive control, not an example
of a non-myopic policy advantage. All depths below optimize the full two-action
menu with repeats; no synthetic observation is a real source outcome.

| Order | h1 absolute error | h2 absolute error | h3 absolute error |
|---|---:|---:|---:|
| 4 | 2.78e-17 | .000888889 | .001417989 |
| 8 | 0 | .000125701 | .000192605 |
| 16 | 0 | .000012843 | .000018958 |

All plans completed and chose the analytic action. Frozen error tolerance is1e-4,
so order4 and order8 FAIL at depths2/3. Order16 passes this particular fixture.
Do not loosen the tolerance because the order8 miss is small. Equality of selected
actions cannot replace the requested value-accuracy criterion.

## Full public reference h2

All8 public-prior h2 plans with the correction completed in1.432-2.294s and satisfy
the necessary family-bound check (no estimate below it by more than1e-10).
For baseball, corrected risk .166209789685 exceeds bound .165759530152;
the earlier uncorrected .165661703681 estimate lay below it. Other dimensions
also pass that necessary check. This removes the observed contradiction, but
does not prove unbiased values or source calibration. Same-dimensional generic
priors are still identical and are not independent scientific evidence.

## Why one-step exactness did not extend

The implemented control variate integrates within-family CURRENT target variance
exactly. At a deeper node, the remaining risk is a different multiple of posterior
noise variance. Q[V-U] therefore still contains a quadratic-in-observation term,
even for a single family; it need not integrate exactly. The observed refinement
is consistent with that limitation, not a failed implementation equivalence test.

A next candidate is a horizon-dependent potential: the family-revelation optimal
risk for the remaining depth. Given the current action, each family's posterior
precision is independent of the observed value. Its optimal remaining design
coefficient is likewise fixed, so the expectation of that potential can still be
computed analytically using the noise-variance martingale. For a single family
this potential equals the exact continuation value. For a mixture, the residual
is the excess value caused by family uncertainty and still needs numerical checks.

Implement that candidate only as an explicit alternative with matching signed
tree corrections, scalar checks and unchanged analytic/error thresholds. Do not
claim it is qualified merely from this derivation. Its extra design calculations
need bounded caching and profiling. H3 node growth, independent mixture refinement,
source calibration, full paired protocol and useful LLM proposals remain unsolved.
No scientific/pruning permission, cap increase or lower-order result rescue here.

The audit script passed lint and completed all9 analytic plans and8 public h2
plans. The underlying implementation's177-test verification remains the preceding
turn's evidence; no new code correctness claim is inferred from the failed
accuracy cases. No paid calls, measurements or active processes; account/ledger
unchanged, automation paused, full project goal unfinished.
