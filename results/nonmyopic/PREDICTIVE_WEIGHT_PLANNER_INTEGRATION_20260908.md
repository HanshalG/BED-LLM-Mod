# Predictive weights enter the planner

The previous turn made concrete budget-safety progress. This turn connects the
prequential pipeline-weight design to the existing ordinary-horizon solver.
No experiment endpoint was opened. Authenticated cumulative usage remains
220.376693994, balance $24.623306006; September 8 ledger validates at zero spend.

## Implemented contract

`environments/chembench_mopen/predictive_mixture.py` accepts named finite
predictive pools and explicit normalized group weights. Particle mass is group
weight times the pool's current-history particle probability. Enlarging one pool
does not increase its group's total mass. Equivalent duplicated particles with
split weights preserve predictions and planning values; arbitrary new particles
can legitimately change the within-group predictive distribution.

The adapter copies likelihoods, targets and masses into the existing
`FiniteBeliefModel`, used directly by `HorizonPlanner`. No second solver is added.
Structures remain fixed throughout a planning call. Imagined likelihood updates
condition the joint group/particle distribution without mutating a live
`PrequentialMixture`. After a real outcome, the original sealed forecasts earn
their one weight update; freshly generated pools inherit those earned group
weights, not extra credit for explaining the observation that generated them.

Shared axis shapes and target weights are checked. Semantic correspondence of
action, outcome and target IDs is still the caller's responsibility. A zero-mass
group has no conditional forecast and returns None; no reset or invented
probability is supplied. A fresh runner must explicitly handle eliminated groups.

## Verification and limits

60 combined horizon, prequential and adapter tests pass in 2.64 seconds; lint
passes. Tests cover a weight-dependent selected query, invariance under a 50-fold
particle duplication, imagined-versus-real weight separation, post-observation
refresh without double credit, unsupported observations, copied snapshots, and
independent persistent-world policy enumeration at h1/h2/h3.

These are constructed software tests, not an empirical proposal or horizon gate.
Pipeline weights are not a posterior over physical mechanisms. They provide a
declared predictive mixture; calibration must be established on fresh evidence.
The snapshot does not anticipate future structural discovery, and its simulation
can still disagree with an actual refreshed updater. The fresh controlled
proposer/predictive-weight study, its semantic gate, and paired policy endpoints
remain incomplete. No closed chemistry, Number Game or scene configuration is
reopened. Automation remains paused and no paid inference is authorized.
