# Continuous horizon reference: correct small cases, runtime gate not cleared

## Implementation

`raw_horizon.py` now optimizes all contingent decisions through the declared
horizon, or enumerates precommitted sequences for the open-loop control. Both
use the same raw likelihood, fixed targets and posterior-variance objective.
No repeats, full available action menu, deterministic tie ordering, maximum
depth three, bounded per-call caches, and a shared global evaluation/time budget.
Future real structural refresh is not modelled and no LLM is called.

Each recursion level receives tolerance/horizon for integration. Conditional on
those uniform numerical error estimates being valid, the minimum operator does
not amplify value error; the sum of level allowances bounds estimated root error.
The root result reports action values, unresolved selections and an estimated
selection-regret allowance. These are not rigorous certificates: QUADPACK error
estimates can miss features, especially in approximate nested integrands. Exact
uninformative measurements are skipped algebraically, not approximately pruned.

Unlike the finite solver's explicit tree, this continuous reference represents
the contingent policy by recursive optimization at each integration state; it
does not serialize an infinite tree. A future pilot must log its actually visited
histories, root values and representative counterfactual continuations.

## Tests And Bounded Benchmark

Focused raw-horizon/integration/belief/finite-horizon tests: 65/65 in10.97s.
An independent sufficient-statistic calculation verifies two informative Gaussian
measurements. Three-step adaptive/open-loop plateau tests verify one informative
plus two genuinely uninformative designs. Budget exhaustion and input validation
pass. These do not establish a positive continuous adaptivity gap.

The first two-step test hit the initial200,000 evaluation cap. A separately
declared development benchmark allows1,000,000 evaluations and30seconds per plan,
with tolerance1e-5 and two equally weighted latent particles. For h1/h2/h3 it
offers exactly h exchangeable informative Gaussian designs. This is an integration
complexity check, NOT an equal-budget efficacy or monotonicity experiment.

| Horizon | Status | Evaluations | Seconds | Expected risk |
| --- | --- | --- | --- | --- |
| 1 | completed | 462 | 0.011 | 0.22927603 |
| 2 | completed | 427812 | 9.732 | 0.21139193 |
| 3 | global resource cap | no completed plan | 22.810 | absent |

The result is `reference_not_qualified`, saved with source hashes and atomic
per-depth checkpoints in `chembench_raw_horizon/20260908-v1/`. No failed plan is
replaced with a shallower answer. No chemistry response matrix was opened.

## Implication For The Full Plan

Nested adaptive quadrature is a small validation reference, not a viable pilot
engine at this cap. Increasing the cap again is not the next experiment. The
next implementation must retain actual contingent h1/h2/h3 and the full menu,
but use a bounded approximation with independent decision-level refinement.
Keep the adaptive reference for selected low-dimensional checks. Do not deploy
the previously failed order-nine rule unchanged or call it qualified.

Completion audit of the approved three-part package:
- A: finite numerical reference complete; continuous integration individually
  tested, but practical informative h3 and positive continuous adaptivity remain
  unqualified. The new runtime failure changes the next engineering action.
- B: eight-world source panel still unopened; exact chemistry prior/noise/targets,
  six-arm end-to-end runtime and sealed output protocol remain incomplete.
- C: new proposal semantic gate and paired LLM pilot have not opened; productive
  compute controls, held-out proposal benefit and powered confirmation remain.

No overall completion claim, paid authorization, model calls, source outcomes,
cluster use, old-endpoint reopening or automation changes. No active background
benchmark remains after the saved terminal result.
