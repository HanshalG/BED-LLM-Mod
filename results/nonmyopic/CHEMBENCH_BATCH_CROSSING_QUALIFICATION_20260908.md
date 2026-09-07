# Batched crossing-aware integration: synthetic qualification passed

## Development Evidence

The previous goal turn was progress: it banked a bounded scalar implementation
and its unresolved numerical refinement error. This continuation adds batched
Bellman evaluation with bounded live arrays and a global state/time budget.
Scalar-versus-batched tests cover every root value at depths1/2/3, adaptive and
open-loop, varying batch sizes, zero and tiny support, and resource exhaustion.
No greedy continuation or particle truth index is introduced.

Uniform32/64 quantiles were tested first and remain a saved failure at commit
7783a27b. Batching reproduced their values while reducing the32-branch h3 example
from17.83s to0.69s. Nevertheless64 branches had one-step error0.002359, above the
unchanged0.001 limit. Increasing resolution alone did not fix the narrow
posterior transition regions.

The successor changes the integration rule, not observations, likelihoods or
thresholds: solve pairwise Gaussian posterior-density crossing equations, map
their roots through the predictive CDF, and apply a composite Gauss-Legendre
rule over those probability intervals. All hypotheses stay in the full likelihood
and posterior. At least two nodes per interval are required; too many crossings
fail closed instead of silently discarding hypotheses. Quadrature intervals
below1e-14 are coalesced numerically. Total branch counts remain32/64.

## Prespecified Check

The successor reuses opened synthetic fixtures as numerical-method development,
not new independent empirical validation. Before execution it fixes error and
regret caps0.001, positive constructed adaptivity floor0.001, maximum5M states
and60s per plan,180s panel, and a64MiB tensor workspace allowance. The latter is
an internal allocation estimate, not an operating-system RSS guarantee.

| Check | 32 branches | 64 branches |
| --- | --- | --- |
| Maximum one-step absolute error | 0.00045343 | 0.00018813 |
| Three-step sufficient-statistic error | 0.00006533 | 0.00001358 |
| Three-step seconds | 1.617 | 9.382 |
| Constructed adaptive expected risk | 0.00005311 | 0.00022674 |
| Constructed open-loop expected risk | 0.05316603 | 0.05315912 |
| Constructed planned adaptivity gap | 0.05311292 | 0.05293238 |

All four gates pass: one-step reference accuracy, three-step reference accuracy,
32/64 value refinement, and positive adaptivity with outcome-dependent next
assays. Root values for two prospectively selected observation positions are
saved, not hand-picked examples. The Gaussian h3 check is not a same-budget
depth efficacy experiment. The constructed regime example establishes planned
adaptation, not superiority under real receding-horizon deployment.

Artifacts:
- `chembench_batch_refinement/20260908-v1/RESULT.json`: preserved uniform failure.
- `chembench_crossing_refinement/20260908-v1/RESULT.json`: synthetic_refinement_passed,
  with source hashes and atomic per-order checkpoints.

Focused combined numerical tests:83/83 in33.21s, including crossing count failure
and scalar/batch equivalence for the state-dependent rules. Numerical tolerances
are not rigorous mathematical certificates. Broader chemistry mixtures may exceed
the crossing budget or fail calibration; this panel does not claim otherwise.

## Full-Plan Completion Audit

Numerical deliverable A now includes the finite exact reference and a tested
bounded continuous candidate with actual contingent h1/h2/h3 and open-loop
controls. The eight-world source pilot B remains unopened. Its exact source
sampling, prior, raw-noise law, target distribution, paired random numbers,
complete-arm runtime budget and endpoint rules must be frozen before responses.
Do not shrink parameter uncertainty merely to satisfy the crossing cap.

LLM deliverable C remains unopened and requires the new useful-proposal semantic
gate, real-history updates, productive compute controls, paired full-policy
endpoints and honest uncertainty. The stronger path-dependent discovery goal is
still unachieved. No paid authorization, LLM calls, chemistry outcomes, cluster
use, old endpoint reopening or automation reactivation occurred.
