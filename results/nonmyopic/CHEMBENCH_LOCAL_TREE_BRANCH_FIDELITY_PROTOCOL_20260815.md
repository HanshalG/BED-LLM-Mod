# ChemBench Local-Tree Branch-Fidelity Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Validate the continuous-observation branch approximation required by a local
non-myopic scenario tree. This gate asks whether a small deterministic branch
set ranks candidate next assays like high-sample posterior-predictive Monte
Carlo.

The test uses the V3-authorized 512-particle SMC posterior on prospective
in-prior source worlds. It is correct-structure and contains no structural
proposal transition, planning depth, LLM, API, network call, or new benchmark
endpoint.

## Frozen Development Panel

Use these 12 compound mechanisms on easy, medium, and hard, for 36 cases:

1. `c10_mm_competitive_arrhenius`
2. `c23_pingpong_arrhenius`
3. `c33_hill_competitive`
4. `c37_hill_arrhenius`
5. `c48_sinh_competitive`
6. `c65_ordered_bi_bi`
7. `c67_allosteric_act`
8. `c70_mixed_inhibition`
9. `c71_coop_inhibition`
10. `c73_metal_activation`
11. `c78_allosteric_act_arrhenius`
12. `c93_fractal_competitive`

This panel was selected before branch results to cover saturation, inhibition,
temperature, two-substrate, Hill/sinh, allosteric, cooperative, metal, and
fractal parameter geometries, including V1-V3's difficult allosteric family.

## Root Posterior

- Regenerate exact source-only truth and observation seeds from V1/V3.
- Condition only on the first four frozen history assays:
  `C_A=0.02`, `C_A=100`, `C_I=50,C_A=0.1`, and
  `C_I=50,C_A=100`.
- Fit two independent V3 SMC banks with 512 particles, scrambled Sobol
  initialization, full-covariance rejuvenation, ESS target 0.6, and three moves
  per rung.
- Candidate next actions are the remaining 14 frozen assays.
- Held-out terminal risk uses the first 128 rows of each tier's frozen
  512-query matrix.

## Risk Functional

For a posterior over parameter particles, terminal Bayes risk is mean
posterior variance of `log1p(rate)` over the 128 held-out query designs. For
each candidate next assay, estimate expected risk after its observation.

This is the same task-facing uncertainty that the later tree will minimize; it
does not use entropy as a proxy.

## Monte Carlo Reference

For each bank, case, and action:

1. Draw 2,048 common-random-number outcomes from the posterior-predictive
   mixture: sample a weighted particle and add transformed-rate Gaussian noise
   under the frozen likelihood.
2. Reweight all 512 particles for each simulated observation.
3. Compute the resulting terminal posterior-predictive Bayes risk.
4. Average over outcomes.

Use seed base `2026083700`, stably mixed with bank, difficulty, domain, and
action. The same reference draws define all branch approximations for that
action.

## Quantile Approximation

Sort the 2,048 reference outcomes and form equal-probability bins. Represent
each bin by its median observation, reweight the posterior once, and average
child risks by empirical bin mass.

Evaluate 3, 5, and 9 branches. Nine branches are the primary candidate; 3 and 5
are cost diagnostics only.

## Frozen Gates

All primary conditions use nine branches and must pass for both SMC banks.

1. Every V1-compatible root hash, posterior, outcome, weight, and risk is
   finite and reproducible.
2. Median within-case Spearman correlation across the 14 action values is at
   least 0.90 for each bank.
3. At least 90% of cases have within-case Spearman at least 0.80 for each bank.
4. In at least 90% of cases, approximation top-one regret is at most 3% of the
   root terminal risk for each bank.
5. Mean top-one regret is at most 1% of root risk for each bank.
6. The two banks select the same approximate best action in at least 75% of
   cases.
7. Nine branches are nonworse than five branches on median Spearman and mean
   normalized top-one regret for both banks.

## Decision

A pass freezes nine posterior-predictive branches for source-only local-tree
d1/d2/d3 development.

A failure blocks deeper planning. Diagnose action-specific multimodality or
increase branch adaptivity without changing the calibrated posterior or
reading a depth endpoint. Do not compensate with more rollouts, deeper trees,
or LLM calls.
