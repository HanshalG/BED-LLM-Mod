# ChemBench Posterior-State Branch-Fidelity Protocol

Date frozen: 2026-08-15 (Europe/London)

## Motivation

The frozen raw-observation quantile gate failed. Nine branches had low
top-action regret but did not meet all-action rank coverage, and its two-bank
argmin agreement requirement was not attainable by the reference itself: the
two independent 2,048-draw references selected the same exact action in only
24/36 cases (66.67%). This successor does not relax the rank or regret gates.
It changes the numerical architecture and replaces exact finite-sample argmin
agreement with component-bank regret for one pooled belief.

## Frozen Inputs

- Bind raw-quantile result SHA-256
  `e4533d76af6ab9be344bbf72806762eb695a425e7664a272105b3d075f706b16`.
- Use the same 12 mechanism families, three tiers, source-only truths, first
  four observations, 14 actions, 128 task targets, likelihood, V3 bank seeds,
  and two 512-particle posteriors as the predecessor.
- Concatenate the two equally weighted banks into one 1,024-particle pooled
  root posterior. This pooled posterior is the canonical planning belief.
- Draw 2,048 pooled posterior-predictive outcomes per action with common random
  numbers from seed base `2026083800`.

## Posterior-State Branches

For every simulated outcome, reweight the pooled particles and compute, over
the 128 task targets:

1. child posterior predictive means of `log1p(rate)`;
2. child posterior predictive variances of `log1p(rate)`.

Scale mean coordinates by root predictive standard deviation and variance
coordinates by root predictive variance, with a `1e-8` denominator floor.
Center each coordinate across the 2,048 outcomes. These 256 coordinates are
the child belief feature; the scalar observation alone is not the clustering
state.

Run deterministic farthest-first initialized Lloyd clustering with nine
clusters and at most 30 iterations. Empty clusters retain their prior center.
For each cluster, choose the actual outcome whose feature is nearest its final
centroid. Its reweighted posterior and scalar observation are the realizable
branch state and residual-history representative. Branch probability is the
empirical cluster fraction.

Also evaluate the predecessor's nine equal-mass raw-observation medians on the
same pooled outcomes as a non-gating architecture baseline.

## Reference And Regret

The pooled 2,048-outcome mean child task Bayes risk is the action-value
reference. Spearman and top-one regret compare nine posterior-state branches
against this pooled reference.

For numerical robustness, take the action selected by pooled posterior-state
branches and evaluate its reference regret under each predecessor component
bank's saved 2,048-draw action values, normalized by that bank's root risk.
This asks whether one pooled action is practically safe under both calibrated
finite-particle realizations; exact component argmin equality is not required.

## Frozen Gates

All conditions must pass:

1. Exact source, V1, V3, predecessor-result, posterior, outcome, weight, and
   risk bindings are finite and reproducible.
2. Median within-case pooled action-value Spearman is at least 0.90.
3. At least 90% of cases have pooled within-case Spearman at least 0.80.
4. At least 90% of cases have pooled normalized top-one regret at most 3%.
5. Mean pooled normalized top-one regret is at most 1%.
6. For each component bank, at least 90% of pooled selected actions have
   component-normalized reference regret at most 3%, and mean regret is at
   most 1%.
7. Posterior-state branches are nonworse than raw-observation quantiles on
   median Spearman and mean normalized top-one regret.

## Decision

A pass freezes the pooled 1,024-particle belief and nine posterior-state
branches for source-only local-tree d1/d2/d3 development. A failure blocks
depth and triggers a branch-budget or posterior-representation diagnosis.

This gate contains no planning depth, structural support proposal, LLM, API,
network call, or benchmark endpoint.
