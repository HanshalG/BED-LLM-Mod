# ChemBench Posterior-Sampling Fidelity Protocol

Date frozen: 2026-08-15 (Europe/London)

## Purpose

Determine whether a practical common-random-number posterior-sampling budget
can reproduce the saved 2,048-outcome one-step action values. This is the
numerical entry gate for posterior-sampling MCTS after fixed nine-branch
quadrature failed rank coverage.

This protocol contains no planning depth, structural proposal transition, LLM,
API, network call, benchmark endpoint, or paid resource.

## Frozen Bindings

- Posterior-state predecessor result SHA-256:
  `637c031946268e8015687d5eb4af4d037bfc1399300e56c0e50bdf049b4fab07`.
- Authorized V3 result SHA-256:
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`.
- Use the predecessor's exact 36 source-only cases, first four observations,
  two V3 512-particle banks, pooled 1,024-particle belief, 14 candidate actions,
  128 task targets, likelihood, and pooled outcome seed base `2026083800`.

Regenerate every pooled posterior and require its particle/weight SHA-256 to
match the predecessor before evaluating a sample estimate.

## Monte Carlo Estimates

For each case and action, regenerate the exact ordered 2,048 posterior-
predictive outcomes used by the predecessor. Reweight the pooled posterior and
compute terminal task Bayes risk for each outcome.

Evaluate nested sample counts `32`, `64`, `128`, and `256` in each of four
disjoint replicates. Replicate `r` uses the prefix of length `n` from the
256-outcome block starting at `256*r`, for `r = 0,1,2,3`. Thus every estimate
is nested within replicate and the four 256-sample estimates are independent
conditional on the frozen posterior-predictive stream.

The saved predecessor 2,048-outcome expected risks are the action-value
reference. Do not replace or recompute the reference after seeing sample
results. Common random numbers are shared across methods and sample counts,
not across actions whose predictive distributions differ.

## Metrics

For every sample count and replicate, compute across the 36 cases:

- median within-case Spearman over 14 action risks;
- fraction of cases with Spearman at least 0.80;
- fraction with sampled top-one reference regret at most 3% of pooled root
  risk;
- mean pooled normalized top-one regret;
- selected-action reference regret under each saved component bank, normalized
  by that bank's root risk.

Also report exact selected-action agreement across all six replicate pairs and
the 1,024-sample four-replicate ensemble as diagnostics. They are not primary
gates.

## Frozen Gates

The 256-sample budget opens posterior-sampling tree development only if **all
four replicates** independently satisfy:

1. every binding, posterior, outcome, weight, and risk is finite and
   reproducible;
2. median within-case Spearman is at least 0.90;
3. at least 90% of cases have Spearman at least 0.80;
4. at least 90% of cases have normalized top-one regret at most 3%;
5. mean pooled normalized top-one regret is at most 1%;
6. for each component bank, at least 90% of selected actions have normalized
   component-reference regret at most 3%, and mean regret is at most 1%.

Counts 32/64/128 are diagnostics and may identify a cheaper later setting, but
cannot open depth in this first implementation. There is no monotonicity gate
across finite Monte Carlo prefixes.

## Decision

A pass freezes 256 trajectories per root action as the initial MCTS value
budget and authorizes an oracle-support horizon-opportunity gate. It does not
authorize LLM calls.

A failure blocks MCTS depth at this action-panel size. Reduce or pre-screen the
action set prospectively, improve variance reduction, or redesign the horizon
environment before any LLM spend; do not increase depth to average away a
failed first link.
