# Number Game Predictive-Risk Independent-Tree Replication Preregistration

Date frozen: 2026-07-28, before any replication-tree response.

The development tree and cross-model target holdout are fully exposed. This
replication therefore regenerates every planning and target support.

## Frozen Design

- Eight independent planning trees with seeds `26070..26077`.
- Each tree uses Gemini 2.5 Flash, nonreasoning, temperature `0.7`, strict
  schema, and a fixed request seed.
- Each tree has one 24-rule initial support and both branch supports for eight
  root candidates: 17 planning calls.
- Each tree has one independent GPT-5.4 nonreasoning target-support call at
  temperature `0.7`, with seeds `26170..26177`.
- Total: exactly 18 calls per tree and 144 accepted calls.
- The safe executable grammar, filtering, extension deduplication, candidate
  construction, uniform particles, and greedy second-query EIG are unchanged.
- Predictive-risk BED minimizes terminal posterior-predictive Brier over the
  current initial particles treated as simulated truths.
- Baselines are myopic EIG, classical fixed-support depth two, uniform over two
  seeded PTS roots, and exact uniform choice over all eight candidate roots.
- Every policy uses the same realized regenerated branch support after its
  first query. Only first-query selection differs.
- Primary endpoints are the fresh GPT-5.4 targets. Targets novel to each
  planning support are a required transfer diagnostic.
- Aggregate metrics weight trees equally. Confidence intervals resample whole
  trees 50,000 times with seed 26270; target rows within a tree are not treated
  as independent.
- Raw responses remain private. Compiled rules, seeds, hashes, per-tree
  results, accounting, and aggregate metrics are public.

## Structural Gates

All eight trees must have:

- at least 16 valid unique initial rules;
- at least eight valid unique rules in every branch;
- at least 16 valid unique target rules, including eight novel extensions;
- exact request/attempt accounting, zero reasoning tokens, and zero forced
  exits.

Total cost must not exceed `$2.00`. The full live provider balance is usable;
there is no reserve.

## Scientific Pass Criteria

All must pass:

1. Predictive-risk BED selects a different root from myopic EIG on at least
   six of eight trees and from fixed-support depth two on at least six.
2. Versus myopic EIG, aggregate Brier improves by at least 5%, the whole-tree
   bootstrap interval for the Brier difference lies below zero, and at least
   six trees are wins.
3. Versus myopic EIG, aggregate best-rule Hamming improves by at least 5%, its
   whole-tree interval lies below zero, and mean exact-extension coverage does
   not decrease.
4. Versus fixed-support depth two, aggregate Brier improves by at least 5% and
   its whole-tree interval lies below zero.
5. Aggregate Brier improves by at least 5% versus exact uniform random and by
   at least 5% versus PTS; the PTS whole-tree interval lies below zero.
6. Across extension-novel targets, mean Brier and mean Hamming differences
   versus myopic are both negative.

No tree, seed, target, baseline, threshold, or aggregation rule will be removed
or changed after responses. A full pass supports the paper's LLM-native
non-myopic result. Failure is reported as the independent-tree result.
