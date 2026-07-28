# Number Game Predictive-Risk Powered Replication Preregistration

Date frozen: 2026-07-28, before any powered-replication response.

The eight-tree V2 comparison is fully exposed. It established large,
whole-tree-significant gains over myopic EIG, fixed-support depth two, and
uniform random, but its 2.84% gain over PTS had an interval crossing zero. The
failed multi-draw development leaves the successful one-draw method unchanged.

## Frozen Design

- 32 entirely fresh planning trees with Gemini seeds `26400..26431`.
- 32 entirely fresh GPT-5.4 target supports with seeds `26500..26531`.
- Exact V2 one-draw method: nonreasoning, temperature `0.7`, strict schema,
  24-rule initial support, both branches of eight roots, terminal predictive
  Bayes-risk selection, and greedy-EIG second query.
- Exact V2 controls: myopic EIG, classical fixed-support depth two, uniform
  over two seeded PTS roots, and exact uniform over all eight candidates.
- Exactly 18 successful responses per tree, 576 total. Explicit zero-cost
  provider errors may receive the already-preregistered identical-payload
  transport retry.
- Primary inference uses only these 32 fresh trees, weighted equally, with
  50,000 whole-tree bootstrap samples. The old eight trees are not pooled into
  the primary result.
- Raw responses remain private. Compiled rules, seeds, hashes, per-tree
  metrics, and aggregate results are public.
- Total cost cap: `$3.00`; no reserve.

## Structural Gates

All trees must retain at least 16 initial rules, eight rules in every branch,
16 target rules, and eight target extensions novel to planning support. Exact
attempt accounting, zero reasoning tokens, and zero forced exits are required.

## Scientific Pass Criteria

All must pass:

1. Predictive-risk BED differs from myopic and fixed-depth-two on at least
   24/32 trees.
2. Versus PTS: at least 2% aggregate Brier gain, wholly negative whole-tree
   Brier interval, at least 20/32 Brier wins, positive Hamming gain, and wholly
   negative Hamming interval.
3. Versus myopic: at least 10% Brier gain, wholly negative interval, at least
   24/32 wins, at least 15% Hamming gain, and no mean coverage loss.
4. Versus fixed depth two: at least 10% Brier gain and wholly negative
   interval.
5. Versus exact uniform random: at least 5% Brier gain and wholly negative
   interval.
6. Extension-novel targets have negative mean Brier and Hamming differences
   versus myopic.

The 2% PTS magnitude is a prospective powered-effect threshold, not a
retroactive change to V2's failed 5% conjunction. V2 remains formally null
against its own preregistration regardless of this result.
