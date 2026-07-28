# Number Game Cross-Planner Replication Preregistration

Date frozen: 2026-07-28, before any replication response.

The powered primary study used Gemini 2.5 Flash to generate the planning
support and branch-conditioned successor supports, with GPT-5.4 generating
independent target concepts. This replication swaps model families to test
whether the result depends on Gemini owning the counterfactual belief dynamics.

One separate seed `26890/26990` mechanics-only smoke was run before freezing.
Only schema yield, novelty, transport, reasoning, and cost were inspected:
20 valid initial rules, at least 14 valid rules per branch, 22 target rules,
13 novel targets, 18/18 responses, zero retries/reasoning/forced exits, and
`$0.05210415`. Its policy endpoints are excluded.

## Frozen Design

- 32 fresh GPT-5.4 Mini planning trees, seeds `27000..27031`.
- 32 fresh Gemini 2.5 Flash target supports, seeds `27100..27131`.
- The exact powered-study method is unchanged: nonreasoning, temperature
  `0.7`, strict schema, 24-rule initial support, both branches of eight roots,
  terminal predictive Bayes-risk selection, and greedy-EIG second query.
- Controls remain myopic EIG, classical fixed-support depth two, uniform over
  two seeded PTS roots, and exact uniform over all eight candidates.
- Exactly 18 successful responses per tree, 576 total. Only explicit
  zero-cost provider errors may receive the existing identical-payload
  transport retry.
- Primary inference weights the 32 fresh trees equally and uses 50,000
  whole-tree bootstrap samples. No previous tree or smoke is pooled.
- Raw responses remain private. Compiled rules, seeds, hashes, per-tree
  metrics, and aggregates are public.
- Total cost cap: `$3.00`; no reserve.

## Structural Gates

Every tree must retain at least 16 initial rules, eight rules in every branch,
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

Failure does not alter the original powered result. Passing establishes
planning-model-family robustness under a role-swapped target generator; it
does not establish robustness beyond these two model families or beyond the
restricted executable Number Game grammar.
