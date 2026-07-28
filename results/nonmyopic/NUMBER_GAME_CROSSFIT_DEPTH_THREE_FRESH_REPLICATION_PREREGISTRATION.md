# Number Game Cross-Fitted Depth-Three Fresh Replication Preregistration

Date frozen: 2026-07-28, before any response under this protocol.

## Motivation And Independence

The fixed-policy endpoint-precision study passed every frozen gate: depth
three improved Brier `1.8619%` over equally cross-fitted depth two with a
whole-tree interval `[-0.004996, -0.001160]`, while Hamming, coverage, novel
targets, and ranking fidelity also improved. Its policies were fixed before
512 fresh endpoint responses, but its planning trees had already been opened
by the preceding single-endpoint null.

This replication removes that remaining caveat. Planning trees, validation
supports, selected roots, and endpoint supports are all fresh under one frozen
protocol. No prior tree, root, validation response, or endpoint response enters
the primary statistic.

## Frozen Design

- Thirty-two fresh GPT-5.4 Mini planning trees, seeds `30600..30631`.
- Per tree, eight fresh Gemini 2.5 Flash validation supports with unique seeds
  `30700..30955`.
- Per tree, sixteen fresh Gemini 2.5 Flash endpoint supports: one seed from
  `31000..31031` plus fifteen unique seeds per tree from `31100..31579`.
- Both models are nonreasoning at temperature `0.7`.
- Exactly 49 planning responses, eight validation responses, and sixteen
  endpoint responses per tree: 73 per tree and 2,336 total.
- The hypothesis prompt, strict parser, eight candidate roots, analytical
  likelihoods, Bayesian update, and query policy are unchanged.
- First and second branch refreshes retain consistent parent particles and
  union them with valid newly generated hypotheses.
- Each validation draw receives equal weight. Cross-fitted depth-three risk is
  mean validation Brier after three adaptive queries; cross-fitted depth-two
  risk is mean validation Brier after two, on the same draws.
- Roots minimize the corresponding validation risk, with the frozen `1e-12`
  numerical tie tolerance and source root-order tie-break.
- Each endpoint draw receives equal weight. The 16 draws are averaged within
  tree before whole-tree aggregation and bootstrap resampling.
- Both selected roots deploy through the same fully retained three-query tree.
- Primary comparison: cross-fitted depth three versus cross-fitted depth two.
- Controls: in-sample retained depth three, in-sample depth two, myopic EIG,
  fixed-support depth three, exact uniform random, and PTS.
- Projected cost is `$6.90`; hard cumulative cap is `$7.60`.
- Authenticated balance before freeze is `$29.925036144`. No reserve is held.

## Frozen Conjunctive Gates

All gates must pass:

1. exactly 2,336 accepted responses with exact attempt accounting;
2. zero reasoning tokens and zero forced exits;
3. total reported cost at most `$7.60`;
4. every initial, validation, and endpoint support has at least 16 valid
   hypotheses;
5. every tree has at least 128 endpoint hypotheses novel to its initial
   support across the 16 endpoint draws;
6. every retained first branch has at least eight hypotheses and every
   retained second branch has at least four;
7. cross-fitted depth-three and depth-two roots differ on at least 12/32
   trees;
8. depth-three Brier improves at least `1%` over depth two, its whole-tree
   bootstrap difference interval is strictly below zero, and it wins at least
   12/32 trees;
9. mean Hamming and coverage do not regress versus depth two;
10. novel-target mean Brier, Hamming, and coverage do not regress;
11. depth three improves at least `2%` over in-sample retained depth three and
    its Brier interval is below zero;
12. depth three improves at least `5%` over myopic EIG and fixed-support depth
    three;
13. depth three improves at least `3%` over PTS and its Brier interval is below
    zero;
14. depth-three source-risk versus endpoint Brier Spearman is at least `0.7`;
    and
15. depth-three Spearman exceeds depth two by at least `0.15`.

Any failure is a gated null for this exact fresh replication. There will be no
tree substitution, seed reuse, endpoint extension, draw-count change,
threshold repair, or rerun under this protocol.
