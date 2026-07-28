# Number Game Cross-Fit Endpoint-Precision Preregistration

Date frozen: 2026-07-28, before any response under this protocol.

## Motivation And Boundary

The fresh 32-tree cross-fitted depth-three confirmation is a gated null:
Brier improves `1.54%` over an equally cross-fitted depth-two selector and
wins 13 trees, but the tree-bootstrap interval is
`[-0.005115, 0.000214]`. Cross-fitting itself is strongly validated against
the in-sample depth-three selector (`4.07%`, interval below zero), and
depth-three root ranking is substantially stronger than depth two
(`0.771` versus `0.527` Spearman).

The opened endpoint used one independent Gemini support per tree. This
extension measures the same already-fixed policy roots more precisely. It does
not regenerate planning trees, change a root, alter a prompt, tune a
threshold, or include the opened endpoint in its primary statistic.

## Frozen Design

- Source public `RESULT.json` SHA-256:
  `1081da1e8381b88f7cd3fcd905b5ed539047863bcc087d686e509ed8f82d794a`.
- Source public `TREES.json` SHA-256:
  `cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9`.
- All 32 cross-fitted depth-three and depth-two roots remain exactly fixed.
- Generate 16 entirely fresh Gemini 2.5 Flash endpoint supports per tree:
  512 independent responses with unique seeds `29000..29511`.
- Nonreasoning, temperature `0.7`, unchanged initial-support prompt and strict
  parser.
- Each endpoint draw receives equal weight, regardless of parsed support
  size. Hypotheses within a draw receive equal weight.
- The original endpoint seeds `28400..28431` are excluded from the new
  primary statistic and retained only as the prior null.
- Primary comparison: fixed cross-fitted depth-three root versus fixed
  cross-fitted depth-two root, both deployed through the same fully retained
  three-query tree.
- In-sample depth three, in-sample depth two, myopic EIG, fixed-support depth
  three, exact uniform random, and PTS remain fixed controls.
- Whole planning trees remain the bootstrap unit; the 16 endpoint draws are
  averaged within tree before resampling.
- Projected cost is `$1.14`; hard cumulative cap is `$1.50`.
- Authenticated balance before freeze is `$31.071351944`. No reserve is held.

## Frozen Conjunctive Gates

All gates must pass:

1. exactly 512 accepted responses with exact attempt accounting;
2. zero reasoning tokens and zero forced exits;
3. total reported cost at most `$1.50`;
4. all 16 endpoint supports per tree contain at least 16 valid hypotheses;
5. each tree has at least 128 endpoint hypotheses novel to its initial
   support across the 16 draws;
6. depth-three Brier improves at least `1%` over depth two, its tree-bootstrap
   difference interval is strictly below zero, and it wins at least 12/32
   trees;
7. mean Hamming does not regress versus depth two;
8. novel-target mean Brier and Hamming do not regress;
9. depth three improves at least `2%` over the fixed in-sample depth-three
   selector and its Brier interval is below zero;
10. depth three improves at least `5%` over myopic EIG;
11. depth-three source-risk versus high-precision endpoint Brier Spearman is
    at least `0.7`; and
12. depth-three Spearman exceeds depth two by at least `0.15`.

Coverage remains reported but not gated. Any failure is a gated null for this
fixed-policy endpoint-precision extension. There will be no draw-count change,
seed reuse, endpoint inclusion, threshold repair, or rerun under this
protocol.
