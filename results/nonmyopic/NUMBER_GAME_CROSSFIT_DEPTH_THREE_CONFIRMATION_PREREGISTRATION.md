# Number Game Cross-Fitted Depth-Three Confirmation Preregistration

Date frozen: 2026-07-28, before any response under this protocol.

## Motivation And Development Boundary

The powered full-retention run changed 13/20 depth-two roots but improved
Brier only `0.44%`, with 5 wins and 8 losses. A post-hoc cross-fit audit then
evaluated root risk on target supports from other trees while keeping each
tree's own endpoint held out. The prespecified sensitivity table was
development-only. Eight validation draws produced a `3.20%` Brier gain for
depth three over an equally cross-fitted depth-two selector, 10 wins, one
loss, Brier interval below zero, and risk-ranking Spearman `0.850` versus
`0.446`.

This fresh run tests that one change. It does not alter the hypothesis prompt,
parser, candidate roots, branch generation, posterior, retention rule, query
policy, or held-out endpoint.

## Frozen Fresh Design

- Thirty-two fresh GPT-5.4 Mini planning trees, seeds `28300..28331`.
- Per tree, eight fresh Gemini 2.5 Flash validation supports with unique seeds
  `28500..28755`.
- Per tree, one separate Gemini 2.5 Flash endpoint support, seeds
  `28400..28431`.
- Both models are nonreasoning at temperature `0.7`.
- Exactly 49 planning responses, eight validation responses, and one endpoint
  response per tree: 58 per tree and 1,856 total.
- First and second refreshes retain consistent parent particles and union them
  with valid newly generated hypotheses.
- The eight validation draws receive equal weight. Within each draw,
  hypotheses receive equal weight.
- Cross-fitted depth-three risk is the mean validation Brier after three
  adaptive queries.
- Cross-fitted depth-two risk is the mean validation Brier after two adaptive
  queries, using the same eight draws.
- Roots minimize the corresponding validation risk. Values within `1e-12`
  of the numeric minimum use the existing source root order; this prevents
  machine-epsilon flips among extension-symmetric roots.
- Both selected roots are deployed through the same fully retained
  three-query policy tree on the ninth, held-out endpoint support.
- Primary comparison: cross-fitted depth three versus cross-fitted depth two.
- Key selection control: cross-fitted depth three versus the original
  in-sample full-retention depth-three selector.
- Myopic EIG, fixed-support depth three, exact uniform random, and PTS are
  reported controls.
- Brier is primary. Hamming is conjunctive corroboration versus cross-fitted
  depth two. Exact-extension coverage is reported but not gated because it is
  a support proxy rather than the proper predictive-risk objective, and the
  open development audit showed a small coverage tradeoff.
- Whole trees are the aggregation and bootstrap unit.
- Only zero-cost provider-error responses may receive the existing
  identical-payload retry. Malformed normal stops fail closed.
- Projected cost is `$5.80`; hard cumulative cap is `$6.50`.
- The authenticated balance before freeze is `$36.858332144`. No reserve is
  held; this run uses about one day of the four-day budget.

## Frozen Conjunctive Gates

All gates must pass:

1. exactly 1,856 accepted responses with exact attempt accounting;
2. zero reasoning tokens and zero forced exits;
3. total reported cost at most `$6.50`;
4. every initial and endpoint support has at least 16 valid hypotheses and at
   least eight endpoint hypotheses are novel to the initial support;
5. all eight validation supports per tree have at least 16 valid hypotheses;
6. every retained first branch has at least eight hypotheses and every
   retained second branch has at least four;
7. cross-fitted depth-three and depth-two roots differ on at least 12/32
   trees;
8. depth-three Brier improves at least `1.5%` over depth two, its
   tree-bootstrap difference interval is strictly below zero, and it wins at
   least 10/32 trees;
9. mean Hamming does not regress versus cross-fitted depth two;
10. novel-target mean Brier and Hamming do not regress;
11. cross-fitting improves at least `2%` over the original in-sample
    depth-three selector, its Brier interval is below zero, and it wins at
    least 10/32 trees;
12. cross-fitted depth three improves at least `5%` over myopic EIG;
13. cross-fitted depth-three risk has mean source-to-endpoint Spearman at
    least `0.7`; and
14. its mean Spearman exceeds cross-fitted depth two by at least `0.15`.

Any failure is a gated null for this exact cross-fitted depth-three method.
There will be no threshold repair, support-count change, selective tree
removal, seed reuse, or rerun under this protocol.
