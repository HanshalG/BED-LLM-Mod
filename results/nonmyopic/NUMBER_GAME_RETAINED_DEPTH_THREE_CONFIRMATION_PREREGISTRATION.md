# Number Game Retained Depth-Three Confirmation Preregistration

Date frozen: 2026-07-28, before any confirmation response.

## Development Boundary

The open eight-tree depth-three V2 dataset was used to design retained
rejuvenation and choose the `0.005` Brier risk-set tolerance. Its endpoints are
development data and are excluded from this confirmation.

The serving interface is unchanged from the prior exact-400-response V2 run
and its excluded exact-50-response smoke. Retention and risk-set selection are
deterministic local operations, so no additional serving smoke is required.

## Frozen Fresh Design

- Six fresh GPT-5.4 Mini planning trees: seeds `27800..27805`.
- Six fresh Gemini 2.5 Flash target supports: seeds `27900..27905`.
- Both models are nonreasoning at temperature `0.7`.
- Exactly 50 accepted responses per tree and 300 total.
- The existing hard executable-constraint prompt and parser are unchanged.
- First-step support is freshly generated.
- Second-step support is the deduplicated union of newly generated valid
  hypotheses and consistent parent hypotheses.
- Candidate selection retains roots within `0.005` absolute simulated Brier
  risk of the minimum, then minimizes simulated Hamming, maximizes simulated
  coverage, and uses deterministic root order.
- Every policy is deployed through the same retained-rejuvenation branches on
  the independently generated target support.
- Controls: pure-Brier retained depth three, generated-only depth-three
  selector, parent-only depth-three selector, predictive-risk depth two,
  myopic EIG, fixed-support depth three, exact uniform random root, and seeded
  positive-test strategy.
- Whole trees are the unit of aggregation and bootstrap resampling.
- Only explicit provider-error responses with zero reported cost may receive
  the existing identical-payload retry. Malformed normal stops fail closed.
- Hard run cap: `$1.07`. Live balance before freezing is
  `$1.080791044`; no reserve is retained per user instruction.

## Frozen Conjunctive Gates

All gates must pass:

1. exactly 300 accepted responses with exact attempt accounting;
2. zero reasoning tokens and zero forced exits;
3. total reported cost at most `$1.07`;
4. every tree has at least 16 initial, eight first-branch, 16 target, and
   eight novel-target hypotheses;
5. every retained second branch has at least four hypotheses;
6. live and public retained-support minima agree;
7. candidate and depth-two roots differ on at least 3/6 trees;
8. candidate Brier improves at least 1% over depth two and wins at least 3/6;
9. mean Hamming and coverage do not regress versus depth two;
10. candidate and parent-only roots differ on at least 2/6, candidate Brier
    improves at least 1%, and candidate wins at least 2/6;
11. candidate and generated-only roots differ on at least 2/6 and candidate
    mean Brier is lower;
12. candidate mean Brier is lower than myopic EIG;
13. mean within-tree source-risk/target-Brier Spearman is at least `0.4`; and
14. mean pairwise concordance is at least `0.65`.

Any failure is a gated null for this exact retained depth-three method. There
will be no threshold repair, selective tree removal, or rerun on these seeds.
