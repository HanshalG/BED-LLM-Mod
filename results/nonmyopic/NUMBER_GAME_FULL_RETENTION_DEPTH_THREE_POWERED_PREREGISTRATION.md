# Number Game Full-Retention Depth-Three Powered Preregistration

Date frozen: 2026-07-28, before any response under this protocol.

## Motivation And Development Boundary

The fresh six-tree retained-rejuvenation confirmation preserved the initial
generated-only first refresh and retained consistent parent hypotheses only at
the second refresh. It produced a directional depth-three gain over depth two:
the selected roots differed on 2/6 trees, both changes won, and the remaining
four trees tied exactly. The run was nevertheless a frozen gated null.

This experiment tests the single remaining support-loss mechanism: retain
consistent particles at both refreshes. The first observation now forms a
deduplicated union of newly generated valid rules and consistent initial
particles. The greedy second query is selected on that merged support. The
second observation then forms the same kind of union from newly generated
rules and consistent first-step parent particles.

The open six-tree paired Brier differences versus depth two were
`[-0.019211786, 0, 0, 0, 0, -0.001230068]`, with mean `-0.003406976` and
sample standard deviation `0.007758362`. The normal-approximation calculation
`(1.96 * SD / abs(mean))^2` gives `19.921`, fixing the sample size at 20
trees. These open endpoints were used only for sample-size planning. They were
not used to change the previously chosen `0.005` risk-set tolerance or define
the prospective effect thresholds.

## Frozen Fresh Design

- Twenty fresh GPT-5.4 Mini planning trees: seeds `28000..28019`.
- Twenty fresh Gemini 2.5 Flash target supports: seeds `28100..28119`.
- Both models are nonreasoning at temperature `0.7`.
- Exactly 50 accepted responses per tree and 1,000 total.
- The existing hard executable-constraint prompt, strict parser, eight-root
  candidate construction, and target generation are unchanged.
- First support refresh: valid generated rules plus consistent initial
  particles, deduplicated by extension.
- Second support refresh: valid generated rules plus consistent first-step
  parent particles, deduplicated by extension.
- Candidate selection retains roots within `0.005` absolute simulated Brier
  risk of the minimum, then minimizes simulated Hamming, maximizes simulated
  coverage, and uses deterministic root order.
- Every policy is deployed on the same fully retained branch supports and
  independently generated target support.
- Primary comparison: predictive-Bayes-risk depth three versus
  predictive-Bayes-risk depth two.
- Controls: retained-parent-only depth three, second-refresh-generated-only
  depth three, myopic EIG, fixed-support depth three, exact uniform random
  root, and the seeded positive-test strategy.
- The generated-only control isolates the second refresh only: its first
  refresh remains retained because changing the first support also changes the
  greedy second query. Fixed-support and parent-only controls separately
  measure no generation and no second-refresh generation.
- PTS remains reported as a descriptive control but is not a promotion gate.
- Whole trees are the unit of aggregation and bootstrap resampling.
- Only explicit provider-error responses with zero reported cost may receive
  the existing identical-payload retry. Malformed normal stops fail closed.
- Hard run cap: `$3.60`. The projected cost from the fresh six-tree serving
  rate is about `$3.25`. Per user instruction, no balance reserve is retained
  and the full available OpenRouter balance may be used.
- The live balance at freeze time is `$0.127224544`; therefore no paid call is
  authorized until the account is topped up enough to execute the complete
  protocol.

The prior exact-300 serving run qualifies the unchanged models, prompts,
schema, and transport path. Full retention is deterministic local inference,
so no paid smoke is required.

## Frozen Conjunctive Gates

All gates must pass:

1. exactly 1,000 accepted responses with exact attempt accounting;
2. zero reasoning tokens and zero forced exits;
3. total reported cost at most `$3.60`;
4. every tree has at least 16 initial, 16 target, and eight novel-target
   hypotheses;
5. every fully retained first and second branch has at least eight hypotheses;
6. live and public retained-support minima agree;
7. mean support size strictly increases over generated-only support at both
   refreshes on every tree;
8. candidate and depth-two roots differ on at least 6/20 trees;
9. candidate Brier improves at least 1% over depth two, the tree-bootstrap
   Brier-difference interval is strictly below zero, and candidate wins at
   least 6/20 trees;
10. mean Hamming and coverage do not regress versus depth two;
11. candidate and parent-only roots differ on at least 5/20, candidate Brier
    improves at least 1%, and candidate wins at least 5/20;
12. candidate and second-refresh-generated-only roots differ on at least 3/20
    and candidate mean Brier is lower;
13. candidate Brier improves at least 5% over myopic EIG;
14. candidate Brier improves at least 5% over fixed-support depth three and
    its tree-bootstrap Brier-difference interval is strictly below zero;
15. candidate mean Brier is lower than exact uniform random;
16. mean within-tree source-risk/target-Brier Spearman is at least `0.4`;
17. mean pairwise concordance is at least `0.65`; and
18. novel targets have no mean Brier, Hamming, or coverage regression versus
    depth two.

Any failure is a gated null for this exact full-retention depth-three method.
There will be no threshold repair, selective tree removal, seed reuse, or
rerun under this protocol.
