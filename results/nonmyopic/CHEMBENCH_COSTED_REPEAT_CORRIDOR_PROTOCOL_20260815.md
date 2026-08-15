# ChemBench costed-repeat corridor protocol

Date frozen: 2026-08-15

## Purpose

Test whether realistic stochastic assay precision and costed repeat allocation turn
the verified ChemBench structural-discovery opportunity into a robust monotonic
d1/d2/d3 policy ladder.

This is a genuinely new zero-call environment after the equal-cost 1%-noise
staged-compound corridor closed. It changes held-out parameter versions, target
queries, observation model, action space, budget, and planner transition before
opening any new response. It does not modify or rescue the closed result.

A pass authorizes only a separately frozen LLM atomic-edit semantic gate.

## Source and cohort

Use pinned ActiveSciBench-Chem commit
`acf160eb6c96897748dd92b152703b59b74efc05` and the same complete structural
stratum of nine standard inhibitor/product + Arrhenius + pH-bell compounds:

1. `c19_mm_competitive_arrhenius_ph`
2. `c20_mm_uncompetitive_arrhenius_ph`
3. `c21_mm_noncompetitive_arrhenius_ph`
4. `c22_mm_product_arrhenius_ph`
5. `c32_pingpong_competitive_arrhenius_ph`
6. `c44_hill_competitive_arrhenius_ph`
7. `c45_hill_noncompetitive_arrhenius_ph`
8. `c59_sinh_competitive_arrhenius_ph`
9. `c60_sinh_noncompetitive_arrhenius_ph`

Use `easy`, `medium`, and `hard` for 27 paired cells. Inference particles remain
source versions `v0`, `v1`, and `v2`. Use previously unopened `v4` as the held-out
truth version. No truth or difficulty cell may be removed.

Use all source-defined standard structures `c0` through `c64`, the public atomic
compiler, complete three-particle structure proposals, and the same prohibition on
truth-particle proposal and multi-edit jumps.

## Stochastic observation model

An individual well observes

```text
z = log1p(rate) + Normal(0, sigma_log^2)
```

with `sigma_log = sqrt(log(1 + 0.10^2))`, corresponding to a 10% log-normal
coefficient of variation. This is fixed before inspecting v4 responses.

An action chooses a base assay and repeat count `r` in `{1, 2, 4}`. Its observation
is the mean of the `r` log-rate replicates, with standard deviation
`sigma_log / sqrt(r)`, and costs exactly `r` wells.

For each base assay, freeze low/high category thresholds to the inference-particle
tertiles of `log1p(rate)` using v0-v2 only. Apply those same thresholds to every
truth and repeat count. Thus held-out v4 values do not define the observation
codec.

## Base assay protocols

Use these ten source-defined diagnostic extremes:

1. `C_A=0.02`
2. `C_A=100`
3. `C_I=50,C_A=0.1`
4. `C_I=50,C_A=100`
5. `C_B=0.01`
6. `C_B=100`
7. `C_P=20`
8. `T=278`
9. `pH=4`
10. `pH=10`

Each combines with all three repeat counts, yielding 30 costed actions. After a
base assay is chosen, all three repeat variants of that base assay become
unavailable. The total execution budget is eight wells.

## Target

Terminal loss is mean squared error in `log1p(rate)` on 512 source-box queries per
difficulty using new seeds:

- easy: `2026084201`
- medium: `2026084202`
- hard: `2026084203`

Queries and truth values remain paired across every policy and depth.

## Support transition

After an averaged categorical observation, enumerate source-supported structures
one public atomic edit from represented support, marginalize each candidate's
history likelihood over its v0-v2 particles, and add all three particles of the
highest-evidence candidate. Add at most one structure per decision. Never add a
held-out v4 truth particle, skip an edit, or prune/re-propose within the eight-well
gate.

## Cost-aware policy ladder

d1, d2, and d3 denote one, two, and three future **decisions**, not wells. Every
policy is executed receding-horizon until no feasible action remains under the same
eight-well budget.

At each state, build one deterministic shared action shortlist:

1. evaluate every feasible action's expected one-step terminal risk under the
   current prospective belief and atomic support transition;
2. rank by absolute one-step risk reduction, then stable action identity;
3. separately rank by one-step risk reduction per well, then stable identity;
4. take the stable union of the top four from each ranking, capped at eight actions.

All depths and controls use the identical shortlist rule. The planner exactly
integrates the three categorical outcomes within this frozen shortlist. Choosing
an action subtracts its repeat cost and removes every repeat variant of that base
assay.

Evaluate:

- dynamic d1, d2, and d3;
- fixed-support d1;
- full-compositional-support d1;
- random feasible costed policy;
- immutable replay of every oracle proposal; and
- a cost-blind control that treats every repeat choice as cost one but is charged
  its actual well cost during execution.

The last control tests whether any gain comes from modeling resource allocation,
not merely from lower-noise actions.

## Prospective gate

Use paired truth-cell terminal losses with practical tie tolerance `1e-8`. Every
condition must pass:

1. finite normalized likelihoods, beliefs, and losses;
2. thresholds depend only on v0-v2 inference particles;
3. every proposal is one complete three-particle atomic edit;
4. no v4 truth particle is proposed;
5. immutable proposal replay is exact;
6. every policy spends at most eight wells and never reuses a base assay;
7. d2 mean MSE is at least 5% below d1;
8. d3 mean MSE is at least 5% below d2;
9. d2 paired wins exceed losses;
10. d3 paired wins exceed losses;
11. d3 beats d1 on at least 21 of 27 cells;
12. d2 improves at least two difficulty means;
13. d3 improves all three difficulty means;
14. d1/d2 and d2/d3 root actions each differ on at least one difficulty;
15. at least one depth link changes root repeat count;
16. d2 aggregate risk remains above `1e-8`;
17. dynamic d3 is at least 20% better than fixed-support d1;
18. dynamic d3 is no worse than 1.5 times full-support d1;
19. dynamic d3 is at least 3% better than the cost-blind control; and
20. random policy is worse than dynamic d3.

Failure closes this exact v4 cohort, noise, thresholds, protocols, repeat set,
budget, target seeds, shortlist, transition, controls, and gates. No observed
failure may be repaired by changing noise, repeat choices, budget, shortlist width,
cells, or thresholds.

## Authority after pass

A pass opens only a prospectively frozen 12--24 branch LLM semantic gate using the
public atomic edit vocabulary and quantitative task-anchored residual fingerprint.
That gate must include history-blind, answer-flip, and recursive-expanded-pool
controls before any d1/d2/d3 LLM policy run.
