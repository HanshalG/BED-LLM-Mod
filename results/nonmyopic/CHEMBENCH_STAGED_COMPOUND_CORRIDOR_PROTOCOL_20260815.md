# ChemBench staged-compound corridor protocol

Date frozen: 2026-08-15

## Purpose

Test, before any LLM response, whether a public compositional ChemBench model space
contains a robust non-myopic structural-discovery opportunity at policy depths one,
two, and three.

This is a zero-call mechanics gate. A pass may authorize only a separately frozen
LLM atomic-edit semantic gate. It does not authorize a policy endpoint or a paper
efficacy claim.

## Source and scientific stratum

Use the already pinned ActiveSciBench-Chem source commit
`acf160eb6c96897748dd92b152703b59b74efc05` and its compound-rate factory. The
source preserves several standard compound domains outside its standard 57-domain
benchmark because pH-bell activity requires estimating two independent pKa
constants. This difficulty is the scientific reason for the stratum; no planner
outcome is inspected when selecting it.

The truth structures are every source-defined standard compound containing:

- one standard substrate core;
- one supported inhibitor or product-inhibition mechanism;
- Arrhenius temperature dependence; and
- pH-bell activity.

This yields exactly nine structures:

1. `c19_mm_competitive_arrhenius_ph`
2. `c20_mm_uncompetitive_arrhenius_ph`
3. `c21_mm_noncompetitive_arrhenius_ph`
4. `c22_mm_product_arrhenius_ph`
5. `c32_pingpong_competitive_arrhenius_ph`
6. `c44_hill_competitive_arrhenius_ph`
7. `c45_hill_noncompetitive_arrhenius_ph`
8. `c59_sinh_competitive_arrhenius_ph`
9. `c60_sinh_noncompetitive_arrhenius_ph`

Evaluate all nine at `easy`, `medium`, and `hard`, for 27 paired truth cells. Use
source parameter versions `v0`, `v1`, and `v2` as inference particles and held-out
`v3` as each truth. No task or cell may be removed after evaluation.

## Public compositional model space

Use all source-defined standard primitive and compound structures `c0` through
`c64`, including the source-preserved pH worlds. Exclude later novel-core families.

A public structure is represented as:

```text
substrate core
+ optional inhibitor/product mechanism
+ optional Arrhenius factor
+ optional pH-bell factor
```

The deterministic compiler owns the equation, required parameters, bounds,
canonical signature, and inverse edit. One atomic edit may:

- add or remove one inhibitor/product, Arrhenius, or pH-bell component; or
- replace the substrate core while preserving the modifier set.

Only source-supported compatible compositions are executable in this first gate.
The compiler may not consult a truth identity.

## Parameter uncertainty

Flatten each structure's `v0`, `v1`, and `v2` parameterizations into three
equal-prior inference particles. The held-out `v3` truth particle participates in
the prospective world distribution and truth replay but is never proposed into
the inference support.

This is a finite exact Bayesian parameter approximation, not a point-parameter
registry lookup. Every proposed structure adds all three inference particles at
once.

## Actions, observations, and target

Use the existing 18 frozen equal-cost ChemBench assays:

- baseline;
- four substrate concentrations;
- four inhibitor-by-substrate conditions;
- three second-substrate concentrations;
- two product concentrations;
- two temperatures; and
- two pH values.

Use source 1% Gaussian rate noise and the existing three-outcome categorical
likelihood construction. Action groups are substrate, inhibitor, second substrate,
product, temperature, and pH. An assay may be used at most once. Execution budget
is four assays.

Terminal loss is mean squared error in `log1p(rate)` on 512 deterministic target
queries per difficulty, sampled over the complete source query box with seeds:

- easy: `2026084001`
- medium: `2026084002`
- hard: `2026084003`

The same target queries are used for every policy and depth within a difficulty.

## Oracle support transition

Initial support is the frozen nine primitive structures already used by the
project. Each contributes its three inference parameter particles.

After a simulated or realized `(assay, outcome)`:

1. enumerate structures exactly one public atomic edit from any represented
   structure;
2. exclude already represented structures;
3. score each candidate by marginal history likelihood over its three inference
   particles;
4. select the highest-evidence candidate with stable signature/name tie-breaking;
5. add exactly its three inference particles; and
6. update the posterior over all represented structure-parameter particles.

No transition may add a held-out truth particle, skip an edit level, use a truth
identifier, or add more than one structure. The represented pool is not pruned in
this four-step opportunity gate, so no candidate can disappear and be re-proposed.

## Policies and controls

Evaluate exact finite policy improvement at levels d1, d2, and d3, each executed
receding-horizon for the same four-assay budget. Planning integrates over all 27
held-out truth worlds and all three categorical outcomes.

Also evaluate:

- fixed-support d1 with no structural proposal;
- full-compositional-support d1 with every inference particle represented from the
  start; and
- immutable proposal-cache replay of the primary oracle results.

The policy state and proposer input must contain no truth identifier.

## Prospective gate

Use paired held-out truth-cell terminal losses. Practical ties are absolute
differences at most `1e-8`.

All conditions must pass:

1. finite normalized likelihoods, beliefs, and terminal losses;
2. exact source reconstruction and immutable proposal replay;
3. every accepted proposal is one executable atomic edit and contains exactly its
   three non-truth inference particles;
4. no held-out truth particle is ever proposed;
5. d2 mean terminal MSE is at least 5% lower than d1;
6. d3 mean terminal MSE is at least 5% lower than d2;
7. d2 paired wins exceed losses;
8. d3 paired wins exceed losses;
9. d3 beats d1 on at least 21 of 27 truth cells, excluding ties;
10. d2 improves the per-difficulty mean on at least two of three difficulties;
11. d3 improves the per-difficulty mean on all three difficulties;
12. d1 and d2 root actions differ on at least one difficulty;
13. d2 and d3 root actions differ on at least one difficulty;
14. aggregate d2 MSE remains greater than `1e-8`, preventing a numerical-zero
    depth-three opportunity;
15. dynamic d3 is at least 20% better than fixed-support d1; and
16. dynamic d3 is no worse than 1.5 times full-support d1.

Failure closes this exact cohort, versions, action set, seeds, transition, and
thresholds. It may motivate a prospectively new costed-repeat or assay-panel
environment, but no threshold, cell, seed, or structure may be changed to rescue
this result.

## Authority after a pass

A pass authorizes implementation and prospective freezing of a small atomic-edit
semantic gate. That gate must compare residual-aware against history-blind
proposals, test simulated-answer obedience, and include recursive states in which a
generated composition is already represented. No LLM policy endpoint opens until
that separate gate passes.
