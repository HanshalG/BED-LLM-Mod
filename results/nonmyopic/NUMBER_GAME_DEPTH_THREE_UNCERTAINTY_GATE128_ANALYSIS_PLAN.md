# Number Game Depth-Three Uncertainty Gate-128 Analysis Plan

Date frozen: 2026-07-29, after the four source outcomes and pooled first-link
analysis were open, but before computing any uncertainty-gated policy endpoint.

This is a retrospective, target-blind calibration analysis. It cannot rescue
or replace any source study's registered status.

## Question

Does the depth-three versus depth-two null arise because depth three deploys
root changes whose simulated advantage is not stable across its eight
independent validation supports?

## Fixed Sources

- four disjoint 32-tree Number Game blocks;
- two Qwen 3.7 Plus and two GPT-5.4 Mini planning-generator blocks;
- eight already-generated validation supports per tree;
- the same 33 exact Tenenbaum--Griffiths endpoint concepts per tree;
- the already-fixed cross-fitted depth-two and depth-three roots.

All source `RESULT.json` and `TREES.json` files are hash-pinned in the
analysis code.

## Fixed Gate

For every tree where the depth-two and depth-three roots differ:

1. recompute the depth-three policy Brier risk of both roots separately on
   each of the eight validation supports;
2. define per-draw advantage as depth-two-root risk minus depth-three-root
   risk, so positive favors depth three;
3. choose depth three only when at least six of eight advantages are strictly
   positive and their sample mean minus one standard error is strictly
   positive;
4. otherwise fall back to the existing depth-two root.

When the two original roots agree, keep that shared root. No endpoint value,
concept label, family identity, or source-study outcome enters the gate.
There is one fixed rule and no threshold sweep.

## Fixed Analysis

Report:

- depth-three acceptances and depth-two fallbacks;
- candidate-versus-depth-two exact-canonical Brier mean, relative reduction,
  wins/ties/losses, and 20,000-sample block-stratified interval;
- the same comparison by planner family and source block;
- preservation of the gated policy's advantage over myopic EIG;
- predicted-versus-realized advantage correlation among accepted root
  changes.

Bootstrap seed is `59000`; every replicate resamples 32 trees independently
inside each included source block.

There are no confirmatory pass thresholds because the endpoint families and
aggregate depth null were already observed. A positive result is
retrospective evidence for uncertainty-aware deployment, not a new
monotonic-depth confirmation. A null or adverse result closes this fixed
gate without threshold tuning.

Model calls and cost are zero.
