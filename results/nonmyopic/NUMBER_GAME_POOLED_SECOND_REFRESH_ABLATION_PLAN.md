# Number Game Pooled Second-Refresh Ablation Plan

Date frozen: 2026-07-29, after the fresh pooled result was known and before
computing any alternative support-policy root or exact endpoint comparison.

This is a retrospective zero-call mechanism analysis. It cannot rescue or
reclassify the source `gated_null`.

## Question

Does adding the LLM's newly regenerated second-step hypotheses improve root
selection beyond merely filtering and retaining the parent support?

## Fixed Replay

For each of the 32 hash-bound fresh pooled trees, use the same:

- eight candidate roots;
- retained first-step supports and resulting second queries;
- eight cross-fit validation supports;
- exact 33-concept per-root endpoint outcomes.

Recompute depth-three root selection with exactly three second-step supports:

1. `merged_retained_generated`: the deployed union of consistent parent
   hypotheses and new LLM generations;
2. `parent_only`: consistent parent hypotheses, excluding every new
   second-step generation;
3. `generated_only`: new second-step generations, excluding retained parents.

The first comparison is primary and isolates second-step regeneration.
Generated-only is a diagnostic for the complementary value of retention.
Queries, validation draws, target bank, first-step support, and endpoint
evaluation remain fixed.

## Frozen Analysis

Report selected-root differences, exact endpoint Brier, relative reduction,
wins/ties/losses, and a 20,000-sample paired tree-bootstrap interval using
seed `63800`.

The merged-support mechanism is positive against parent-only only if:

- roots differ on at least 12/32 trees;
- relative Brier reduction is at least 2%;
- the paired Brier-difference interval is below zero;
- wins minus losses is at least eight.

No alternate support subset, tree subset, threshold, endpoint, or query
reconstruction will be tried. Model calls and cost are zero.
