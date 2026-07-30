# Number Game Dynamic-Support Mechanism-96 Analysis Plan

Date frozen: 2026-07-30

## Purpose

This is a zero-call retrospective mechanism audit of the completed fresh
96-tree Qwen dynamic-support study. The study's endpoint and formal status are
already known, so this analysis cannot rescue, relabel, or replace its
preregistered result.

The audit asks whether answer-conditioned LLM support regeneration improves
the first planning link in the way required by the paper's mechanism claim:
does the dynamic-support simulator rank candidate first queries better than a
compute-matched fixed-support simulator, and do its changed selections realize
lower exact-canonical Brier risk?

## Frozen Sources

- `RESULT.json` SHA256:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- `TREES.json` SHA256:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- source tree count: `96`
- candidate roots per tree: `8`
- endpoint: the same exact 33-concept canonical bank already recorded in the
  source result

The analysis must fail before producing output if either source hash, tree
count, tree seed, candidate-root set, or selected-root record differs.

## Frozen Analyses

### Candidate-Ranking Fidelity

For every tree, recompute the fixed-support depth-three risk map from the
initial support. Compare it with:

- the stored cross-fitted dynamic-support depth-three risk map; and
- stored exact-canonical per-root endpoint Brier.

Report tree-mean Spearman correlation and pairwise concordance for dynamic and
fixed risk, together with paired tree-bootstrap intervals for the
dynamic-minus-fixed differences.

### Changed-Root First Link

For every tree where dynamic and fixed support select different roots, report:

- dynamic predicted advantage:
  `dynamic_risk[fixed_root] - dynamic_risk[dynamic_root]`;
- fixed counter-advantage:
  `fixed_risk[dynamic_root] - fixed_risk[fixed_root]`;
- score-reversal margin: the sum of those two nonnegative quantities;
- realized advantage:
  `endpoint_brier[fixed_root] - endpoint_brier[dynamic_root]`;
- wins, ties, and losses;
- Spearman correlations from each simulated margin to realized advantage.

Report paired tree-bootstrap intervals for mean realized advantage and both
margin-to-realized correlations.

### Realized Candidate-Set Regret

For each tree, define oracle candidate risk as the minimum exact-canonical
per-root Brier among the same eight candidate roots. Report dynamic and fixed
selected-root regret relative to that oracle, their paired difference, and a
tree-bootstrap interval. This is a diagnostic oracle, not a deployable policy.

### Support-Regeneration Descriptives

Using extension hashes only:

- first-refresh novelty is the number and fraction of generated hypotheses not
  present in the answer-consistent filtered initial support;
- second-refresh novelty is the number and fraction of generated hypotheses
  not present in the answer-consistent filtered first-stage retained support.

Report all-branch summaries and summaries for branches under the dynamic- and
fixed-selected roots. Correlate dynamic-minus-fixed selected-root novelty with
realized advantage on changed-root trees. These novelty correlations are
descriptive and may not be used as a positive mechanism gate.

## Inference

- Bootstrap seed: `86000`.
- Bootstrap samples: `20000`.
- Resampling unit: one complete tree, with 96 draws per replicate.
- Intervals: percentile 95% intervals.
- Ties use tolerance `1e-15`.
- No multiplicity-adjusted confirmatory claim is made.

The first-link evidence is called directionally coherent only if:

1. changed-root mean realized advantage is positive with an interval above
   zero;
2. dynamic candidate-ranking fidelity exceeds fixed-support fidelity on at
   least one of Spearman or concordance without reversing on the other; and
3. dynamic selected-root regret is lower with an interval below zero.

Score-to-realized correlations and novelty analyses explain strength or
failure of the link but cannot override those three conditions.

## Output

The implementation will write one public `RESULT.json` under:

`results/nonmyopic/number_game_dynamic_support_mechanism96/`

It makes zero model calls and costs `$0`.
