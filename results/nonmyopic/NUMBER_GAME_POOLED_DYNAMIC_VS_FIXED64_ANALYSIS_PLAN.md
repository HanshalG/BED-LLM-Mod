# Number Game Pooled Dynamic-vs-Fixed Support-64 Analysis Plan

Date frozen: 2026-07-29, after the pooled policy synthesis and before
computing this cross-cohort comparison.

## Question

Does the full depth-three planner with path-dependent LLM support
regeneration outperform the same depth-three lookahead restricted to the
initial fixed LLM support?

This is broader than the second-refresh ablation: the candidate regenerates
and retains support after both simulated observations, while the baseline
filters one fixed initial support. Both use the same Qwen-generated initial
belief, candidate roots, exact transition logic, eight independent Gemini
validation supports, three-query horizon, and exact 33-concept endpoint.

## Fixed Sources

Bind the two disjoint pooled-Qwen 32-tree results:

- cohort one `RESULT.json` SHA-256
  `cef6ded08050e6df07de62bbbf548635f229a8bf73fac149b4d0bd960f7b0253`;
- cohort two `RESULT.json` SHA-256
  `71281c483297e7ee4cc011d0ba795dabd28ec2d8598366771e11afee2af2a459`.

Use the already stored `crossfit_depth_three` and
`fixed_support_depth_three` selected-policy endpoints. No root, target, tree,
support, or endpoint may be recomputed under a different rule.

## Frozen Analysis

Use 20,000 cohort-stratified bootstrap replicates with seed `64700`,
resampling 32 trees independently within each cohort.

Report pooled and per-cohort:

- candidate and baseline Brier, mean paired difference, relative reduction,
  interval, and wins/ties/losses;
- Hamming and exact-target coverage differences with intervals;
- cohort-one minus cohort-two effect contrasts.

Call the retrospective comparison positive only if:

- both source mean Brier differences are below zero;
- pooled relative Brier reduction is at least `3%`;
- the pooled paired Brier interval is below zero;
- pooled Brier wins are at least `28/64`.

Hamming and coverage are corroborating and may be null or adverse.

## Scope

This is a zero-call, zero-cost retrospective synthesis. It cannot rescue or
reclassify either source and does not replace a fresh dynamic-vs-fixed
primary. A full pass authorizes designing one separately preregistered fresh
confirmation with more validation draws; it does not authorize changing the
semantic support generator after seeing this result.

There is one bootstrap seed and no subset, weighting, transform, threshold,
endpoint, or model sweep.
