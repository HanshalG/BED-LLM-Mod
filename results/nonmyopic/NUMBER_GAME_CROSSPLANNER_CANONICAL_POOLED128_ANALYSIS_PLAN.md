# Number Game Cross-Planner Canonical Pooled-128 Analysis Plan

Date frozen: 2026-07-29, after all four source block outcomes were open.
This is retrospective robustness analysis and cannot rescue or replace any
source study's registered status.

## Question

Does path-dependent depth-three planning beat myopic EIG on the exact
canonical concept bank across both Qwen 3.7 Plus and GPT-5.4 Mini planning
generators?

## Fixed Data And Analysis

- Two independent 32-tree Qwen blocks, hash-bound through their public
  results.
- Two independent 32-tree GPT-5.4 Mini blocks, hash-bound through the
  canonical replay-64 result.
- Verify all 128 tree seeds are disjoint.
- Preserve all four block summaries.
- Pool separately within each planner family and across all four blocks.
- Use a stratified bootstrap that independently resamples 32 trees within
  every source block.
- Report every matched control, Hamming, coverage, ranking, and depth-two
  diagnostic.

Descriptive robustness checks require every block to improve Brier over
myopic by at least 8%, have an interval below zero, and win at least 20
trees; both model families and the 128-tree pool must improve by at least
10% with intervals below zero; the pooled comparison must have at least 80
wins.

Any pooled depth-three versus depth-two result remains descriptive because
the already-open block effects are heterogeneous. Model calls and cost are
zero.
