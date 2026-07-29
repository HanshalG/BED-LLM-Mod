# Number Game Pooled Replication Synthesis-64 Analysis Plan

Date frozen: 2026-07-29, after both disjoint pooled-Qwen 32-tree cohorts
completed and before any cross-cohort aggregate was computed.

## Fixed Sources

The analysis binds:

1. the first pooled-Qwen policy result, public `RESULT.json` SHA-256
   `cef6ded08050e6df07de62bbbf548635f229a8bf73fac149b4d0bd960f7b0253`,
   and its matched second-refresh ablation, SHA-256
   `77fa26cc9d4599804521081bcb201c6d9cf1dfdc0d56d5f51487314d8e70f95c`;
2. the independent fresh prospective second-refresh confirmation, public
   `RESULT.json` SHA-256
   `71281c483297e7ee4cc011d0ba795dabd28ec2d8598366771e11afee2af2a459`.

Each cohort has 32 disjoint trees, two independently seeded Qwen planning
draws per history, eight Gemini validation draws per tree, and the exact same
33-concept endpoint. No model response, tree, target, root, or endpoint may be
added, removed, or recomputed under a different rule.

## Frozen Analyses

Use 20,000 cohort-stratified bootstrap replicates with seed `64600`, sampling
32 trees with replacement independently within each cohort.

Report:

- pooled full depth-three versus myopic Brier, Hamming, coverage, wins/ties/
  losses, relative reductions, and stratified bootstrap intervals;
- pooled merged versus parent-only second-support Brier, root differences,
  wins/ties/losses, relative reduction, and stratified interval;
- pooled merged versus generated-only second-support metrics;
- each source effect unchanged;
- the cohort-one minus cohort-two mean Brier-difference contrast and its
  stratified bootstrap interval for both support ablations.

The policy effect is descriptively robust only if both source cohorts retain
their preregistered depth-three-versus-myopic efficacy passes and the pooled
Brier interval is below zero.

The prospective second-refresh result remains the binding replication test.
No pooled estimate, heterogeneity estimate, or generated-only comparison may
rescue or reclassify its `gated_null` mechanism primary. The retrospective
source also remains retrospective.

## Accounting

This is a zero-call, zero-cost analysis over hash-bound public artifacts.
There is one bootstrap seed, no threshold sweep, no alternate subset,
weighting, endpoint, transform, or support definition. The result will be
reported once, including null or adverse findings.
