# Number Game Qwen External-Canonical Pooled-64 Result

Run: `number-game-qwen-external-canonical-pooled64-20260729T070712Z`

Status: **retrospective robustness positive**. This zero-call synthesis does
not alter either source cohort's registered `gated_null` status.

## Replicated Effect

Across 64 disjoint fresh Qwen planning trees, with the bootstrap stratified
by the two independent 32-tree cohorts:

- depth-three Brier: `0.1048436`;
- myopic-EIG Brier: `0.1191378`;
- relative reduction: **11.998%**;
- stratified interval: `[-0.019592,-0.009358]`;
- wins/ties/losses: **50/2/12**;
- one-sided exact sign p-value excluding ties: `6.07e-7`.

Both individual cohorts independently exceeded 8% Brier reduction, had
whole-tree intervals below zero, and won at least 20/32 trees. Every
descriptive robustness check passes.

## Controls And Boundary

- Fixed-support depth three: `4.493%` Brier reduction,
  interval `[-0.009262,-0.000805]`.
- Positive-test strategy: `6.468%`,
  interval `[-0.011315,-0.003217]`.
- Uniform random candidate root: `8.276%`,
  interval `[-0.012093,-0.006728]`.
- Cross-fitted depth two: `2.563%`, but interval
  `[-0.005719,+0.000164]`; monotonic depth remains unsupported.
- Mean rank correlation: `0.305` for depth three versus `0.185` for depth
  two.
- Pooled Hamming is directionally better, but its interval crosses zero;
  exact coverage is also imprecise.

## Provenance

The analysis hash-binds the two source `RESULT.json` artifacts:

- confirmation V1:
  `370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b`;
- replication V2:
  `a03c5a6f6e01af403ce27ef7984e29176bf0aa5f40caeae2bb6d213ce8c5dc83`.

It makes zero model calls and costs `$0`. Both source composite nulls and
their retry counts remain unchanged.

Public `RESULT.json` SHA-256:
`1fca68eb22440ea91c6f486aa730d2e46a4600e0a55894e1941f7db42c121afe`.
