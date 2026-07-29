# Number Game Cross-Planner Canonical Pooled-128 Result

Run: `number-game-crossplanner-canonical-pooled128-20260729T072206Z`

Status: **retrospective cross-planner robustness positive**. This synthesis
does not alter any source study's registered status.

## Main Result

Across four disjoint 32-tree blocks and two planning-generator families:

- depth-three Brier: `0.1064787`;
- myopic-EIG Brier: `0.1216563`;
- relative reduction: **12.476%**;
- four-block stratified interval: `[-0.019190,-0.011322]`;
- wins/ties/losses: **95/5/28**;
- one-sided exact sign p-value excluding ties: `5.08e-10`;
- mean Hamming difference: `-0.00454`, interval
  `[-0.00717,-0.00187]`.

Every 32-tree block independently exceeds 8% Brier reduction, has a
whole-tree interval below zero, and wins at least 20 trees.

## Planner Families

- Qwen 3.7 Plus: `12.00%`, interval `[-0.01972,-0.00934]`,
  50/2/12 wins/ties/losses.
- GPT-5.4 Mini: `12.93%`, interval `[-0.02210,-0.01041]`,
  45/3/16.

Both family-level gains exceed 10% with intervals below zero.

## Controls And Depth Boundary

- Fixed-support depth three: `5.851%` gain,
  interval `[-0.00994,-0.00336]`.
- Positive-test strategy: `7.373%`,
  interval `[-0.01122,-0.00576]`.
- Uniform random candidate root: `8.909%`,
  interval `[-0.01232,-0.00845]`.
- Mean exact-bank rank correlation: `0.404` for depth three versus `0.266`
  for depth two.
- Cross-fitted depth two: only `1.817%`, interval
  `[-0.00418,+0.00030]`, 44/52/32 wins/ties/losses.

The cross-planner evidence robustly supports path-dependent non-myopic
selection over myopic EIG. It does not support monotonic improvement from
depth two to depth three.

## Scope

This is a retrospective synthesis of already-open, hash-bound policy blocks.
The bootstrap independently resamples 32 trees inside each of the four
blocks. It makes zero model calls and costs `$0`.

Public `RESULT.json` SHA-256:
`42c189bbf4d0039b78c5fbc7d48774169744975b151c598f2619b9cb73a3e2f3`.
