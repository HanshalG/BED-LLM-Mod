# Number Game Pooled Dynamic-vs-Fixed Support-64 Result

Run: `number-game-pooled-dynamic-vs-fixed64-20260729T114252Z`

Status: **retrospective dynamic-support positive**. All four frozen Brier
gates pass.

## Primary Comparison

Across two disjoint 32-tree pooled-Qwen cohorts, the full path-dependent
depth-three planner is compared with depth-three planning on the fixed
initial LLM support:

- dynamic-support Brier: `0.103632`;
- fixed-support Brier: `0.107712`;
- relative Brier reduction: `3.79%`;
- stratified bootstrap difference:
  `[-0.0078516, -0.0004650]`;
- wins/ties/losses: `30/12/22`.

The source mean differences are `-0.0049767` and `-0.0031841`, so both
cohorts are directionally positive. Their cohort-one-minus-two contrast is
`-0.0017926`, with interval `[-0.0091847, 0.0056413]`; there is no evidence
that the Brier effect differs between cohorts.

The result passes every frozen requirement: both source effects are below
zero, pooled reduction exceeds `3%`, the interval is below zero, and there
are at least `28/64` wins.

## Corroborating Metrics

Hamming is directionally adverse by `5.52%`, with dynamic-minus-fixed mean
`0.0012704`, but its interval `[-0.0011767, 0.0036566]` crosses zero.

Exact-target coverage is nearly unchanged: dynamic support is `0.095`
percentage points lower, with interval `[-0.0217803, 0.0208333]`. The source
coverage directions differ.

## Interpretation

This isolates a broader path-dependent belief effect than the failed
second-refresh replication. Candidate and baseline have the same initial
LLM support, root candidates, depth-three horizon, validation targets, and
exact endpoint. The candidate alone asks the LLM to regenerate support after
simulated observations and retains compatible parent hypotheses. Its lower
exact posterior-predictive Brier therefore cannot be attributed merely to
deeper planning on a fixed hypothesis set.

The analysis is retrospective and cannot reclassify either source. It
authorizes a separately preregistered fresh confirmation with increased
validation draws, where dynamic versus fixed support is the primary.

Model calls: `0`. Cost: `$0`.

Public `RESULT.json` SHA-256:
`c71fa24d25650065a97ccb5caf69e1a5dd3b1522f0c4f7d483f0aa98fa961290`.
