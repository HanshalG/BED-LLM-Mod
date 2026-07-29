# Number Game Pooled Second-Refresh Ablation Result

Run: `number-game-pooled-second-refresh-ablation-20260729T104014Z`

Status: **retrospective second-refresh mechanism positive**.

## Primary Result

The full merged-support planner and parent-only planner use the same 32 fresh
trees, retained first-step supports, second queries, eight cross-fit
validation draws, and exact 33-concept outcomes. They differ only in whether
new LLM-generated second-step hypotheses are included.

Merged support beats parent-only:

- selected roots differ on `22/32` trees;
- merged Brier: `0.104586`;
- parent-only Brier: `0.112588`;
- relative Brier reduction: `7.11%`;
- paired tree-bootstrap difference:
  `[-0.0141984, -0.0023230]`;
- wins/ties/losses: `15/10/7`;
- wins minus losses: `8`.

Every frozen mechanism gate passes.

## Generated-Only Diagnostic

Merged support also improves over generated-only by `2.43%`, but the paired
interval `[-0.0074033, 0.0015938]` crosses zero. Roots differ on `18/32`
trees, with `11/14/7` wins/ties/losses.

This complementary null matters: new generations provide value over
parent-only filtering, but discarding compatible parent hypotheses is not a
reliable improvement. The useful belief update is the union of retained
support and path-dependent LLM regeneration.

## Interpretation

This is the most direct evidence in the project that the LLM's path-dependent
belief dynamics are functionally important. The candidate questions,
trajectories, validation targets, and exact endpoint are fixed; the effect
comes from newly generated hypotheses changing simulated continuation value
and therefore root selection.

The analysis is retrospective and makes zero model calls, so it cannot rescue
the source confirmation's `gated_null` status or establish a prospective
7.11% effect. It does establish a matched mechanism result that cannot be
reduced to extra rollout width, a different target bank, or a different
question trajectory.

Public `RESULT.json` SHA-256:
`77fa26cc9d4599804521081bcb201c6d9cf1dfdc0d56d5f51487314d8e70f95c`.
