# Number Game Qwen First-Link Failed-Prefix-36 Analysis Plan

Date frozen: 2026-07-29, after the 64-tree run failed during tree 37 and
before reconstructing or scoring any canonical endpoint for its 36 complete
trees.

This is an underpowered zero-call diagnostic. It cannot rescue, replace, or
change the failed 64-tree confirmation.

## Fixed Source

- Private raw checkpoint SHA-256:
  `6c2106ae7a676d0dbd4a7eb8e9b34cea20775e1f8a763132f43476452a1dc2a8`.
- Exactly the first 36 complete trees from seeds `60100..60135`.
- Mechanics-only target seeds `60200..60235`.
- Eight validation supports per tree, seeds `61100..61387`.
- The failed tree 37 and every later planned tree are excluded because they
  are incomplete or absent, not because of any scientific outcome.

The parse failure happened before canonical scoring and exposed no endpoint
metric.

## Frozen Replay

1. Replay the saved initial, 16 first-step, 32 second-step, target, and eight
   validation responses through the original strict parser.
2. Reconstruct the original retained-rejuvenation trees, candidate roots,
   and cross-fitted depth-two/depth-three risk tables.
3. Score every root once on all 33 exact Tenenbaum--Griffiths concepts over
   `0..100`, with equal concept and tree weight.
4. Verify that replay makes zero model calls and that every saved response
   batch is consumed exactly once.

## Fixed Analysis

On changed depth-three versus myopic roots, report:

- changed-root count;
- mean simulated and realized Brier advantage;
- wins/ties/losses;
- simulated-to-realized Spearman;
- 20,000 tree-bootstrap intervals for realized advantage and Spearman,
  seed `61800`.

Also report the exact policy-level depth-three comparisons with myopic EIG,
cross-fitted depth two, fixed-support depth three, PTS, and random root.

There are no pass thresholds. The original 64-tree gates may be shown only
as historical context and are not evaluated on this prefix. Positive,
negative, or imprecise results are reported unchanged. No continuation,
seed substitution, response repair, or paid call is allowed.
