# Number Game Qwen Planner 31-Tree Replay Preregistration

Date frozen: 2026-07-28, after the immutable formal serving failure and before
replaying any endpoint.

## Scope

The failed Qwen-planner formal run checkpointed 31 complete trees before one
final-tree Gemini endpoint response truncated. This audit replays exactly
those 31 complete raw trees with zero model calls.

It is post-hoc development evidence only. It cannot repair, resume, relabel,
or replace the failed 32-tree formal run.

Frozen sources:

- private raw checkpoint SHA-256
  `bed953cb8163f98fa046c42a713e6556571047a73bff0a9293696636c03d7fc1`;
- complete formal run log SHA-256
  `48523bcf77b5ebaf653cefa781e90ec93a502d65eaa814af9216a94356c3bd4e`.

## Replay

For each of the first 31 frozen seeds:

- replay the exact Qwen initial, first-refresh, and second-refresh responses;
- retain consistent parent hypotheses at both refreshes;
- replay the exact Gemini target, eight validation, and fifteen additional
  endpoint responses;
- recompute cross-fitted depth-three and depth-two roots;
- evaluate the same three-query policies on all sixteen endpoint draws;
- report all existing Brier, Hamming, coverage, novel-target, control, and
  rank-fidelity metrics.

No text normalization, response replacement, new generation, root repair,
threshold tuning, or use of the incomplete 32nd tree is allowed.

## Spend Decision Gate

A separately frozen full replication with another independent endpoint
provider is justified only if all original scientific gates pass on the 31
complete trees:

- depth-three and depth-two roots differ on at least 12 trees;
- depth three improves Brier by at least 1%, its whole-tree 95% interval is
  below zero, and it wins at least 12 trees;
- Hamming and coverage do not regress;
- novel-target Brier, Hamming, and coverage do not regress;
- depth three beats myopic and fixed-support depth three by at least 5%, and
  PTS by at least 3%, with all intervals below zero; and
- depth-three Spearman rank fidelity is at least 0.7 and exceeds depth two by
  at least 0.15.

If any gate fails, do not buy another Qwen-planner replication. Report the
development result and close this route. If every gate passes, the next study
must use fresh planning trees and a separately frozen independent target
provider; these 31 trees remain development-only.
