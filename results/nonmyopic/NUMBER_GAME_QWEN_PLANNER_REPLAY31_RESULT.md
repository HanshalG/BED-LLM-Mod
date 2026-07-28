# Number Game Qwen Planner 31-Tree Replay Result

Date completed: 2026-07-28.

Status: **post-hoc development gate failed; formal failure unchanged**.

## Scope

This zero-call audit replays the 31 complete trees checkpointed before the
Qwen-planner formal run's final-tree Gemini endpoint truncation. It excludes
the incomplete 32nd tree and uses the exact frozen planning, validation, and
16-draw endpoint responses.

The replay does not repair, resume, or relabel the failed formal run. It was
prospectively allowed only as a spend-decision gate for a new independent
provider replication.

## Primary Proper-Score Result

| Policy | Mean Brier | Mean Hamming |
|---|---:|---:|
| Cross-fitted depth three | 0.164628 | 0.065279 |
| Equally cross-fitted depth two | 0.174717 | 0.063897 |

Depth three improves posterior-predictive Brier by `5.7745%`. The mean paired
difference is `-0.010089`, with whole-tree bootstrap 95% CI
`[-0.014129, -0.006345]`. It wins 21/31 trees and chooses a different root on
22/31.

The depth-three source risk ranks independent endpoint Brier very faithfully:

- depth-three Spearman: `0.9293`, bootstrap `[0.9078, 0.9493]`;
- depth-two Spearman: `0.5131`, bootstrap `[0.3902, 0.6283]`;
- depth-three concordance: `0.9251`;
- depth-two concordance: `0.7039`.

## Frozen Gate Misses

Three conjunction gates fail:

- Hamming worsens by `2.1623%`: candidate `0.065279` versus depth-two
  `0.063897`; the difference interval `[-0.004026, 0.006628]` crosses zero.
- Exact-extension coverage falls `1.3492` percentage points.
- On novel targets, Brier improves by `0.007498`, but Hamming worsens by
  `0.002810` and coverage falls `2.9161` points.

One replayed tree also contains 125 endpoint hypotheses novel to its initial
support, below the original formal mechanical threshold of 128. This was not
part of the separately frozen scientific spend gate, but it would have been
an additional formal conjunction miss had the serving run completed.

Because the spend gate required every scientific condition, no new
Qwen-planner replication is authorized.

## Controls

The cross-fitted depth-three policy improves Brier over:

- myopic EIG by `10.4407%`, CI `[-0.024241, -0.014330]`;
- fixed-support depth three by `8.5194%`, CI
  `[-0.019514, -0.011263]`;
- PTS by `12.6282%`, CI `[-0.027673, -0.020047]`;
- in-sample retained depth three by `2.4365%`, CI
  `[-0.006295, -0.002133]`; and
- in-sample predictive-risk depth two by `6.4044%`, CI
  `[-0.014898, -0.007880]`.

All 31 initial supports contain at least 21 valid rules, retained first
branches at least 10, retained second branches at least seven, validation
supports at least 20, and endpoint supports at least 20.

## Interpretation

The third planning model family reproduces the intended proper-score
mechanism: independent Monte Carlo support makes depth-three root risk highly
predictive of held-out posterior-predictive Brier, and non-myopic depth three
beats depth two and all controls by large margins.

However, Brier, Hamming, and exact support coverage are different losses.
Qwen's depth-three selector improves calibrated ensemble prediction while
sometimes losing the closest exact rule. The frozen all-metric conjunction
therefore fails. This is useful planner-family evidence for the primary
proper score, but not a formal all-gates replication.

Accounting and artifacts:

- model calls during replay: `0`;
- original accepted responses replayed: `2,263`;
- original response cost represented: `$3.35121368`;
- public `RESULT.json` SHA-256:
  `a645c92126c9b5fd69e28532fb621028781dcf19d8ec890cd7758e6d1df710f2`;
- public `TREES.json` SHA-256:
  `7abae080ec286b1991a941bf1de0ceb4ea0d3997a648a43ae4bd6ceeaa6c151a`;
- public `ENDPOINTS.json` SHA-256:
  `aa93e20f8d1b0cc3a7bfdd1ec255f0b429a7d3fa98a67ee00098632b81540a9b`.
