# Number Game Fresh Qwen External-Canonical Confirmation Result

## Decision

**Composite gated null because of two recovered provider-error retries. All
seven prespecified scientific primary gates and all six diagnostics passed.**

This is the sole preregistered fresh-tree run. It is not repaired, resumed,
relabelled, or rerun.

## Scientific Primary

| Comparison | D3 Brier | Baseline Brier | Relative gain | 95% whole-tree interval | D3 tree wins |
|---|---:|---:|---:|---:|---:|
| Cross-fitted depth two | 0.104564 | 0.109634 | 4.62% | [-0.009545, -0.001040] | 14/32 |
| Myopic EIG | 0.104564 | 0.117981 | 11.37% | [-0.019039, -0.007844] | 25/32 |

All frozen efficacy gates pass:

- depth-three and depth-two roots differ on 19/32 trees;
- the depth-three gain over depth two exceeds 1%, its whole-tree interval is
  below zero, and it wins at least 12 trees;
- the depth-three gain over myopic exceeds 5%, its interval is below zero,
  and it wins at least 16 trees.

This is a fresh planner-family and external-target proper-score result. The
Qwen planner generated all initial and twice answer-conditioned belief
supports. Gemini generated eight independent validation supports per tree for
cross-fitted root selection. Efficacy was scored only on the complete fixed
33-concept Tenenbaum--Griffiths bank.

## Prespecified Diagnostics

All six diagnostic booleans are favorable:

- Hamming improves by 3.20% and exact truth-extension coverage rises by 0.95
  percentage points versus depth two;
- depth three directionally beats fixed-support depth three by 5.14%;
- depth three beats PTS by 6.27%, with CI
  `[-0.013329, -0.000753]`;
- depth three beats uniform random root by 8.36%, with CI
  `[-0.013367, -0.005702]`;
- external-bank rank fidelity is higher for depth three: mean Spearman 0.287
  versus 0.083 for depth two.

Fixed-support depth three is directionally worse, but its interval
`[-0.011900, 0.000283]` narrowly includes zero. PTS Hamming and coverage are
better than the selected depth-three policy even though its Brier is worse;
the claim therefore remains about calibrated posterior prediction.

## Operational Gate

The run accepted exactly 1,856 model responses:

- 1,568 Qwen planning responses;
- 32 generated targets used only for source-tree mechanics;
- 256 Gemini validation responses;
- zero endpoint-generation calls;
- zero reasoning tokens and zero forced exits.

OpenRouter returned two transient provider errors. The already-registered
transport retry path recovered them, producing 1,858 HTTP attempts for 1,856
accepted responses. Retry count 2 is within the frozen cap 8, but the separate
`zero_provider_error_retries` gate is false. Since status required every
mechanics gate plus every scientific primary gate, the public artifact is
correctly `gated_null`.

This operational failure does not change any accepted response or endpoint,
but it prevents calling the run an all-gates preregistered pass. No rerun is
authorized.

## Cost And Provenance

- OpenRouter cost: `$2.3876747`, below the `$4.00` cap.
- Model calls: `1,856` accepted; `1,858` HTTP attempts.
- `RESULT.json` SHA-256:
  `370e1c2923e56fb6a8344558db0a69bd5f86a8b013da7d9452675df380f7f12b`.
- `TARGETS.json` SHA-256:
  `9e788da25b8431f457d044e9f7724bcea77312ca989aaf94b92001a21bf01a44`.
- `TREES.json` SHA-256:
  `39b79f391ae3b613d15794c9dd6c86ef02eb96907fbaa33591b157b2ac19cc63`.
- Private raw checkpoint is retained locally and excluded from the paper
  package and git.
- Authenticated OpenRouter balance after the run: `$15.628982893`; the
  user-reported additional `$40` is still not visible at the credit endpoint.
