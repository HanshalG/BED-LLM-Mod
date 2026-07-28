# Number Game Cross-Fitted Depth-Three Fresh Replication Result

Date completed: 2026-07-28.

Status: **formal conjunctive null; primary monotonic-depth result
independently replicates**.

## Fresh Design And Transport

All 32 GPT-5.4 Mini planning trees, 256 Gemini validation supports, selected
roots, and 512 Gemini endpoint supports are fresh. No tree, root, or response
from the prior endpoint-precision study enters this result.

The run completed exactly 2,336 accepted responses in 2,337 HTTP attempts,
with one transport retry, zero provider-error retries, zero reasoning tokens,
and zero forced exits. Reported cost was `$6.8473362`, below the frozen
`$7.60` cap.

Public artifact hashes:

- `RESULT.json`:
  `25e0939164f6806b30481e48a88372f22639f6065096cb59978600509d43a3d8`
- `TREES.json`:
  `197dcfe3cb48eeb9a4b656d0f9b20ef2e0b45ac8d1df0ec4bd9358e07af18802`
- `ENDPOINTS.json`:
  `103626763fe25987541a48adb93ac1dea600f53ecf64b95d1d4dbc2d66efc824`
- private raw responses, retained locally and not committed:
  `a867ba9a402487fea7450866db7bb21d214c499c1bd943165204904190891680`

## Primary Result

| Policy | Mean Brier | Mean Hamming | Candidate Brier gain | Brier wins | Paired Brier difference 95% CI |
|---|---:|---:|---:|---:|---:|
| Cross-fitted depth three | 0.158978 | 0.075879 | - | - | - |
| Cross-fitted depth two | 0.163569 | 0.077998 | 2.81% | 13/32 | [-0.008248, -0.001538] |
| In-sample retained depth three | 0.161905 | 0.080443 | 1.81% | 15/32 | [-0.004817, -0.001254] |
| In-sample depth two | 0.167620 | 0.081168 | 5.16% | 22/32 | [-0.012289, -0.005354] |
| Myopic EIG | 0.173392 | 0.081485 | 8.31% | 27/32 | [-0.018987, -0.010063] |
| Fixed-support depth three | 0.171139 | 0.081365 | 7.11% | 25/32 | [-0.017212, -0.007823] |
| Uniform random | 0.171767 | 0.081592 | 7.45% | 32/32 | [-0.014388, -0.011169] |
| Positive-test strategy | 0.171905 | 0.080798 | 7.52% | 30/32 | [-0.016018, -0.009869] |

Cross-fitted depth three and depth two choose different roots on 16/32 trees.
Depth three wins 13 of those changed-root comparisons and loses three; the
other 16 trees tie because the roots are identical. It improves mean Brier by
`2.8070%`, with the whole-tree interval strictly below zero. Mean Hamming
improves `2.7159%`, and coverage increases `1.5818` percentage points.

Novel targets also improve: candidate-minus-baseline Brier is `-0.004266`,
Hamming is `-0.004288`, and coverage rises `3.0613` points.

## Ranking Fidelity

The fresh validation supports strongly rank the independent 16-draw endpoint:

- depth-three Spearman: `0.9174`, bootstrap `[0.8839, 0.9457]`;
- depth-two Spearman: `0.4479`, bootstrap `[0.3073, 0.5789]`;
- depth-three pairwise concordance: `0.9118`;
- depth-two pairwise concordance: `0.6763`.

This independently reproduces the first-link mechanism and the monotonic-depth
endpoint effect. The LLM remains load-bearing in initial hypothesis
generation, both path-conditioned retained refreshes, and independent prior
sampling for policy risk.

## Formal Gate Status

Twenty-three of 25 frozen gates pass. Two auxiliary gates fail:

1. one of 512 endpoint responses parsed to 14 valid hypotheses, below the
   frozen per-response minimum of 16; its tree still averaged `22.06` valid
   hypotheses per draw and contained 204 novel endpoint hypotheses; and
2. cross-fitting improves `1.808%` over the in-sample retained depth-three
   selector, below the frozen `2%` magnitude threshold, although its
   whole-tree confidence interval is strictly below zero.

The primary depth-three-versus-depth-two effect, win count, Hamming, coverage,
novel-target, ranking, control, transport, and budget gates all pass. The
overall status remains a conjunctive null without threshold repair or response
removal. Scientifically, this is an independent fresh-tree replication of the
paper's monotonic LLM-native depth claim, with two disclosed auxiliary
qualification misses.
