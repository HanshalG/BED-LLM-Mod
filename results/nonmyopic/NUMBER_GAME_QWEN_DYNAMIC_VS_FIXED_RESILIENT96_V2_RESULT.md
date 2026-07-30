# Number Game Qwen Dynamic-vs-Fixed Resilient-96 V2 Result

Date: 2026-07-30

## Status

**Gated null overall; all preregistered scientific gates pass.** The fresh
powered comparison confirms a Brier advantage from path-dependent LLM support
regeneration over compute-matched fixed-support depth-three planning. Three of
96 trees miss the separately frozen all-branches support floor, so the
composite protocol remains `gated_null`.

## Primary Dynamic-Support Result

Across 96 entirely fresh trees and the exact 33-concept canonical endpoint:

| Metric | Dynamic-support d3 | Fixed-support d3 | Difference |
|---|---:|---:|---:|
| Brier | 0.1033323 | 0.1070127 | -0.0036803 |
| Hamming | 0.0235586 | 0.0228367 | +0.0007219 |

Dynamic support:

- reduces mean Brier by `3.439%`, above the frozen `3%` gate;
- has paired 95% tree-bootstrap Brier difference
  `[-0.0064155, -0.0009591]`;
- changes the selected root on `72/96` trees;
- records `50/24/22` Brier wins/ties/losses.

All four frozen dynamic-support gates pass. Hamming worsens `3.16%`, and the
coverage difference is `-0.347` percentage points with an interval crossing
zero. Those metrics were diagnostics and do not alter the Brier primary.

## Co-Required Policy Efficacy

Dynamic depth three also passes all preregistered gates against myopic EIG:

- Brier `0.1033323` versus `0.1166878`;
- `11.445%` relative reduction;
- paired 95% interval `[-0.0163675, -0.0103286]`;
- `72/6/18` wins/ties/losses;
- Hamming improves `13.63%`.

This independently preserves the paper's robust non-myopic-over-myopic policy
result.

## Mechanics Boundary

Exactly three trees miss the all-branches support floor:

| Tree seed | Minimum first support | Minimum second support | Required |
|---:|---:|---:|---:|
| 80034 | 11 | 25 | 12 / 8 |
| 80070 | 10 | 19 | 12 / 8 |
| 80086 | 15 | 7 | 12 / 8 |

The other 93 trees pass. Across all trees, median first/second minima are
`24/19`, and all 1,536 validation supports contain at least 20 rules.

The three mechanics-failing trees have mean dynamic-minus-fixed Brier
`-0.000087`; they do not carry the favorable endpoint. Removing the single
most favorable tree from all 96 leaves mean difference `-0.003174` and
`49/24/22` wins/ties/losses. These are post-result diagnostics only and cannot
rescue or reclassify the composite null.

## Serving And Cost

- exactly `11,040` accepted requests and parsed provider draws;
- exactly `11,057` HTTP attempts and `17` transparent provider-error retries;
- zero fallback-seed transitions, reasoning tokens, forced exits, or
  item-salvaged draws;
- every accepted response strict JSON;
- cost `$14.00459672`, below the `$15.75` cap;
- provider-visible remaining balance after completion:
  `$15.404915749`.

The fallback mechanism was available but never invoked: every provider error
resolved within the original identical-seed retry schedule.

## Interpretation

This is fresh preregistered evidence for the causal Brier value of planning
over answer-conditioned LLM belief regeneration. It is not an all-gates
confirmation because the support-floor conjunction fails. The paper should
therefore say:

- the powered scientific dynamic-support primary passes;
- the full protocol remains a mechanics-qualified gated null;
- regeneration improves calibrated predictive Brier, while Hamming and
  coverage do not show corresponding gains.

## Artifact Hashes

- `RESULT.json`:
  `04177da415498d765baa2b460f6d732a5162b118776df4d5019f7b540bfce36e`
- `TREES.json`:
  `8535df7eba5437c564cc6df56816fcee0a6991c3358fdaa6b62c6ca96948da82`
- `TARGETS.json`:
  `2fd09b75bcce734e17f237ced9a49e97e13e290e2f5229b9354ad7a8a1bdafd6`
- Private raw responses:
  `c0a95d6c43ccc95e2a07a275e7800b451d653092f82d33c8dfcd8ddbbed6ac2d`
