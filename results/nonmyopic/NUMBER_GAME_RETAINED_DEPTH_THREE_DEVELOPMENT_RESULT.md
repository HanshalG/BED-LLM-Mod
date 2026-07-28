# Number Game Retained Depth-Three Development Result

Date: 2026-07-28

This is a post-hoc, zero-call method development over the open eight-tree
depth-three V2 dataset. It is not a confirmation result.

## Repair

The failed generated-only implementation discarded a valid parent support at
the second refresh and replaced it with one new 24-rule LLM sample. Retained
rejuvenation instead:

1. filters the parent support by the second observation;
2. keeps all consistent parent hypotheses;
3. adds every valid newly generated hypothesis; and
4. deduplicates by executable extension.

The LLM still creates every first-step support and proposes new hypotheses at
the second step. Retention prevents a noisy rejuvenation draw from deleting the
entire belief state.

Finite-particle root selection uses a development-derived risk set. It finds
the minimum simulated terminal Brier risk, retains roots within `0.005`, then
minimizes simulated Hamming and maximizes simulated exact-extension coverage.
The tolerance is now frozen for fresh evaluation.

## Open Development Result

| Control | Candidate Brier | Control Brier | Relative gain | Brier wins | Hamming difference | Coverage difference |
|---|---:|---:|---:|---:|---:|---:|
| depth-two predictive risk | 0.17100 | 0.17593 | 2.80% | 4/8 | -0.00627 | +0.03286 |
| parent-only depth three | 0.17100 | 0.18052 | 5.27% | 5/8 | -0.00881 | +0.00059 |
| generated-only depth three | 0.17100 | 0.17421 | 1.84% | 3/8 | -0.00807 | +0.04011 |
| myopic EIG | 0.17100 | 0.18897 | 9.51% | 7/8 | -0.01964 | +0.06235 |
| fixed-support depth three | 0.17100 | 0.19302 | 11.41% | 6/8 | -0.02513 | +0.06852 |
| exact uniform random | 0.17100 | 0.18962 | 9.82% | 8/8 | -0.01997 | +0.04308 |
| positive-test strategy | 0.17100 | 0.18754 | 8.82% | 8/8 | -0.01372 | +0.04447 |

Candidate roots differ from depth two on 5/8 trees, parent-only on 5/8,
generated-only on 4/8, and myopic on 8/8. On targets absent from initial
support, candidate-minus-depth-two differences remain favorable: `-0.00555`
Brier, `-0.01395` Hamming, and `+0.01141` coverage.

Across the eight roots within each tree, retained-support simulated risk ranks
independent-target Brier at mean Spearman `0.673`
(`[0.506, 0.821]`) and pairwise concordance `0.754`
(`[0.661, 0.839]`).

## Mechanics

Generated-only second supports have minimum size `0`; retained rejuvenation
raises every tree's minimum to `7` or more. Mean merged support size ranges
from `19.19` to `21.53`. Each branch adds an average of `6.28` to `7.66`
consistent parent extensions not already present in its generated sample.

This result licenses one fresh six-tree confirmation. The fresh protocol and
all thresholds are frozen separately before responses.

Public result SHA-256:
`fd5d0587022461805d3311468949f3a40309f90c13f14569d1000596981bd078`
