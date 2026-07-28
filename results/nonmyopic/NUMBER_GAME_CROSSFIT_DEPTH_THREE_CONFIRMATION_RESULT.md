# Number Game Cross-Fitted Depth-Three Confirmation Result

Date completed: 2026-07-28.

Status: **gated null with a directional depth-three effect**.

## Transport

The fresh 32-tree run completed exactly 1,856 accepted requests in 1,856 HTTP
attempts. There were zero retries, provider errors, reasoning tokens, or
forced exits. Reported cost was `$5.7869802`, below the frozen `$6.50` cap.

Public artifact hashes:

- `RESULT.json`:
  `1081da1e8381b88f7cd3fcd905b5ed539047863bcc087d686e509ed8f82d794a`
- `TREES.json`:
  `cf239683033be3fbfed6a449f2aaa1ad1bcbe57d935ca21cf43e46ba938ea7f9`
- private raw responses, retained locally and not committed:
  `79d4632033df6969279722b2ed08300a52d18c2406b7e98a46fbc84d9eb46c6f`

## Primary Result

| Policy | Mean Brier | Mean Hamming | Candidate Brier gain | Brier wins | Paired Brier difference 95% CI |
|---|---:|---:|---:|---:|---:|
| Cross-fitted depth three | 0.155900 | 0.068244 | - | - | - |
| Cross-fitted depth two | 0.158345 | 0.070834 | 1.54% | 13/32 | [-0.005115, 0.000214] |
| In-sample retained depth three | 0.162508 | 0.074659 | 4.07% | 18/32 | [-0.010523, -0.003243] |
| In-sample depth two | 0.161333 | 0.074799 | 3.37% | 20/32 | [-0.008110, -0.002845] |
| Myopic EIG | 0.172307 | 0.077454 | 9.52% | 30/32 | [-0.020525, -0.012639] |
| Fixed-support depth three | 0.167074 | 0.075887 | 6.69% | 22/32 | [-0.017111, -0.005850] |
| Uniform random | 0.170763 | 0.076122 | 8.70% | 31/32 | [-0.017574, -0.012201] |
| Positive-test strategy | 0.167904 | 0.072498 | 7.15% | 30/32 | [-0.016135, -0.008258] |

Cross-fitted depth three changed 19/32 cross-fitted depth-two roots and met
the frozen `1.5%` effect threshold with 13 wins. It also improved mean Hamming
by `3.66%`, had exactly equal mean coverage to numerical precision, and
improved novel-target Brier, Hamming, and coverage. The Brier interval
nevertheless crossed zero by `0.000214`, so the primary conjunction fails.

One endpoint support contained seven hypotheses novel to its initial support,
below the frozen minimum of eight. All initial, target, and validation support
sizes otherwise passed; every retained first and second branch passed its
minimum.

## Ranking And Mechanism

Independent validation substantially repaired root scoring:

- depth-three source-to-endpoint Spearman: `0.7708`, bootstrap
  `[0.7046, 0.8318]`;
- depth-two Spearman: `0.5268`, bootstrap `[0.4278, 0.6176]`;
- depth-three pairwise concordance: `0.8237`;
- depth-two pairwise concordance: `0.7165`.

Cross-fitting also beat the original in-sample depth-three selector by
`4.07%` Brier with a confidence interval below zero, 18 wins, an `8.59%`
Hamming gain, and a one-point coverage gain. This confirms the proposed
finite-support argmin diagnosis even though the incremental depth comparison
remains underpowered/noisy.

Across the 19 trees where the cross-fitted depth policies selected different
roots, depth three won 13 and lost six; 13 additional trees selected the same
root and tied exactly. The mean paired difference was `-0.002445` with sample
standard deviation `0.007781`, implying a normal-approximation sample size of
about 39 trees for a zero-crossing test at the observed effect.

## Interpretation

This run is not a formal monotonic-depth positive. It is stronger evidence
for the first link: independent LLM prior draws rank depth-three roots much
more faithfully than in-sample source particles, and the selected policy
dominates myopic and in-sample controls.

The residual uncertainty is concentrated in the held-out endpoint estimate,
which uses one Gemini support of roughly 20--23 hypotheses per tree. The next
measurement fixes all 32 policy trees and selected roots, excludes this opened
endpoint from its primary statistic, and averages 16 entirely fresh endpoint
draws per tree. That is a separately preregistered endpoint-precision
extension, not a repair or reinterpretation of this gated null.
