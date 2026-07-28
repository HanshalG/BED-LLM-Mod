# Number Game Cross-Fit Endpoint-Precision Result

Date completed: 2026-07-28.

Status: **all preregistered gates pass**.

## Fixed-Policy Design

This study binds the fresh 32-tree cross-fit confirmation by its public
`RESULT.json` and `TREES.json` hashes. It does not regenerate a planning tree
or change any selected root. For each fixed tree, it generates 16 entirely
fresh Gemini 2.5 Flash endpoint supports and averages endpoint metrics across
draws before aggregating over trees. The single endpoint support opened by the
prior gated-null confirmation is excluded from every primary statistic.

The run completed exactly 512 accepted responses in 512 HTTP attempts, with
zero retries, provider errors, reasoning tokens, or forced exits. Reported
cost was `$1.1463158`, below the frozen `$1.50` cap.

Public artifact hashes:

- `RESULT.json`:
  `47e684b11ac5c6340ff4f063ff7a14db8b8d0bd18b184d61f91e315903e4809e`
- `ENDPOINTS.json`:
  `0e6bd789b28a77f2ca47a13015fc80d46686daf14f3a2540caebe8663eff17c0`
- private raw responses, retained locally and not committed:
  `bc64658ffab6142c9db6c95ef64f04299718dff537d90e3bacf663dea8695655`

## Primary Result

| Policy | Mean Brier | Mean Hamming | Candidate Brier gain | Brier wins | Paired Brier difference 95% CI |
|---|---:|---:|---:|---:|---:|
| Cross-fitted depth three | 0.157258 | 0.070823 | - | - | - |
| Cross-fitted depth two | 0.160242 | 0.073173 | 1.86% | 13/32 | [-0.004996, -0.001160] |
| In-sample retained depth three | 0.162964 | 0.074714 | 3.50% | 18/32 | [-0.008397, -0.003255] |
| In-sample depth two | 0.162988 | 0.077110 | 3.52% | 22/32 | [-0.008121, -0.003567] |
| Myopic EIG | 0.172334 | 0.079183 | 8.75% | 30/32 | [-0.018172, -0.012185] |
| Fixed-support depth three | 0.168149 | 0.076634 | 6.48% | 23/32 | [-0.015123, -0.007005] |
| Uniform random | 0.170921 | 0.076917 | 7.99% | 32/32 | [-0.015683, -0.011749] |
| Positive-test strategy | 0.168622 | 0.074463 | 6.74% | 30/32 | [-0.014667, -0.008391] |

Depth three and depth two select different roots on 19/32 fixed trees. Depth
three wins 13 of those comparisons, loses six, and ties on the 13 trees with
the same root. Mean Brier improves `1.8619%`; the whole-tree bootstrap
interval is strictly below zero. Mean Hamming improves `3.2111%`, and exact
support coverage rises `0.2733` percentage points.

Novel endpoint hypotheses improve consistently: candidate-minus-baseline
Brier is `-0.006235`, Hamming is `-0.004213`, and coverage rises `0.7331`
points. Every endpoint response parses to at least 20 valid rules, and every
tree contains at least 148 novel endpoint hypotheses across its 16 draws.

## Ranking Fidelity

Independent eight-draw source risk strongly ranks the high-precision
endpoint:

- depth-three Spearman: `0.9226`, bootstrap `[0.8973, 0.9449]`;
- depth-two Spearman: `0.5677`, bootstrap `[0.4725, 0.6540]`;
- depth-three pairwise concordance: `0.9174`;
- depth-two pairwise concordance: `0.7321`.

This directly supports the first link: independent LLM prior sampling makes
the depth-three simulator's root ordering substantially more faithful than
depth two's. The earlier one-support endpoint was a noisy measurement of that
ordering, not evidence that the ranking disappeared.

## Claim Boundary

This is the first frozen Number Game result in the project where a deeper
non-myopic horizon beats an equally cross-fitted shallower horizon. The LLM is
load-bearing: it generates the open-ended initial support, both
branch-conditioned belief transitions, and independent Monte Carlo prior
draws. Exact code supplies labels, Bayesian updates, and policy evaluation.

The result does not erase the preceding single-endpoint confirmation null.
The decision to increase endpoint precision was made after observing that
null. Scientific protection comes from fixing every policy root, excluding
the opened endpoints, preregistering the draw count and gates, and using 512
fresh responses. A wholly independent planning-tree replication remains the
strongest next robustness check.
