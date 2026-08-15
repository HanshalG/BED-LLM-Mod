# ChemBench Adaptive-SMC V3 Result

Date: 2026-08-15 (Europe/London)

## Status

**Source-only posterior calibration passed; opened-v4 stress failed.** The
frozen source-first protocol therefore authorizes the 512-particle posterior
representation for local scenario-tree development on prospectively in-prior
worlds. It does not authorize v4 planner evaluation or an LLM efficacy claim.

No LLM, API, or network call was made. Cost was $0. All 288 replayed case hashes
match V1 exactly.

## Source-Only Pass

| Tier | Bank 1 MSE | Bank 2 MSE | Mean MSE | Evidence rho | Median evidence delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 0.007452 | 0.008503 | 0.007978 | 0.9940 | 0.533 nats |
| Medium | 0.034625 | 0.042272 | 0.038449 | 0.9948 | 0.598 nats |
| Hard | 0.037257 | 0.029303 | 0.033280 | 0.9965 | 0.608 nats |
| **Aggregate** | **0.026445** | **0.026693** | **0.026569** | **0.9964** | **0.578 nats** |

Every source gate passes:

- all V1 hashes match;
- every run reaches temperature 1 in at most 22 rungs;
- acceptance is 0.25-0.31 and every run accepts rejuvenation proposals;
- evidence rho is above 0.95 and median difference below 1 nat on each tier;
- MSE improves 37.52% over V2, 92.87% over static 16, and 85.06% over static
  100;
- source bank ratios are 1.14 easy, 1.22 medium, and 1.27 hard, all below the
  frozen 1.5 threshold;
- SMC remains better than both static baselines on every tier.

This resolves both V1's evidence instability and V2's source-hard posterior
prediction instability without changing the prior, likelihood, histories, or
gates.

## Opened-v4 Stress Failure

| Tier | Bank 1 MSE | Bank 2 MSE | Mean MSE | Bank ratio |
| --- | ---: | ---: | ---: | ---: |
| Easy | 0.011797 | 0.006564 | 0.009180 | 1.80 |
| Medium | 0.007828 | 0.015305 | 0.011566 | 1.96 |
| Hard | 0.040565 | 0.043257 | 0.041911 | 1.07 |
| **Aggregate** | **0.020063** | **0.021709** | **0.020886** | **1.08** |

V4 remains far better than static 16/100 and V1 SMC, and its aggregate banks
agree. However, two frozen conditions fail:

- easy and medium exceed the per-tier 1.5 bank-ratio threshold;
- aggregate MSE 0.020886 is 25.59% worse than the V2 reference 0.016630.

The V2 reference was itself bank-unstable, so its lower mean was not reliable
evidence that 256 particles were better. Nevertheless the prospective V3
condition is binding and cannot be waived.

## Interpretation

The numerical dependency is solved for the actual intended prospective model:
truth parameters drawn from a declared broad prior and withheld from inference
particles. It is not solved for the benchmark's v4 parameter shift, where 1.75%
of coordinates are outside the frozen prior and query-relevant modes remain
family-specific.

Use V3 only for source-only development and for a future sealed cohort sampled
prospectively from the same declared prior. Do not reuse v4 for a depth or LLM
claim. A future out-of-prior robustness study would need a separately frozen
heavy-tailed or hierarchical prior, not post-hoc widening.

## Next Authorized Work

Integrate immutable 512-particle SMC snapshots into a local root-sampled
scenario tree using source-only worlds. The first tree gate should compare
branch approximations against high-sample one-step Monte Carlo before any d2/d3
result. Structure evidence and proposal transitions remain disabled until this
branch-fidelity gate passes.

## Artifacts

- V3 result SHA256:
  `dce0832190aa8b76c9b345220a9043edb6e622db3b8cf0f524b68b5c06c783c1`
- V3 runner SHA256:
  `b421eae495bfe058ed1c1d5c30570cba9d00a2bbf5a057356600807fa7c34162`
- V3 test SHA256:
  `7affb2549fccb8a2854416b7310073397f078e51f9eb895d9aa5a527cc8af884`
- Focused tests before execution: 31 passed.
