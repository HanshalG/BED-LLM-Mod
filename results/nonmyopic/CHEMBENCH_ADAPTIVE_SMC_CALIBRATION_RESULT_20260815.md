# ChemBench Adaptive-SMC Calibration Result

Date: 2026-08-15 (Europe/London)

## Status

**Failed closed on the predeclared evidence-stability conjunction.** The
parameter-posterior predictive result is strongly positive, every health and
support condition passes, and SMC beats both static baselines by a large
margin. However, the two 100-particle SMC banks do not meet the frozen
cross-bank evidence thresholds, so this exact configuration is not authorized
for structure weighting or the non-myopic planner.

No LLM, API, or network call was made. Cost was $0. The opened-v4 stress cohort
was reused; no new scientific endpoint was opened.

## Frozen Setup

- Correct structure supplied for all cases.
- 48 compound mechanisms x easy/medium/hard.
- Source-only in-prior cohort plus already-open v4 stress cohort: 288 cases.
- v0-v3-only transformed parameter boxes, expansion factor 1.5.
- Eight fixed mechanism-covering assays with common 1% observation noise.
- 512 held-out query assays per tier.
- Static importance sampling with 16 and 100 particles.
- Two adaptive-tempered SMC banks with 100 particles, ESS target 0.6, three
  bounded Metropolis moves per rung, and at most 80 rungs.

## Predictive Result

### Source-only in-prior cohort

| Tier | Static 16 MSE | Static 100 MSE | SMC MSE | SMC vs 16 | SMC vs 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 0.280307 | 0.247426 | 0.004551 | -98.38% | -98.16% |
| Medium | 0.367830 | 0.096001 | 0.040965 | -88.86% | -57.33% |
| Hard | 0.469185 | 0.190049 | 0.094168 | -79.93% | -50.45% |
| **Aggregate** | **0.372441** | **0.177825** | **0.046561** | **-87.50%** | **-73.82%** |

SMC versus static 16 has 128 wins, zero practical ties, and 16 losses.

### Opened-v4 stress cohort

| Tier | Static 16 MSE | Static 100 MSE | SMC MSE | SMC vs 16 | SMC vs 100 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 0.107334 | 0.037523 | 0.016570 | -84.56% | -55.84% |
| Medium | 0.263574 | 0.150303 | 0.034484 | -86.92% | -77.06% |
| Hard | 0.457858 | 0.146447 | 0.052799 | -88.47% | -63.95% |
| **Aggregate** | **0.276255** | **0.111424** | **0.034618** | **-87.47%** | **-68.93%** |

SMC versus static 16 has 132 wins and 12 losses. Only 1.75% of v4 parameter
coordinates lie outside the prospectively frozen prior overall: 0% easy,
1.87% medium, and 3.37% hard. Prior hashes are identical across source-only and
v4 cohorts.

## Posterior Health

- All 576 SMC fits reach temperature 1.
- Maximum tempering rungs are 24 source-only and 27 v4, below the cap of 80.
- Aggregate acceptance is approximately 0.19-0.26 by tier and bank.
- Every run accepts at least one rejuvenation proposal.
- Invalid bounded proposals average approximately 0.18-0.26.
- Static importance sampling collapses: median ESS is exactly 1.0 for both 16
  and 100 particles on every tier and cohort.

This confirms that adaptive tempering and rejuvenation are necessary under the
broad prior and released 1% noise model. Merely raising a static particle count
from 16 to 100 is not enough.

## Failed Evidence Gates

The frozen source-only thresholds were Spearman correlation at least 0.95 and
median absolute difference at most 1.0 nat between independent SMC banks.

| Tier | Evidence Spearman | Median absolute difference | Required |
| --- | ---: | ---: | ---: |
| Easy | 0.9292 | 1.6960 nats | >=0.95 / <=1.0 |
| Medium | 0.8999 | 1.9542 nats | >=0.95 / <=1.0 |
| Hard | 0.9294 | 2.2005 nats | >=0.95 / <=1.0 |

The disagreement is outlier-driven rather than a general numerical failure.
Examples include a 30.46-nat difference on medium Hill competitive and a
30.11-nat difference on hard Hill product Arrhenius. Some of these cases also
show large posterior-predictive bank differences. This is consistent with two
100-particle banks finding different narrow modes after the static prior has
effective sample size one.

## Decision

Close the exact random-initialized 100-particle configuration. Preserve the
strong result that adaptive parameter inference is the correct direction, but
do not use these log evidences as structure probabilities.

The next zero-call configuration should address mode coverage directly:

1. use scrambled low-discrepancy prior initialization at a power-of-two
   particle count;
2. increase to at least 256 particles;
3. use a regularized full-covariance proposal in transformed coordinates rather
   than independent coordinate steps;
4. compare two independent banks on both evidence and posterior predictions;
5. retain every assay, cohort, seed family, likelihood, and predictive threshold
   that does not concern the changed numerical sampler.

No threshold may be relaxed. A successor must be frozen before running its new
particles.

## Artifacts

- Result JSON SHA256:
  `34b28639ce9b8625a69b8a5ba74e6e4f78573c0c1829f70ffb1c080895d1ae8f`
- SMC implementation SHA256:
  `719058e6031cd87173776df655bb48e80c1a11fda7a30b63290abb651231e745`
- Calibration runner SHA256:
  `c7bad1563b709a67505a2a10b622fc2268c2be159d8b1d6d092b836ce2aa52de`
- Focused tests: 27 passed.
