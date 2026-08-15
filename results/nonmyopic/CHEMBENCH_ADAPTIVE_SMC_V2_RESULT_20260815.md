# ChemBench Adaptive-SMC V2 Result

Date: 2026-08-15 (Europe/London)

## Status

**Failed closed on one prospective posterior-prediction stability condition.**
V2 fixes every V1 evidence-stability failure and improves aggregate prediction,
but the two source-only hard-tier banks differ by more than the frozen 1.5 MSE
ratio. This exact 256-particle estimator is not authorized for planner state.

No LLM, API, or network call was made. Cost was $0. All 288 prior, truth,
history, and query hashes exactly match V1.

## Evidence Stability

The unchanged V1 requirements were Spearman at least 0.95 and median absolute
log-evidence difference at most 1.0 nat on every source-only tier.

| Tier | V1 Spearman | V2 Spearman | V1 median delta | V2 median delta |
| --- | ---: | ---: | ---: | ---: |
| Easy | 0.9292 | 0.9831 | 1.696 | 0.569 |
| Medium | 0.8999 | 0.9779 | 1.954 | 0.968 |
| Hard | 0.9294 | 0.9939 | 2.200 | 0.816 |

All evidence conditions pass. Scrambled Sobol coverage plus full-covariance
rejuvenation removes the narrow-mode evidence failure seen in V1.

## Predictive Result

### Source-only

| Tier | V2 SMC MSE | Static 16 | Static 100 | Bank 1 | Bank 2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Easy | 0.004691 | 0.280307 | 0.247426 | 0.005686 | 0.003696 |
| Medium | 0.045983 | 0.367830 | 0.096001 | 0.046566 | 0.045401 |
| Hard | 0.076903 | 0.469185 | 0.190049 | 0.038541 | 0.115265 |
| **Aggregate** | **0.042526** | **0.372441** | **0.177825** | **0.030264** | **0.054788** |

Aggregate V2 improves 8.67% over V1 SMC, 88.58% over static 16, and 76.09%
over static 100. It has 129 wins and 15 losses versus static 16.

### Opened v4

V2 aggregate MSE is 0.016630, improving 51.96% over V1 SMC, 93.98% over
static 16, and 85.08% over static 100. Tier MSE is 0.010226/0.008257/0.031406
for easy/medium/hard.

## Sole Failed Gate

The frozen condition required the larger bank MSE to be no more than 1.5 times
the smaller bank MSE on every cohort/tier. Source-only hard is
0.115265/0.038541 = 2.99. Source-only easy and v4 easy/hard also exceed 1.5.

This mismatch is not strongly associated with evidence disagreement. The main
source-hard contribution is `c78_allosteric_act_arrhenius`: bank MSE
1.0095 versus 3.5813 while log evidence differs by only 0.338 nat. Similar
likelihood modes make different held-out forecasts because eight fixed assays
do not identify every query-relevant parameter direction.

## Health

- All runs reach temperature 1 in at most 26 rungs.
- Acceptance is approximately 0.25-0.31 across cohorts, tiers, and banks.
- Every run accepts rejuvenation proposals.
- All V1 hashes match.
- Every evidence, health, baseline, and aggregate-MSE gate passes.

## Decision

Close exact V2. The correct next test is more posterior integration, not a
threshold change: two 512-particle scrambled-Sobol/full-covariance banks on the
same histories. Run source-only first and open the already-available v4 replay
only if source evidence and bank-prediction stability pass. Preserve the 1.5
bank-ratio requirement.

If 512 particles still fails, stop scaling particles and treat the remaining
uncertainty as a design problem: add posterior-predictive action search that
targets bank disagreement before building the non-myopic policy ladder.

## Artifacts

- V2 result SHA256:
  `868f24fac0dda206a2f5387c314cb1bca79757b915ebea6f9446b62ca676b469`
- SMC implementation SHA256:
  `08a14879f52016e73751fc345bee2b8fd70136a0ed995c23285dfc9eab2d87bd`
- V2 runner SHA256:
  `deba1c353b1de648d08de06b9ba7388a7014a1cc6f80b8b4d9b4ff6851426833`
- Focused tests before execution: 30 passed.
