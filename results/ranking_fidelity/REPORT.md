# Ranking-Fidelity Interim Report

This is the Phase 1 gate for Path A: does configured StrategyEIG rollout
scoring rank candidate strategies by realized information gain?

The current runs use the Phase 2 configured scorer: common rollout controls,
fixed-common analytical support, and final LLM refresh disabled for scoring.

## Completed Evidence

### Fixed Pilot

Run: `rankfid4b_configured_pilot_t3_s1304_hfix`

- Trials: 3
- Probe states: 9 (`rounds 0, 3, 6`)
- Deployments per strategy/depth: 4
- Depths: 2, 3, 5

| depth | Spearman entropy | 95% bootstrap CI | Spearman RMSE | SNR | top-1 entropy regret |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.634 | [0.487, 0.783] | 0.033 | 1.762 | 0.159 |
| 3 | 0.673 | [0.543, 0.806] | -0.118 | 1.607 | 0.235 |
| 5 | 0.653 | [0.523, 0.789] | -0.009 | 2.430 | 0.186 |

### First Gate Chunk

Run: `rankfid4b_gate_configured_t20_m8_o04`

- Trials: 4
- Probe states: 12 (`rounds 0, 3, 6`)
- Deployments per strategy/depth: 8
- Depths: 2, 3, 5

| depth | Spearman entropy | 95% bootstrap CI | Spearman RMSE | SNR | top-1 entropy regret |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.507 | [0.269, 0.710] | -0.245 | 1.873 | 0.243 |
| 3 | 0.558 | [0.355, 0.733] | -0.345 | 2.259 | 0.146 |
| 5 | 0.603 | [0.369, 0.804] | -0.261 | 1.990 | 0.211 |

## Current Read

The configured scorer is clearing the Phase 1 entropy-ranking gate in the
completed evidence: all completed mean Spearman entropy correlations are above
0.5. This supports continuing Path A for entropy/information-gain ranking.

The realized RMSE link is not established. RMSE rank correlations are near zero
or negative, which means better predicted entropy reduction is not reliably
translating to lower realized RMSE in these diagnostics.

## In Flight

The full configured gate is split into five four-trial chunks. The offset-4
chunk is complete. Offset chunks 0 and 8 have been relaunched with corrected
comma-separated arguments and are running. Offset chunks 12 and 16 are pending.

The full T=20 aggregate is not complete yet; this report is intentionally
interim and should be replaced/extended after all chunks finish.
