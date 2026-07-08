# Phase 1 Ranking-Fidelity Gate: 26B A4B

Status: **proceed, with caveats**

This is the completed Path A Phase 1 gate for the thinking-enabled
`google/gemma-4-26B-A4B-it` ranking-fidelity diagnostic.

## Run

- Remote checkout: `/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z`
- Config: `configs/config_strategy_ranking_fidelity_26b_a4b.yaml`
- Run directories: `runs/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_o*`
- Trials: 20
- Probe rounds: 0, 3, 6
- Records: 60/60
- Depths: 2, 3, 5
- Candidate strategies per probe: 8
- Deployments per strategy/depth: 8
- Score variant: `configured`

## Aggregate Metrics

Mean +/- standard error over all 60 probe records:

| Depth | Entropy Spearman | Truth-log-prob Spearman | RMSE Spearman | Top-1 entropy regret | SNR |
|---:|---:|---:|---:|---:|---:|
| 2 | 0.380 +/- 0.045 | 0.371 +/- 0.040 | -0.088 +/- 0.051 | 0.326 +/- 0.048 | 1.692 +/- 0.431 |
| 3 | 0.440 +/- 0.041 | 0.373 +/- 0.045 | -0.080 +/- 0.045 | 0.289 +/- 0.043 | 2.504 +/- 0.586 |
| 5 | 0.396 +/- 0.044 | 0.362 +/- 0.044 | -0.035 +/- 0.050 | 0.211 +/- 0.044 | 2.327 +/- 0.603 |

## Round-Broken Metrics

| Round | Depth | Entropy Spearman | Truth-log-prob Spearman | RMSE Spearman | Top-1 entropy regret | SNR |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 2 | 0.324 +/- 0.080 | 0.327 +/- 0.068 | -0.224 +/- 0.090 | 0.404 +/- 0.086 | 0.499 +/- 0.097 |
| 0 | 3 | 0.419 +/- 0.069 | 0.311 +/- 0.085 | -0.159 +/- 0.082 | 0.317 +/- 0.052 | 0.653 +/- 0.119 |
| 0 | 5 | 0.353 +/- 0.068 | 0.309 +/- 0.069 | -0.131 +/- 0.081 | 0.164 +/- 0.040 | 0.636 +/- 0.105 |
| 3 | 2 | 0.413 +/- 0.068 | 0.352 +/- 0.051 | -0.083 +/- 0.097 | 0.399 +/- 0.083 | 1.071 +/- 0.179 |
| 3 | 3 | 0.527 +/- 0.072 | 0.421 +/- 0.079 | -0.060 +/- 0.070 | 0.322 +/- 0.072 | 2.613 +/- 1.208 |
| 3 | 5 | 0.441 +/- 0.081 | 0.405 +/- 0.080 | -0.070 +/- 0.084 | 0.261 +/- 0.066 | 1.419 +/- 0.259 |
| 6 | 2 | 0.402 +/- 0.086 | 0.436 +/- 0.085 | 0.043 +/- 0.071 | 0.174 +/- 0.075 | 3.506 +/- 1.192 |
| 6 | 3 | 0.374 +/- 0.073 | 0.389 +/- 0.070 | -0.022 +/- 0.081 | 0.228 +/- 0.096 | 4.248 +/- 1.177 |
| 6 | 5 | 0.394 +/- 0.081 | 0.373 +/- 0.080 | 0.095 +/- 0.089 | 0.209 +/- 0.108 | 4.925 +/- 1.662 |

## Interpretation

The first link is positive: rollout-estimated StrategyEIG ranks strategies
above chance by realized entropy reduction and by truth log-posterior
probability. Depth 3 has the highest overall entropy rank correlation, and
depth 5 has the lowest top-1 entropy regret.

RMSE rank correlation is near zero, which is expected because RMSE is a noisier
downstream localization readout. It should not be used as the Phase 1 gate
metric.

The gate therefore supports continuing to the constrained/locality Path A depth
sweep. It does not by itself prove a monotonic depth effect.

## RMSE Repair Analysis

The realized-realized repair analysis was computed from the aggregate records at
`results/ranking_fidelity/RMSE_REPAIR.md`.

| Depth | entropy-drop vs RMSE-drop Spearman | truth-log-prob vs RMSE-drop Spearman | RMSE SNR |
|---:|---:|---:|---:|
| 2 | 0.101 +/- 0.058 | 0.083 +/- 0.056 | 0.572 +/- 0.164 |
| 3 | 0.073 +/- 0.055 | 0.068 +/- 0.051 | 0.424 +/- 0.051 |
| 5 | 0.103 +/- 0.049 | 0.098 +/- 0.050 | 0.474 +/- 0.050 |

Expected posterior RMSE could not be recomputed from the current aggregate JSONL:
The aggregate ranking-fidelity JSONL stores candidate-level realized entropy drops, point-RMSE drops, truth-log-probability means, and query-distance diagnostics, but it does not store final posterior hypothesis supports/probabilities for each deployment. Expected posterior RMSE cannot be recomputed exactly without those posterior states.
