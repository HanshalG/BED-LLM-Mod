# Path B Gate 0: Task-Loss Ranking Fidelity

- Status: **FAIL**
- Best depth: 1
- Best task scorer rho: 0.231
- Gate threshold: 0.300
- Records: 60 from `results/ranking_fidelity/rankfid26b_a4b_gate_v2ghs_configured_t20_m8_aggregate_records.jsonl`
- Config: `configs/config_strategy_ranking_fidelity_26b_a4b.yaml`
- LLM calls: none

## Correlations

Macro-average within-probe Spearman rho; brackets are a 95% bootstrap CI over probe states.

| Depth | task scorer vs realized posterior-risk drop | entropy scorer vs realized posterior-risk drop | task scorer vs point-dRMSE | realized-risk vs point-dRMSE ceiling |
|---:|---:|---:|---:|---:|
| 1 | 0.231 [0.098, 0.360] | NA | 0.249 [0.130, 0.371] | 0.165 [0.020, 0.301] |

## Estimator Diagnostics

| Depth | mean score SE / between-candidate SD | task-vs-smooth rho at round 0 | task-vs-smooth rho at round 3 | task-vs-smooth rho at round 6 |
|---:|---:|---:|---:|---:|
| 1 | 0.330 | 0.055 | 0.305 | 0.335 |

## Interpretation

The gating comparison is the first numeric column. The final column is the rankability ceiling: it measures how well the smooth posterior-risk endpoint itself ranks noisy point-RMSE gains.

This is the specified local fallback, not a reconstruction of the original LLM rollout trajectories. The legacy records preserve candidate roots, histories, and hidden states, but not rollout posterior snapshots or future query trajectories. Each stored root is therefore replayed with the same no-LLM analytic executor: subsequent queries maximize posterior predictive log-signal variance over that probe's stored root set.

Prior importance ESS fraction: mean 0.3702, minimum 0.0003. MCMC rejuvenation acceptance: mean 0.432.
