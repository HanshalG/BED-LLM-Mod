# Constrained Support-Grid MPP30 Summary

This is the combined result from split Slurm jobs `102226`, `102227`, and `102228`.
The runs used `google/gemma-4-26B-A4B-it` on `msc` / `oat11`, seed `1304`, 30 paired
trials, 6 rounds, branch-decoy/local-bump source dynamics, max step radius `0.5`,
analytical posterior updates, fixed deployed support, analytic rollout future queries,
support-grid candidate generation, and StrategyEIG depths 1/3/5 with matched-compute
myopic controls.

The combined local artifact is:

`runs/loc_branch_decoy_local_constrained_supportgrid_mpp30_26b_a4b_split/fixed_root_depth_sweep_metrics.json`

## Final Metrics

| policy | RMSE mean | RMSE sd | truth log-prob mean | truth log-prob sd | entropy mean | selected EIG mean | realized entropy drop mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| `EIG` | 0.4118 | 0.8546 | -1.6721 | 1.2565 | 0.9440 | 0.2179 | 0.2672 |
| `naive` | 0.1665 | 0.4101 | -1.4840 | 1.2429 | 1.1335 | 0.0000 | 0.2082 |
| `naive+belief` | 1.1374 | 1.2273 | -2.5780 | 1.0952 | 1.6696 | 0.0000 | 0.0082 |
| `StrategyEIG-d1` | 0.9288 | 1.2102 | -2.5317 | 0.9205 | 1.6728 | 0.0054 | 0.0022 |
| `StrategyEIG-d3` | 1.1270 | 1.2928 | -2.5897 | 1.0222 | 1.4487 | 0.0558 | 0.0620 |
| `StrategyEIG-d5` | 0.5417 | 0.9990 | -1.9823 | 1.2845 | 1.1856 | 0.2453 | 0.1516 |
| `StrategyEIG-myopic-d3` | 0.9288 | 1.2102 | -2.5317 | 0.9205 | 1.6728 | 0.0054 | 0.0022 |
| `StrategyEIG-myopic-d5` | 0.9288 | 1.2102 | -2.5317 | 0.9205 | 1.6728 | 0.0054 | 0.0022 |

## Mean RMSE Trace

| policy | r1 | r2 | r3 | r4 | r5 | r6 |
|---|---:|---:|---:|---:|---:|---:|
| `EIG` | 0.9483 | 0.7077 | 0.6178 | 0.7723 | 1.0110 | 0.4118 |
| `naive` | 0.9787 | 0.9787 | 0.8887 | 0.8169 | 0.2530 | 0.1665 |
| `StrategyEIG-d1` | 0.9281 | 0.9281 | 0.9281 | 1.0110 | 1.0126 | 0.9288 |
| `StrategyEIG-d3` | 1.0576 | 0.9681 | 1.0454 | 0.9653 | 0.8881 | 1.1270 |
| `StrategyEIG-d5` | 1.1728 | 1.1693 | 1.1692 | 1.0007 | 1.0857 | 0.5417 |
| `StrategyEIG-myopic-d5` | 0.9281 | 0.9281 | 0.9281 | 1.0110 | 1.0126 | 0.9288 |

## Paired Final Deltas vs EIG

Positive RMSE delta means worse than greedy EIG. Positive truth-log-prob delta means
better than greedy EIG.

| policy | RMSE delta mean | RMSE 95% CI | RMSE Wilcoxon p | truth-log-prob delta mean | truth-log-prob 95% CI | truth Wilcoxon p |
|---|---:|---:|---:|---:|---:|---:|
| `StrategyEIG-d1` | 0.5170 | [0.1766, 0.8915] | 0.0084 | -0.8596 | [-1.2684, -0.4540] | 0.0013 |
| `StrategyEIG-d3` | 0.7153 | [0.1938, 1.2343] | 0.0076 | -0.9176 | [-1.2686, -0.5366] | 0.0002 |
| `StrategyEIG-d5` | 0.1299 | [-0.1364, 0.3956] | 0.0121 | -0.3102 | [-0.7401, 0.1600] | 0.0804 |
| `StrategyEIG-myopic-d5` | 0.5170 | [0.1823, 0.9061] | 0.0084 | -0.8596 | [-1.2674, -0.4472] | 0.0013 |
| `naive` | -0.2453 | [-0.5530, 0.0028] | 0.5049 | 0.1881 | [-0.2284, 0.6031] | 0.3877 |
| `naive+belief` | 0.7256 | [0.3272, 1.1353] | 0.0059 | -0.9059 | [-1.3765, -0.4800] | 0.0007 |

## Interpretation

This constrained MPP30 run does not support a claim that StrategyEIG beats greedy EIG or
naive prompting. It does support a narrower internal depth/objective claim: d5 is much
better than d1, d3, and the matched-compute myopic controls on final RMSE, truth
log-probability, entropy, selected EIG, and realized entropy drop. The clean paper framing
is therefore about partial translation from ranking-fidelity and non-myopic scoring into
deployment gains, with the analytic greedy and naive baselines still stronger in this run.
