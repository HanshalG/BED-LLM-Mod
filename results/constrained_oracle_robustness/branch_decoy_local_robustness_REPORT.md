# Constrained Oracle Robustness Sweep

This CPU-only sweep maps where non-myopic planning helps in the constrained
branch-decoy/local-bump environment. Values are planner minus greedy final
RMSE, so negative cells favor the depth planner.

## Base Configuration

- Trials per cell: 100
- Rounds: 6
- Particles: 64
- Grid size: 13
- Arena: 2.2
- Source prior: `branch_decoy`
- Source radius: 2.2
- Planner depth: 2
- Planning support size: 8
- Signal amplitude: 8.0
- Seed: 1304
- Highlighted operating point: lengthscale 0.5, max step radius 0.5, noise sd 0.15

## Cells

| lengthscale | max step radius | noise sd | planner - greedy final RMSE | planner - lawnmower final RMSE | win rate vs greedy |
|---:|---:|---:|---:|---:|---:|
| 0.35 | 0.4 | 0.1 | -0.0003 | -0.0040 | 0.240 |
| 0.35 | 0.4 | 0.15 | -0.0003 | -0.0023 | 0.210 |
| 0.35 | 0.4 | 0.25 | -0.0003 | -0.0020 | 0.280 |
| 0.35 | 0.5 | 0.1 | 0.0001 | -0.0010 | 0.220 |
| 0.35 | 0.5 | 0.15 | 0.0000 | -0.0025 | 0.160 |
| 0.35 | 0.5 | 0.25 | -0.0005 | -0.0017 | 0.370 |
| 0.35 | 0.7 | 0.1 | -0.0000 | -0.0054 | 0.370 |
| 0.35 | 0.7 | 0.15 | -0.0000 | -0.0036 | 0.380 |
| 0.35 | 0.7 | 0.25 | 0.0001 | -0.0040 | 0.280 |
| 0.5 | 0.4 | 0.1 | -0.0226 | -0.6734 | 0.210 |
| 0.5 | 0.4 | 0.15 | -0.1905 | -0.4343 | 0.390 |
| 0.5 | 0.4 | 0.25 | 0.0036 | -0.0071 | 0.180 |
| 0.5 | 0.5 | 0.1 | -0.0790 | -0.6935 | 0.200 |
| 0.5 | 0.5 | 0.15 | -0.1751 | -0.2864 | 0.380 |
| 0.5 | 0.5 | 0.25 | -0.0002 | 0.0026 | 0.140 |
| 0.5 | 0.7 | 0.1 | -0.0001 | -0.7340 | 0.240 |
| 0.5 | 0.7 | 0.15 | -0.0096 | -0.7865 | 0.400 |
| 0.5 | 0.7 | 0.25 | -0.0105 | -0.6485 | 0.340 |
| 0.75 | 0.4 | 0.1 | 0.0108 | -0.4962 | 0.230 |
| 0.75 | 0.4 | 0.15 | 0.0209 | -0.6027 | 0.180 |
| 0.75 | 0.4 | 0.25 | -0.0037 | -0.6623 | 0.300 |
| 0.75 | 0.5 | 0.1 | 0.0106 | -0.5766 | 0.290 |
| 0.75 | 0.5 | 0.15 | 0.0020 | -0.6227 | 0.210 |
| 0.75 | 0.5 | 0.25 | -0.0029 | -0.7324 | 0.290 |
| 0.75 | 0.7 | 0.1 | -0.0005 | -0.1113 | 0.280 |
| 0.75 | 0.7 | 0.15 | -0.0005 | -0.1899 | 0.210 |
| 0.75 | 0.7 | 0.25 | -0.0013 | -0.3937 | 0.290 |

## Figure

![Robustness heatmap](plots/constrained_oracle_robustness/branch_decoy_local_robustness_heatmap.png)

