# LLM Cost vs Depth

| label | method | depth | trials | rounds | calls | prompt tokens | completion tokens | total tokens | tokens/trial | tokens/trial-round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_root_depth_sweep_metrics | StrategyEIG fixed-root sweep | 1..5 | 30 | 6 | 1008 | 1357255 | 2512265 | 3869520 | 128984 | 21497.3 |
| fixed_root_depth_sweep_metrics | StrategyEIG fixed-root sweep | 1..5 | 30 | 6 | 1023 | 1367782 | 2389992 | 3757774 | 125259.1 | 20876.5 |

Note: fixed-root depth sweeps report total run cost for the full depth set; use separate single-depth runs for strict per-depth wall-clock/token accounting.

## StrategyEIG vs brute-force n-step EIG proxy

Candidate-tree proxy: brute-force depth-d EIG expands candidate sets at `1 + B + ... + B^(d-1)` nodes per deployed decision; StrategyEIG uses one generated strategy-root set per deployed decision plus rollout simulations.

| run | depth | B | rollouts | decisions | StrategyEIG root sets | StrategyEIG rollout paths | StrategyEIG simulated steps | brute-force candidate sets | brute-force leaf sequences | BF/Strategy root-set ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_root_depth_sweep_metrics | 1 | 2 | 8 | 180 | 180 | 2880 | 2880 | 180 | 360 | 1 |
| fixed_root_depth_sweep_metrics | 2 | 2 | 8 | 180 | 180 | 2880 | 5760 | 540 | 720 | 3 |
| fixed_root_depth_sweep_metrics | 3 | 2 | 8 | 180 | 180 | 2880 | 8640 | 1260 | 1440 | 7 |
| fixed_root_depth_sweep_metrics | 4 | 2 | 8 | 180 | 180 | 2880 | 11520 | 2700 | 2880 | 15 |
| fixed_root_depth_sweep_metrics | 5 | 2 | 8 | 180 | 180 | 2880 | 14400 | 5580 | 5760 | 31 |
| fixed_root_depth_sweep_metrics | 1 | 2 | 8 | 180 | 180 | 2880 | 2880 | 180 | 360 | 1 |
| fixed_root_depth_sweep_metrics | 2 | 2 | 8 | 180 | 180 | 2880 | 5760 | 540 | 720 | 3 |
| fixed_root_depth_sweep_metrics | 3 | 2 | 8 | 180 | 180 | 2880 | 8640 | 1260 | 1440 | 7 |
| fixed_root_depth_sweep_metrics | 4 | 2 | 8 | 180 | 180 | 2880 | 11520 | 2700 | 2880 | 15 |
| fixed_root_depth_sweep_metrics | 5 | 2 | 8 | 180 | 180 | 2880 | 14400 | 5580 | 5760 | 31 |
