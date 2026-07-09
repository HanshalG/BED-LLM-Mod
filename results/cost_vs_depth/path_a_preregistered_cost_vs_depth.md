# LLM Cost vs Depth

| label | method | depth | trials | rounds | calls | prompt tokens | completion tokens | total tokens | tokens/trial | tokens/trial-round |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| config_location_branch_decoy_local_final50_26b_a4b | StrategyEIG planned fixed-root sweep | 1..5 | 50 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |
| config_location_branch_decoy_local_unconstrained_final50_26b_a4b | StrategyEIG planned fixed-root sweep | 1..5 | 50 | 6 | 0 | 0 | 0 | 0 | 0 | 0 |

Note: fixed-root depth sweeps report total run cost for the full depth set; use separate single-depth runs for strict per-depth wall-clock/token accounting.
Planned config rows report algorithmic scaling only; token totals stay zero until completed run logs are supplied.

## StrategyEIG vs brute-force n-step EIG proxy

Candidate-tree proxy: brute-force depth-d EIG expands candidate sets at `1 + B + ... + B^(d-1)` nodes per deployed decision; StrategyEIG uses one generated strategy-root set per deployed decision plus rollout simulations.

| run | depth | B | rollouts | decisions | StrategyEIG root sets | StrategyEIG rollout paths | StrategyEIG simulated steps | brute-force candidate sets | brute-force leaf sequences | BF/Strategy root-set ratio |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| config_location_branch_decoy_local_final50_26b_a4b | 1 | 10 | 16 | 300 | 300 | 48000 | 48000 | 300 | 3000 | 1 |
| config_location_branch_decoy_local_final50_26b_a4b | 2 | 10 | 16 | 300 | 300 | 48000 | 96000 | 3300 | 30000 | 11 |
| config_location_branch_decoy_local_final50_26b_a4b | 3 | 10 | 16 | 300 | 300 | 48000 | 144000 | 33300 | 300000 | 111 |
| config_location_branch_decoy_local_final50_26b_a4b | 4 | 10 | 16 | 300 | 300 | 48000 | 192000 | 333300 | 3000000 | 1111 |
| config_location_branch_decoy_local_final50_26b_a4b | 5 | 10 | 16 | 300 | 300 | 48000 | 240000 | 3333300 | 30000000 | 11111 |
| config_location_branch_decoy_local_unconstrained_final50_26b_a4b | 1 | 10 | 16 | 300 | 300 | 48000 | 48000 | 300 | 3000 | 1 |
| config_location_branch_decoy_local_unconstrained_final50_26b_a4b | 2 | 10 | 16 | 300 | 300 | 48000 | 96000 | 3300 | 30000 | 11 |
| config_location_branch_decoy_local_unconstrained_final50_26b_a4b | 3 | 10 | 16 | 300 | 300 | 48000 | 144000 | 33300 | 300000 | 111 |
| config_location_branch_decoy_local_unconstrained_final50_26b_a4b | 4 | 10 | 16 | 300 | 300 | 48000 | 192000 | 333300 | 3000000 | 1111 |
| config_location_branch_decoy_local_unconstrained_final50_26b_a4b | 5 | 10 | 16 | 300 | 300 | 48000 | 240000 | 3333300 | 30000000 | 11111 |
