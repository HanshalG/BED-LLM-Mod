# Location Depth Sweep Contrast Report

This report compares the Phase 4 constrained and unconstrained fixed-root depth sweeps.

## Headline Constrained Depths

Paper-facing subset: greedy EIG and StrategyEIG depths 1, 3, and 5 on the constrained task.

| policy | final RMSE | final RMSE std | paired final RMSE delta vs EIG | 95% CI | Wilcoxon p |
|---|---:|---:|---:|---:|---:|
| `EIG` | 0.4118 | 0.8546 | n/a | n/a | n/a |
| `StrategyEIG-d1` | 0.9288 | 1.2102 | 0.5170 | [0.1766, 0.8915] | 0.0084 |
| `StrategyEIG-d3` | 1.1270 | 1.2928 | 0.7153 | [0.1938, 1.2343] | 0.0076 |
| `StrategyEIG-d5` | 0.5417 | 0.9990 | 0.1299 | [-0.1364, 0.3956] | 0.0121 |

## Constrained

- Config: `/auto/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z/configs/config_location_branch_decoy_local_supportgrid_mpp30_26b_a4b.yaml`
- Trials: 30
- Rounds: 6
- Source prior: `branch_decoy`
- Signal model: `local_bump`
- Max step radius: 0.5
- Questioner model: `google/gemma-4-26B-A4B-it`
- Host: `None`
- SLURM job: `None`

## LLM Token Usage

- Calls: 1008; prompt tokens: 1357255; completion tokens: 2512265; total tokens: 3869520

| call type | calls | prompt tokens | completion tokens | total tokens |
|---|---:|---:|---:|---:|
| `batched_chat` | 850 | 714752 | 2418131 | 3132883 |
| `chat` | 30 | 18090 | 90085 | 108175 |
| `forced_final` | 128 | 624413 | 4049 | 628462 |

| policy | metric | final value | paired final delta vs EIG | 95% CI | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| `naive` | `source_rmse` | 0.1665 | -0.2453 | [-0.5530, 0.0028] | 0.5049 |
| `naive` | `expected_posterior_rmse` | 0.3796 | 0.0143 | [-0.2477, 0.2601] | 0.9508 |
| `naive` | `posterior_entropy` | 1.1335 | 0.1895 | [-0.0369, 0.4453] | 0.2410 |
| `naive` | `truth_log_probability` | -1.4840 | 0.1881 | [-0.2284, 0.6031] | 0.3877 |
| `naive+belief` | `source_rmse` | 1.1374 | 0.7256 | [0.3272, 1.1353] | 0.0059 |
| `naive+belief` | `expected_posterior_rmse` | 0.9682 | 0.6029 | [0.3443, 0.8474] | 0.0002 |
| `naive+belief` | `posterior_entropy` | 1.6696 | 0.7256 | [0.5113, 0.9425] | 0.0000 |
| `naive+belief` | `truth_log_probability` | -2.5780 | -0.9059 | [-1.3765, -0.4800] | 0.0007 |
| `EIG` | `source_rmse` | 0.4118 | n/a | n/a | n/a |
| `EIG` | `expected_posterior_rmse` | 0.3653 | n/a | n/a | n/a |
| `EIG` | `posterior_entropy` | 0.9440 | n/a | n/a | n/a |
| `EIG` | `truth_log_probability` | -1.6721 | n/a | n/a | n/a |
| `StrategyEIG-d1` | `source_rmse` | 0.9288 | 0.5170 | [0.1766, 0.8915] | 0.0084 |
| `StrategyEIG-d1` | `expected_posterior_rmse` | 1.0731 | 0.7078 | [0.4733, 0.9437] | 0.0001 |
| `StrategyEIG-d1` | `posterior_entropy` | 1.6728 | 0.7288 | [0.5156, 0.9476] | 0.0000 |
| `StrategyEIG-d1` | `truth_log_probability` | -2.5317 | -0.8596 | [-1.2684, -0.4540] | 0.0013 |
| `StrategyEIG-d3` | `source_rmse` | 1.1270 | 0.7153 | [0.1938, 1.2343] | 0.0076 |
| `StrategyEIG-d3` | `expected_posterior_rmse` | 1.0045 | 0.6392 | [0.3637, 0.9209] | 0.0001 |
| `StrategyEIG-d3` | `posterior_entropy` | 1.4487 | 0.5048 | [0.2793, 0.7287] | 0.0006 |
| `StrategyEIG-d3` | `truth_log_probability` | -2.5897 | -0.9176 | [-1.2686, -0.5366] | 0.0002 |
| `StrategyEIG-d5` | `source_rmse` | 0.5417 | 0.1299 | [-0.1364, 0.3956] | 0.0121 |
| `StrategyEIG-d5` | `expected_posterior_rmse` | 0.4741 | 0.1088 | [-0.1477, 0.3493] | 0.0507 |
| `StrategyEIG-d5` | `posterior_entropy` | 1.1856 | 0.2416 | [0.0189, 0.4611] | 0.0558 |
| `StrategyEIG-d5` | `truth_log_probability` | -1.9823 | -0.3102 | [-0.7401, 0.1600] | 0.0804 |
| `StrategyEIG-myopic-d3` | `source_rmse` | 0.9288 | 0.5170 | [0.1767, 0.9081] | 0.0084 |
| `StrategyEIG-myopic-d3` | `expected_posterior_rmse` | 1.0731 | 0.7078 | [0.4497, 0.9424] | 0.0001 |
| `StrategyEIG-myopic-d3` | `posterior_entropy` | 1.6728 | 0.7288 | [0.5290, 0.9324] | 0.0000 |
| `StrategyEIG-myopic-d3` | `truth_log_probability` | -2.5317 | -0.8596 | [-1.2662, -0.4548] | 0.0013 |
| `StrategyEIG-myopic-d5` | `source_rmse` | 0.9288 | 0.5170 | [0.1823, 0.9061] | 0.0084 |
| `StrategyEIG-myopic-d5` | `expected_posterior_rmse` | 1.0731 | 0.7078 | [0.4636, 0.9463] | 0.0001 |
| `StrategyEIG-myopic-d5` | `posterior_entropy` | 1.6728 | 0.7288 | [0.5214, 0.9509] | 0.0000 |
| `StrategyEIG-myopic-d5` | `truth_log_probability` | -2.5317 | -0.8596 | [-1.2674, -0.4472] | 0.0013 |

## Unconstrained

- Config: `/auto/users/hanyal/BED-LLM-Mod-qwen-strategy-b500-noeager-20260601T210610Z/configs/config_location_branch_decoy_local_unconstrained_supportgrid_mpp30_26b_a4b.yaml`
- Trials: 30
- Rounds: 6
- Source prior: `branch_decoy`
- Signal model: `local_bump`
- Max step radius: None
- Questioner model: `google/gemma-4-26B-A4B-it`
- Host: `None`
- SLURM job: `None`

## LLM Token Usage

- Calls: 1023; prompt tokens: 1367782; completion tokens: 2389992; total tokens: 3757774

| call type | calls | prompt tokens | completion tokens | total tokens |
|---|---:|---:|---:|---:|
| `batched_chat` | 843 | 640155 | 2298185 | 2938340 |
| `chat` | 30 | 18090 | 88398 | 106488 |
| `forced_final` | 150 | 709537 | 3409 | 712946 |

| policy | metric | final value | paired final delta vs EIG | 95% CI | Wilcoxon p |
|---|---|---:|---:|---:|---:|
| `naive` | `source_rmse` | 0.0889 | -0.0126 | [-0.0218, -0.0050] | 0.0051 |
| `naive` | `expected_posterior_rmse` | 0.1526 | 0.0733 | [-0.0185, 0.1999] | 0.1812 |
| `naive` | `posterior_entropy` | 0.9236 | -0.1596 | [-0.3585, 0.0418] | 0.0532 |
| `naive` | `truth_log_probability` | -1.4963 | 0.2206 | [-0.2897, 0.7202] | 0.3135 |
| `naive+belief` | `source_rmse` | 0.0951 | -0.0063 | [-0.0146, 0.0006] | 0.1235 |
| `naive+belief` | `expected_posterior_rmse` | 0.0877 | 0.0084 | [-0.0017, 0.0201] | 0.5237 |
| `naive+belief` | `posterior_entropy` | 1.2249 | 0.1416 | [0.0440, 0.2550] | 0.0672 |
| `naive+belief` | `truth_log_probability` | -1.7486 | -0.0318 | [-0.3280, 0.2492] | 0.9672 |
| `EIG` | `source_rmse` | 0.1014 | n/a | n/a | n/a |
| `EIG` | `expected_posterior_rmse` | 0.0793 | n/a | n/a | n/a |
| `EIG` | `posterior_entropy` | 1.0832 | n/a | n/a | n/a |
| `EIG` | `truth_log_probability` | -1.7169 | n/a | n/a | n/a |
| `StrategyEIG-d1` | `source_rmse` | 0.1029 | 0.0015 | [-0.0127, 0.0164] | 0.8939 |
| `StrategyEIG-d1` | `expected_posterior_rmse` | 0.0791 | -0.0001 | [-0.0151, 0.0146] | 1.0000 |
| `StrategyEIG-d1` | `posterior_entropy` | 1.1244 | 0.0412 | [-0.0855, 0.1685] | 0.5509 |
| `StrategyEIG-d1` | `truth_log_probability` | -1.5831 | 0.1337 | [-0.2485, 0.5105] | 0.7734 |
| `StrategyEIG-d3` | `source_rmse` | 0.1077 | 0.0063 | [-0.0097, 0.0222] | 0.6661 |
| `StrategyEIG-d3` | `expected_posterior_rmse` | 0.0904 | 0.0111 | [-0.0017, 0.0253] | 0.1442 |
| `StrategyEIG-d3` | `posterior_entropy` | 1.1455 | 0.0623 | [-0.0220, 0.1662] | 0.4590 |
| `StrategyEIG-d3` | `truth_log_probability` | -1.8626 | -0.1457 | [-0.4953, 0.2556] | 0.1499 |
| `StrategyEIG-d5` | `source_rmse` | 0.1028 | 0.0014 | [-0.0128, 0.0148] | 0.8336 |
| `StrategyEIG-d5` | `expected_posterior_rmse` | 0.0749 | -0.0044 | [-0.0252, 0.0124] | 0.8693 |
| `StrategyEIG-d5` | `posterior_entropy` | 1.0291 | -0.0541 | [-0.2079, 0.0982] | 0.3765 |
| `StrategyEIG-d5` | `truth_log_probability` | -1.6319 | 0.0850 | [-0.3358, 0.5751] | 0.7892 |
| `StrategyEIG-myopic-d3` | `source_rmse` | 0.1029 | 0.0015 | [-0.0127, 0.0167] | 0.8939 |
| `StrategyEIG-myopic-d3` | `expected_posterior_rmse` | 0.0791 | -0.0001 | [-0.0159, 0.0146] | 1.0000 |
| `StrategyEIG-myopic-d3` | `posterior_entropy` | 1.1244 | 0.0412 | [-0.0912, 0.1746] | 0.5509 |
| `StrategyEIG-myopic-d3` | `truth_log_probability` | -1.5831 | 0.1337 | [-0.2480, 0.5311] | 0.7734 |
| `StrategyEIG-myopic-d5` | `source_rmse` | 0.1029 | 0.0015 | [-0.0126, 0.0171] | 0.8939 |
| `StrategyEIG-myopic-d5` | `expected_posterior_rmse` | 0.0791 | -0.0001 | [-0.0153, 0.0149] | 1.0000 |
| `StrategyEIG-myopic-d5` | `posterior_entropy` | 1.1244 | 0.0412 | [-0.0832, 0.1784] | 0.5509 |
| `StrategyEIG-myopic-d5` | `truth_log_probability` | -1.5831 | 0.1337 | [-0.2291, 0.5305] | 0.7734 |
