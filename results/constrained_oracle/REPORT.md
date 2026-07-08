# Constrained Oracle Check Interim Report

This is the Phase 3 cheap non-LLM sanity check for the locality-constrained
location-finding environment. The goal is to find a geometry where a grid
depth planner beats grid greedy EIG by a clear margin before spending LLM
compute on StrategyEIG depth sweeps.

All runs use analytic likelihood, fixed first query at the origin, paired
hidden states/support/noise between greedy and planner, and the constrained
action grid enforced by `max_step_radius`.

## Current Best Geometry

Best validated candidate so far:

- `source_prior: branch_decoy`
- `signal_model: local_bump`
- `signal_lengthscale: 0.5`
- `signal_amplitude: 8.0`
- `num_sources: 1`
- `arena: 2.5`
- `source_radius: 2.2`
- `max_step_radius: 0.5`
- `noise_sd: 0.15`
- `num_rounds: 6`
- `planner_depth: 2`
- `planning_support_size: 6`

This is the first structural myopic-trap setting that survived a larger paired
validation run. The key change is finite-range sensing: far-away hypotheses are
nearly indistinguishable until the agent commits movement toward a branch. Under
that signal model, depth-2 planning reaches informative branch locations earlier
than greedy EIG.

| run | T | greedy final RMSE | planner final RMSE | planner - greedy final RMSE | planner - greedy RMSE AUC | planner win rate |
|---|---:|---:|---:|---:|---:|---:|
| `constrained_oracle_branch_decoy_local_r22_l05_t120_d2` | 120 | 0.5606 | 0.1529 | -0.4077 | -0.9306 | 0.525 |
| `tune_branch_local_r2.2_l0.5_t20` | 20 | not reported | not reported | -0.4682 | -1.0959 | 0.650 |
| `constrained_oracle_corners_arena3_d2_t20` | 20 | 0.1372 | 0.0473 | -0.0899 | -1.4671 | 0.500 |
| `constrained_oracle_corners_arena3_d2_t200` | 200 | 0.0746 | 0.0664 | -0.0082 | -0.5438 | 0.375 |
| `constrained_oracle_corners_arena3_d2_t80_r8` | 80 | 0.3113 | 0.2658 | -0.0456 | -0.1748 | 0.275 |
| `constrained_oracle_corners_r15_d2_t200_r5` | 200 | 0.3364 | 0.2940 | -0.0424 | -0.2069 | 0.275 |

The paired win rate is only slightly above chance because many trials are
near-ties, but the mean final-RMSE and AUC reductions are large. This satisfies
the cheap-oracle requirement well enough to justify implementing the same
finite-range signal option in the real environment before Phase 4.

Figure: `plots/constrained_oracle/constrained_oracle_branch_decoy_local_r22_l05_t120_d2_rmse.png`

## Other Geometries Tried

| geometry | outcome |
|---|---|
| Normal prior, radius 1.0-ish | Greedy-friendly; final RMSE nearly zero for both policies. |
| Ring prior | Planner did not improve RMSE. |
| Axis endpoints | Small final-RMSE gains in some short runs; not robust. |
| Fork prior | Slight entropy and RMSE AUC gains; final-RMSE gap too small. |
| Corners with 2 sources | Planner lost on RMSE. |
| Tighter radius 0.25 | Mostly slowed both policies equally. |
| Larger arena/source radius | Preserved entropy but did not create a clear RMSE gap. |
| Short-horizon corners (`num_rounds=5`) | Larger mean final-RMSE reduction, but only 27.5% paired win rate at T=200. |
| Larger planning support (`planning_support_size=12`) | Improved entropy/AUC in a 20-trial check, but final-RMSE gap remained small. |
| Branch-decoy prior with inverse-square signal | Not separately validated; expected to remain greedy-friendly because far-field signal is too informative. |
| Branch-decoy prior with local-bump signal | Clear depth-2 advantage at radius 2.2 and lengthscale 0.5. |

## Current Conclusion

The locality constraint implementation is working, and the oracle script now
produces reproducible JSONL, summary JSON, and plots. Simple source-prior
geometries with the original inverse-square signal remain mostly greedy-friendly.
The structural finite-range signal variant creates a clear oracle gap:
depth-2 planner final RMSE is 0.1529 vs greedy 0.5606 over 120 paired trials.

Next implementation direction: add the `local_bump` signal option to the actual
location-finding environment behind config flags, preserve the current
inverse-square signal as default, then run the paired depth sweep on this
branch-decoy/local-bump constrained task.
