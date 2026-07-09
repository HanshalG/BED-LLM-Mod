# Support-Grid Candidate Pilot: 26B A4B, 1 Trial x 6 Rounds

Run: `loc_branch_decoy_local_constrained_supportgrid_26b_a4b_t1r6`  
Job: `102208` on `gh200` / `oat21`  
Commit/tag: `path-a-support-grid-candidates-20260709`  
Artifacts: `runs/loc_branch_decoy_local_constrained_supportgrid_26b_a4b_t1r6/`

## Purpose

This was a throughput and mechanism smoke test after replacing LLM candidate generation
with deterministic `support_grid` candidates. It is not statistical evidence for a depth
claim because `n=1`.

## Key Counters

| counter | value |
|---|---:|
| completed rounds | 6 / 6 |
| decision rows | 48 |
| LLM usage events | 34 |
| total tokens | 126,519 |
| forced thinking exits | 5 |
| LLM candidate-generation calls | 0 |
| support-grid candidate calls | 6 |
| LLM strategy-location rollout calls | 0 |
| LLM belief-refresh requests | 0 |
| fixed-support belief updates | 6 |

The previous expensive paths are gone: future rollout queries are analytic, deployed
belief updates reweight fixed support, and EIG/StrategyEIG candidate generation no longer
calls the LLM in this pilot mode. Remaining cost is initial belief generation, naive
baseline actions, and StrategyEIG strategy/root proposals.

## Final Metrics

| policy | final RMSE | final entropy | final truth log prob |
|---|---:|---:|---:|
| `EIG` | 0.092 | 2.181 | -2.476 |
| `StrategyEIG-d1` | 2.473 | 2.181 | -2.505 |
| `StrategyEIG-d3` | 2.473 | 2.181 | -2.498 |
| `StrategyEIG-d5` | 0.092 | 1.618 | -1.896 |
| `StrategyEIG-myopic-d3` | 2.473 | 2.181 | -2.505 |
| `StrategyEIG-myopic-d5` | 2.473 | 2.181 | -2.505 |
| `naive` | 0.092 | 2.181 | -2.505 |
| `naive+belief` | 2.473 | 2.181 | -2.477 |

## Mechanism Readout

The qualitative first-link behavior is promising in this one trial:

- `StrategyEIG-d5` selected the distant branch at round 0 with root `[2.2, 1.76]`, then
  kept choosing intermediate movement-constrained waypoints toward that branch.
- `StrategyEIG-d1` and both myopic controls repeatedly stayed near the local mode around
  `[-1.43, 0]` / the x-axis.
- The d5 trajectory eventually reached the same low RMSE as greedy `EIG`, while d1/d3 and
  myopic controls did not.

This is exactly the qualitative behavior Path A needs, but a 3-trial support-grid pilot is
the next minimum check before scaling to MPP30.

## Next Step

When GH200 queue pressure is low, run the same support-grid configuration with
`--num-trials 3 --num-rounds 6`. Do not use this single-trial smoke result as a depth
claim in the paper.
