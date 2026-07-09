# Path B Gate 0 Diagnosis

## Decision

Gate 0 **fails**. Do not launch Gate 1 or implement the arbitration policy on the
strength of these records.

The preregistered threshold was macro-average within-probe Spearman rho at or above
approximately 0.3 between the estimated posterior-RMSE reduction and the realized
posterior-RMSE reduction. The canonical common-support depth-1 replay reaches rho
0.231 with a 95% bootstrap CI of [0.098, 0.360]. It uses 60 probe states, 256 scoring
rollouts, 256 realized deployments, 100,000 prior particles, a 256-particle posterior
support, and no LLM calls.

## Sensitivity Trail

| Replay | Support | Depth | Task scorer rho vs smooth target | Outcome |
|---|---|---:|---:|---|
| Initial diagnostic | independently resampled | 2 | 0.170 | fail |
| Initial diagnostic | independently resampled | 3 | 0.191 | fail |
| Initial diagnostic | independently resampled | 5 | -0.026 | fail |
| 256-sample convergence | independently resampled | 1 | 0.278 | fail |
| 256-sample convergence | independently resampled | 3 | 0.145 | fail |
| 1024-sample convergence | independently resampled | 1 | 0.262 | fail |
| Canonical support repair | common prior + systematic resampling | 1 | 0.231 | fail |

The rollout-noise explanation was falsified. At depth 1, increasing the integration
budget from 256 to 1024 reduced mean score SE relative to between-candidate spread from
0.316 to 0.162, but rho decreased from 0.278 to 0.262. Replacing independent support
resampling with a common prior pool and deterministic systematic resampling also did not
recover the gate.

## Where The Link Breaks

For the canonical depth-1 replay, rho is 0.055 at the prior state (round 0), 0.305 at
round 3, and 0.335 at round 6. The task-loss scorer becomes useful after observations
concentrate the posterior, but before any data it cannot rank which query will reduce
risk for the particular hidden truth drawn in that trial. This is not fixed by more
rollouts because the scorer correctly averages over the prior while the realized target
is conditional on one unknown truth.

The point-RMSE rankability ceiling is also low: realized posterior-risk reduction versus
realized point-RMSE reduction has rho 0.165 in the canonical replay. Thus even the smooth
target only weakly ranks the primary endpoint at these probe states. Later-round strata
are encouraging but were not preregistered as a separate gate and are not used to claim
a pass.

## Scope And Provenance

The legacy Path A records retain candidate roots, histories, and hidden states, but not
rollout posterior snapshots or future LLM query trajectories. This analysis therefore
uses the specified local fallback. At depth 1 there is no continuation-policy mismatch:
each stored candidate root is scored and deployed directly with the exact location
likelihood and a prior-particle posterior. The estimated scorer never accesses the
hidden truth. Realized targets use observations generated from the fixed recorded truth.

Canonical artifacts:

- `results/ranking_fidelity/PATH_B_GATE0_TASK_LOSS.md`
- `results/ranking_fidelity/path_b_gate0_task_loss.json`
- `plots/ranking_fidelity/path_b_gate0_task_loss.png`

This supports the Goal playbook's Gate-0-failure framing: entropy and task-loss rollout
scores both fail to robustly rank realized localization gains. The evidence does not
authorize a claim that task-aligned arbitration will beat naive or greedy EIG.
