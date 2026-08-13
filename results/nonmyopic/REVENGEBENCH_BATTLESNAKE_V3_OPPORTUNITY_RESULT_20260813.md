# RevengeBench BattleSnake V3 opportunity result

Date: 2026-08-13

Status: **complete arena; no robust horizon opportunity**

OpenRouter calls/cost: **0 / $0**

## Mechanics

The deterministic CLI repair passed the previously failing cell in three fresh
arms. The full frozen BattleSnake cohort then completed 27 target/probe/seed
cells and 54 fresh games. Every paired arm, terminal result, retained target
state/action sequence, and candidate action-distance input is exact. All 27
true-hypothesis self-distances are zero. Retained decisions range from 12 to the
frozen cap of 64.

## Opportunity result

All three probe likelihoods are nondegenerate, but BattleSnake does not satisfy
the non-myopic gate:

| Beta | Myopic first | Depth-two first | Depth-two utility | Margin over myopic policy |
|---:|---:|---:|---:|---:|
| 0.5 | 2 | 1 | 0.00208684 | 0.00000000 |
| 1.0 | 2 | 1 | 0.00832252 | 0.00000000 |
| 2.0 | 1 | 1 | 0.03264775 | 0.00000000 |

At beta 0.5 and 1.0, depth two changes only the order of probes 1 and 2. Both
policies use the same unordered pair and have exactly equal expected terminal
entropy reduction. At beta 2.0 their first probe also agrees. The frozen gate
requires a changed first probe at every beta and at least 0.01 nat gain, so this
arena is non-qualifying.

This is a clean arena-level negative, not a planner failure claim. It means the
selected BattleSnake intervention set contains no robust adaptive two-step
advantage under the frozen likelihood family.
