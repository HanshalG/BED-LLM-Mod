# RevengeBench execution-opportunity V2 mechanics result

Date: 2026-08-13

Status: **infrastructure inconclusive before scientific scoring**

OpenRouter calls/cost: **0 / $0**

## Result

The first BattleSnake smoke cell passed native counterfactual validity: the true
hypothesis reproduced all 44 retained target actions exactly, while both
off-diagonal hypotheses differed on 20.45% of decisions.

The full V2 launch then failed its mandatory fresh-arm gate before a distance
matrix or planner result was accepted. For hidden hypothesis 0, probe 1, and
paired seed 0, two fresh arms assigned the fixed `target` and `probe` names to
opposite starting corners at turn zero. The trajectories consequently retained
33 versus 18 target decisions. True-hypothesis self-distance remained zero in
both arms, but the behavioral trajectories were not equal.

Pinned CLI source localizes the defect: `buildSnakesFromOptions` returns a Go map,
and `initializeBoardFromArgs` iterates that map to construct the ordered snake-ID
list consumed by seeded starting-position assignment. Go map iteration is not
stable. The CLI `--seed` deterministically shuffles positions, but not the
mapping from player name to those positions. Runtime UUIDs compound the issue.

No V2 arena likelihood, policy utility, endpoint result, or horizon conclusion is
reported. The aborted private trajectories are not serialized and cannot be
reused by a successor.

## Prospective successor

V3 may change only the pinned BattleSnake mechanics CLI by:

1. assigning deterministic IDs from the frozen game seed and player option
   index; and
2. sorting map-derived snake states by fixed player name before every
   behavior-relevant traversal.

V3 must restart BattleSnake from scratch and retain every V2 hypothesis, probe,
seed, parser, likelihood, planner, threshold, and endpoint gate unchanged.
