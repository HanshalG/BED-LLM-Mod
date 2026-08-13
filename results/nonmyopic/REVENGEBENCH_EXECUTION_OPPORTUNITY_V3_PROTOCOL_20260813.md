# RevengeBench deterministic-mechanics execution-opportunity V3 protocol

Date: 2026-08-13

Status: **frozen before any repaired-engine trajectory**

## Motivation

V2 is infrastructure-inconclusive before scientific scoring because the pinned
BattleSnake CLI iterates a Go map when associating fixed player names with
seeded starting positions. The same explicit game seed can therefore put target
and probe in opposite starting corners across fresh processes.

## Exact mechanics repair

Apply only `patches/revengebench_battlesnake_deterministic_players.patch` to
upstream BattleSnake commit `aaf48003cfa5034a14d7053bb0ccc0bc6ae99cee`.
The patch:

- generates snake IDs as `revengebench-<game-seed>-<option-index>` when the CLI
  test-only ID generator is absent; and
- sorts `SnakeState` values by fixed player name, then ID, before initialization,
  request, update, and exported-request traversal.

It does not alter a target/probe policy, board rule, random seed, starting-point
algorithm, food process, action parser, distance function, or terminal scoring.
Build the repaired ARM CLI in the existing pinned Python/Flask runtime and bind
the resulting image ID.

Before opening the full cohort, the repaired CLI must pass at least three fresh
arms of the same frozen BattleSnake cell with identical canonical initial named
positions, complete target states/actions, terminal result, and candidate
action-distance inputs. Failure closes BattleSnake for V3.

## Unchanged experiment

Use the exact V2 manifest and protocol for BattleSnake, Halite, and HuskyBench:
the same three hidden hypotheses, three selected probes, paired probe seeds,
endpoint seeds, first-64-decision cap, native parsers/distances, betas
`{0.5,1.0,2.0}`, uniform prior, exact depth-two planner, compute-matched receding
myopic, fixed, random, and all calibration/separation gates.

The headline opportunity bar remains unchanged: at least two of three arenas
must change the depth-two first probe at every beta, gain at least `0.01` nats in
every changed arena/beta, and pass held-out endpoint separation.

All BattleSnake V2 trajectories are discarded rather than reused. V3 authorizes
zero model calls, zero OATML cluster use, and no efficacy claim.
