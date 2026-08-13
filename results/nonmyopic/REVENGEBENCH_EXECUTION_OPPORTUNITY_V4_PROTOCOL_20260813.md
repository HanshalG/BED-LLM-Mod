# RevengeBench Halite feeder-corrected execution-opportunity V4 protocol

Date: 2026-08-13

Status: **frozen before any corrected-feeder Halite trajectory**

## Exact repair

V4 inherits the complete V3 manifest and changes only the public Halite offline
feeder's frame timing:

```text
GetInit input: player tag, dimensions, productions, frame 0
read bot name
for recorded action i = 0..63:
    send replay frame i to GetFrame
    read action i
```

Frame 0 is intentionally sent once in the initialization packet and again to
the first `GetFrame()` call. This matches the pinned public C SDK lifecycle. The
old release behavior (`send frame i+1; compare with move i`) is forbidden.

No game engine, target/probe source, seed, dimension, wall-clock shim, player
order, parser, action distance, likelihood, beta, planner, threshold, or endpoint
changes. All V3 Halite trajectories are discarded and cannot be reused.

## Decision rule

BattleSnake remains complete and non-qualifying. Therefore both Halite and
HuskyBench must independently satisfy the unchanged robust changed-first-probe,
at-least-0.01-nat, and endpoint-separation gates for the overall opportunity
stage to pass. Any self-distance, replay, compile, or parser failure remains
fail-closed. V4 authorizes zero model calls and no efficacy claim.
