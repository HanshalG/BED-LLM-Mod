# RevengeBench deterministic replay result

Date: 2026-08-13

Status: **pass (4/5 arenas; one infrastructure exclusion)**

OpenRouter calls/cost: **0 / $0**

OATML cluster use: **none**

## Result

Four arenas pass the frozen fresh-process replay gate:

| Arena | Complete trajectory per arm | Target actions per arm | Exact paired hash |
|---|---:|---:|---|
| BattleSnake | 303 states | 302 | states `84adcd1c...`, actions `f3bc3b92...` |
| Halite | 301 frames | 8,463 nonzero owned-cell moves | `3fa4e2b9...` |
| HuskyBench | 3 games | 9 | `96b2ab70...` |
| RoboCode | 875 turns | 724 nontrivial state transitions | record `40d5d04c...`, score `0f8aa9ce...` |

For every passing arena, the two fresh arms have identical canonical target
states, target actions, trajectory lengths, terminal results, normalized scores,
and action-distance evaluation inputs. The adjudicator therefore advances the
route to the prospectively frozen source-opportunity audit.

RobotRumble is `infrastructure_pending`: its pinned release contains an x86-64
engine binary and the local ARM Docker runtime cannot execute the pinned amd64
image. This is the single infrastructure exclusion allowed by the frozen gate;
it is not counted as scientific failure.

## Reproducibility repairs

The repairs affect only process-level replay mechanics:

- Halite uses the pinned public SDK header omitted from generated target folders
  and a preload shim that freezes wall-clock calls while preserving monotonic
  engine timing.
- HuskyBench uses a Python startup shim that freezes `random`, wall-clock time,
  and UUID generation.
- RoboCode receives the frozen JVM random seed.
- BattleSnake uses the previously banked pinned public server template repair.

All successful evidence comes from fresh processes. Pre-game build and launch
attempts that produced no valid trajectory were discarded and were not included
in adjudication.

## Scope

This result establishes deterministic paired mechanics only. It does not show a
non-myopic structural gap, valid semantic beliefs, planner efficacy, or held-out
endpoint benefit. No target source, raw trajectory, released prior outcome, or
target provenance is serialized in the public result.

Machine result: `revengebench_deterministic_replay/AUDIT.json`

Machine-result SHA-256: `f0713d796a9f011cde0c24bf3d3c29c6359bbcb257e7f5cddfd8ba6231a0585a`
