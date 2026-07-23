# Range-Gated Rock Hierarchical H5 S0 Result

The preregistered Gemma 4 26B thinking smoke passed every serving, mechanics,
and mechanism gate.

## Frozen Setup

- Architecture: LLM target assignment, deterministic shortest-path compiler,
  exact h5 verifier.
- Focused prior: rock 6 has `p_good=.5`; all other rocks have `p_good=.005`.
- Seed: `24239`.
- Cells: 10 distinct strict focused-prior beliefs.
- Projection: none.
- Preregistration:
  `RANGE_GATED_ROCK_DEPTH5_GOAL_COMPILER_PREREGISTRATION.md`.

## Result

| Gate | Result |
| --- | ---: |
| Legal five-action cells with distinct targets | 10/10 |
| Rock 6 assigned to fixed `move-NORTH` root | 9/10 |
| `North, West, West, West, check-6` compiled | 9/10 |
| Exact h5 route selected | 9/10 |
| Accepted on the first logical attempt | 10/10 |
| LLM calls during scoring | 0 |

Gemma assigned rock 6 to the north root in nine cells. The deterministic
compiler generated the exact load-bearing h5 route in those nine cells, and the
exact verifier selected it each time. In the remaining cell, Gemma assigned
rock 4 to north and rock 6 to a check root. The frozen 8/10 threshold therefore
passed without correction or projection.

Every first reasoning pass reached its length allowance. The repaired generic
adapter made one bounded non-reasoning final request for each cell, and all ten
returned valid target JSON. No validation retry was needed.

## Usage

- Physical requests: 20.
- Prompt tokens: 29,876.
- Completion tokens: 42,495.
- Reasoning tokens: 26,453.
- Forced exits: 10.
- Forced-final requests/successes: 10/10.
- Cost: `$0.01897481`.
- Project spend after S0: `$40.31059099245983 / $110`.

## Interpretation

The hierarchical semantic interface transfers from h4 to the deeper h5
opportunity: Gemma identifies the load-bearing target/root pair in 90% of fresh
belief cells while deterministic routing handles four-step spatial
composition. This is serving and mechanism evidence only. Per preregistration,
the pass authorizes unchanged S1 seed `24240`; no h5 policy trajectory is
authorized yet.
