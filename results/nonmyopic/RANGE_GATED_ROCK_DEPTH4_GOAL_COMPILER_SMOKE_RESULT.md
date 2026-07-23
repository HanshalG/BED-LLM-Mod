# Range-Gated Rock Hierarchical H4 S0 Result

The preregistered Gemma 4 26B thinking smoke passed every serving, mechanics,
and mechanism gate.

## Frozen Setup

- Architecture: LLM target assignment, deterministic shortest-path compiler,
  exact h4 verifier.
- Seed: `24228`.
- Cells: 10 distinct corner-start range-gated beliefs.
- Projection: none.
- Preregistration:
  `RANGE_GATED_ROCK_DEPTH4_GOAL_COMPILER_PREREGISTRATION.md`.

## Result

| Gate | Result |
| --- | ---: |
| Legal four-plan cells with distinct targets | 10/10 |
| Rock 4 assigned to fixed `move-NORTH` root | 10/10 |
| `North, North, North, check-4` compiled | 10/10 |
| Exact h4 route selected | 10/10 |
| Accepted logical cells | 10/10 |
| LLM calls during scoring | 0 |

Gemma assigned rock 4 to the north root on every cell. The deterministic
compiler therefore generated the exact load-bearing h4 route on every cell, and
the unchanged exact verifier selected it every time. The remaining three targets
varied with belief and root.

Three initial logical responses were provider truncation notices rather than
JSON. The single frozen correction recovered all three. Nine bounded forced-final
continuations were requested and all nine succeeded. Every physical request is
accounted.

## Usage

- Physical requests: 22.
- Prompt tokens: 31,173.
- Completion tokens: 54,161.
- Reasoning tokens: 37,616.
- Forced exits: 12.
- Forced-final requests/successes: 9/9.
- Cost: `$0.02324779`.
- Project spend after S0: `$40.26268143245984 / $110`.

## Interpretation

This isolates the previous h4 failure to low-level action-sequence composition
and serving, not semantic sensing-goal selection. It is still serving/mechanism
evidence only. Per preregistration, the pass authorizes fresh S1 seed `24229`;
no h4 trajectory is authorized yet.
