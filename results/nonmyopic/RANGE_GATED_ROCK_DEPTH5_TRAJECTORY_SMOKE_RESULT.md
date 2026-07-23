# Focused Range-Gated Rock H5 Trajectory S0 Result

The preregistered Gemma 4 26B thinking late-state smoke passed every serving,
mechanics, and exact-root gate.

## Frozen Setup

- Architecture: semantic target assignment, deterministic three-slot routing,
  exact h5 verification.
- Seed: `24244`.
- Cells: 12 fixed trajectory-prefix states covering the empty history, the first
  three movement prefixes, and one- through three-repeat remote rock-6 outcomes.
- Projection: none.
- Preregistration:
  `RANGE_GATED_ROCK_DEPTH5_TRAJECTORY_PREREGISTRATION.md`.

## Result

| Gate | Result |
| --- | ---: |
| Valid four-plan cells with distinct targets | 12/12 |
| Selected root matches exhaustive h5 | 12/12 |
| Empty history selects registered five-action route | Yes |
| Accepted on first logical attempt | 12/12 |
| LLM calls during exact scoring | 0 |

The interface stayed aligned after both movement and belief updates, including
the deepest repeated-check histories. Every reasoning pass reached the registered
allowance; each bounded non-reasoning final returned valid target JSON. No
validation retry or projection was used.

## Usage

- Physical requests: 24.
- Prompt tokens: 34,974.
- Completion tokens: 50,982.
- Reasoning tokens: 34,730.
- Forced exits: 12.
- Forced-final requests/successes: 12/12.
- Cost: `$0.02126545`.
- Project spend after S0: `$40.36448656245983 / $110`.

## Interpretation

This closes the late-state interface risk left by the one-state h5 proposal gate:
semantic target assignments remain sufficient for exact h5 root recovery across
all registered trajectory prefixes. It is serving and mechanics evidence only.
Per preregistration, the pass authorizes unchanged 50-pair trajectory seed
`24245`; no trajectory endpoint has yet been observed.
