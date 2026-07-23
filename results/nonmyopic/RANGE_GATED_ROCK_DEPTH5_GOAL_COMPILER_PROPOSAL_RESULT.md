# Range-Gated Rock Hierarchical H5 S1 Result

The preregistered Gemma 4 26B thinking proposal gate and independent exact
audit passed every frozen scientific and mechanics gate.

## Frozen Setup

- Architecture: LLM semantic target assignment, deterministic shortest-path
  compiler, exact h5 verifier.
- Focused prior: rock 6 has `p_good=.5`; all other rocks have `p_good=.005`.
- S1 seed: `24240`.
- Cells: 16 distinct strict h5-over-h4 opportunities.
- Producer bootstrap seed: `24241`; audit bootstrap seed: `24242`.
- Projection: none.
- Preregistration:
  `RANGE_GATED_ROCK_DEPTH5_GOAL_COMPILER_PREREGISTRATION.md`.

## Result

| Paired comparison | Mean gain | Producer 95% CI | Audit 95% CI | W/T/L |
| --- | ---: | ---: | ---: | ---: |
| LLM targets vs matched random targets | +.330423 | [+.180731, +.450178] | [+.181306, +.450274] | 12/3/1 |
| LLM h5 vs identical compiled plans scored at h4 | +.446940 | [+.387222, +.477018] | [+.387222, +.477018] | 15/1/0 |
| LLM h5 vs strongest exact-h4 root closed at h5 | +.440652 | [+.381796, +.470080] | [+.381796, +.470080] | 15/0/1 |
| Exact h5 opportunity recovery | .937398 | [.812193, 1.000000] | [.812193, 1.000000] | 15/0/1 |

Gemma assigned rock 6 to the fixed north root and the compiler produced and
selected `North, West, West, West, check-6` in 15/16 cells, for a `.9375`
exact-route selection rate. Every cell independently remained a strict h5
opportunity, and exact scoring made no LLM calls.

All 16 target responses were valid on the first logical attempt. Each reasoning
pass reached the registered length allowance, and all 16 bounded non-reasoning
final requests succeeded. No correction retry or projection was used.

## Retained Miss

Cell 11 is the sole miss. After the secondary-rock history
`check-7=bad, check-7=good`, Gemma assigned rock 4 rather than rock 6 to the
north root. The compiler produced `North, North, North, check-4, check-4`;
its exact value was `.023781`, below the exact h5 value `.494632` and slightly
below the strongest exact-h4-root value `.024551`. The resulting recovery was
`-.00164`. This cell remains in every endpoint and interval.

## Audit

The independent audit:

- recompiled every target assignment and replayed every record;
- recomputed all LLM and control plan values;
- reproduced all producer comparisons apart from the independently seeded
  bootstrap intervals;
- independently recovered positive lower bounds for all three paired controls;
- verified the route and recovery gates and the producer result.

Every audit mechanics check passed.

## Usage

- Physical requests: 32.
- Prompt tokens: 47,787.
- Completion tokens: 67,971.
- Reasoning tokens: 42,955.
- Forced exits: 16.
- Forced-final requests/successes: 16/16.
- Cost: `$0.03263012`.
- Project spend after S1: `$40.34322111245983 / $110`.

## Interpretation

This is positive h5 proposal-quality evidence for non-myopic LLM-Modulo BED.
The LLM sees semantic uncertainty and geometry but no EIG, values, rankings,
routes, or actions. It identifies the load-bearing sensing target in 15/16
fresh strict opportunities; deterministic routing turns that target into the
four enabling moves, and exact h5 verification recovers 93.7% of the available
gain while beating matched random targets and two h4 controls.

The result is not yet a receding-horizon trajectory claim. A separate fresh-seed
paired trajectory protocol is required to show that cached or repeatedly
generated hierarchical h5 proposals improve realized entropy and truth-log
posterior over d4 and matched random controls.
