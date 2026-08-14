# ChemBench M-open Mechanics V1 Development Amendment

Date frozen: 2026-08-14 (Europe/London)

No `v3`, `v4`, or `v5` response matrix has been constructed. This amendment is
based only on a disposable screen over the already opened `easy/v2`,
`medium/v2`, and `hard/v2` source states.

## Development Screen

After fixing represented-mass underflow and retaining evidence-weighted reserve
models in posterior prediction, the registry-oracle screen produced:

| Horizon | Aggregate terminal MSE |
| --- | ---: |
| d1 | 0.07614260 |
| d2 | 0.04056918 |
| d3 | 0.03670768 |

The reductions are 46.72% for d2 versus d1 and 9.52% for d3 versus d2. d2 and
d1 selected different roots on all three opened slices. At the inherited
floating comparison tolerance `1e-12`, d3 versus d2 had 45 wins, 81 ties, and
45 losses. Many counted differences are numerically real but scientifically
negligible: at an absolute terminal log-rate MSE tolerance of `1e-6`, the same
comparison is 42 wins, 90 ties, and 39 losses.

The screen made no model or network calls, cost `$0`, wrote only disposable
artifacts under `/tmp`, and has no authority over the untouched cohort.

## Frozen Clarifications

1. Reserve structures retain their evidence and participate in posterior
   prediction and terminal risk. `live` versus `reserve` controls prompt and
   proposal management, not whether a discovered model is silently assigned
   zero predictive mass.
2. Represented conditional weights are stored separately from represented
   joint mass. If unknown support dominates enough for joint masses to
   underflow, conditional represented forecasts remain mathematically defined;
   no pseudocount is added to model evidence.
3. The paired truth-cell gate uses a prospectively frozen practical tie
   tolerance of `1e-6` absolute terminal log-rate MSE. Exact `1e-12` counts are
   still reported descriptively. Aggregate 5% thresholds and every other V1
   condition are unchanged.

The `1e-6` tolerance is less than 0.003% of the opened d3 aggregate MSE and far
below the benchmark's `0.01` RMSLE exact-accuracy scale. It prevents floating
and immaterial forecast differences from determining a directional cell gate;
it does not alter any action, belief, proposal, loss, or aggregate threshold.

This amendment must be committed and pushed with the implementation, focused
tests, independent verifier, and exact protocol hashes before the one permitted
`v3` command.
