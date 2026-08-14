# ChemBench M-open Opportunity V1 Mechanics Terminal

Date: 2026-08-14 (Europe/London)

## Decision

Close V1 as an implementation failure. It is not a scientific result and does
not test whether planning depth helps on ChemBench.

## What Happened

The protocol was pushed at commit `91a8a55d` and the implementation was pushed
at commit `9348c658` before validation. Source verification and focused tests
passed. The one permitted command then opened only the first frozen slice,
`easy/v1`, and stopped during truth-conditional replay before writing a result.

`ExactPlanner.belief_key` rounded posterior probabilities to 12 decimal places
for memoization. A reachable branch assigned the true model less than that
amount, rounded it to zero, and later attempted to condition on an observation
that was possible under that truth but had zero represented mixture mass. The
run raised on the resulting `NaN` consistency check.

The exception exposed one incomplete aggregate scalar in the console:

```text
d2 prior-averaged risk for easy/v1: 0.016291327081876972
```

No d1/d3 comparison, truth-cell table, gate, result JSON, or paper-facing
metric was produced. The planned result path does not exist.

## Accounting And Sealing

```text
model calls: 0
API calls:   0
cost:        $0
opened:      easy/v1 response matrix and one incomplete aggregate scalar
unopened:    easy/v2; medium/v0-v2; hard/v0-v2
```

The `easy/v1` slice is spent and excluded from every successor. V1 may not be
rerun or repaired in place.

## Authorized Successor

A V2 successor may use only the seven still-unopened slices. It must retain the
same source, assays, noise, four-experiment budget, exact terminal utility,
horizons, query seeds, and 5% aggregate thresholds. To avoid weakening the
original 75% slice requirement, each successive depth comparison must win at
least six of seven slices. V2 may change only posterior representation from
rounded to exact normalized floating-point weights, with an adversarial test
that proves tiny positive mass is preserved.

