# ChemBench M-open Non-Myopic Opportunity V2 Protocol

Date frozen: 2026-08-14 (Europe/London)

This successor inherits
`CHEMBENCH_MOPEN_NONMYOPIC_OPPORTUNITY_PROTOCOL_20260814.md` byte-for-science,
subject only to the mechanics correction and cohort exclusion below.

## V1 Exclusion

V1 opened `easy/v1` and failed before producing a result because 12-decimal
belief-key rounding erased a low-mass truth. The exact V1 interface and slice
are terminally closed. Require the V1 terminal report and do not read or reuse
its partially exposed aggregate scalar.

## Mechanics Correction

`ExactPlanner.belief_key` must normalize nonnegative weights and preserve their
full floating-point values. It must not round, floor, clip positive values to
zero, or add pseudocounts. A focused test must show that a normalized weight of
order `1e-20` remains positive and that prior-averaged risk equals the mean of
truth-conditional replay.

No likelihood, action, objective, budget, tie rule, source adapter, or gate may
change.

## Untouched V2 Cohort

| Slice | Query seed |
| --- | ---: |
| easy/v2 | 2026081502 |
| medium/v0 | 2026081503 |
| medium/v1 | 2026081504 |
| medium/v2 | 2026081505 |
| hard/v0 | 2026081506 |
| hard/v1 | 2026081507 |
| hard/v2 | 2026081508 |

This is 399 paired truth cells (`7 x 57`). All remain unopened at freeze time.

## Frozen V2 Gate

All conditions are conjunctive:

1. Every inherited source, design, noise, query, objective, and replay check
   passes and all values are finite.
2. Aggregate mean loss improves by at least 5% for d2 versus d1.
3. Aggregate mean loss improves by at least 5% for d3 versus d2.
4. Each successive comparison improves slice-mean loss on at least six of the
   seven untouched slices.
5. d3 improves over d1 on all seven slices.
6. Each successive comparison wins on strictly more truth cells than it loses,
   excluding ties within `1e-12`.

V2 must be committed and pushed with focused tests before any of these seven
response matrices are constructed. Its output path must differ from V1. A pass
has the same narrow authority as V1: it authorizes only zero-call M-open
mechanics, not a model call or efficacy claim.

