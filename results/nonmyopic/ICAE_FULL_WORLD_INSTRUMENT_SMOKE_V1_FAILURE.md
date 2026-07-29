# ICAE Full-World Instrument V1 Failure

Date: 2026-07-29

Status: **failed closed on indexed-row ordering after one call; no scientific
endpoint was reached**.

## Frozen Run

- preregistration commit: `4e68219`
- run: `icae-full-world-instrument-20260729T055207Z`
- selected task: `realcode@185` (Go)
- public failure SHA-256:
  `b968e3507bdb900ea76db221875b22af0222ef5c83fc1729a37b3cbebf2bc05e`
- private partial-response SHA-256:
  `99042c27d684ebf542b365bee178fb4624041acd78d59466c29fdad6feefe928`

## Result

| Metric | Result |
|---|---:|
| accepted requests before failure | 1 |
| HTTP attempts | 1 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 546 / 1,191 |
| cost | `$0.01923` |

The response contained eight unique `world_index` values, six unique
`question_index` values, and world probabilities summing to 100. Their array
orders were:

```text
worlds:    1, 2, 3, 4, 5, 6, 7, 0
questions: 1, 2, 3, 4, 5, 0
```

V1 had frozen both arrays to appear in numeric index order, so it stopped
before call two. No likelihood, branch, controller, hidden-coverage, or
executable endpoint was opened.

## Interpretation

This is an indexed-collection codec failure. Since every row carries its own
unique index, array position has no scientific meaning. V1 nevertheless
remains failed and is not normalized or resumed.

One prospective V2 may define all explicitly indexed response collections as
unordered keyed sets and canonicalize them before use. It must use a new
mechanics task and fresh model seeds; all belief semantics, scientific gates,
and the exact ten-call limit otherwise remain unchanged.
