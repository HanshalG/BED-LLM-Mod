# ICAE First-Link Instrument V1 Failure

Date: 2026-07-29

Status: **failed closed on the preregistered matcher-ID ordering rule before
the hidden endpoint opened**.

## Frozen Run

- preregistration commit: `804def3`
- run: `icae-first-link-instrument-20260729T053546Z`
- selected task: `realcode@259` (Java)
- public failure SHA-256:
  `1b8d88e2fd4c6ce6cca7e41e10a7a7947c69f3c478a72790e2dbc512ad4941fa`

## Result

| Metric | Result |
|---|---:|
| accepted requests before failure | 7 |
| HTTP attempts | 7 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 9,679 / 4,214 |
| cost | `$0.06164225` |
| development-or-later tasks opened | no |
| executable endpoint opened | no |

The seventh response contained known, unique trigger IDs, but returned them in
a different order from the released trigger catalog. V1 had frozen the
matcher parser to require catalog order, so it stopped immediately with:

```text
actual match.matched_ids is not in catalog order
```

Calls 8--10 were never made. In particular, the realized-history refresh and
both duplicate hidden-requirement endpoint judgments were not observed.

## Interpretation

This is a representation failure, not evidence for or against the scientific
first-link hypothesis. Trigger matches are semantically a set; their sequence
does not affect the exact-response controller. Nevertheless, accepting or
reordering the V1 response after seeing it would violate its frozen parser, so
V1 remains failed and is not rescored.

A distinct V2 may prospectively define matcher IDs as an unordered unique set
that is canonicalized to catalog order. It must use an untouched mechanics
task and fresh selection/model seeds. All scientific gates and the exact
ten-call limit otherwise remain unchanged.
