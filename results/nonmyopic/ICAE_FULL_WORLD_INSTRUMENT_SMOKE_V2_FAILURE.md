# ICAE Full-World Instrument V2 Failure

Date: 2026-07-29

Status: **failed closed on duplicate semantic keys after four calls; no hidden
endpoint was reached**.

## Frozen Run

- preregistration commit: `cd12bef`
- run: `icae-full-world-instrument-v2-20260729T055518Z`
- selected task: `realcode@015` (Python)
- public failure SHA-256:
  `203c6801b08cca3395e498ed4cfdfc9fa7514e847c6d892c66953783ad433341`
- private partial-response SHA-256:
  `5c9d42e9a09f0c14fd1b04c348605ed301b1f8e48b27ac1a258d824b2efa62be`

## Result

| Metric | Result |
|---|---:|
| accepted requests before failure | 4 |
| HTTP attempts | 4 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 6,046 / 3,977 |
| cost | `$0.06406` |

The initial support, six hypothetical answers, and complete likelihood matrix
all parsed. The positive-branch refresh then returned:

```text
world indices:    1, 2, 3, 4, 5, 6, 7, 5
question indices: 1, 2, 3, 4, 5, 5
probability sum:  100
```

V2 correctly rejected the duplicate keys. Calls 5--10 were not made. No
fallback branch, controller response, hidden coverage, or executable endpoint
was opened.

## Interpretation

Ignoring array order fixed V1's incidental-order failure, but model-generated
bookkeeping IDs remain unreliable. Reassigning either duplicate after seeing
the response would be a repair, so V2 remains failed and is not resumed.

The final principled transport variant is a compact positional codec with no
model-generated identifiers: code assigns world/question identity from
fixed-size arrays, and matrices are nested fixed-size arrays. This changes
serialization only, not the full-world scientific method. It requires a new
already-open task and fresh model seeds.
