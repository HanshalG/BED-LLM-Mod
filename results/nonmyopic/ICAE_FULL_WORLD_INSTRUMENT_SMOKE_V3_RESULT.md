# ICAE Selective Full-World Instrument V3 Result

Date: 2026-07-29

Status: **scientific gated null; the exact full-world ICAE route is closed**.

## Frozen Run

- preregistration commit: `c29d910`
- run: `icae-full-world-instrument-v3-20260729T060045Z`
- selected task: `realcode@212` (C#)
- public serving SHA-256:
  `52ecfd958d88e083fa648a9a42003fe718d9d64590570814ec135d5af3a14ed7`
- private raw SHA-256:
  `840b0ec0d39099a4eeb06f95418703158850ce24c8e66e92871bce8bffdb00f5`

## Accounting

| Metric | Result |
|---|---:|
| accepted requests / expected | 10 / 10 |
| HTTP attempts / expected | 10 / 10 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 18,624 / 4,668 |
| cost | `$0.0892018` |

The compact positional codec passed all transport and schema checks. The
instrument reached every branch, controller, and hidden-coverage stage without
repair or retry.

## Positive Representation Result

Full-world particles solve the atomic-support saturation problem:

| Diagnostic | Result |
|---|---:|
| weighted hidden coverage | `0.2764` |
| minimum / maximum world coverage | `0.2143` / `0.4286` |
| world coverage range | `0.2143` |
| effective worlds, initial | `7.669` |
| effective worlds, positive branch | `7.862` |
| effective worlds, fallback branch | `7.669` |
| effective worlds, actual history | `7.553` |

The endpoint is both unsaturated and discriminative across worlds. Semantic
likelihoods use 15 distinct values. Positive, fallback, and realized histories
each change all six questions; all three refreshed supports differ exactly
from the initial support and from one another. The actual question matches two
released triggers and produces an exact controller response.

## Formal Failures

Two frozen gates fail:

1. The retention evaluator marks all eight initial worlds represented in both
   positive and fallback refreshed supports (`8/8` versus `8/8`), so the
   branch-retention difference is zero.
2. Duplicate hidden-coverage judgments disagree in 28 of 112 cells (`25%`).
   Probability-weighted coverage moves from `0.2764` to `0.0914`, an absolute
   change of `0.185`.

The exact clause sets have zero overlap between initial/positive,
initial/fallback, and positive/fallback supports. Thus the `8/8` retention
result is not caused by copying complete worlds; the semantic criterion
(`>=4/6` clauses preserved by any refreshed world) is too permissive or the
judge is too broad. More importantly, the independent endpoint itself is not
stable enough to measure policy effects.

## Interpretation

This is a useful partial engineering result but not an efficacy instrument.
Complete competing worlds provide selective posterior mass and a
non-saturated target, validating the representation change. The remaining
LLM semantic evaluators do not provide reproducible transition or endpoint
measurements: the retention map is branch-insensitive while duplicate hidden
coverage changes by far more than any plausible policy effect.

Per the preregistration, do not rerun this task, relax retention, majority-vote
the endpoint, or open an ICAE paired cohort. A publishable ICAE result now
requires an externally executable endpoint and deterministic compatibility
between observations and worlds, rather than another LLM judge over free text.
The exact full-world semantic-judge route is closed.
