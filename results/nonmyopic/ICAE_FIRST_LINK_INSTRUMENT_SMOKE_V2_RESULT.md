# ICAE Model-Aware First-Link Instrument V2 Result

Date: 2026-07-29

Status: **gated null; the exact ICAE first-link instrument is closed**.

## Frozen Run

- preregistration commit: `f15954d`
- run: `icae-first-link-instrument-v2-20260729T054230Z`
- selected task: `realcode@023` (Ruby)
- public serving SHA-256:
  `52ecf603772281f24be1bd92a1c3b2e729a54623cb8f4516558847536bef6b13`
- private raw SHA-256:
  `1214b1dc82aa853a505827cd7c23014271ba7fc72572a02652b1d599a61a9785`

An earlier zero-call launch used the wrong local source-root argument and
failed before reading the task record or contacting a model. It cost `$0`,
opened no endpoint, and is preserved under run
`icae-first-link-instrument-v2-20260729T054210Z` with public failure SHA-256
`891b306bac95f751bffa94684a64c4ccd5ed02b2619876a467b7047511bcb5f2`.
The unchanged preregistered code, task, models, and seeds were then run with
the correct hash-pinned ICAE checkout.

## Accounting

| Metric | Result |
|---|---:|
| accepted requests / expected | 10 / 10 |
| HTTP attempts / expected | 10 / 10 |
| retries / provider-error retries | 0 / 0 |
| reasoning tokens / forced exits | 0 / 0 |
| prompt / completion tokens | 12,357 / 5,293 |
| cost | `$0.08007025` |

The prospective set-valued matcher codec worked: the realized first question
matched exactly one released trigger, code returned the exact stored response,
and the realized history changed all six generated followup questions.

## Formal Failures

Two preregistered scientific gates failed:

| Gate | Result | Required |
|---|---:|---:|
| positive branch changed questions | 6 / 6 | at least 4 |
| fallback branch changed questions | 2 / 6 | at least 4 |
| positive/fallback supports differ | yes | yes |
| hidden endpoint coverage | 9 / 9 (`1.00`) | `[0.20, 0.90]` |

The branch-retention proxy did detect a small path effect: the positive branch
retained 10 of 12 initial hypotheses and the fallback branch retained 11, so
the frozen one-hypothesis minimum was met. The generated semantic likelihood
matrix was also nonconstant, with 30 distinct values.

## Interpretation

The controller is no longer the bottleneck. The remaining failure is the
LLM-generated belief instrument itself:

- a fallback observation barely changed the next-question set and retained
  nearly the entire initial support; and
- the final 12-particle support was broad enough for the independent judge to
  mark every released hidden requirement covered.

That combination leaves no calibrated endpoint range for comparing myopic
and non-myopic policies. The support generator demonstrates path dependence,
but mostly by paraphrasing or adding broad requirements rather than making
selective, falsifiable belief updates. Relaxing the four-question or
unsaturated-coverage thresholds after observing this task would convert the
instrument into a self-confirming metric.

Per the preregistration, do not rerun this task, alter its prompts or
thresholds, or proceed to a paired ICAE policy measurement. A future ICAE
route would need a genuinely different belief representation with selective
mass or externally executable particles, not another matcher repair.
