# HiL-Bench Support-Expansion Smoke Result

Run on 2026-07-25 using GPT-5.4 non-thinking through OpenRouter.

## Result

**V1 failed closed during policy-response parsing. No scientific endpoint was
loaded or measured.**

- Two initial and six regeneration responses completed: 8 physical requests and
  8 HTTP attempts.
- Retry count, reasoning tokens, and forced exits were all zero.
- One returned refresh object contained a hypothesis outside the frozen
  12--280-character range.
- The strict parser rejected it before either hidden blocker registry was
  loaded and before either post-freeze judge call.
- Adapter cost was `$0.050995`.

The failure checkpoint has SHA-256
`2ad638e8b443ace9e2743958111df51455f03fd3058b3a432f15db1d08fc1d90`.
Because refresh responses were checkpointed only after the batch-wide parse in
V1, the offending response was not persisted. That ordering bug is now fixed,
but it does not authorize response recovery, a parser amendment, or a rerun of
V1.

## Interpretation

This is a serving-grammar and observability failure, not evidence for or against
observation-conditioned support expansion. Initial responses parsed, all paid
calls were non-thinking and retry-free, and hidden endpoints remained sealed.
The exact V1 interface and cases are closed.

Any follow-up must be a separately frozen protocol on fresh development tasks.
The justified change is to remove the non-load-bearing free-text hypothesis
length constraint, while retaining exact support count, distinct questions,
the matched no-evidence control, delayed endpoint loading, all scientific
gates, and the hard budget reserve.

Public failure:
`results/nonmyopic/hil_bench_support_expansion_smoke/hil-support-expansion-smoke-20260725T162000Z/SMOKE_FAILURE.json`.
