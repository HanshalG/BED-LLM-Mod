# HiL-Bench Support-Expansion V2 Smoke Result

Run on 2026-07-25 using GPT-5.4 non-thinking through OpenRouter.

## Result

**V2 failed closed during policy-response parsing. No hidden endpoint or
support-recovery metric was loaded.**

- Eight policy requests and eight HTTP attempts completed.
- Retry count, reasoning tokens, and forced exits were all zero.
- Both initial responses and four of six regeneration responses were valid.
- The two evidence-conditioned refreshes for `sql_45` were malformed at their
  final character: each returned four complete question strings and then `}`
  without closing the `blocker_questions` array.
- Both responses had provider finish reason `stop`, so this was not output-token
  truncation.
- The strict parser rejected the batch before either hidden blocker registry or
  either judge call was loaded.
- Adapter cost was `$0.035035`.

The preserved raw checkpoint has SHA-256
`58cce71b99e607b8523f8820167ce4e3e33c6065c5faa2ce2f53192a2ccbd944`.
No closing bracket was inserted, no first-object extraction was used, and no
endpoint was computed diagnostically.

## Interpretation

Removing V1's non-load-bearing free-text hypothesis label did not yield a
reliable exact interface. V2 failed on malformed JSON in two of six refreshes,
despite a flat question-only schema, zero reasoning, temperature zero, ample
output allowance, and `stop` completion.

Provider-enforced structured output is not an available repair on this route:
the separately preregistered GPT-5.4 Chat and Responses strict-schema smokes
already returned OpenRouter 404 with `require_parameters=true`. A permissive
closing-bracket repair would violate V2 and would conceal a 2/6 generation
failure rate.

The HiL-Bench route is therefore closed for now. Its zero-call structural result
remains valuable: the release contains genuine progressive evidence and hidden
external blockers. What remains unmeasured is the load-bearing causal link that
observation-conditioned LLM support recovers those blockers. Reopen only if a
reliable structured-output serving path becomes available; do not spend on a V3
prompt/parser rerun.

Public failure:
`results/nonmyopic/hil_bench_support_expansion_v2_smoke/hil-support-expansion-v2-smoke-20260725T164500Z/SMOKE_FAILURE.json`.
