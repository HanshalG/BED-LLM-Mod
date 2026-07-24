# Paprika Structured Unlock Apparatus Amendment

Date: 2026-07-24

The first development execution reached all 12 final semantic-coverage calls but wrote
no endpoint records. Parsing stopped because one response reported a
`best_hypothesis_index` outside its support. The index is explanatory metadata; the
frozen endpoint and every pass threshold depend only on `best_match_score`.

Before any gate metric was inspected, the parser was changed to:

- preserve every valid `[0,1]` score unchanged;
- retain the raw reported index;
- set an invalid index to null and mark `best_hypothesis_index_valid=false`;
- continue to reject changed support IDs/order, missing scores, out-of-range scores, or
  missing reasons;
- preserve the raw response if a future coverage row still fails.

No task, seed, model, prompt, support size, semantic threshold, or gate criterion
changed. The failed apparatus attempt used 229 requests, zero reasoning, and
`$0.03862585`. Its only artifact is
`paprika_structured_unlock/development_seed24287_20260724/DEVELOPMENT_FAILURE.json`.
The rerun uses a new run ID and remains subject to the original `$2` cap.

## Second Apparatus Failure

The v2 execution also wrote no endpoint records. It stopped before semantic judging
because one refreshed list still failed the eight-unique-string parser after bounded
repairs. It used 171 Gemma requests, zero reasoning, and `$0.01201205`.

Inspection of the earlier serving failure had already established the equivalent schema
drift: Gemma sometimes expresses each requested hypothesis as
`{"cause":"...","remedy":"..."}` instead of one combined string. Before another run,
the unlock runner was changed to normalize either representation into the same
cause-and-remedy sentence and still require exactly eight unique, nonempty hypotheses.
This normalization is applied identically to initial and refreshed supports. It does
not add, drop, judge, or rewrite semantic content and does not alter any endpoint or
gate threshold. The v2 failure is preserved at
`paprika_structured_unlock/development_v2_seed24287_20260724/DEVELOPMENT_FAILURE.json`.

The v3 execution reached the same pre-judge refinement stage and again produced no
endpoint records. It used 171 Gemma requests, zero reasoning, and `$0.01183416`. The
normalizer still required exactly eight returned items and recognized only literal
`cause`/`remedy` keys. Before v4 it was extended, without changing retained support
size, to accept common equivalent field names (`problem`, `hypothesis`, `solution`,
`fix`, and `recommended_action`) and to retain the first eight unique hypotheses when
an over-complete list is returned. Fewer than eight usable semantic hypotheses still
trigger bounded repair and then fail closed with parsed-count telemetry. The v3 failure
is preserved at
`paprika_structured_unlock/development_v3_seed24287_20260724/DEVELOPMENT_FAILURE.json`.
