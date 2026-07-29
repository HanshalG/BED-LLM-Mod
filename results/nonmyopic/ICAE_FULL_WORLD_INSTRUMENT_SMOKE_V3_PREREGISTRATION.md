# ICAE Selective Full-World Instrument Smoke V3

Date: 2026-07-29

## Purpose

Run the final transport-qualified version of the selective full-world
instrument. V1 failed because valid explicit indices arrived out of array
order. V2 accepted arbitrary order but failed when a branch refresh duplicated
world and question IDs. Neither run reached the hidden endpoint, and neither
is normalized, repaired, or resumed.

V3 removes all model-generated bookkeeping identifiers. Code assigns world
and question identity from fixed-size positional arrays. Likelihoods,
retention, and hidden coverage use nested fixed-size matrices. Released ICAE
trigger IDs remain because they are semantic environment actions, not
bookkeeping. The full-world belief semantics and every scientific gate remain
unchanged.

## Frozen Task And Models

Apply the same public mechanics eligibility rule and select the task minimizing
`SHA256("52000:<alias>")`:

- `realcode@212` (C#).

This task was already opened by the mechanics source audit but has not been
used by either full-world attempt.

Models:

- GPT-5.4 non-thinking, seed `52100`: world generation and refresh;
- GPT-5.4 Mini non-thinking, seed `52200`: likelihood matrices, retention
  vectors, semantic trigger match, and duplicate coverage matrices.

## Exact Ten Calls And Gates

The exact ten stages and scientific thresholds remain identical:

1. initial eight-world/six-question belief;
2. six hypothetical positive answers;
3. 8-by-6 likelihood matrix;
4. positive branch refresh;
5. fallback branch refresh;
6. two branch-retention vectors;
7. realized exact-controller match;
8. realized refresh;
9. 8-by-N per-world hidden-coverage matrix;
10. exact endpoint repeat.

All must pass:

- exact 10 accepted/HTTP requests and zero retries, reasoning, or forced
  exits;
- valid unique full-world supports, probability sum one, effective world
  count at least `2.0`;
- at least three likelihood values;
- both branches and actual history change at least four questions;
- branch supports differ and retention counts differ by at least one;
- realized root matches a released trigger;
- duplicate endpoint matrices are exact;
- probability-weighted coverage is in `[0.20, 0.90]`;
- per-world coverage range is at least `0.10`;
- cost is at most `$0.50`.

No response repair, reissue, retry, task/model/seed substitution,
development-or-later partition, or executable test endpoint is allowed.
Passage authorizes only a separately frozen paired measurement on an unopened
cohort. Scientific failure closes the full-world route; another serialization
variant is not authorized.
