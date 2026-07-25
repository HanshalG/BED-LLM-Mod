# HiL-Bench Support-Expansion V2 Smoke Preregistration

Frozen on 2026-07-25 after V1 failed before endpoints and before any V2 model
response.

## Amendment

V1's free-text `hypothesis` label had a frozen 12--280-character constraint.
One of eight otherwise complete policy responses violated it, and V1 closed
without loading a blocker registry. The label was not used by retrieval,
matching, or any scientific metric.

V2 removes that field entirely. Each generated support is exactly four distinct
single-topic clarification questions. The questions themselves encode the
candidate blocker support and are the only objects evaluated. This avoids an
unnecessary free-text surface rather than normalizing or broadening a failed
response.

## Fresh Cases And Unchanged Protocol

- Official HiL-Bench commit
  `352d14c861f2531949dfa91848d4b2fe46b8a247`.
- Fresh development tasks `sql_72` and `sql_45`; V1 tasks are excluded.
- Seed `24394`.
- GPT-5.4 through OpenRouter, temperature zero, explicitly non-thinking.
- Exactly eight policy calls: initial support/searches, matched problem-only
  regeneration, and two evidence-conditioned regenerations per task.
- Hidden registries load only after all eight policy responses freeze.
- Exactly two post-freeze GPT-5.4 matching calls; ten physical requests total.
- Same BM25 top-five observation transition, four-question support size,
  two-root width, matched no-evidence control, scientific metrics, thresholds,
  `$0.75` cap, and no repair/reissue.
- No OatML cluster use.

## Frozen Gates

All must pass:

- exactly ten physical requests and HTTP attempts;
- zero retries, reasoning tokens, and forced exits;
- all question/search/judge objects parse with exact counts and distinct
  single-topic questions;
- hidden blocker data is unavailable to all policy calls;
- the two retrieved document lists differ on each task;
- all four generated supports differ on each task;
- at least one task's best evidence branch matches a blocker absent from both
  problem-only supports;
- at least one such new blocker has official type `business info`;
- best-evidence matched-blocker total exceeds both initial and matched
  no-evidence totals;
- cost is at most `$0.75`.

Failure closes V2 without parser, prompt, cohort, threshold, or model repair.
Passage authorizes only a separately preregistered paired planning experiment.
It is not itself evidence that non-myopia beats a myopic policy.

## Budget

V1 spent `$0.050995`. The local ledger is `$86.96040621920734` spent against
`$101.14330481920742`, while the pre-V1 authenticated credit balance was
`$43.475391484`. V2's hard cap preserves the `$25` through-Monday reserve.
