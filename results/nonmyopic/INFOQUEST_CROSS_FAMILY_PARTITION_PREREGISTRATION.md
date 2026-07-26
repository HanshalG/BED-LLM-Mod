# InfoQuest Cross-Family Partition Preregistration

Frozen after the cached-answer first-link diagnostic and before any
cross-family scorer response.

## Purpose and Inputs

The cached-answer diagnostic isolates the failure at question ranking.
V3 used one GPT-5.4 response to regenerate support and estimate the weights and
answer partitions that exact EIG then scored. This development gate tests the
specific hypothesis that same-model support/likelihood coupling creates
self-confirming but uncalibrated entropy.

It binds the V3 support source and common histories:

- V3 public/private SHA-256:
  `0cfbe3d1590001af508d351d131d12d747f2fcadf0448e2e7868809f40d968f7`
  and
  `134a9659dc5f86df3dba6e18fdc8c72a79f3524b166d2cd52d91a7fe88747e1e`;
- common-history public/private SHA-256:
  `140f77447da9bd3d83e8a7be7fd45dc8fc792422d4e1cef398f6d8460d13ccc3`
  and
  `c4dec013386083afb4279754f4ffb1b0a2f6559bb2a2cf1863d80a849fe6e93e`.

Only V3's 30 regenerated K8 supports and 30 exact-copy fixed K8 supports are
reused. V3 weights, partitions, scores, choices, simulator answers, checklist
judgments, and metrics are not estimator inputs.

## Frozen Estimator

For every dynamic and fixed support, a fresh Gemini-2.5-Flash non-reasoning call
sees the observed history, eight hypotheses, and four remaining roots. It emits
exactly five arrays: eight posterior weights and four eight-hypothesis semantic
answer partitions. It does not choose a root and never sees the checklist.

An exact deterministic entropy scorer chooses `A-D`. The matched-world answer
for that root is taken from the frozen counterfactual answer bank, so there are
no fresh simulator calls and identical choices receive identical answers. Six
fresh GPT-5.4 Mini checklist calls score the paired endpoints; identical paths
must receive identical bits.

## Gates and Budget

A synthetic gate must pass exactly three requests/HTTP attempts (two Gemini
partition scores and one checklist), every parser and identical-path invariant,
zero retry/reasoning/forced exits, and a `$0.03` cap.

Only a serving pass authorizes one mechanics run:

- 30 dynamic Gemini likelihood calls;
- 30 fixed Gemini likelihood calls;
- 0 support-generation and 0 simulator calls;
- 6 GPT-5.4 Mini checklist calls;

for exact 66 physical requests/HTTP attempts and a `$0.15` cap.

The same twelve V3 scientific gates are unchanged. Any pass remains
disclosed-world development evidence and requires separately preregistered
fresh-record confirmation before a positive paper claim. Failure closes this
exact cross-family route without repair, model swap, threshold change, or
rerun.

The pre-Monday operational allowance is `$2.08299670`; maximum route cost is
`$0.18`. OatML jobs: `0`.
