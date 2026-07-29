# GuessingGame Path-BED Serving Smoke Preregistration

Date: 2026-07-29

Status: **frozen before any new model response**.

## Authorization

The zero-call source audit passed every frozen gate. This smoke binds public
manifest SHA-256
`8d1269e197c9cf05865852353b16eeadb36cc8db6231af1d429b7650b263e7b8`
and opens only its five serving cases.

## Interface

For each serving target, make two independent structured retrieval calls:

- one receives only the released material question and answer;
- one receives only the released primary-function question and answer.

Each request supplies the complete released 858-object vocabulary under
zero-based integer IDs. The target is uniformly drawn from that vocabulary,
but no target marker, target ID, unasked answer, mechanics case, development
case, or confirmation case is supplied.

Each response must contain exactly 32 unique vocabulary IDs and integer
weights in `[1,100]`. Response order is semantically irrelevant; rank is
defined prospectively by descending weight and then ascending ID. No
`uniqueItems` schema keyword is used because uniqueness is checked locally.

## Exact Ten Calls

- model: `openai/gpt-5.4-mini`;
- non-thinking;
- seed `39500`;
- temperature `.3`;
- concurrency `10`;
- exactly five material plus five function requests;
- no retry, repair, continuation, reissue, model substitution, or reasoning
  fallback;
- projected cost `$0.05`; hard cap `$0.20`.

## Frozen Gates

All gates are conjunctive:

- exactly 10 accepted requests and 10 HTTP attempts;
- zero retries, provider-error retries, reasoning tokens, and forced exits;
- all ten schemas parse with exactly 32 unique valid IDs;
- every support has at least four distinct weights and max/min weight ratio at
  least `2`;
- material-only retrieval contains the exact hidden target on at least two of
  five cases, including at least one target in its top 16;
- function-only retrieval contains the exact hidden target on at least four of
  five cases, including at least three targets in its top 16;
- every target appears in at least one of its two supports;
- material and function supports have Jaccard overlap at most `.75` on every
  case; and
- total cost is at most `$0.20`.

Failure closes this exact model/prompt/case/seed/support-size interface. There
is no semantic repair, target removal, threshold change, or rerun. A provider
schema rejection before a response may authorize a prospectively frozen
transport-only codec change.

Passage authorizes only a separately frozen 10-case mechanics experiment
comparing material-first and function-first belief order. It does not
authorize development or confirmation.

## Dry Verification

Before any live response:

- source and serving tests must pass;
- an exact ten-call deterministic fixture must pass; and
- implementation, tests, and this preregistration must be committed and
  pushed.
