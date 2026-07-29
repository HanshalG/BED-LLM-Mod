# Number Game Qwen Item-Isolated Serving Preregistration

Date frozen: 2026-07-29, before seed `62000` or any successor-study seed is
sent to a model.

## Motivation

The prospective 64-tree confirmation completed 36 trees and then failed on a
single unterminated JSON string, despite requesting a strict JSON schema. The
scientific endpoint was never scored. This successor changes only malformed
response handling; it does not continue the old run or reuse its responses.

## Frozen Codec

The model still receives the same strict JSON schema and open executable-rule
prompt. A valid outer document is parsed exactly as before. If and only if the
outer JSON is invalid:

1. independently decode complete JSON objects in source order;
2. retain only objects with exactly `name` and `expression`;
3. pass those objects through the unchanged executable, observation-
   consistency, and extension-deduplication filters;
4. count every absent or incomplete item as rejected.

There is no string completion, semantic repair, response continuation, or
inference about an incomplete item. Existing support minima remain binding, so
insufficient complete valid items still fail closed.

## Exact-10 Serving Gate

- Model: `qwen/qwen3.7-plus`, nonreasoning, seed `62000`.
- Same two initial, four one-observation, and four two-observation generic
  histories as the linked-support V2 smoke.
- Exactly 10 accepted requests, 10 HTTP attempts, zero retry/provider retry,
  zero reasoning/forced exit, cost at most `$0.10`.
- Both initial supports have at least 16 valid hypotheses; every conditioned
  draw has at least four; merged first/second supports have at least eight/four.
- All ten live smoke responses must be valid strict JSON. Synthetic unit tests,
  rather than a paid malformed response, verify deterministic item isolation.

A full pass alone authorizes a separately frozen fresh 32-tree confirmation.
Failure closes this codec/model route without prompt, seed, threshold, or
response repair.

## Conditional Fresh Cohort

If the smoke passes, freeze and run fresh tree seeds `62100..62131`, target
seeds `62200..62231`, eight validation-support seeds per tree from
`62300..62555`, and 20,000 bootstrap draws with seed `62600`. Use the same
retained-rejuvenation policy and exact 33-concept canonical endpoint.

The prospective primary will be first-link fidelity against myopic EIG:

- at least 28 changed roots;
- mean realized changed-root advantage at least `0.008` and bootstrap lower
  bound above zero;
- Spearman at least `0.25` and bootstrap lower bound above zero;
- wins minus losses at least eight;
- policy Brier reduction versus myopic at least 8%, paired tree interval below
  zero, and at least 20 wins.

Transport and support gates will require exactly 1,856 accepted scientific
requests, at most 16 transparent provider retries, no reasoning/forced exits,
at most eight item-salvaged responses, all original support minima, and a
`$3.25` run cap. Depth two and other policy controls remain diagnostic.

The cohort is independent of every prior tree. It cannot rescue or reclassify
the failed 64-tree run.
