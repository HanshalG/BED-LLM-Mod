# Bongard OpenWorld Luna Transport-Retry Amendment

Date frozen: 2026-08-08, before any Bongard serving, mechanics,
development, or confirmation response.

## Motivation

The independent Aug 8 RegretBench chain failed before a scientific result when
one DeepSeek response ended with `IncompleteRead` after transport retries. The
unopened Bongard implementation currently makes `zero_retries` and exact
HTTP-attempt equality scientific gates. That converts transient network
delivery into a semantic null even when the final accepted responses are
complete, strict, and otherwise valid.

Transport reliability is not the estimand. A retry is acceptable only when it
is triggered before a usable response is available and repeats the same model,
messages, structured schema, request seed, and decoding configuration. It may
not depend on response content or endpoint labels.

## Frozen Change

For Bongard Luna serving, mechanics, development, and confirmation, replace
the three gates requiring exact HTTP attempts, zero retries, and zero provider
error retries with the following conjunction:

1. Usage counts are non-negative integers.
2. The number of accepted responses exactly equals the preregistered expected
   response count.
3. `http_attempts == accepted_responses + retry_count`.
4. `retry_count <= max(4, ceil(0.02 * accepted_responses))`.
5. `0 <= provider_error_retries <= retry_count`.

Frozen manifests separately report accepted-response maxima and worst-case HTTP
attempt maxima. Development blocks bind `344` accepted responses and `351`
attempts; confirmation blocks bind `688` accepted responses and `702` attempts.
At `$0.004` per reserved attempt, the confirmation maximum exposure is
`$2.808`, below the unchanged `$4.75` run cap and `$5.00` daily cap.

The existing adapter performs transport retries inside one request method and
reuses the exact serialized request. Its structured provider-error retry also
reuses the same payload and is allowed only for a zero-cost response with
`finish_reason=error`. The hard per-attempt reservation is acquired separately
for every HTTP attempt.

## Unchanged Boundaries

This amendment does not change:

- tasks, partitions, images, prompts, histories, hypotheses, or labels;
- models, reasoning settings, seeds, temperature, response schemas, or token
  limits;
- accepted-response counts or scientific endpoint counts;
- dynamic, myopic, fixed, matched-update, shuffled, history-blind, or random
  policies;
- common-random-number pairing or task-preserving terminal batches;
- parsing, semantic-support, branch-obedience, terminal-obedience, privacy, or
  finite-score gates;
- efficacy thresholds, bootstrap seeds, confirmation authorization, or claim
  tiers;
- per-request reservations, run caps, or the account-wide `$5.00` daily cap.

Retries remain fully reported in public usage records. Exceeding the frozen
bound, violating attempt accounting, returning a malformed response, or
exceeding any budget still fails closed. A transport failure that never yields
the exact accepted-response set still banks a failure and cannot be resumed or
rerun in place.

## Interpretation

The amendment removes an infrastructure-only veto; it cannot turn a malformed,
semantically invalid, unpaired, privacy-violating, or scientifically negative
run into a pass. Zero retries remains the preferred observed outcome and is
reported descriptively, but is no longer itself a scientific requirement.
