# Bamboogle Cached-Search v2 Contingent Preregistration

## Motivation

Version 1 stopped after one of five prompt-only initial responses appended one
extra closing brace. Four responses were exact; no retrieval or scientific
endpoint ran. Version 2 changes only transport to the repository's existing
provider-enforced strict JSON Schema path. It does not relax parsing, extract
an object, repair a response, change a prompt, alter support size, or change a
scientific threshold.

This document freezes both phases before either phase makes a call.

## Phase A: Structured-Serving Gate

- Same five released mechanics tasks and non-reasoning `openai/gpt-5.4`.
- Same initial messages, temperature `0.0`, and 512-token maximum.
- Exactly five physical calls through
  `chat_complete_messages_batched_structured`.
- Strict JSON Schema requires the same 24 named fields, disallows additional
  properties, constrains weights to integers in `1..100`, and constrains
  queries to nonempty strings of at most 200 characters.
- The unchanged parser additionally requires eight normalized-unique
  hypotheses, weights summing exactly to 100, and all eight root/fixed queries
  to be normalized-unique.
- Zero model retries, response repairs, reasoning tokens, or forced exits.
- No Wikipedia request and no scientific endpoint.
- Cost cap: `$0.15`.

Phase A passes only if all five responses parse and every serving/count/cost
condition holds. Failure closes v2. There is no third serving attempt.

## Phase B: Mechanics v2

Phase B runs only if Phase A passes. Its complete scientific protocol and all
eleven conjunctive gates are exactly those in
`BAMBOOGLE_CACHED_SEARCH_MECHANICS_PREREGISTRATION.md`.

The only implementation change is that each of the four model batches uses a
phase-specific strict JSON Schema:

- initial belief/root/fixed-query schema;
- root-refresh belief/adaptive-query schema; and
- terminal-belief schema for both adaptive and fixed terminals.

The exact model-call count remains `65`; retrieval count remains `60`;
scientific/model retries and repairs remain zero; the cost cap remains
`$0.75`. Phase A responses are a serving-only diagnostic and are not reused
by Phase B. Phase B gets fresh independent initial calls so its raw tree is
internally coherent.

Passing Phase B authorizes only a separately committed opportunity20
protocol. Development and holdout values remain sealed. No OatML resource is
used in either phase.
