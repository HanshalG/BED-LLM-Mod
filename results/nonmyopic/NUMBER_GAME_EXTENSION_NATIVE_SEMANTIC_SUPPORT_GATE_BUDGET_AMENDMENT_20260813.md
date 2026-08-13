# Number Game Extension-Native Semantic-Support Gate Budget Amendment

Date frozen: 2026-08-13, before implementation and before any response under the
extension-native semantic-support interface.

## Reason

The parent protocol froze a `$0.08` stage cap while allowing 7,000 proposal-output
tokens and 3,000 audit-output tokens per request. At the frozen request counts and
catalog prices, those output ceilings alone can exceed the stage cap. That conflicts
with the account-wide rule that the full worst-case exposure must be reserved before
dispatch.

## Binding Change

Only the per-request output ceilings and the dependency-valid reservation ordering
change:

- proposal requests: 3,300 output tokens rather than 7,000;
- semantic-audit requests: 1,200 output tokens rather than 3,000;
- proposal exposure is authorized before proposal dispatch; after all proposals parse,
  the exact description-dependent audit messages are constructed and audit exposure is
  authorized before audit dispatch.

All models, nonreasoning modes, prompts, schemas, histories, seeds, probes, request
counts, zero-retry rule, concurrency, parsers, thresholds, privacy rules, and decision
rules in the parent protocol remain unchanged.

## Worst-Case Authorization

Immediately before each phase, the executor must reread the phase's exact model catalog
entry and authenticated account usage. It constructs all ten exact request payloads
for that phase and computes conservative exposure as:

```text
sum(UTF-8 bytes in exact serialized messages) * input_price_per_token
+ request_count * output_token_cap * output_price_per_token
```

Treating every UTF-8 input byte as one billable token is deliberately conservative.
Before proposals, proposal exposure must be at most `$0.08`. Before audits, the
accepted proposal cost plus exact audit exposure must be at most `$0.08`. At both
points, live account-wide London-day spend plus all still-authorized exposure must be
at most `$5.00`. The executor also rereads usage and reserves the individual request's
full exposure immediately before every paid HTTP attempt. Any catalog, route, usage,
balance, payload-count, hash, or exposure mismatch fails before that phase or request.

This amendment may only reduce available completion length. It does not authorize a
retry, prompt repair, partial semantic judgment, threshold change, or a second run.
